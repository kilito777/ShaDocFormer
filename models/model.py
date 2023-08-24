import numbers

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# Layer Norm
def to_3d(x):
    return einops.rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


# Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)

        return x


## Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class MaskFormer(nn.Module):
    def __init__(self, num_trans_blocks):
        super(MaskFormer, self).__init__()

        model = [nn.Conv2d(1, 4, 3, padding=1),
                 nn.InstanceNorm2d(4),
                 nn.LeakyReLU(),
                 nn.Conv2d(4, 16, 3, padding=1),
                 nn.LeakyReLU()]

        for _ in range(num_trans_blocks):
            model += [
                TransformerBlock(dim=16, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')]

        model += [nn.Conv2d(16, 4, 3, padding=1),
                  nn.LeakyReLU(),
                  nn.Conv2d(4, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, bin_x):
        flattened = bin_x.view(-1)

        hist = torch.histc(flattened, bins=256, min=0, max=1)

        pmf = hist / hist.sum()
        cdf = torch.cumsum(pmf, dim=0)

        cumulative_means = torch.cumsum(pmf * torch.arange(0, 256).to(bin_x.device), dim=0)
        total_mean = cumulative_means[-1]
        inter_class_variances = (total_mean * cdf - cumulative_means) ** 2 / (cdf * (1.0 - cdf) + 1e-12)
        otsu_threshold = torch.argmax(inter_class_variances) / 255

        otsu_mask = torch.ones_like(bin_x, device=bin_x.device)
        otsu_mask[bin_x > otsu_threshold] = 0

        trans_mask = torch.sigmoid(self.model(otsu_mask))
        trans_threshold = F.avg_pool2d(trans_mask, kernel_size=trans_mask.shape[2:]).item()

        trans_mask = torch.ones_like(bin_x, device=bin_x.device)
        trans_mask[bin_x > trans_threshold] = 0

        res_mask = torch.clamp(otsu_mask + trans_mask, min=0, max=1)

        return res_mask


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, bias=True, groups=1, norm='in',
                 nonlinear='relu'):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias,
                                dilation=dilation)
        self.norm = norm
        self.nonlinear = nonlinear

        if norm == 'bn':
            self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.normalization = nn.InstanceNorm2d(out_channels, affine=False)
        else:
            self.normalization = None

        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            self.activation = None

    def forward(self, x):
        out = self.conv2d(self.reflection_pad(x))
        if self.normalization is not None:
            out = self.normalization(out)
        if self.activation is not None:
            out = self.activation(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, channels, k, nonlinear='relu'):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.k = k
        self.nonlinear = nonlinear

        self.linear1 = nn.Linear(channels, channels // k)
        self.linear2 = nn.Linear(channels // k, channels)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            raise ValueError

    def attention(self, x):
        N, C, H, W = x.size()
        out = torch.flatten(self.global_pooling(x), 1)
        out = self.activation(self.linear1(out))
        out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)

        return out.mul(x)

    def forward(self, x):
        return self.attention(x)


class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Aggreation, self).__init__()
        self.attention = SelfAttention(in_channels, k=8, nonlinear='relu')
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,
                              nonlinear='leakyrelu',
                              norm=None)

    def forward(self, x):
        return self.conv(self.attention(x))


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, interpolation_type='bilinear'):
        super(SPP, self).__init__()
        self.conv = nn.ModuleList()
        self.num_layers = num_layers
        self.interpolation_type = interpolation_type

        for _ in range(self.num_layers):
            self.conv.append(
                ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, dilation=1, nonlinear='leakyrelu',
                          norm=None))

        self.fusion = ConvLayer((in_channels * (self.num_layers + 1)), out_channels, kernel_size=3, stride=1,
                                norm='False', nonlinear='leakyrelu')

    def forward(self, x):

        N, C, H, W = x.size()
        out = []

        for level in range(self.num_layers):
            out.append(F.interpolate(self.conv[level](
                F.avg_pool2d(x, kernel_size=2 * 2 ** (level + 1), stride=2 * 2 ** (level + 1),
                             padding=2 * 2 ** (level + 1) % 2)), size=(H, W), mode=self.interpolation_type))

        out.append(x)

        return self.fusion(torch.cat(out, dim=1))


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


# Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim, bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()

        dim = 16
        self.patch_embed = OverlapPatchEmbed(3, dim)

        self.down_1 = Downsample(dim)
        self.down_2 = Downsample(dim * 2)
        self.down_3 = Downsample(dim * 2 * 2)

        self.block_down_1_1 = ConvLayer(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, dilation=8,
                                        norm=None, nonlinear='leakyrelu')
        self.block_down_1_2 = ConvLayer(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, dilation=8,
                                        norm=None, nonlinear='leakyrelu')
        self.aggreation_down_1 = Aggreation(in_channels=dim * 2 * 3, out_channels=dim * 2)

        self.block_down_2_1 = ConvLayer(in_channels=dim * 2 * 2, out_channels=dim * 2 * 2, kernel_size=3, stride=1,
                                        dilation=4,
                                        norm=None, nonlinear='leakyrelu')
        self.block_down_2_2 = ConvLayer(in_channels=dim * 2 * 2, out_channels=dim * 2 * 2, kernel_size=3, stride=1,
                                        dilation=4,
                                        norm=None, nonlinear='leakyrelu')
        self.aggreation_down_2 = Aggreation(in_channels=dim * 2 * 2 * 3, out_channels=dim * 2 * 2)

        self.block_down_3_1 = ConvLayer(in_channels=dim * 2 * 4, out_channels=dim * 2 * 4, kernel_size=3, stride=1,
                                        dilation=2,
                                        norm=None, nonlinear='leakyrelu')
        self.block_down_3_2 = ConvLayer(in_channels=dim * 2 * 4, out_channels=dim * 2 * 4, kernel_size=3, stride=1,
                                        dilation=2,
                                        norm=None, nonlinear='leakyrelu')
        self.aggreation_down_3 = Aggreation(in_channels=dim * 2 * 4 * 3, out_channels=dim * 2 * 4)

        # self.middle = SPP(in_channels=dim * 2 * 4, out_channels=dim * 2 * 4, num_layers=4, interpolation_type='bicubic')

        self.middle = nn.Sequential(*[NAFBlock(c=dim * 2 * 4) for _ in range(3)])

        self.block_up_2_1 = ConvLayer(in_channels=dim * 2 * 2, out_channels=dim * 2 * 2, kernel_size=3, stride=1,
                                      dilation=2,
                                      norm=None, nonlinear='leakyrelu')
        self.block_up_2_2 = ConvLayer(in_channels=dim * 2 * 2, out_channels=dim * 2 * 2, kernel_size=3, stride=1,
                                      dilation=2,
                                      norm=None, nonlinear='leakyrelu')
        self.aggreation_up_2 = Aggreation(in_channels=dim * 2 * 2 * 3, out_channels=dim * 2 * 2)

        self.block_up_1_1 = ConvLayer(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, dilation=4,
                                      norm=None, nonlinear='leakyrelu')
        self.block_up_1_2 = ConvLayer(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, dilation=4,
                                      norm=None, nonlinear='leakyrelu')
        self.aggreation_up_1 = Aggreation(in_channels=dim * 2 * 3, out_channels=dim * 2)

        self.coefficient_3_2 = nn.Parameter(torch.ones((2, dim * 2 * 2)), requires_grad=True)
        self.coefficient_2_1 = nn.Parameter(torch.ones((2, dim * 2)), requires_grad=True)
        self.coefficient_1_0 = nn.Parameter(torch.ones((2, dim)), requires_grad=True)

        # self.latent = nn.Sequential(
        #     *[TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
        #       for _ in range(1)])
        self.latent = SPP(in_channels=dim, out_channels=dim, num_layers=4, interpolation_type='bicubic')

        self.skip_3_2 = nn.Conv2d(dim * 2 * 2, dim * 2 * 2, kernel_size=1, bias=False)
        self.skip_2_1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False)

        self.up_3 = Upsample(int(dim * 2 * 4))
        self.up_2 = Upsample(int(dim * 2 * 2))
        self.up_1 = Upsample(int(dim * 2))

        self.refinement = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for _ in range(1)])

        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        inp_enc = self.patch_embed(x)

        inp_enc_level1_0 = self.down_1(inp_enc)

        out_enc_level1_0 = self.block_down_1_1(inp_enc_level1_0)
        out_enc_level1_1 = self.block_down_1_2(out_enc_level1_0)
        out_agg_down_1 = self.aggreation_down_1(
            torch.cat((inp_enc_level1_0, out_enc_level1_0, out_enc_level1_1), dim=1))

        inp_enc_level2_0 = self.down_2(out_agg_down_1)

        out_enc_level2_0 = self.block_down_2_1(inp_enc_level2_0)
        out_enc_level2_1 = self.block_down_2_2(out_enc_level2_0)
        out_agg_down_2 = self.aggreation_down_2(
            torch.cat((inp_enc_level2_0, out_enc_level2_0, out_enc_level2_1), dim=1))

        inp_enc_level3_0 = self.down_3(out_agg_down_2)

        out_enc_level3_0 = self.block_down_3_1(inp_enc_level3_0)
        out_enc_level3_1 = self.block_down_3_2(out_enc_level3_0)
        out_agg_down_3 = self.aggreation_down_3(
            torch.cat((inp_enc_level3_0, out_enc_level3_0, out_enc_level3_1), dim=1))

        out_agg_down_3 = self.middle(out_agg_down_3)

        inp_enc_level3_1 = self.up_3(out_agg_down_3)

        inp_enc_level2_1 = self.coefficient_3_2[0, :][None, :, None, None] * out_agg_down_2 + self.coefficient_3_2[1,
                                                                                              :][None, :, None,
                                                                                              None] * inp_enc_level3_1
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)

        out_enc_level2_0 = self.block_up_2_1(inp_enc_level2_1)
        out_enc_level2_1 = self.block_up_2_2(out_enc_level2_0)
        out_agg_up_2 = self.aggreation_up_2(torch.cat((inp_enc_level2_1, out_enc_level2_0, out_enc_level2_1), dim=1))

        out_enc_level2_1 = self.up_2(out_agg_up_2)

        inp_enc_level1_1 = self.coefficient_2_1[0, :][None, :, None, None] * out_agg_down_1 + self.coefficient_2_1[1,
                                                                                              :][None, :, None,
                                                                                              None] * out_enc_level2_1
        inp_enc_level1_1 = self.skip_2_1(inp_enc_level1_1)

        out_enc_level1_0 = self.block_up_1_1(inp_enc_level1_1)
        out_enc_level1_1 = self.block_up_1_2(out_enc_level1_0)
        out_agg_up_1 = self.aggreation_up_1(torch.cat((inp_enc_level1_1, out_enc_level1_0, out_enc_level1_1), dim=1))

        out_enc_level1_1 = self.up_1(out_agg_up_1)

        out_fusion = self.latent(inp_enc)

        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion + self.coefficient_1_0[1, :][None, :, None,
                                                                             None] * out_enc_level1_1

        out = self.refinement(out)

        out = torch.sigmoid(self.output(out))

        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mask = MaskFormer(num_trans_blocks=3)

        self.refine = Refine()

    def forward(self, bin_x, x):
        mask = self.mask(bin_x)

        x_res = mask * x

        x_res = self.refine(x_res)

        res = x_res * mask + x * (1 - mask)

        return res


if __name__ == '__main__':
    model = Model().cuda()
    img = Image.open('test.jpg').convert('RGB')
    img = TF.to_tensor(img).cuda()
    img = TF.resize(img, (512, 512)).unsqueeze(0)
    g_img = TF.rgb_to_grayscale(img)
    out = model(g_img, img)
    print(out.shape)
