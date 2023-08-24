import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import einops
import numbers
import math


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


class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Aggreation, self).__init__()
        self.attention = SelfAttention(in_channels, k=8, nonlinear='relu')
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,
                              nonlinear='PReLU',
                              norm='in')

    def forward(self, x):
        return self.conv(self.attention(x))


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


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, interpolation_type='bilinear'):
        super(SPP, self).__init__()
        self.conv = nn.ModuleList()
        self.num_layers = num_layers
        self.interpolation_type = interpolation_type

        for _ in range(self.num_layers):
            self.conv.append(
                ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, dilation=1, nonlinear='PReLU',
                          norm=None))

        self.fusion = ConvLayer((in_channels * (self.num_layers + 1)), out_channels, kernel_size=3, stride=1,
                                norm='False', nonlinear='PReLU')

    def forward(self, x):

        N, C, H, W = x.size()
        out = []

        for level in range(self.num_layers):
            out.append(F.interpolate(self.conv[level](
                F.avg_pool2d(x, kernel_size=2 * 2 ** (level + 1), stride=2 * 2 ** (level + 1),
                             padding=2 * 2 ** (level + 1) % 2)), size=(H, W), mode=self.interpolation_type))

        out.append(x)

        return self.fusion(torch.cat(out, dim=1))


# ResMLP's normalization
class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


# Color Normalization
class Aff_channel(nn.Module):
    def __init__(self, dim, channel_first=True):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2


class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock_ln(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=Aff_channel, init_values=1e-4):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.norm1 = Aff_channel(dim)
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = Aff_channel(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        # print(x.shape)
        norm_x = x.flatten(2).transpose(1, 2)
        # print(norm_x.shape)
        norm_x = self.norm1(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = x + self.drop_path(self.gamma_1 * self.conv2(self.attn(self.conv1(norm_x))))
        norm_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm2(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.gamma_2 * self.mlp(norm_x))
        return x


class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            # nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(out_channels // 2),
            # nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DHAN(nn.Module):
    def __init__(self, channels=16):
        super(DHAN, self).__init__()

        self.fusion = ConvLayer(in_channels=3, out_channels=channels, kernel_size=1, stride=1, norm='in',
                                nonlinear='PReLU')

        # Stage0
        self.block0_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, norm='in',
                                  nonlinear='PReLU')
        self.block0_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, norm='in',
                                  nonlinear='PReLU')
        # self.block0_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,
        # 1), padding = (1,1))

        self.aggreation0_rgb = Aggreation(in_channels=channels * 2, out_channels=channels)
        self.aggreation0_mas = Aggreation(in_channels=channels * 2, out_channels=channels)

        # Stage1
        self.block1_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=2,
                                  norm='in', nonlinear='PReLU')
        self.block1_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=4,
                                  norm='in', nonlinear='PReLU')
        # self.block1_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,
        # 1), padding = (4,4), dilation = (4,4))

        self.aggreation1_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        self.aggreation1_mas = Aggreation(in_channels=channels * 3, out_channels=channels)

        # Stage2
        self.block2_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=8,
                                  norm='in', nonlinear='PReLU')
        self.block2_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=16,
                                  norm='in', nonlinear='PReLU')
        # self.block2_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,
        # 1), padding = (16,16), dilation = (16,16))

        self.aggreation2_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        self.aggreation2_mas = Aggreation(in_channels=channels * 3, out_channels=channels)

        # Stage3
        self.block3_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=32,
                                  norm='in', nonlinear='PReLU')
        self.block3_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=64,
                                  norm='in', nonlinear='PReLU')
        # self.block3_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,
        # 1), padding = (64,64), dilation = (64,64))

        self.aggreation3_rgb = Aggreation(in_channels=channels * 4, out_channels=channels)
        self.aggreation3_mas = Aggreation(in_channels=channels * 4, out_channels=channels)

        # Stage4
        self.spp_img = SPP(in_channels=channels, out_channels=channels, num_layers=4, interpolation_type='bicubic')
        self.spp_mas = SPP(in_channels=channels, out_channels=channels, num_layers=4, interpolation_type='bicubic')

        self.block4_1 = nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, inp):
        out = self.fusion(inp)

        # Stage0
        out0_1 = self.block0_1(out)
        out0_2 = self.block0_2(out0_1)

        agg0_rgb = self.aggreation0_rgb(torch.cat((out0_1, out0_2), dim=1))
        agg0_mas = self.aggreation0_mas(torch.cat((out0_1, out0_2), dim=1))

        out0_2 = agg0_rgb.mul(torch.sigmoid(agg0_mas))
        # out0_2 = self.dropout(out0_2)

        # Stage1
        out1_1 = self.block1_1(out0_2)
        out1_2 = self.block1_2(out1_1)

        agg1_rgb = self.aggreation1_rgb(torch.cat((agg0_rgb, out1_1, out1_2), dim=1))
        agg1_mas = self.aggreation1_mas(torch.cat((agg0_mas, out1_1, out1_2), dim=1))

        out1_2 = agg1_rgb.mul(torch.sigmoid(agg1_mas))
        # out1_2 = self.dropout(out1_2)

        # Stage2
        out2_1 = self.block2_1(out1_2)
        out2_2 = self.block2_2(out2_1)

        agg2_rgb = self.aggreation2_rgb(torch.cat((agg1_rgb, out2_1, out2_2), dim=1))
        agg2_mas = self.aggreation2_mas(torch.cat((agg1_mas, out2_1, out2_2), dim=1))

        out2_2 = agg2_rgb.mul(torch.sigmoid(agg2_mas))
        # out2_2 = self.dropout(out2_2)

        # Stage3
        out3_1 = self.block3_1(out2_2)
        out3_2 = self.block3_2(out3_1)

        agg3_rgb = self.aggreation3_rgb(torch.cat((agg1_rgb, agg2_rgb, out3_1, out3_2), dim=1))
        agg3_mas = self.aggreation3_mas(torch.cat((agg1_rgb, agg2_rgb, out3_1, out3_2), dim=1))

        # Stage4
        spp_rgb = self.spp_img(agg3_rgb)
        spp_mas = self.spp_mas(agg3_mas)

        spp_rgb = spp_rgb.mul(torch.sigmoid(spp_mas))

        out_rgb = (self.block4_1(spp_rgb))

        alpha = torch.sigmoid(out_rgb[:, -1, :, :].unsqueeze(1))

        out_rgb = inp.mul(alpha).add(out_rgb[:, :-1, :, :].mul(1 - alpha)).clamp(0, 1)

        return out_rgb


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


# Dual Gated Feed-Forward Networ
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


# Axis-based Multi-head Self-Attention

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


# Axis-based Transformer Block
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


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


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
    def __init__(self, c=48, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
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


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__()

        dim = 32

        pre = [nn.Conv2d(3, 16, 3, padding=1),
               nn.InstanceNorm2d(16),
               nn.LeakyReLU(),
               nn.Conv2d(16, dim, 3, padding=1),
               nn.LeakyReLU()]

        for _ in range(5):
            pre += [ResidualBlock(dim)]

        self.pre = nn.Sequential(*pre)

        self.naf = nn.Sequential(*[NAFBlock(c=dim) for _ in range(3)])
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66,
                             bias=False, LayerNorm_type='WithBias') for _ in range(3)])

        pos = [nn.Conv2d(dim * 2, 16, 3, padding=1),
               nn.LeakyReLU(),
               nn.Conv2d(16, 3, 3, padding=1)]
        self.pos = nn.Sequential(*pos)

    def forward(self, inp):
        inter = self.pre(inp)
        res_1 = self.naf(inter)
        res_2 = self.transformer(res_1)
        res = self.pos(torch.cat((res_1, res_2), dim=1))
        return res


class LocalNet(nn.Module):
    def __init__(self, in_dim=3, dim=16):
        super(LocalNet, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1),
                   nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU()]

        self.branch1 = nn.Sequential(*blocks1)
        self.branch2 = DHAN()

        self.conv_fuss = nn.Conv2d(int(in_dim * 2), in_dim, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inp):
        # short cut connection
        branch1 = self.branch1(self.relu(self.conv1(inp)))

        branch2 = self.branch2(inp)

        res = self.conv_fuss(torch.cat((branch1, branch2), dim=1))

        return res


if __name__ == '__main__':
    model = GlobalNet().cuda()
    t = torch.randn(1, 3, 256, 256).cuda()
    out1 = model(t)
    print(out1.shape)
