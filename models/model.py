import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from models.mask import ThresholdFormer
from models.refine import Refiner


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mask = ThresholdFormer(num_trans_blocks=3)

        self.refine = Refiner()

    def forward(self, bin_x, x):
        mask = self.mask(bin_x)

        x_res = torch.cat((x, mask), dim=1)

        res = self.refine(x_res)

        return res


if __name__ == '__main__':
    model = Model().cuda()
    img = Image.open('test.jpg').convert('RGB')
    img = TF.to_tensor(img).cuda()
    img = TF.resize(img, (512, 512)).unsqueeze(0)
    g_img = TF.rgb_to_grayscale(img)
    out = model(g_img, img)
    print(out.shape)
