import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from models.common import SPPCSPC


class Deformable_Conv2D(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super(Deformable_Conv2D, self).__init__()
        self.kernel = k
        self.stride = s
        self.padding = p
        self.conv = nn.Conv2d(c1, c2, k, s, p)  # 原卷积

        self.conv_offset = nn.Conv2d(c1, 2 * k * k, kernel_size=k, stride=s, padding=p)
        init_offset = torch.Tensor(np.zeros([2 * k * k, c1, k, k]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  # 初始化为0

        self.conv_mask = nn.Conv2d(c1, k * k, kernel_size=k, stride=s, padding=p)
        init_mask = torch.Tensor(np.zeros([k * k, c1, k, k]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  # 初始化为0.5

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))  # 保证在0到1之间
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.conv.weight, mask=mask,
                                            stride=(self.stride, self.stride), padding=(self.padding, self.padding))
        return out


class SPPCSPC_Defor(SPPCSPC):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Deformable_Conv2D(c1, c_, 1, 1, 0)
        self.cv2 = Deformable_Conv2D(c1, c_, 1, 1, 0)
        self.cv3 = Deformable_Conv2D(c_, c_, 3, 1)
        self.cv4 = Deformable_Conv2D(c_, c_, 1, 1, 0)
        self.cv5 = Deformable_Conv2D(4 * c_, c_, 1, 1, 0)
        self.cv6 = Deformable_Conv2D(c_, c_, 3, 1)
        self.cv7 = Deformable_Conv2D(2 * c_, c2, 1, 1, 0)


if __name__ == '__main__':
    x = torch.randn(1, 128, 160, 160)
    model = SPPCSPC_Defor(128, 128)

    start = time.time()
    y = model(x)
    end = time.time()
    print(y.shape)
    print(end - start)
