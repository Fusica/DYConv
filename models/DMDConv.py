import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.init import calculate_gain

import time
import thop


class FilterNorm(nn.Module):
    def __init__(self, in_channels, kernel_size, nonlinearity='linear', running_std=False, running_mean=False):
        assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        b = x.size(0)
        c = self.in_channels
        x = x.reshape(b, c, -1)
        x = x - x.mean(dim=2).reshape(b, c, 1)
        x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
        x = x.reshape(b, -1)
        if self.runing_std:
            x = x * self.std[None, :]
        else:
            x = x * self.std
        if self.runing_mean:
            x = x + self.mean[None, :]

        return x


def build_spatial_branch(in_channels, kernel_size, head=1,
                         nonlinearity='relu', stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, head * kernel_size ** 2, 1, stride=stride)
    )


def build_channel_branch(in_channels, kernel_size,
                         nonlinearity='relu', se_ratio=0.25):
    assert se_ratio > 0
    mid_channels = int(in_channels * se_ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels, mid_channels, 1),
        nn.ReLU(True),
        nn.Conv2d(mid_channels, in_channels * kernel_size ** 2, 1))


class DDFPack(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, dilation=1, head=1,
                 se_ratio=0.2, nonlinearity='relu', kernel_combine='mul'):
        super(DDFPack, self).__init__()
        assert kernel_size > 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.head = head
        self.kernel_combine = kernel_combine

        self.spatial_branch = build_spatial_branch(
            in_channels, kernel_size, head, nonlinearity, stride)

        self.channel_branch = build_channel_branch(
            in_channels, kernel_size, nonlinearity, se_ratio)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.head
        k = self.kernel_size
        s = self.stride
        channel_filter = self.channel_branch(x).reshape(b*g, c//g, k, k)
        channel_filter = torch.sigmoid(channel_filter)
        spatial_filter = self.spatial_branch(x).reshape(b*g, -1, h//s, w//s)
        spatial_filter = torch.sigmoid(spatial_filter)
        x = x.reshape(b*g, c//g, h, w)
        aggregated_filter = channel_filter * spatial_filter
        x = x * aggregated_filter
        return x.reshape(b, c, h//s, w//s)


if __name__ == '__main__':
    x = torch.randn(4, 512, 20, 20)
    model = DDFPack(512)

    start = time.time()
    y = model(x)
    end = time.time()
    print(y.shape)
    print(end - start)

    flops, params = thop.profile(model, inputs=(x,))
    print('FLOPs: ' + str(flops) + ', Params: ' + str(params))