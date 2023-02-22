import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFENaive(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=2):
        super(CARAFENaive, self).__init__()
        self.scale_factor = scale_factor
        self.comp_factor = scale_factor * scale_factor
        self.conv = nn.Conv2d(in_planes, out_planes * self.comp_factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Upsample the low-resolution feature map
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        # Use convolution to obtain content-aware masks
        mask = self.conv(x_up)
        mask = mask.view(b, self.comp_factor, -1, h, w)
        mask = F.softmax(mask, dim=1)

        # Reassemble the low-resolution features in a content-aware way
        x_reassembled = torch.einsum('bkihw, bkchw->bchw', mask, x_up.view(b, self.comp_factor, c, h, w))

        return x_reassembled


class CARAFEScale(nn.Module):
    def __init__(self, in_planes, out_planes, up_factor):
        super(CARAFEScale, self).__init__()

        # Define the convolutional layers for each scale
        self.scales = nn.ModuleList()
        for scale in range(up_factor):
            kernel_size = 3 + scale * 2
            padding = (kernel_size - 1) // 2
            scale_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1),
                nn.Conv2d(out_planes, out_planes * scale ** 2, kernel_size=kernel_size, padding=padding),
                nn.PixelShuffle(scale),
                nn.Conv2d(out_planes, out_planes, kernel_size=1)
            )
            self.scales.append(scale_conv)

    def forward(self, x):
        # Compute the scale indices based on the upscaling factor
        n, c, h, w = x.size()
        up_factor = len(self.scales)
        scale_indices = [i * 2 + 1 for i in range(up_factor)]

        # Compute the output for each scale
        scale_outputs = []
        for scale, scale_conv in enumerate(self.scales):
            scale_input = x[:, :, ::scale_indices[scale], ::scale_indices[scale]]
            scale_output = scale_conv(scale_input)
            scale_outputs.append(scale_output)

        # Compute the content-aware masks and reassemble the feature map
        content_aware_masks = torch.stack(scale_outputs, dim=1)
        content_aware_masks = torch.softmax(content_aware_masks, dim=1)
        content_aware_masks = content_aware_masks.view(n, up_factor ** 2, h, w)
        y = torch.einsum('bnhw, bhw->bnhw', content_aware_masks, x)
        return y


class CARAFEpp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, channels_groups=4):
        super(CARAFEpp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.channels_groups = channels_groups
        self.mid_channels = out_channels * scale_factor ** 2

        # Define convolutional layers
        self.scale_conv_list = nn.ModuleList()
        for i in range(scale_factor):
            kernel_size = 3 + i * 2
            padding = (kernel_size - 1) // 2
            scale_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.mid_channels, kernel_size=1),
                nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel_size, padding=padding,
                          groups=channels_groups),
                nn.PixelShuffle(scale_factor)
            )
            self.scale_conv_list.append(scale_conv)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * (scale_factor ** 2 + 1), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Compute the output for each scale
        scale_output_list = []
        for i in range(self.scale_factor):
            scale_input = x[:, :, ::self.scale_factor, ::self.scale_factor]
            scale_output = self.scale_conv_list[i](scale_input)
            scale_output_list.append(scale_output)

        # Compute content-aware attention weights
        content_aware_list = []
        for i in range(self.scale_factor):
            content_aware = torch.einsum('bcwh, bckh -> bwhk', [x, scale_output_list[i]])
            content_aware = F.softmax(content_aware, dim=-1)
            content_aware = torch.einsum('bwhk, bckh -> bcwh', [content_aware, scale_output_list[i]])
            content_aware_list.append(content_aware)

        # Assemble the final feature map
        output = torch.cat([x] + content_aware_list, dim=1)
        output = self.bottleneck(output)
        return output


if __name__ == '__main__':
    x = torch.randn(1, 256, 80, 80)
    model = CARAFEpp(in_channels=256, out_channels=128, scale_factor=2)

    start = time.time()
    model(x)
    end = time.time()
    print(x.shape)
    print(end - start)
