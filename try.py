import torch
import torch.nn as nn
import numpy as np

# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])
#
# grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")  # vs indexing="xy"
#
# print(grid_x)
# print(grid_y)

input = torch.randn(1, 128, 160, 160)


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv = nn.Conv2d(128, 256, 3, 2, 1)

    def forward(self, x):
        return self.cv(x)


test = Test()
output = test(input)

print(output.shape)
