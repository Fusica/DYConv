import torch
import torch.nn as nn
import numpy as np

x = torch.randn(1, 128, 1, 1)

flatten = nn.Flatten()
linear = nn.Linear(128, 256)

y = linear(x)

print(y.shape)
