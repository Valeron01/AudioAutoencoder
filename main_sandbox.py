import torch
from torch import nn


conv = nn. ConvTranspose1d(
    64, 2, 3, 2, 1, 1
)


result = conv(torch.zeros(1, 64, 128))
print(result.shape)