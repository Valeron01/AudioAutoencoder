import typing

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import pytorch_lightning.utilities.model_summary
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: typing.Tuple[int, int] = None):
        super(ResBlock, self).__init__()
        if stride is None:
            stride = (1, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_features, out_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        self.branch = nn.Conv2d(in_features, out_features, kernel_size=stride, stride=stride) \
            if stride != (1, 1) or in_features != out_features else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.branch(x)



if __name__ == '__main__':
    ae = LitAutoEncoder()
    print(pl.utilities.model_summary.ModelSummary(ae))
    inputs = torch.randn(10, 1, 192, 256)
    result = ae(inputs)
    print(result.shape)
