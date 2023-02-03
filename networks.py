import typing

import torch
from torch import nn
import pytorch_lightning as pl
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


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super(LitAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 15, 1, 7),  # 256
            nn.ReLU(inplace=True),

            ResBlock(64, 128, (2, 2)),  # 128
            ResBlock(128, 256, (2, 2)),  # 64
            ResBlock(256, 512, (2, 2)),  # 32
            ResBlock(512, 512, (2, 2)),  # 16
            ResBlock(512, 512, (2, 2)),  # 8
            # ResBlock(1024, 1024, (2, 2)),  # 4
            # ResBlock(1024, 1024),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            # ResBlock(1024, 1024),  # 4
            # nn.ConvTranspose2d(1024, 1024, 2, 2),  # 8

            ResBlock(512, 512),
            nn.ConvTranspose2d(512, 512, 2, 2),  # 16

            ResBlock(512, 512),
            nn.ConvTranspose2d(512, 256, 2, 2),   # 32

            ResBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 2, 2),   # 64

            ResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 2, 2),  # 128

            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 2, 2),  # 256

            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


if __name__ == '__main__':
    ae = LitAutoEncoder()
    inputs = torch.randn(10, 1, 256, 256)
    result = ae(inputs)
    print(result.shape)
