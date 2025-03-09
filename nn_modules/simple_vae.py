import einops
import torch
from torch import nn


def get_up_down_block(in_channels, out_channels, stride, kernel_size):
    if stride < 1:
        upsample_rate = 1 / stride
        return nn.Sequential(
            nn.Upsample(scale_factor=upsample_rate),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding="same")
        )
    else:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        self.block = nn.Sequential(
            get_up_down_block(in_channels, out_channels, stride, kernel_size),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.identity = nn.Identity() if in_channels == out_channels and stride == 1 else get_up_down_block(
            in_channels, out_channels, stride, kernel_size
        )

    def forward(self, x):
        return self.block(x) + self.identity(x)


class LearnableConvolutionalEmbedding(nn.Module):
    def __init__(self, n_channels, kernel_size, n_groups):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size, 1, padding="same", groups=n_groups)

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, inner_channels, n_heads):
        super().__init__()
        self.n_heads = n_heads

        self.conv_in = nn.Conv1d(inner_channels, inner_channels * 3, 3, 1, 1)
        self.conv_out = nn.Sequential(
            nn.Conv1d(inner_channels, inner_channels, 1),
            nn.GroupNorm(1, inner_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        b, c, l = x.shape

        q, k, v = self.conv_in(x).chunk(3, 1)

        q = einops.rearrange(q, "b (nh dh) l->b nh l dh", nh=self.n_heads).contiguous()
        k = einops.rearrange(k, "b (nh dh) l->b nh l dh", nh=self.n_heads).contiguous()
        v = einops.rearrange(v, "b (nh dh) l->b nh l dh", nh=self.n_heads).contiguous()

        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attention = einops.rearrange(attention, "b nh l dh->b (nh dh) l", b=b, l=l, nh=self.n_heads)

        return self.conv_out(attention) + x


class Encoder(nn.Module):
    def __init__(
            self,
            inner_channels,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim
    ):
        super().__init__()
        assert len(kernel_sizes) == len(strides)

        self.stem = nn.Sequential(
            nn.Conv1d(1, inner_channels, 17, 1, "same"),
            nn.GroupNorm(4, inner_channels),
            nn.SiLU(inplace=True)
        )

        self.downsample_blocks = nn.Sequential(
            LearnableConvolutionalEmbedding(inner_channels, 129, 16),
            *[ResidualBlock(inner_channels, inner_channels, stride, kernel_size) for stride, kernel_size in zip(
                strides, kernel_sizes
            )]
        )
        self.transformer = nn.Sequential(
            *[TransformerBlock(inner_channels, n_heads) for _ in range(n_transformer_blocks)])
        self.conv_out = nn.Conv1d(inner_channels, z_dim * 2, 1, 1)

    def forward(self, x):
        stem = self.stem(x)
        conv_features = self.downsample_blocks(stem)
        transformer = self.transformer(conv_features)
        z = self.conv_out(transformer).chunk(2, 1)
        return z


class Decoder(nn.Module):
    def __init__(
            self,
            inner_channels,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim
    ):
        super().__init__()
        assert len(kernel_sizes) == len(strides)

        self.stem = nn.Sequential(
            nn.Conv1d(z_dim, inner_channels, 1, 1),
            nn.GroupNorm(4, inner_channels),
            nn.SiLU(inplace=True)
        )
        self.transformer = nn.Sequential(
            LearnableConvolutionalEmbedding(inner_channels, 7, 16),
            *[TransformerBlock(inner_channels, n_heads) for _ in range(n_transformer_blocks)]
        )

        self.upsample_blocks = nn.Sequential(
            *[ResidualBlock(inner_channels, inner_channels, stride, kernel_size) for stride, kernel_size in zip(
                strides, kernel_sizes
            )]
        )

        self.conv_out = nn.Conv1d(inner_channels, 1, 15, 1, padding="same")

    def forward(self, x):
        stem = self.stem(x)
        transformer = self.transformer(stem)
        conv_features = self.upsample_blocks(transformer)
        z = self.conv_out(conv_features)
        return z


class AudioAutoencoder(nn.Module):
    def __init__(
            self, inner_channels,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim
    ):
        super().__init__()
        self.encoder = Encoder(inner_channels, kernel_sizes, strides, n_transformer_blocks, n_heads, z_dim)
        upsample_strides = [1 / i for i in reversed(strides)]
        upsample_kernels = list(reversed(kernel_sizes))

        self.decoder = Decoder(inner_channels, upsample_kernels, upsample_strides, n_transformer_blocks, n_heads, z_dim)

    def encode(self, x):
        return self.encoder(x[:, None])

    def decode(self, z):
        return self.decoder(z).squeeze(1)


if __name__ == '__main__':
    model = AudioAutoencoder(512, [11, 3, 3, 3, 3, 3, 3], [5, 2, 2, 2, 2, 2, 2], 5, 8, 8).cuda()
    with torch.autocast("cuda", torch.float16), torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        z = model.encode(torch.rand(4, 1, 16000 * 12).cuda())[0]
        print(z.shape)
        decoded = model.decode(z)
        print(decoded.shape)
        decoded.mean().backward()

