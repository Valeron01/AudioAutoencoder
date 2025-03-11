import einops
import torch
from torch import nn


def get_up_down_block(in_channels, out_channels, stride, kernel_size):
    if stride < 1:
        upsample_rate = int(1 / stride)
        return nn.Sequential(
            nn.Upsample(scale_factor=upsample_rate),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding="same")
        )
    else:
        stride = int(stride)
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        self.block = nn.Sequential(
            get_up_down_block(in_channels, out_channels, stride, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
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
            nn.LeakyReLU(inplace=True)
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
            n_channels_list,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim
    ):
        super().__init__()
        assert len(kernel_sizes) == len(strides) == (len(n_channels_list) - 1)

        self.stem = nn.Sequential(
            nn.Conv1d(1, n_channels_list[0], 9, 1, "same"),
            nn.BatchNorm1d(n_channels_list[0]),
            nn.LeakyReLU(inplace=True)
        )

        self.downsample_blocks = nn.Sequential(
            *[ResidualBlock(in_channels, out_channels, stride, kernel_size) for
              stride, kernel_size, in_channels, out_channels in zip(
                    strides, kernel_sizes, n_channels_list[:-1], n_channels_list[1:]
                )],
            LearnableConvolutionalEmbedding(n_channels_list[-1], 129, 16),
        )
        self.transformer = nn.Sequential(
            *[TransformerBlock(n_channels_list[-1], n_heads) for _ in range(n_transformer_blocks)])
        self.conv_out = nn.Conv1d(n_channels_list[-1], z_dim, 1, 1)

    def forward(self, x, return_features=False):
        if not return_features:
            stem = self.stem(x)
            conv_features = self.downsample_blocks(stem)
            transformer = self.transformer(conv_features)
            z = self.conv_out(transformer)
            return z
        else:
            stem = self.stem(x)
            intermediate_features = [stem]
            block_result = stem
            for block in self.downsample_blocks:
                block_result = block(block_result)
                intermediate_features.append(block_result)

            for block in self.transformer:
                block_result = block(block_result)
                intermediate_features.append(block_result)
            z = self.conv_out(block_result)

            return z, intermediate_features


class Decoder(nn.Module):
    def __init__(
            self,
            n_channels_list,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim
    ):
        super().__init__()
        assert len(kernel_sizes) == len(strides)

        self.stem = nn.Sequential(
            nn.Conv1d(z_dim, n_channels_list[0], 1, 1),
            nn.GroupNorm(4, n_channels_list[0]),
            nn.LeakyReLU(inplace=True)
        )
        self.transformer = nn.Sequential(
            LearnableConvolutionalEmbedding(n_channels_list[0], 7, 16),
            *[TransformerBlock(n_channels_list[0], n_heads) for _ in range(n_transformer_blocks)]
        )

        self.upsample_blocks = nn.Sequential(
            *[
                ResidualBlock(in_channels, out_channels, stride, kernel_size) for
                stride, kernel_size, in_channels, out_channels in zip(
                    strides, kernel_sizes, n_channels_list[:-1], n_channels_list[1:]
                )]
        )

        self.conv_out = nn.Conv1d(n_channels_list[-1], 1, 15, 1, padding="same")

    def forward(self, x):
        stem = self.stem(x)
        transformer = self.transformer(stem)
        conv_features = self.upsample_blocks(transformer)
        z = self.conv_out(conv_features)
        return z


class AudioVAE(nn.Module):
    def __init__(
            self,
            n_channels_list,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim
    ):
        super().__init__()
        self.encoder = Encoder(n_channels_list, kernel_sizes, strides, n_transformer_blocks, n_heads, z_dim * 2)
        upsample_strides = [1 / i for i in reversed(strides)]
        upsample_kernels = list(reversed(kernel_sizes))

        self.decoder = Decoder(n_channels_list, upsample_kernels, upsample_strides, n_transformer_blocks, n_heads, z_dim)

    def encode(self, x):
        return self.encoder(x[:, None]).chunk(2, 1)

    def decode(self, z):
        return self.decoder(z).squeeze(1)


if __name__ == '__main__':
    n_channels_list = [64, 128, 128, 256, 256, 384, 384, 512, 512, 512, 512, 768, 768, 768]
    strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    kernel_sizes = [3] * len(strides)
    model = AudioVAE(n_channels_list, kernel_sizes, strides, 1, 8, 2).cuda().eval()
    with torch.autocast("cuda", torch.float16), torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        source = torch.rand(4, 16384 * 6).cuda()
        z = model.encode(source)[0]
        decoded = model.decode(z)
        print(z.shape)
        print(decoded.shape)
        print(decoded.mean())
        print(16384 * 12)
        assert source.shape == decoded.shape
        print(decoded.numel() / z.numel())
        print(decoded.shape[1] / z.shape[2])
