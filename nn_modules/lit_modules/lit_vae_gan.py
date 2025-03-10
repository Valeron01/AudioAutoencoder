import lightning as pl
import torch.optim

from nn_modules.simple_vae import AudioVAE, Encoder
from torch import nn


class LitSimpleVAEGAN(pl.LightningModule):
    def __init__(
            self,
            n_channels_list,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim,
            lr,
            weight_decay,
            kl_weight,
            disc_weight,
            feature_matching_weight
    ):
        super().__init__()
        self.disc_weight = disc_weight
        self.feature_matching_weight = feature_matching_weight
        self.lr = lr
        self.kl_weight = kl_weight
        self.weight_decay = weight_decay
        self.autoencoder = AudioVAE(n_channels_list, kernel_sizes, strides, n_transformer_blocks, n_heads, z_dim)
        self.discriminator = Encoder(n_channels_list, kernel_sizes, strides, n_transformer_blocks, n_heads, 1)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.autoencoder.parameters(), self.lr, weight_decay=self.weight_decay
        ), torch.optim.AdamW(
            self.discriminator.parameters(), self.lr, weight_decay=self.weight_decay
        )

    def training_step(self, batch, *args, **kwargs):
        real_audios = batch

        generator_optimizer, discriminator_optimizer = self.optimizers()

        self.toggle_optimizer(discriminator_optimizer)
        self.autoencoder.eval()
        with torch.inference_mode():
            mean, log_var = self.autoencoder.encode(real_audios)
            std = (log_var / 2).exp()
            sample = mean + torch.randn_like(std) * std
            fake_audios = self.autoencoder.decode(sample)
        inputs_concated = torch.cat([real_audios, fake_audios], dim=0)
        labels_reals, labels_fakes = self.discriminator(inputs_concated[:, None]).chunk(2, 0)

        loss_reals = nn.functional.mse_loss(labels_reals, torch.ones_like(labels_reals))
        loss_fakes = nn.functional.mse_loss(labels_fakes, torch.zeros_like(labels_fakes))
        total_discriminator_loss = (loss_fakes + loss_reals) * 0.5

        discriminator_optimizer.zero_grad()
        self.manual_backward(total_discriminator_loss)
        torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 1)
        discriminator_optimizer.step()
        self.untoggle_optimizer(discriminator_optimizer)

        self.toggle_optimizer(generator_optimizer)
        self.discriminator.eval().requires_grad_(False)
        self.autoencoder.train()

        mean, log_var = self.autoencoder.encode(real_audios)
        std = (log_var / 2).exp()
        sample = mean + torch.randn_like(std) * std
        fake_audios = self.autoencoder.decode(sample)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        discriminator_outputs, discriminator_features = self.discriminator(fake_audios[:, None], return_features=True)
        loss_generator_adversarial = nn.functional.mse_loss(
            discriminator_outputs, torch.ones_like(discriminator_outputs)
        )
        with torch.no_grad():
            _, discriminator_real_features = self.discriminator(real_audios[:, None], return_features=True)

        features_loss = sum([
            nn.functional.mse_loss(i, j) for i, j in zip(discriminator_features[4:], discriminator_real_features[4:])
        ])

        total_generator_loss = (features_loss * self.feature_matching_weight +
                                loss_generator_adversarial * self.disc_weight +
                                kl_loss * self.kl_weight)

        generator_optimizer.zero_grad()
        self.manual_backward(total_generator_loss)
        torch.nn.utils.clip_grad_norm(self.autoencoder.parameters(), 1)
        generator_optimizer.step()

        self.discriminator.train().requires_grad_(True)
        self.autoencoder.train().requires_grad_(True)

        self.log("train_features_loss", features_loss, prog_bar=True)
        self.log("total_generator_loss", total_generator_loss, prog_bar=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True)
        self.log("train_loss_reals", loss_reals)
        self.log("train_loss_fakes", loss_fakes)
        self.log("train_total_discriminator_loss", total_discriminator_loss, prog_bar=True)
        self.log("train_loss_generator_adversarial", loss_generator_adversarial)

    def validation_step(self, batch, *args, **kwargs):
        source_audios = batch

        mean, log_var = self.autoencoder.encode(source_audios)
        std = (log_var / 2).exp()
        sample = mean + torch.randn_like(std) * std
        decoded = self.autoencoder.decode(sample)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        reconstruction_loss = torch.nn.functional.mse_loss(decoded, source_audios)

        total_loss = reconstruction_loss + kl_loss * self.kl_weight
        self.log("val_reconstruction", reconstruction_loss)
        self.log("val_loss", total_loss)
        self.log("val_kl_loss", kl_loss)
