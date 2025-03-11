import lightning as pl
import torch.optim

from nn_modules.simple_vae import AudioVAE


class LitSimpleVAE(pl.LightningModule):
    def __init__(
            self,
            n_channels_list,
            kernel_sizes, strides,
            n_transformer_blocks,
            n_heads,
            z_dim,
            lr,
            weight_decay,
            kl_weight
    ):
        super().__init__()
        self.lr = lr
        self.kl_weight = kl_weight
        self.weight_decay = weight_decay
        self.autoencoder = AudioVAE(n_channels_list, kernel_sizes, strides, n_transformer_blocks, n_heads, z_dim)
        self.automatic_optimization = True
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, *args, **kwargs):
        source_audios = batch

        mean, log_var = self.autoencoder.encode(source_audios)
        std = (log_var / 2).exp()
        sample = mean + torch.randn_like(std) * std
        decoded = self.autoencoder.decode(sample)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        reconstruction_loss = torch.nn.functional.mse_loss(decoded, source_audios)

        total_loss = reconstruction_loss + kl_loss * self.kl_weight
        self.log("train_reconstruction", reconstruction_loss, prog_bar=True)
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, *args, **kwargs):
        source_audios = batch

        mean, log_var = self.autoencoder.encode(source_audios)
        std = (log_var / 2).exp()
        sample = mean + torch.randn_like(std) * std
        decoded = self.autoencoder.decode(sample)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        reconstruction_loss = torch.nn.functional.mse_loss(decoded, source_audios)

        total_loss = reconstruction_loss + kl_loss * self.kl_weight
        self.log("val_reconstruction", reconstruction_loss, prog_bar=True)
        self.log("val_loss", total_loss)
        self.log("val_kl_loss", kl_loss)
