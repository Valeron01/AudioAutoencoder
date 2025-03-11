import glob
import os.path

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger, ModelCheckpoint
from data_utils.data_utils_raw_audio import build_datasets
from nn_modules.lit_modules.lit_simple_vae import LitSimpleVAE
from nn_modules.lit_modules.lit_vae_gan import LitSimpleVAEGAN


def main():
    train_audios = glob.glob(
        "/mnt/LxData/AudioDatasetWAV/*.*"
    ) + glob.glob(
        "/mnt/LxData/*/CroppedVideos25FPS/*/*.*"
    )

    validation_audios = ["/mnt/LxData/AudioDatasetWAV/Audio000013.wav"]
    train_audios = [i for i in train_audios if i not in validation_audios]
    train_dataset_length = 10000
    validation_dataset_length = 500
    batch_size = 4
    samples_count = 16_384 * 6
    target_sample_rate = 16_384
    tensorboard_folder_path = "/mnt/LxData/AudioVAEGAN"
    n_channels_list = [64, 128, 128, 256, 256, 384, 384, 512, 512, 512, 512, 768, 768, 768]
    strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    n_transformer_blocks = 1
    n_heads = 8
    z_dim = 2
    kl_weight = 1e-6
    lr = 5e-5
    weight_decay = 0.01
    disc_weight = 0.001
    feature_matching_weight = 1
    path_to_pretrained_checkpoint = "/mnt/LxData/AudioVAE/checkpoints/version_050/last.ckpt"

    train_dataset, validation_dataset = build_datasets(
        train_audios, validation_audios, train_dataset_length, validation_dataset_length,
        samples_count, target_sample_rate
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    val_dataloader = DataLoader(
        validation_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True
    )

    logger = TensorBoardLogger(tensorboard_folder_path)
    checkpointer = ModelCheckpoint(
        os.path.join(logger.save_dir, "checkpoints", f"version_{logger.version:03d}"),
        save_last=True, save_top_k=2, monitor="val_loss"
    )

    trainer = pl.Trainer(
        accelerator="gpu", logger=logger, callbacks=[checkpointer], min_epochs=100,
        precision="16-mixed", log_every_n_steps=4, accumulate_grad_batches=4
    )

    model = LitSimpleVAEGAN(
        n_channels_list=n_channels_list,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_transformer_blocks=n_transformer_blocks,
        n_heads=n_heads,
        z_dim=z_dim,
        lr=lr, weight_decay=weight_decay, kl_weight=kl_weight,
        disc_weight=disc_weight, feature_matching_weight=feature_matching_weight
    )
    loaded_vae = LitSimpleVAE.load_from_checkpoint(path_to_pretrained_checkpoint)
    model.autoencoder.load_state_dict(loaded_vae.autoencoder.state_dict())

    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
