import random

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import networks
from build_dataset import audio_files_to_mel_spectrogram, AudioDataset
import torch.utils.data

from callbacks import TestCallback

dataset = audio_files_to_mel_spectrogram([
    r"C:\Users\elect\PycharmProjects\AutoEncoder\audio\audio.wav",
    r"D:\NeuralNetsProjects\audioGan\dataset\ncs_small_16384.wav",
    r"D:\NeuralNetsProjects\audioGan\dataset\ls_16384.wav",
    r"D:\NeuralNetsProjects\audioGan\dataset\bach_classic.wav"
])

min_value = dataset.min()
max_value = dataset.max()

dataset = (dataset - min_value) / (max_value - min_value)

torch.manual_seed(0)
random.seed(0)
train_dataset = dataset[..., :int(dataset.shape[-1] * 0.9)]
validation_dataset = dataset[..., int(dataset.shape[-1] * 0.9):]

train_dataset = AudioDataset(train_dataset, 256)
validation_dataset = AudioDataset(validation_dataset, 256)


train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=16)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)

network = networks.LitAutoEncoder()
logger = TensorBoardLogger("./runs", name="simplest_aae")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    f"./best_checkpoints/{logger.name}_v{logger.version}", monitor="val_loss",
    verbose=True
)

test_callback = TestCallback(validation_loader, "./predictions", min_value, max_value)

trainer = pl.Trainer(
    logger=logger, gpus=1, default_root_dir="./checkpoints",
    callbacks=[checkpoint_callback, test_callback]
)

trainer.fit(
    network,
    train_loader,
    validation_loader
)

# 0 | encoder | Sequential | 27.4 M
# 1 | decoder | Sequential | 28.0 M
