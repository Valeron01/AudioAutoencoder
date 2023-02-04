import os.path


import torch
import torchaudio
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from torchaudio import transforms


class TestCallback(Callback):
    def __init__(
            self, data_loader, results_path, min_value, max_value,
            sample_rate=22500,
            n_fft=1024,
            n_mels=192,
            win_length=1024,
            hop_length=256,
            n_iter=1204
    ):
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.min_value = min_value
        self.max_value = max_value
        self.data_loader = data_loader
        self.results_path = results_path
        self.sample_rate = sample_rate
        self.n_iter = n_iter

        self.mel_spectrogram_to_audio = torch.nn.Sequential(
            transforms.InverseMelScale(
                n_stft=self.n_fft // 2 + 1,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
            ),
            transforms.GriffinLim(
                n_fft=self.n_fft,
                n_iter=self.n_iter,
                win_length=self.win_length,
                hop_length=self.hop_length
            )
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        count = 0
        for batch in self.data_loader:
            with torch.no_grad():
                batch = batch.to(pl_module.device)
                results = pl_module(batch)

            results = torch.cat([batch, results], dim=-1)

            for spectrogram_index in range(results.shape[0]):
                image = results[spectrogram_index]
                result_path = os.path.join(
                    self.results_path,
                    f"{trainer.logger.name}_v{trainer.logger.version}",
                    f"epoch_{trainer.current_epoch}",
                    f"audio{count:06d}.wav"
                )
                os.makedirs(os.path.dirname(result_path), exist_ok=True)

                image = image * (self.max_value - self.min_value) + self.min_value
                mel_spectrogram = torchaudio.functional.DB_to_amplitude(image.cuda(), 1, 0.5)
                restored_audio = self.mel_spectrogram_to_audio.to(pl_module.device)(mel_spectrogram)

                torchaudio.save(result_path, restored_audio.cpu(), self.sample_rate)

                count += 1

                return
