import os.path


import torch
import torchaudio
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from torchaudio import transforms


class TestCallback(Callback):
    def __init__(self, data_loader, results_path, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.data_loader = data_loader
        self.results_path = results_path

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
                inverse_mel = transforms.InverseMelScale(n_mels=256, n_stft=579).cuda()(mel_spectrogram)
                griffin_lim = transforms.GriffinLim(n_fft=1157, n_iter=1024).cuda()(inverse_mel)

                torchaudio.save(result_path, griffin_lim.cpu(), 16_000)

                count += 1

                return
