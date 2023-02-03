import random
import typing
import torch.utils.data
import torch
import torchaudio
from torchaudio import transforms


def audio_files_to_mel_spectrogram(
        paths: typing.List[str],
        target_sample_rate: int = 16_000,
        verbose: bool = True
) -> torch.Tensor:
    assert len(paths) != 0

    results = []
    total_duration = 0
    for path in paths:
        waveform, sample_rate = torchaudio.load(path)
        total_duration += waveform.shape[-1] / sample_rate
        waveform = waveform[[0], :].cuda()
        waveform = transforms.Resample(sample_rate, new_freq=target_sample_rate).cuda()(waveform)

        spectrogram = transforms.Spectrogram(n_fft=1157).cuda()(waveform)

        mel_spectrogram = transforms.MelScale(
            n_mels=256, n_stft=579, sample_rate=target_sample_rate
        ).cuda()(spectrogram)

        mel_spectrogram = transforms.AmplitudeToDB("db")(mel_spectrogram)

        results.append(mel_spectrogram.cpu())

    if verbose:
        print(total_duration)
    return torch.cat(results, -1)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, mel_spectrogram, window_size: int = 256, random_access: bool = True):
        self.mel_spectrogram = mel_spectrogram
        self.window_size = window_size
        self.length = mel_spectrogram.shape[-1] // window_size
        self.random_access = random_access

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.random_access:
            index = random.randint(0, self.mel_spectrogram.shape[-1] - self.window_size)
            return self.mel_spectrogram[..., index:index + self.window_size]

        return self.mel_spectrogram[..., item * self.window_size:item * self.window_size + self.window_size]
