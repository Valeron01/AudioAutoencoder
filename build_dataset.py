import random
import typing
import torch.utils.data
import torch
import torchaudio
from torchaudio import transforms


def audio_files_to_mel_spectrogram(
        paths: typing.List[str],
        sample_rate=22500,
        n_fft=1024,
        n_mels=192,
        win_length=1024,
        hop_length=256,
        verbose: bool = True,
        seconds_split: int = 1800
) -> torch.Tensor:
    assert len(paths) != 0

    mel_spectrogram_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
    ).cuda()

    results = []
    total_duration = 0
    for path in paths:
        if verbose:
            print("Reading file: ", path)
        initial_waveform, initial_sample_rate = torchaudio.load(path)

        waveforms = torch.split(initial_waveform, sample_rate * seconds_split, -1)

        for waveform in waveforms:
            total_duration += waveform.shape[-1] / initial_sample_rate
            waveform = waveform[[0], :].cuda()
            waveform = transforms.Resample(initial_sample_rate, new_freq=sample_rate).cuda()(waveform)

            mel_spectrogram = mel_spectrogram_transform(waveform)

            spectrogram = transforms.AmplitudeToDB("db")(mel_spectrogram)

            results.append(spectrogram.cpu())

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
