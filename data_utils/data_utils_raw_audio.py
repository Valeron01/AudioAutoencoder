import random

import torch.utils.data
import torchaudio
import tqdm
from torch.utils.data import Dataset


class AudioRawDataset(Dataset):
    def __init__(self, audio_path, samples_count, sample_rate):
        self.samples_count = samples_count

        loaded_audio, loaded_audio_sr = torchaudio.load(audio_path)
        self.loaded_audio = torchaudio.functional.resample(loaded_audio.cuda(), loaded_audio_sr, sample_rate).mean(0).cpu()

    def __len__(self):
        return self.loaded_audio.shape[0] - self.samples_count

    def __getitem__(self, item):
        return self.loaded_audio[item:item + self.samples_count]


class CustomLengthDataset(Dataset):
    def __init__(self, dataset, target_length):
        self.dataset = dataset
        self.target_length = target_length

    def __len__(self):
        return self.target_length

    def __getitem__(self, item):
        return self.dataset[random.randrange(0, len(self.dataset))]


def build_datasets(train_audios, validation_audios, train_dataset_length, validation_dataset_length, samples_count, target_sample_rate):
    train_datasets = [AudioRawDataset(i, samples_count, target_sample_rate) for i in tqdm.tqdm(train_audios)]
    val_datasets = [AudioRawDataset(i, samples_count, target_sample_rate) for i in validation_audios]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    return CustomLengthDataset(train_dataset, train_dataset_length), CustomLengthDataset(val_dataset, validation_dataset_length)
