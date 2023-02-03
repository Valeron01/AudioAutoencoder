import torch
import torchaudio
from torchaudio import transforms
import cv2

from build_dataset import audio_files_to_mel_spectrogram, AudioDataset

dataset = audio_files_to_mel_spectrogram([
    r"C:\Users\elect\PycharmProjects\AutoEncoder\audio\audio.wav",
    r"D:\NeuralNetsProjects\audioGan\dataset\ncs_small_16384.wav",
    r"D:\NeuralNetsProjects\audioGan\dataset\ls_16384.wav",
    r"D:\NeuralNetsProjects\audioGan\dataset\bach_classic.wav"
])

min_value = dataset.min()
max_value = dataset.max()

dataset = (dataset - min_value) / (max_value - min_value)

dataset = AudioDataset(dataset, 256)
image = dataset[0]
print(image.shape)

cv2.imwrite("./spectrogram.png", image[0].numpy() * 255)

image = image * (max_value - min_value) + min_value
mel_spectrogram = torchaudio.functional.DB_to_amplitude(image.cuda(), 1, 0.5)
inverse_mel = transforms.InverseMelScale(n_mels=256, n_stft=579).cuda()(mel_spectrogram)
griffin_lim = transforms.GriffinLim(n_fft=1157, n_iter=1024).cuda()(inverse_mel)

torchaudio.save("./griffin_lim_2.wav", griffin_lim.cpu(), 16_000)
