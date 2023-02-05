import torch
import torchaudio
from torchaudio import transforms

import networks

sample_rate = 22500
n_fft = 1024
n_mels = 192
win_length = 1024
hop_length = 256
n_iter = 1024

min_value = -200
max_value = 96.4569

mel_spectrogram_to_audio = torch.nn.Sequential(
    transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    ),
    transforms.GriffinLim(
        n_fft=n_fft,
        n_iter=n_iter,
        win_length=win_length,
        hop_length=hop_length
    )
).cuda()


mel_spectrogram_transform = transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    n_mels=n_mels,
    win_length=win_length,
    hop_length=hop_length,
).cuda()

network = networks.LitAutoEncoder().load_from_checkpoint(
    "./best_checkpoints/simplest_aae_v31/epoch=22-step=21712.ckpt"
).cuda()

waveform, initial_sample_rate = torchaudio.load(
    r"C:\Users\elect\PycharmProjects\AudioDeepFake\voices\JB\jb_0.wav",
    normalize=True
)
waveform = waveform[[0], initial_sample_rate*670:initial_sample_rate * 700].cuda()


waveform = transforms.Resample(orig_freq=initial_sample_rate, new_freq=sample_rate).cuda()(waveform)
print(waveform.shape)


mel_spectrogram = mel_spectrogram_transform(waveform)

mel_spectrogram = transforms.AmplitudeToDB("db")(mel_spectrogram)
mel_spectrogram = (mel_spectrogram - min_value) / (max_value - min_value)

print(mel_spectrogram.shape)

mel_spectrogram = mel_spectrogram[None, ..., :256].cuda()

with torch.no_grad():
    decoded_sg = network(mel_spectrogram)
print(decoded_sg.shape)

# decoded_sg = mel_spectrogram

decoded_sg = decoded_sg * (max_value - min_value) + min_value
decoded_sg = torchaudio.functional.DB_to_amplitude(decoded_sg, 1, 0.5)


audio = mel_spectrogram_to_audio(decoded_sg[0])

torchaudio.save(f"./restored_v31.wav", audio.cpu(), sample_rate)
print(audio.shape)
