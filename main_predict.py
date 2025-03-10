import torch
import torchaudio

from nn_modules.lit_modules.lit_simple_vae import LitSimpleVAE
from nn_modules.lit_modules.lit_vae_gan import LitSimpleVAEGAN

model = LitSimpleVAEGAN.load_from_checkpoint("/mnt/LxData/AudioVAEGAN/checkpoints/version_000/last.ckpt").eval().requires_grad_(False)
audio, audio_sr = torchaudio.load("/mnt/LxData/AudioDatasetWAV/Audio000013.wav")
audio = torchaudio.functional.resample(audio, audio_sr, 16_000)


start_time = 60
sample_length = 12
inner_sr = 16_000
audio_sample = audio[0, start_time * inner_sr:start_time * inner_sr + sample_length * inner_sr]

audio_sample = audio_sample[None].cuda()
mean, log_var = model.autoencoder.encode(audio_sample)
std = (log_var / 2).exp()
sample = mean# + torch.randn_like(std) * std

decoded = model.autoencoder.decode(sample)
print(decoded.mean())
print(mean.shape)
print(decoded.shape)
torchaudio.save("/mnt/LxData/AudioVAE/Sandbox/ValidationTest_eval_sampled_gan.wav", decoded.cpu(), inner_sr)
