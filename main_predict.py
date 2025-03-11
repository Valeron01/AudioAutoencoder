import torch
import torchaudio

from nn_modules.lit_modules.lit_simple_vae import LitSimpleVAE
from nn_modules.lit_modules.lit_vae_gan import LitSimpleVAEGAN

device = "cpu"
model = LitSimpleVAEGAN.load_from_checkpoint("/mnt/LxData/AudioVAEGAN/checkpoints/version_003/last.ckpt", map_location=device).eval().requires_grad_(False)
audio, audio_sr = torchaudio.load("/mnt/LxData/AudioDatasetWAV/Audio000013.wav")
audio = torchaudio.functional.resample(audio, audio_sr, 16_384)


start_time = 60
sample_length = 24
inner_sr = 16_384
audio_sample = audio[0, start_time * inner_sr:start_time * inner_sr + sample_length * inner_sr]

audio_sample = audio_sample[None].to(device)
with torch.inference_mode(), torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ), torch.autocast(device, torch.float32):
    mean, log_var = model.autoencoder.encode(audio_sample)
    std = (log_var / 2).exp()
    sample = mean + torch.randn_like(std) * std

    decoded = model.autoencoder.decode(sample)
print(decoded.mean())
print(mean.shape)
print(decoded.shape)
torchaudio.save("/mnt/LxData/AudioVAE/Sandbox/ValidationTest_eval_sampled_5_gan_2.wav", decoded.cpu().float(), inner_sr)
