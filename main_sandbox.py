from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

numel = 0
for i in vae.parameters():
    numel += i.numel()

print(numel / 1e6)
