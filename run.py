from diffusion import GaussianDiffusion, Trainer
from unet import UNet

mode = "interpolation" #demultiple, interpolation, denoising
folder = "dataset/"+mode+"/data_train/"
image_size = (64,256)

model = UNet(
        in_channel=2,
        out_channel=1,
        dropout=0.5,
        image_size = 256,
        attn_res=[64, 16]
).to("mps")#.cuda()

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    channels = 1,
    image_size = image_size,
    timesteps = 2000,
    loss_type = 'l2', # L1 or L2
    noise_mix_ratio = None
).to("mps")#.cuda()

trainer = Trainer(
    diffusion,
    mode = mode,
    folder = folder,
    image_size = image_size,
    train_batch_size = 4, #32 for A100; 16 for GTX
    train_lr = 1e-4,
    train_num_steps = 20000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    save_and_sample_every=2500
)

trainer.train()