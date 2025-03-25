import numpy as np
from diffusion import GaussianDiffusion, Trainer
from unet import UNet
from profiledd import Profiles, jamstec_handler, highpass
from refine_train import stn_num_to_n, fit_curves

count = 0
processed_pf = None
combined_pf = None
myfilter = lambda x: highpass(x, 2.0, pf.sampling_rate, poles=4)
for key, value in stn_num_to_n.items():
    if value < 0: continue

    pf = Profiles.load(f'noto/OBS/NT24OBS_J{key}C-1.sgy', jamstec_handler)
    fa_curve = fit_curves[f'{key}']
    new_pf1 = pf[1:4].filter(myfilter).reduction(6.0).diversity_stack(orig_profile_num=True, first_arrival_reference=pf.sampling_rate*0.5+np.pad(fa_curve, (0,pf.shape[2]-len(fa_curve)), mode='edge'))
    new_pf2 = pf[0:5:4].filter(myfilter).reduction(6.0).diversity_stack(orig_profile_num=True, first_arrival_reference=pf.sampling_rate*0.5+np.flip(np.pad(fa_curve, (0,pf.shape[2]-len(fa_curve)), mode='edge')))

    if combined_pf is None:
        processed_pf = pf.filter(myfilter).reduction(6.0)
        combined_pf = Profiles.concatenate((new_pf2[0:1],new_pf1,new_pf2[1:2]))
    else:
        processed_pf = Profiles.concatenate([processed_pf,pf.filter(myfilter).reduction(6.0)])
        combined_pf = Profiles.concatenate([combined_pf,new_pf2[0:1],new_pf1,new_pf2[1:2]])

ds_gt = combined_pf.fragmentize(vclip=50, t_interval=3.07, x_move_ratio=0.2, y_move_ratio=0.2)
ds_data = processed_pf.fragmentize(vclip=300, t_interval=3.07, x_move_ratio=0.2, y_move_ratio=0.2)
ds_data.set_ground_truth(ds_gt)
del processed_pf
del combined_pf


mode = "demultiple" #demultiple, interpolation, denoising
image_size = (64,256)

model = UNet(
        in_channel=2,
        out_channel=1,
        dropout=0.5,
        image_size = 256,
        # attn_res=[64, 16]
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

ds_data.train(diffusion, 10, 16, learning_rate=3e-6, device='mps')
