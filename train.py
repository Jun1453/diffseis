import numpy as np
from diffusion import GaussianDiffusion, Trainer
from unet import UNet
from profiledd import Profiles, jamstec_handler, highpass
from refine_train import stn_num_to_n, fit_curves

count = 0
profiles_data = None
profiles_target = None
myfilter = lambda x: highpass(x, 2.0, pf.sampling_rate, poles=4)

# look up existing dict for corresponding first arrival curve of each OBS
for key, value in stn_num_to_n.items():

    # skip negative values that are set for test set
    if value < 0: continue
    # if int(key) > 1: continue

    # load segy files and preprocess the record section
    pf = Profiles.load(f'noto/OBS/NT24OBS_J{key}C-1.sgy', jamstec_handler)
    pf = pf.filter(myfilter).reduction(6.0)[:,:6000,:]
    
    # load pre-calculated arrival curve and correlate time series origin
    arrival = fit_curves[f'{key}'] + pf.sampling_rate*0.5
    padded_arrival = np.pad(arrival, (0,pf.shape[2]-len(arrival)), mode='edge')

    # apply diversity stacking
    pf_stack3 =   pf[1:4].diversity_stack(orig_profile_num=True, first_arrival_reference=padded_arrival)
    pf_stack2 = pf[0:5:4].diversity_stack(orig_profile_num=True, first_arrival_reference=np.flip(padded_arrival))
    pf_stack_all = Profiles.concatenate((pf_stack2[0:1],pf_stack3,pf_stack2[1:2]))
    norm_pf = Profiles.concatenate([pf[n:n+1].diversity_stack(orig_profile_num=True) for n in range(5)])
    # append new section to existing data
    if profiles_target is None:
        #profiles_data = pf
        profiles_data = norm_pf
        profiles_target = pf_stack_all
    else:
        #profiles_data = Profiles.concatenate((profiles_data, pf))
        profiles_data = Profiles.concatenate((profiles_data, norm_pf))
        profiles_target = Profiles.concatenate((profiles_target, pf_stack_all))

# generate pytorch dataset with fragmentized record section
ds_gt = profiles_target.fragmentize(vclip=25, tmin=0, t_interval=3.57, x_move_ratio=0.2, y_move_ratio=0.2)
ds_data = profiles_data.fragmentize(vclip=25, tmin=0, t_interval=3.57, x_move_ratio=0.2, y_move_ratio=0.2)
ds_data.set_ground_truth(ds_gt)
del profiles_data
del profiles_target


# initialize unet and ddpm
model = UNet(
        in_channel=2,
        out_channel=1,
        dropout=0.5,
        image_size = ds_data.unit_size[1],
        # attn_res=[64, 16]
)

diffusion = GaussianDiffusion(
    model,
    mode = "demultiple",
    channels = 1,
    image_size = ds_data.unit_size,
    timesteps = 2000,
    loss_type = 'l2', # L1 or L2
    noise_mix_ratio = None
)

if __name__ == '__main__':
    ds_data.train(diffusion, 200, 32, gradient_accumulate_every=2, save_every=50, learning_rate=3e-5, results_folder='results/demultiple0722a-oop')
    
    








