import os
import sys
import torch
import random
from pydoc import importfile
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from unet import UNet
from diffusion import GaussianDiffusion, Trainer, Dataset

run_path = sys.argv[1]
n = int(sys.argv[2])
model_name = sys.argv[3] if len(sys.argv) > 3 else "model-final.pt"
playback_speed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

run = importfile(run_path)
work_folder = run.trainer.results_folder
maximum_batch_size = run.trainer.batch_size
# obs_num = 28
# xshift = (obs_num-1)*-2
# to_flip = True
testset_folder = 'dataset/'+run.mode+'/data_test_npy/'
tile_info = importfile(str(testset_folder+'tile_info.py'))
    
parameters = torch.load(str(work_folder/model_name), map_location=torch.device('mps'), weights_only=True)['model']


del parameters['betas']
del parameters['alphas_cumprod']
del parameters['alphas_cumprod_prev']
del parameters['sqrt_alphas_cumprod']
del parameters['sqrt_one_minus_alphas_cumprod']
del parameters['log_one_minus_alphas_cumprod']
del parameters['sqrt_recip_alphas_cumprod']
del parameters['sqrt_recipm1_alphas_cumprod']
del parameters['posterior_variance']
del parameters['posterior_log_variance_clipped']
del parameters['posterior_mean_coef1']
del parameters['posterior_mean_coef2']


def change_key(self, old, new):
    #copy = self.copy()
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v
        
keys = []
for key, value in parameters.items():
    keys.append(key)
    
for i in range(len(keys)):
    change_key(parameters, keys[i], keys[i][11:])
    
run.model.load_state_dict(parameters)


ds = Dataset(testset_folder, image_size=run.image_size, mode=run.mode, file_ext=tile_info.file_ext)

x_len = run.image_size[0]
y_len = run.image_size[1]
x_move = int(x_len*tile_info.x_move_ratio)#int(image_size[0]/8)
y_move = int(y_len*tile_info.y_move_ratio)#int(image_size[1]/8)
canvas_gt= np.ndarray(shape=(y_move*(tile_info.y_tile-1)+y_len, x_move*(tile_info.x_tile-1)+x_len))
canvas_inp = np.ndarray(shape=canvas_gt.shape)
canvas_out = np.ndarray(shape=canvas_gt.shape)
canvas_wt = np.ndarray(shape=canvas_gt.shape)
double_marigin = (playback_speed < 3)

ds_len = len(ds)
# jump to target window num
ds_window_num = tile_info.x_tile*tile_info.y_tile*n
ds_end_num = ds_window_num+tile_info.x_tile*tile_info.y_tile-1

# initialize
current_batch_size = 0
x_loc = np.zeros(maximum_batch_size, dtype=int)
y_loc = np.zeros(maximum_batch_size, dtype=int)

while True:
    # get location for the computing window
    x_loc[current_batch_size] = int((ds_window_num%tile_info.x_tile)*x_move)
    y_loc[current_batch_size] = int(((ds_window_num//tile_info.x_tile)%tile_info.y_tile)*y_move)

    # load data skipping overlapping windows if set on
    if (((x_loc[current_batch_size]%(x_move*playback_speed)==0) and
         ((y_loc[current_batch_size]%(y_move*playback_speed)==0) or 
          (y_loc[current_batch_size]==0) or 
          (y_loc[current_batch_size]+y_len==canvas_gt.shape[0]))) or
        ((y_loc[current_batch_size]%(y_move*playback_speed)==0) and
         ((x_loc[current_batch_size]==0) or 
          (x_loc[current_batch_size]+x_len==canvas_gt.shape[1])))):

        x_in, x_gt = ds[ds_window_num]
        x_in = torch.unsqueeze(x_in, dim=0)
        x_gt = torch.unsqueeze(x_gt, dim=0)
        # start a new batch when empty or concatenate elsewise
        batch_in = x_in if current_batch_size == 0 else torch.cat((batch_in, x_in), dim=0)
        batch_gt = x_gt if current_batch_size == 0 else torch.cat((batch_gt, x_gt), dim=0)
        current_batch_size += 1
    
    # get output matrix when full or before break
    if (ds_window_num+1 >= ds_end_num) or (current_batch_size == maximum_batch_size):
        out = run.diffusion.inference(x_in=batch_in.to("mps"), clip_denoised=False)

        for i in range(current_batch_size):
            inp_2d = batch_in[i,0].cpu().detach().numpy()
            gt_2d = batch_gt[i,0].cpu().detach().numpy()
            out_2d = out[i+current_batch_size,0].cpu().detach().numpy()

            mask = np.ones((64,256))
            if double_marigin:
                if x_loc[i] > 0:
                    mask[:x_move,:] = 0
                    mask[x_move:2*x_move,:] = np.minimum(np.linspace(0,1,x_move+2)[1:-1][:,np.newaxis].repeat(y_len, axis=1), mask[x_move:2*x_move,:])
                if x_loc[i] + x_len < canvas_gt.shape[1]:
                    mask[-x_move:,:] = 0
                    mask[-2*x_move:-x_move,:] = np.minimum(np.linspace(1,0,x_move+2)[1:-1][:,np.newaxis].repeat(y_len, axis=1), mask[-2*x_move:-x_move,:])
                if y_loc[i] > 0:
                    mask[:,:y_move] = 0
                    mask[:,y_move:2*y_move] = np.minimum(np.linspace(0,1,y_move+2)[1:-1], mask[:,y_move:2*y_move])
                if y_loc[i] + y_len < canvas_gt.shape[0]:
                    mask[:,-y_move:] = 0
                    mask[:,-2*y_move:-y_move] = np.minimum(np.linspace(1,0,y_move+2)[1:-1], mask[:,-2*y_move:-y_move])
            else:
                if x_loc[i] > 0:
                    mask[:x_move,:] = np.minimum(np.linspace(0,1,x_move+2)[1:-1][:,np.newaxis].repeat(y_len, axis=1), mask[:x_move,:])
                if x_loc[i] + x_len < canvas_gt.shape[1]:
                    mask[-x_move:,:] = np.minimum(np.linspace(1,0,x_move+2)[1:-1][:,np.newaxis].repeat(y_len, axis=1), mask[-x_move:,:])
                if y_loc[i] > 0:
                    mask[:,:y_move] = np.minimum(np.linspace(0,1,y_move+2)[1:-1], mask[:,:y_move])
                if y_loc[i] + y_len < canvas_gt.shape[0]:
                    mask[:,-y_move:] = np.minimum(np.linspace(1,0,y_move+2)[1:-1], mask[:,-y_move:])


            canvas_gt[ y_loc[i]:y_loc[i]+y_len,x_loc[i]:x_loc[i]+x_len] = gt_2d.T
            canvas_inp[y_loc[i]:y_loc[i]+y_len,x_loc[i]:x_loc[i]+x_len] = inp_2d.T
            # canvas_out[y_loc[i]:y_loc[i]+y_len,x_loc[i]:x_loc[i]+x_len]= out_2d.T
            canvas_out[y_loc[i]:y_loc[i]+y_len,x_loc[i]:x_loc[i]+x_len] += (mask*out_2d).T
            canvas_wt[ y_loc[i]:y_loc[i]+y_len,x_loc[i]:x_loc[i]+x_len] += (mask*np.ones_like(gt_2d)).T


    # clean batch if full
    if (current_batch_size == maximum_batch_size): current_batch_size = 0

    # break when finished
    if (ds_window_num >= ds_end_num): break
    else: ds_window_num += 1

# canvas_inp /= canvas_wt
canvas_out /= canvas_wt
np.save(f'{work_folder}/canvas-fast_gt-{n}.npy', canvas_gt)
np.save(f'{work_folder}/canvas-fast_inp-{n}.npy', canvas_inp) 
np.save(f'{work_folder}/canvas-fast_{str(model_name).replace(".pt","")}-{n}.npy', canvas_out)
# print(i)

# fig, ax = plt.subplots(1,1, figsize=(16,6))
# ax.imshow(canvas_inp, cmap="Greys")
# ax.set_aspect('auto')
# ax.set_axis_off()
# ax.set_title('Input')
# fig.savefig("rebuild_input.png", bbox_inches="tight")

# fig, ax = plt.subplots(1,1, figsize=(16,6))
# ax.imshow(canvas_out, cmap="Greys")
# ax.set_aspect('auto')
# ax.set_axis_off()
# ax.set_title('Output')
# fig.savefig("rebuild_output.png", bbox_inches="tight")

# fig, ax = plt.subplots(1,1, figsize=(16,6))
# ax.imshow(canvas_gt, cmap="Greys")
# ax.set_aspect('auto')
# ax.set_axis_off()
# ax.set_title('Ground Truth')
# fig.savefig("rebuild_gt.png", bbox_inches="tight")

# fig, ax = plt.subplots(1,1, figsize=(16,6))
# ax.imshow(canvas_out - canvas_inp, cmap="Greys", extent=[-10, 90, 7, 0])
# ax.set_aspect('auto')
# ax.set_axis_on()
# ax.set_title('Output')
# fig.savefig("rebuild_output.png", bbox_inches="tight")

# fig, ax = plt.subplots(1,1, figsize=(16,6))
# ax.imshow(canvas_out - canvas_gt, cmap="Greys")
# ax.set_aspect('auto')
# ax.set_axis_off()
# ax.set_title('Output - Ground Truth')
# fig.savefig("rebuild_output-gt.png", bbox_inches="tight")

# data = lambda d: np.flip(d, axis=1) if to_flip else d
# fig, axs = plt.subplots(2,2, figsize=(16,10), gridspec_kw={'wspace': 0.1})
# axs[0,0].imshow(data(canvas_inp), cmap="Greys", extent=[-10+xshift, 90+xshift, 7, 0])
# axs[0,0].set_aspect('auto')
# axs[0,0].set_axis_on()
# axs[0,0].set_title('Input')
# axs[0,0].set_ylabel(' T - X/6.0 (sec)')

# axs[0,1].imshow(data(canvas_gt), cmap="Greys", extent=[-10+xshift, 90+xshift, 7, 0])
# axs[0,1].set_aspect('auto')
# axs[0,1].set_axis_on()
# axs[0,1].set_title('Ground Truth (Diversity Stack)')

# axs[1,0].imshow(data(canvas_out), cmap="Greys", extent=[-10+xshift, 90+xshift, 7, 0])
# axs[1,0].set_aspect('auto')
# axs[1,0].set_axis_on()
# axs[1,0].set_title('Output')
# axs[1,0].set_xlabel('Offset (km)')
# axs[1,0].set_ylabel(' T - X/6.0 (sec)')

# axs[1,1].imshow(data(canvas_out-canvas_gt), cmap="Greys", extent=[-10+xshift, 90+xshift, 7, 0])
# axs[1,1].set_aspect('auto')
# axs[1,1].set_axis_on()
# axs[1,1].set_title('Output - Ground Truth')
# axs[1,1].set_xlabel('Offset (km)')
# fig.savefig(f"rebuild_4plts-{n}.png", bbox_inches="tight")
