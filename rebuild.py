import os
import torch
import random
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from unet import UNet
from diffusion import GaussianDiffusion, Trainer, Dataset

n=15
mode = "demultiple" #demultiple, interpolation, denoising
# folder = './dataset/'+mode+'/data_train/'
folder = 'dataset/'+mode+'/data_test/'

image_size = (64,256)
train_batch_size = 32
    
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

parameters = torch.load("/S/home00/G3506/p0814/diffseis/results/demultiple_0108-no_filt/model-final.pt", map_location=torch.device('mps'))['model']


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
    
model.load_state_dict(parameters)


ds = Dataset(folder, image_size=image_size, mode=mode)

x_move = int(64*0.2)#int(image_size[0]/8)
y_move = int(256*0.2)#int(image_size[1]/8)
canvas_gt= np.ndarray(shape=(y_move*30+256, x_move*36+64))
canvas_inp = np.ndarray(shape=canvas_gt.shape)
canvas_out = np.ndarray(shape=canvas_gt.shape)
canvas_wt = np.ndarray(shape=canvas_gt.shape)

ds_len = len(ds)
for i in range(37*31*(n+1)):
# for i, (x_in) in enumerate(ds):
    if i<37*31*n-(37*31*n%train_batch_size): continue
    if (i % train_batch_size == 0):
        # img = next(dl)
        # inputs = img[i%train_batch_size].to("mps")#.cuda()
        # gt = img[1].to("mps")#.cuda()
        # out = diffusion.inference(x_in=inputs)
        x_start, x_ = ds[i]
        x_start = torch.unsqueeze(x_start, dim=0)
        x_ = torch.unsqueeze(x_, dim=0)
        batch = x_start
        batch_gt = x_

        for j in range(i+1,min(i+train_batch_size,ds_len)):
            x_start, x_ = ds[j]
            x_start = torch.unsqueeze(x_start, dim=0)
            x_ = torch.unsqueeze(x_, dim=0)
            batch = torch.cat((batch, x_start), dim=0)
            batch_gt = torch.cat((batch_gt, x_), dim=0)

        out = diffusion.inference(x_in=batch.to("mps"))
    
    if i<37*31*n: continue

    mask = np.ones(image_size)
    if i % 37 > 0:
        mask[:x_move,:] = 0
        mask[x_move:2*x_move,:] = np.minimum(np.linspace(0,1,x_move+2)[1:-1][:,np.newaxis].repeat(image_size[1], axis=1), mask[x_move:2*x_move,:])
    if i % 37 < 36:
        mask[-x_move:,:] = 0
        mask[-2*x_move:-x_move,:] = np.minimum(np.linspace(1,0,x_move+2)[1:-1][:,np.newaxis].repeat(image_size[1], axis=1), mask[-2*x_move:-x_move,:])
    if (i//37)%31 > 0:
        mask[:,:y_move] = 0
        mask[:,y_move:2*y_move] = np.minimum(np.linspace(0,1,y_move+2)[1:-1], mask[:,y_move:2*y_move])
    if (i//37)%31 < 30:
        mask[:,-y_move:] = 0
        mask[:,-2*y_move:-y_move] = np.minimum(np.linspace(1,0,y_move+2)[1:-1], mask[:,-2*y_move:-y_move])
    # if i == 0:
    #     plt.imshow(mask.T)
    #     plt.colorbar()
    #     # print(mask)
    x_loc = (i%37)*x_move
    y_loc = ((i//37)%31)*y_move
    # canvas_inp[y_loc:y_loc+256,x_loc:x_loc+64] = (inputs[i%train_batch_size,0].cpu().detach().numpy()).T
    # canvas_inp[y_loc:y_loc+256,x_loc:x_loc+64] = (mask*out[i%train_batch_size,0].cpu().detach().numpy()).T
    # canvas_out[y_loc:y_loc+256,x_loc:x_loc+64] = (mask*out[i%train_batch_size+train_batch_size,0].cpu().detach().numpy()).T
    # canvas_inp[y_loc:y_loc+256,x_loc:x_loc+64] += (mask*batch[i%train_batch_size,0].cpu().detach().numpy()).T
    # canvas_wt[y_loc:y_loc+256,x_loc:x_loc+64] += (mask*np.ones_like(batch[i%train_batch_size,0].cpu().detach().numpy())).T

    inp_2d = batch[i%train_batch_size,0].cpu().detach().numpy()
    gt_2d = batch_gt[i%train_batch_size,0].cpu().detach().numpy()
    out_2d = out[i%train_batch_size+train_batch_size,0].cpu().detach().numpy()
    canvas_gt[y_loc:y_loc+256,x_loc:x_loc+64] = gt_2d.T
    canvas_inp[y_loc:y_loc+256,x_loc:x_loc+64] = inp_2d.T
    canvas_out[y_loc:y_loc+256,x_loc:x_loc+64] += (mask*out_2d).T
    canvas_wt[y_loc:y_loc+256,x_loc:x_loc+64] += (mask*np.ones_like(out_2d)).T

    # if i == 370-1: break
    # if i == train_batch_size-1: break
# canvas_inp /= canvas_wt
canvas_out /= canvas_wt
# print(i)

fig, ax = plt.subplots(1,1, figsize=(16,6))
ax.imshow(canvas_inp, cmap="Greys")
ax.set_aspect('auto')
ax.set_axis_off()
ax.set_title('Input')
fig.savefig("rebuild_input.png", bbox_inches="tight")

fig, ax = plt.subplots(1,1, figsize=(16,6))
ax.imshow(canvas_out, cmap="Greys")
ax.set_aspect('auto')
ax.set_axis_off()
ax.set_title('Output')
fig.savefig("rebuild_output.png", bbox_inches="tight")

fig, ax = plt.subplots(1,1, figsize=(16,6))
ax.imshow(canvas_gt, cmap="Greys")
ax.set_aspect('auto')
ax.set_axis_off()
ax.set_title('Ground Truth')
fig.savefig("rebuild_gt.png", bbox_inches="tight")
