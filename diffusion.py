import math
import copy
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random

from torch.utils import data
from torch.amp import GradScaler, autocast
import os
import re

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
from scipy.signal import butter, filtfilt

def natural_noise(shape_like, noise_mix_ratio=None): 
    noise = torch.randn_like(shape_like)
    if not noise_mix_ratio is None:
        for i in range(len(noise)):
            noise[i] = torch.from_numpy(bandpass(noise[i].cpu().numpy(), cutoff=(2.,10.), sample_rate=250, mix_input=noise_mix_ratio).copy())
    return noise

def bandpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 2, mix_input=0):
    b, a = butter(poles, cutoff, 'bandpass', fs=sample_rate)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data + mix_input*np.array(data)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossSigmoidLoss(nn.Module):
    def __init__(self, critical=0, width=0.33, reduction='mean'):
        super().__init__()
        self.sigmoid = lambda x: 1 / (1 + torch.exp(-(x-critical)/width))
        self.reduction = reduction
    def forward(self, input, target):
        # l = torch.sqrt(torch.abs(self.sigmoid(input) - self.sigmoid(target)))
        l = torch.abs(self.sigmoid(input) - self.sigmoid(target))
        if   self.reduction == 'none':
            return l
        elif self.reduction == 'sum':
            return l.sum()
        elif self.reduction == 'mean':
            return l.mean()
        else: raise NotImplementedError("reduction should be either 'none', 'sum', or 'mean'.")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        mode,
        channels,
        image_size,
        timesteps = 2000,
        loss_type = 'l1',
        noise_mix_ratio = None,
    ):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.noise_mix_ratio = noise_mix_ratio
        
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif loss_type == 'l1l2':
            class L1L2Loss(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.L1Loss(reduction='mean')
                    self.l2 = nn.MSELoss(reduction='mean')
                def forward(self, input, target):
                    return 0.5 * (self.l1(input, target) + self.l2(input, target))
            self.loss_func = L1L2Loss()
        elif re.match(r'cross-sigmoid', loss_type):
            if loss_type == 'cross-sigmoid': reduction = 'none'
            else: reduction = loss_type.rsplit('-', 1)[1]
            self.loss_func = CrossSigmoidLoss(reduction=reduction)
        else:
            raise NotImplementedError()

        to_torch = partial(torch.tensor, dtype=torch.float32)

        betas = make_beta_schedule(schedule='linear', n_timestep=timesteps, linear_start=1e-6, linear_end=1e-2)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance( x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()
    
    @torch.no_grad()
    def p_sample_loop(self, x_in, mask=None, clip_denoised=True):
        device = self.betas.device
        x_cond = x_in
        if mask is not None:  
            x_cond = x_in*mask
            
        shape = x_cond.shape
        img = torch.randn(shape, device=device)
        ret_img = x_cond
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, i, clip_denoised=clip_denoised, condition_x=x_cond)
            if mask is not None:
                img = x_cond + img*(1.-mask)
            
        if mask is not None:
            ret_img = torch.cat([ret_img, x_in], dim=0)
        ret_img = torch.cat([ret_img, img], dim=0)
        return ret_img
    
    @torch.no_grad()
    def inference(self, x_in, mask=None, clip_denoised=True):
        return self.p_sample_loop(x_in, mask, clip_denoised)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: natural_noise(x_start, self.noise_mix_ratio))

        # random gama
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)

    def p_losses(self, x_cond, x_start, noise=None):
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1],self.sqrt_alphas_cumprod_prev[t],size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: natural_noise(x_start, self.noise_mix_ratio))

        x_noisy = self.q_sample(x_start=x_start,continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if self.mode == "interpolation":
            # here x_cond -> mask
            x_recon = self.denoise_fn(torch.cat([x_start*x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            loss = self.loss_func(noise, x_recon)
        else:
            x_recon = self.denoise_fn(torch.cat([x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
    

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, mode, file_ext='.png'):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.mode = mode
        self.file_ext = file_ext

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]) if file_ext == '.png' else lambda x: torch.from_numpy(x).unsqueeze(dim=0)

        
    def __len__(self):
        dir_path = self.folder+"data/"
        res = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
        return res
    
    def irregular_mask(self, data, rate=0.5):
        """the mask matrix of random sampling
        Args:
            data: original data patches
            rate: sampling rate,range(0,1)
        """
        n = data.size()[-2]
        mask = torch.torch.zeros(data.size(),dtype=torch.float64)
        
        v = round(n*rate)
        TM = random.sample(range(n),v)
        # mask[:,:,TM]=1 # missing by column 
        mask[:,TM,:]=1 # missing by raw 
        mask = mask.type(torch.HalfTensor)
        return  mask

    def __getitem__(self, index):
        data = self.folder+"data/"+str(index)+self.file_ext
        img_data = Image.open(data) if self.file_ext == '.png' else np.load(data)

        if self.mode == "demultiple":
            label = self.folder+"labels/"+str(index)+self.file_ext
            img_label = Image.open(label) if self.file_ext == '.png' else np.load(label)
            return self.transform(img_data), self.transform(img_label)
        elif self.mode == "interpolation":
            return self.irregular_mask(self.transform(img_data)), self.transform(img_data)
        elif self.mode == "denoising":
            img = self.transform(img_data)
            mean = torch.mean(img)
            std = torch.std(img)
            noise = 0.5*torch.normal(mean, std, size =(img.shape[0], img.shape[1], img.shape[2]))
            img_ = img + noise
            return img_, img
        
        else:
            print("ERROR MODE")

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,       
        mode,
        folder,
        *,
        ema_decay = 0.999,
        image_size = (128,128),
        file_ext = '.png',
        train_batch_size = 32,
        train_lr = 3e-6,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 5000,
        update_ema_every = 1,
        save_and_sample_every = 10000,
        save_and_sample_mode = 'one_rand',
        result_suffix=''
    ):
        super().__init__()
        self.model = diffusion_model
        self.mode = mode
        self.folder = folder
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.save_and_sample_mode = save_and_sample_mode

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset(self.folder, image_size, mode, file_ext)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 1

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        
        results_folder = './results/'+str(self.mode)+str(result_suffix)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        while self.step <= self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                img = next(self.dl)
                inputs = img[0].to("mps")#.cuda()
                gt = img[1].to("mps")#.cuda()
                
                with autocast('mps'):  # Updated usage
                    loss = self.model(inputs, gt).to("mps")
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')
                
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if (self.step == self.train_num_steps) or (self.step != 0 and self.step % self.save_and_sample_every == 0):
                milestone = self.step // self.save_and_sample_every if self.step < self.train_num_steps else 'final'
                is_output_png = self.ds.file_ext == '.png'
                if self.save_and_sample_mode == 'one_rand':
                    inputs_ = torch.unsqueeze(inputs[0], dim=0)
                    if self.mode == "interpolation":
                        gt_ = torch.unsqueeze(gt[0], dim=0)
                        all_images = self.ema_model.inference(x_in=gt_, mask=inputs_, clip_denoised=is_output_png)
                    else:
                        all_images = self.ema_model.inference(x_in=inputs_, clip_denoised=is_output_png)

                    if is_output_png:
                        all_images = (all_images + 1) * 0.5
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 6)
                    else: np.save(str(self.results_folder / f'sample-{milestone}.npy'), all_images.cpu().detach().numpy())
                else:
                    pass
                self.save(milestone)

            self.step += 1

        print('training completed')
