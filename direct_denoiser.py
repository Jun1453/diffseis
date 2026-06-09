"""Deterministic 2D U-Net denoiser (DDPM ablation): one forward pass per patch."""
import torch
from torch import nn

from unet import UNet


class L1L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')
        self.l2 = nn.MSELoss(reduction='sum')

    def forward(self, input, target):
        return 0.9 * (self.l1(input, target) + 0.1 * self.l2(input, target))


class DirectDenoiser(nn.Module):
    """Supervised image-to-image mapper: noisy patch -> clean patch."""

    def __init__(
        self,
        image_size,
        dropout=0.5,
        loss_type='l1l2',
        inner_channel=64,
        channel_mults=(1, 2, 4, 8, 16),
        attn_res=(),
    ):
        super().__init__()
        if isinstance(image_size, (tuple, list)):
            h, w = image_size
        else:
            h = w = image_size
        self.image_size = (h, w)
        self.unet = UNet(
            in_channel=1,
            out_channel=1,
            dropout=dropout,
            image_size=w,
            with_noise_level_emb=False,
            inner_channel=inner_channel,
            channel_mults=channel_mults,
            attn_res=attn_res,
        )
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        elif loss_type == 'l1l2':
            self.loss_func = L1L2Loss()
        else:
            raise NotImplementedError(loss_type)

    def forward(self, x_cond, x_start=None):
        pred = self.unet(x_cond, None)
        if x_start is None:
            return pred
        return self.loss_func(pred, x_start)
