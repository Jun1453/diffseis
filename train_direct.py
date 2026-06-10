"""Train deterministic 2D U-Net baseline (C1 DDPM ablation)."""
import argparse

import numpy as np

from data_noto import FRAGMENT_KWARGS, load_obs_profiles
from direct_denoiser import DirectDenoiser


def parse_args():
    p = argparse.ArgumentParser(description="Train deterministic U-Net baseline")
    p.add_argument(
        "--lopo_enable",
        action="store_true",
        help="Use leave-one-pass-out diversity stacks as training targets",
    )
    return p.parse_args()


args = parse_args()
profiles_data, profiles_target = load_obs_profiles(
    train_only=True,
    lopo_enable=args.lopo_enable,
)

ds_gt = profiles_target.fragmentize(**FRAGMENT_KWARGS)
ds_data = profiles_data.fragmentize(**FRAGMENT_KWARGS)
ds_data.set_ground_truth(ds_gt)
del profiles_data
del profiles_target

model = DirectDenoiser(
    image_size=ds_data.unit_size,
    dropout=0.5,
    loss_type='l1l2',
)

if __name__ == '__main__':
    ds_data.train_direct(
        model,
        200,
        32,
        gradient_accumulate_every=2,
        save_every=10,
        learning_rate=3e-5,
        results_folder='results/baseline/unet-direct',
        trace_mute_ratio=0.1,
    )
