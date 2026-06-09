"""DDPM OOP rebuild — same OBS input/GT pipeline as baseline/rebuild_deepdenoiser.py."""
import argparse
from pathlib import Path

import numpy as np
from diffusion import GaussianDiffusion
from unet import UNet

from baseline.data_noto import FRAGMENT_KWARGS, load_obs_profiles

DEFAULT_RESULT_ROOT = Path("results/demultiple0212-l1l2")


def parse_args():
    p = argparse.ArgumentParser(description="DDPM rebuild (aligned with C1 baseline data loading)")
    p.add_argument("--train_only", action="store_true", help="Process training stations only")
    p.add_argument("--test_only", action="store_true", help="Process test stations only")
    p.add_argument(
        "--result_path",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Output directory (checkpoint expected at <result_path>/model-final.pt)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="model-final.pt",
        help="DDPM checkpoint filename inside --result_path",
    )
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_stations", type=int, default=None)
    p.add_argument(
        "--lopo_enable",
        action="store_true",
        help="Use leave-one-pass-out diversity stacks as training targets",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.train_only and args.test_only:
        raise SystemExit("Use only one of --train_only or --test_only.")

    profiles_data, profiles_target = load_obs_profiles(
        train_only=args.train_only and not args.test_only,
        test_only=args.test_only,
        max_stations=args.max_stations,
        lopo_enable=args.lopo_enable,
    )
    if profiles_data is None:
        raise RuntimeError("No OBS stations matched the requested split.")

    tag = "train" if args.train_only and not args.test_only else ("test" if args.test_only else "all")
    result_root = Path(args.result_path)
    out_dir = result_root / tag if tag != "all" else result_root
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = result_root / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"DDPM checkpoint not found: {ckpt_path}")

    ds_gt = profiles_target.fragmentize(**FRAGMENT_KWARGS)
    ds_data = profiles_data.fragmentize(**FRAGMENT_KWARGS)

    model = UNet(
        in_channel=2,
        out_channel=1,
        dropout=0.5,
        image_size=ds_data.unit_size[1],
    )
    diffusion = GaussianDiffusion(
        model,
        mode="demultiple",
        channels=1,
        image_size=ds_data.unit_size,
        timesteps=2000,
        loss_type="l1l2",
        noise_mix_ratio=None,
    )

    model = model.to(args.device)
    diffusion = diffusion.to(args.device)
    ds_output = ds_data.denoise(diffusion, str(ckpt_path), args.batch_size, args.device)

    profiles_data.write(str(out_dir / "sections.inp"))
    profiles_target.write(str(out_dir / "sections.gt"))

    profiles_out = ds_output.rebuild()
    profiles_out.write(str(out_dir / "sections.out"))

    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_data.rebuild().write(str(out_dir / "rebuild.inp"))
    profiles_out.write(str(out_dir / "rebuild.out"))
    print(f"Saved DDPM rebuild outputs to {out_dir}")


if __name__ == "__main__":
    main()
