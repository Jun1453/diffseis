"""C1 baseline B: deterministic U-Net inference (no DDPM iteration)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from data_noto import FRAGMENT_KWARGS, load_obs_profiles
from direct_denoiser import DirectDenoiser

DEFAULT_CKPT = Path("results/baseline/unet-direct/model-final.pt")
RESULT_DIR = Path("results/baseline/unet-direct/inference")


def parse_args():
    p = argparse.ArgumentParser(description="Deterministic U-Net baseline rebuild")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--lopo_enable",
        action="store_true",
        help="Use leave-one-pass-out diversity stacks as training targets",
    )
    p.add_argument(
        "--trace_mute_ratio",
        type=float,
        default=0,
        help="Fraction of traces randomly muted per patch during inference (0 disables)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    profiles_data, profiles_target = load_obs_profiles(
        train_only=not args.test_only,
        test_only=args.test_only,
        lopo_enable=args.lopo_enable,
    )
    if profiles_data is None:
        raise RuntimeError("No OBS stations matched the requested split.")

    ds_gt = profiles_target.fragmentize(**FRAGMENT_KWARGS)
    ds_data = profiles_data.fragmentize(**FRAGMENT_KWARGS)

    model = DirectDenoiser(image_size=ds_data.unit_size, dropout=0.5, loss_type='l1l2')
    model = model.to('cuda')

    ds_output = ds_data.denoise_direct(
        model, str(args.checkpoint), args.batch_size, 'cuda', trace_mute_ratio=args.trace_mute_ratio
    )

    tag = "test" if args.test_only else "train"
    out_dir = RESULT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_data.rebuild().write(str(out_dir / "rebuild.inp"))
    ds_output.rebuild().write(str(out_dir / "rebuild.out"))
    print(f"Saved deterministic U-Net baseline to {out_dir}")


if __name__ == "__main__":
    main()
