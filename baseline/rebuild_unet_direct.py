"""C1 baseline B: deterministic U-Net inference (no DDPM iteration)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from data_noto import FRAGMENT_KWARGS, load_obs_profiles
from data_noto_mcs import load_mcs_profiles
from direct_denoiser import DirectDenoiser

DEFAULT_CKPT = Path("results/baseline/unet-direct/model-final.pt")
RESULT_DIR = Path("results/baseline/unet-direct/inference")
DATATYPES = ("obs", "mcs", "otj", "nwp")


def parse_args():
    p = argparse.ArgumentParser(description="Deterministic U-Net baseline rebuild")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--train_only", action="store_true", help="Process training stations only")
    p.add_argument("--test_only", action="store_true", help="Process test stations only")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--datatype",
        choices=DATATYPES,
        default="obs",
        help="Input data source (obs, mcs, otj, nwp)",
    )
    p.add_argument("--max_stations", type=int, default=None)
    p.add_argument(
        "--lopo_enable",
        action="store_true",
        help="Use leave-one-pass-out diversity stacks as training targets (obs only)",
    )
    p.add_argument(
        "--trace_mute_ratio",
        type=float,
        default=0,
        help="Fraction of traces randomly muted per patch during inference (0 disables)",
    )
    p.add_argument(
        "--mcs",
        action="store_true",
        help="Shorthand for --datatype mcs",
    )
    return p.parse_args()


def resolve_dataset(train_only: bool, test_only: bool) -> str:
    if train_only and not test_only:
        return "train"
    if test_only:
        return "test"
    return "all"


def resolve_datatype(args) -> str:
    if args.mcs and args.datatype != "obs":
        raise SystemExit("Use only one of --mcs and --datatype.")
    return "mcs" if args.mcs else args.datatype


def load_profiles(datatype: str, train_only: bool, test_only: bool, max_stations, lopo_enable: bool):
    if datatype == "obs":
        return load_obs_profiles(
            train_only=train_only,
            test_only=test_only,
            max_stations=max_stations,
            lopo_enable=lopo_enable,
        )
    if datatype == "mcs":
        if lopo_enable:
            raise SystemExit("--lopo_enable is not supported with --datatype mcs.")
        return load_mcs_profiles(
            train_only=train_only,
            test_only=test_only,
            max_stations=max_stations,
        )
    raise NotImplementedError(f"Dataloader for datatype '{datatype}' is not wired up yet.")


def main():
    args = parse_args()
    if args.train_only and args.test_only:
        raise SystemExit("Use only one of --train_only or --test_only.")

    datatype = resolve_datatype(args)
    dataset = resolve_dataset(args.train_only, args.test_only)
    train_only = args.train_only and not args.test_only

    profiles_data, profiles_target = load_profiles(
        datatype,
        train_only=train_only,
        test_only=args.test_only,
        max_stations=args.max_stations,
        lopo_enable=args.lopo_enable,
    )
    if profiles_data is None:
        raise RuntimeError(f"No {datatype.upper()} stations matched the requested split.")

    ds_gt = profiles_target.fragmentize(**FRAGMENT_KWARGS)
    ds_data = profiles_data.fragmentize(**FRAGMENT_KWARGS)

    model = DirectDenoiser(image_size=ds_data.unit_size, dropout=0.5, loss_type='l1l2')
    model = model.to('cuda')

    ds_output = ds_data.denoise_direct(
        model, str(args.checkpoint), args.batch_size, 'cuda', trace_mute_ratio=args.trace_mute_ratio
    )

    out_dir = RESULT_DIR / f"{datatype}-{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_data.rebuild().write(str(out_dir / "rebuild.inp"))
    ds_output.rebuild().write(str(out_dir / "rebuild.out"))
    print(f"Saved deterministic U-Net baseline ({datatype}-{dataset}) to {out_dir}")


if __name__ == "__main__":
    main()
