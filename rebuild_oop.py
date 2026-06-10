"""DDPM OOP rebuild — same OBS input/GT pipeline as baseline/rebuild_deepdenoiser.py."""
import argparse
from pathlib import Path

from diffusion import GaussianDiffusion
from unet import UNet

from baseline.data_noto import FRAGMENT_KWARGS, load_obs_profiles
from baseline.data_noto_mcs import load_mcs_profiles

DEFAULT_RESULT_ROOT = Path("results/demultiple0212-l1l2")
DATATYPES = ("obs", "mcs", "otj", "nwp")


def parse_args():
    p = argparse.ArgumentParser(description="DDPM rebuild (aligned with C1 baseline data loading)")
    p.add_argument("--train_only", action="store_true", help="Process training stations only")
    p.add_argument("--test_only", action="store_true", help="Process test stations only")
    p.add_argument(
        "--result_path",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Checkpoint directory; outputs are written to <result_path>/{datatype}-{dataset}/",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="model-final.pt",
        help="DDPM checkpoint filename inside --result_path",
    )
    p.add_argument(
        "--datatype",
        choices=DATATYPES,
        default="obs",
        help="Input data source (obs, mcs, otj, nwp)",
    )
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda")
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

    result_root = Path(args.result_path)
    out_dir = result_root / f"{datatype}-{dataset}"
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
    ds_output = ds_data.denoise(
        diffusion, str(ckpt_path), args.batch_size, args.device, trace_mute_ratio=args.trace_mute_ratio
    )

    profiles_data.write(str(out_dir / "sections.inp"))
    profiles_target.write(str(out_dir / "sections.gt"))

    profiles_out = ds_output.rebuild()
    profiles_out.write(str(out_dir / "sections.out"))

    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_data.rebuild().write(str(out_dir / "rebuild.inp"))
    profiles_out.write(str(out_dir / "rebuild.out"))
    print(f"Saved DDPM rebuild outputs ({datatype}-{dataset}) to {out_dir}")


if __name__ == "__main__":
    main()
