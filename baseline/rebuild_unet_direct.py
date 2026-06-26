"""C1 baseline B: deterministic U-Net inference (no DDPM iteration)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import pickle

from data_noto import FRAGMENT_KWARGS, load_obs_profiles
from data_noto_mcs import load_mcs_profiles
from data_nwp import (
    FRAGMENT_KWARGS as NWP_FRAGMENT_KWARGS,
    load_nwp_mcs_profiles,
    load_nwp_obs_profiles,
)
from data_otj import FRAGMENT_KWARGS as OTJ_FRAGMENT_KWARGS, load_otj_obs_profiles, load_otj_profiles
from direct_denoiser import DirectDenoiser
from profiledd import Profiles

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
    p.add_argument(
        "--use_gt_noise_level",
        action="store_true",
        help="Normalize input and ground truth using the target pass median noise level",
    )
    p.add_argument(
        "--source",
        choices=("obs", "mcs"),
        default=None,
        help="For otj/nwp: obs = OBS passes; mcs = MCS input with OBS surrogate target "
        "(default: obs for otj, mcs for nwp)",
    )
    p.add_argument(
        "--input_profiles",
        "--profiles_input",
        type=Path,
        default=None,
        help="Read input Profiles from a saved .npz or pickle file instead of a dataloader.",
    )
    p.add_argument(
        "--target_profiles",
        "--profiles_target",
        "--gt_profiles",
        type=Path,
        default=None,
        help="Read ground-truth Profiles from a saved .npz or pickle file instead of a dataloader.",
    )
    p.add_argument(
        "--output_profiles",
        "--profiles_output",
        type=Path,
        default=None,
        help="Write denoised output Profiles to this path.",
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


def resolve_source(datatype: str, source: str | None) -> str:
    if source is not None:
        return source
    if datatype == "nwp":
        return "mcs"
    if datatype == "otj":
        return "obs"
    return "obs"


def load_profiles(
    datatype: str,
    train_only: bool,
    test_only: bool,
    max_stations,
    lopo_enable: bool,
    use_gt_noise_level: bool,
    source: str = "obs",
):
    if datatype == "obs":
        return load_obs_profiles(
            train_only=train_only,
            test_only=test_only,
            max_stations=max_stations,
            lopo_enable=lopo_enable,
            use_gt_noise_level=use_gt_noise_level,
        )
    if datatype == "mcs":
        if lopo_enable:
            raise SystemExit("--lopo_enable is not supported with --datatype mcs.")
        return load_mcs_profiles(
            train_only=train_only,
            test_only=test_only,
            max_stations=max_stations,
            use_gt_noise_level=use_gt_noise_level,
        )
    if datatype == "otj":
        if lopo_enable:
            raise SystemExit("--lopo_enable is not supported with --datatype otj.")
        if source == "mcs":
            return load_otj_profiles(
                train_only=train_only,
                test_only=test_only,
                max_stations=max_stations,
                use_gt_noise_level=use_gt_noise_level,
            )
        return load_otj_obs_profiles(
            train_only=train_only,
            test_only=test_only,
            max_stations=max_stations,
            use_gt_noise_level=use_gt_noise_level,
        )
    if datatype == "nwp":
        if source == "obs":
            if lopo_enable:
                raise SystemExit("--lopo_enable is not supported with --datatype nwp --source obs.")
            return load_nwp_obs_profiles(
                train_only=train_only,
                test_only=test_only,
                max_stations=max_stations,
                use_gt_noise_level=use_gt_noise_level,
            )
        if lopo_enable:
            raise SystemExit("--lopo_enable is not supported with --datatype nwp.")
        return load_nwp_mcs_profiles(
            train_only=train_only,
            test_only=test_only,
            max_stations=max_stations,
            use_gt_noise_level=use_gt_noise_level,
        )
    raise NotImplementedError(f"Dataloader for datatype '{datatype}' is not wired up yet.")


def fragment_kwargs_for(datatype: str) -> dict:
    if datatype == "nwp":
        return NWP_FRAGMENT_KWARGS
    if datatype == "otj":
        return OTJ_FRAGMENT_KWARGS
    return FRAGMENT_KWARGS


def _profiles_path(path: Path) -> Path:
    """Accept exact paths and Profiles.write stems (which save as <name>.npz)."""
    if path.is_file():
        return path
    npz_path = path.with_suffix(path.suffix + ".npz") if path.suffix else Path(f"{path}.npz")
    if npz_path.is_file():
        return npz_path
    raise FileNotFoundError(f"Profiles file not found: {path}")


def read_profiles(path: Path) -> Profiles:
    """Read a Profiles object from Profiles.write .npz output or a pickle file."""
    path = _profiles_path(path)
    if path.suffix == ".npz":
        return Profiles.read(path)

    try:
        return Profiles.read(path)
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)

    if isinstance(obj, Profiles):
        return obj
    if isinstance(obj, dict) and "data" in obj:
        return Profiles(
            obj["data"],
            sampling_rate=obj.get("sampling_rate"),
            filter_history=obj.get("filter_history", []),
            reduction_vel=obj.get("reduction_vel"),
            offsets=obj.get("offsets"),
            first_arrival_reference=obj.get("first_arrival_reference"),
        )
    raise TypeError(f"Expected a Profiles object or Profiles-like dict in {path}")


def load_profiles_from_files(input_profiles: Path | None, target_profiles: Path | None):
    if input_profiles is None and target_profiles is None:
        return None
    if input_profiles is None or target_profiles is None:
        raise SystemExit("Use --input_profiles and --target_profiles together.")
    return read_profiles(input_profiles), read_profiles(target_profiles)


def main():
    args = parse_args()
    if args.train_only and args.test_only:
        raise SystemExit("Use only one of --train_only or --test_only.")

    datatype = resolve_datatype(args)
    dataset = resolve_dataset(args.train_only, args.test_only)
    train_only = args.train_only and not args.test_only
    source = resolve_source(datatype, args.source)
    if args.source is not None and datatype not in ("otj", "nwp"):
        raise SystemExit("--source applies only to --datatype otj or nwp.")

    file_profiles = load_profiles_from_files(args.input_profiles, args.target_profiles)
    if file_profiles is not None:
        profiles_data, profiles_target = file_profiles
    else:
        profiles_data, profiles_target = load_profiles(
            datatype,
            train_only=train_only,
            test_only=args.test_only,
            max_stations=args.max_stations,
            lopo_enable=args.lopo_enable,
            use_gt_noise_level=args.use_gt_noise_level,
            source=source,
        )
    if profiles_data is None:
        raise RuntimeError(f"No {datatype.upper()} stations matched the requested split.")

    ds_gt = profiles_target.fragmentize(**fragment_kwargs_for(datatype))
    ds_data = profiles_data.fragmentize(**fragment_kwargs_for(datatype))

    model = DirectDenoiser(image_size=ds_data.unit_size, dropout=0.5, loss_type='l1l2')
    model = model.to('cuda')

    ds_output = ds_data.denoise_direct(
        model, str(args.checkpoint), args.batch_size, 'cuda', trace_mute_ratio=args.trace_mute_ratio
    )

    if args.output_profiles is not None:
        out_dir = args.output_profiles.parent
    elif file_profiles is not None:
        out_dir = _profiles_path(args.input_profiles).parent
    else:
        out_dir = RESULT_DIR / f"{datatype}-{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_out = ds_output.rebuild()
    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_data.rebuild().write(str(out_dir / "rebuild.inp"))
    profiles_out.write(str(args.output_profiles if args.output_profiles is not None else out_dir / "rebuild.out"))
    print(f"Saved deterministic U-Net baseline ({datatype}-{dataset}) to {out_dir}")


if __name__ == "__main__":
    main()
