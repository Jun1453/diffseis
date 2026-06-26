"""DDPM OOP rebuild — same OBS input/GT pipeline as baseline/rebuild_deepdenoiser.py."""
import argparse
import pickle
from pathlib import Path

import numpy as np

from diffusion import GaussianDiffusion
from profiledd import Profiles
from unet import UNet

from data_noto import FRAGMENT_KWARGS, load_obs_profiles
from data_noto_mcs import load_mcs_profiles
from data_nwp import FRAGMENT_KWARGS as NWP_FRAGMENT_KWARGS, load_nwp_mcs_profiles, load_nwp_obs_profiles
from data_otj import FRAGMENT_KWARGS as OTJ_FRAGMENT_KWARGS, load_otj_obs_profiles, load_otj_profiles

DEFAULT_RESULT_ROOT = Path("results/demultiple0212-l1l2")
DATATYPES = ("obs", "mcs", "otj", "nwp")


def parse_args():
    p = argparse.ArgumentParser(description="DDPM rebuild (aligned with C1 baseline data loading)")
    p.add_argument("--train_only", action="store_true", help="Process training stations only")
    p.add_argument("--test_only", action="store_true", help="Process test stations only")
    p.add_argument(
        "--result_path",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory; outputs are written to <result_path>/{datatype}-{dataset}/. "
            "With --input_profiles and no --result_path, outputs are written beside the input file."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="model-final.pt",
        help="DDPM checkpoint filename inside --result_path",
    )
    p.add_argument(
        "--datatype",
        "--data_type",
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


def _noise_levels_by_profile(profiles: Profiles) -> np.ndarray:
    """Median absolute amplitude in the first 50 samples for each profile."""
    if profiles.size == 0:
        return np.array([], dtype=float)
    return np.array(
        [np.median(np.abs(np.asarray(profiles[i : i + 1])[:, :50, :])) for i in range(profiles.shape[0])],
        dtype=float,
    )


def print_noise_level_summary(profiles_data: Profiles, profiles_target: Profiles):
    input_levels = _noise_levels_by_profile(profiles_data)
    target_levels = _noise_levels_by_profile(profiles_target)

    def fmt(levels: np.ndarray) -> str:
        if levels.size == 0:
            return "empty"
        return (
            f"median={np.median(levels):.6g}, "
            f"min={np.min(levels):.6g}, max={np.max(levels):.6g}, n={levels.size}"
        )

    print("Noise levels after --use_gt_noise_level normalization (first 50 samples):")
    print(f"  input : {fmt(input_levels)}")
    print(f"  target: {fmt(target_levels)}")


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
    if args.use_gt_noise_level:
        print_noise_level_summary(profiles_data, profiles_target)

    result_root = Path(args.result_path) if args.result_path is not None else DEFAULT_RESULT_ROOT
    if file_profiles is not None and args.result_path is None:
        out_dir = _profiles_path(args.input_profiles).parent
    else:
        out_dir = result_root / f"{datatype}-{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = result_root / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"DDPM checkpoint not found: {ckpt_path}")

    ds_gt = profiles_target.fragmentize(**fragment_kwargs_for(datatype))
    ds_data = profiles_data.fragmentize(**fragment_kwargs_for(datatype))

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

    if file_profiles is None:
        profiles_data.write(str(out_dir / "sections.inp"))
        profiles_target.write(str(out_dir / "sections.gt"))

    profiles_out = ds_output.rebuild()

    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_data.rebuild().write(str(out_dir / "rebuild.inp"))
    profiles_out.write(str(out_dir / "rebuild.out"))
    print(f"Saved DDPM rebuild outputs ({datatype}-{dataset}) to {out_dir}")


if __name__ == "__main__":
    main()
