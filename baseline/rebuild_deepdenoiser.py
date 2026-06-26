"""C1 baseline A: trace-wise DeepDenoiser, then save full record sections.

Usage:
  python baseline/rebuild_deepdenoiser.py --model pretrained --train_only
  python baseline/rebuild_deepdenoiser.py --model finetuned --test_only
  python baseline/rebuild_deepdenoiser.py --model finetuned --refinetune --finetune_epochs 10
  python baseline/rebuild_deepdenoiser.py --model finetuned --mcs --test_only
  python baseline/rebuild_deepdenoiser.py --model finetuned --datatype nwp --test_only
"""
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
from baseline.deepdenoiser_bridge import (
    DEFAULT_MODEL_DIR,
    denoise_profiles_tracewise,
    ensure_finetuned_checkpoint_dir,
    export_finetune_dataset,
    import_from_predict_results,
    log_checkpoint_resolution,
    read_finetune_training_fs,
    resolve_pretrained_model_dir,
    run_finetune,
    weights_match_pretrained,
    write_training_manifest,
)
from profiledd import Profiles

RESULT_ROOT = Path("results/baseline/deepdenoiser")
DEFAULT_FINETUNE_DATA = RESULT_ROOT / "finetune_data"
DEFAULT_FINETUNED_MODEL = DEFAULT_FINETUNE_DATA / "finetuned_model"
DATATYPES = ("obs", "mcs", "nwp")


def parse_args():
    p = argparse.ArgumentParser(description="DeepDenoiser trace-wise baseline rebuild")
    p.add_argument(
        "--model",
        choices=("pretrained", "finetuned"),
        default="pretrained",
        help="Use published weights or fine-tuned checkpoint",
    )
    p.add_argument("--train_only", action="store_true", help="Process training stations only")
    p.add_argument("--test_only", action="store_true", help="Process test stations only")
    p.add_argument(
        "--datatype",
        "--data_type",
        choices=DATATYPES,
        default="obs",
        help="Input data source (obs, mcs, or nwp)",
    )
    p.add_argument(
        "--mcs",
        action="store_true",
        help="Shorthand for --datatype mcs",
    )
    p.add_argument(
        "--finetuned_model_dir",
        type=Path,
        default=DEFAULT_FINETUNED_MODEL,
        help="Fine-tuned checkpoint directory (default: finetune_data/finetuned_model)",
    )
    p.add_argument(
        "--refinetune",
        action="store_true",
        help="Re-run fine-tuning before inference (default: use existing checkpoint)",
    )
    p.add_argument("--finetune_epochs", type=int, default=5, help="Epochs when --refinetune")
    p.add_argument("--max_stations", type=int, default=None)
    p.add_argument(
        "--lopo_enable",
        action="store_true",
        help="Use leave-one-pass-out diversity stacks as training targets",
    )
    p.add_argument(
        "--skip_denoise",
        action="store_true",
        help="Reuse work/dd_output/results from a finished predict run (save/rebuild only)",
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
        help=(
            "For --datatype nwp: obs = OBS passes; "
            "mcs = MCS input with OBS target (default: mcs)"
        ),
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


def resolve_datatype(args) -> str:
    if args.mcs and args.datatype != "obs":
        raise SystemExit("Use only one of --mcs and --datatype.")
    return "mcs" if args.mcs else args.datatype


def resolve_source(datatype: str, source: str | None) -> str:
    if source is not None:
        return source
    if datatype == "nwp":
        return "mcs"
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
    if datatype == "nwp":
        if source == "obs":
            return load_nwp_obs_profiles(
                train_only=train_only,
                test_only=test_only,
                max_stations=max_stations,
                lopo_enable=lopo_enable,
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
    return FRAGMENT_KWARGS


def output_tag_for(datatype: str, source: str, dataset: str) -> str:
    if datatype == "obs":
        return dataset
    if datatype == "nwp" and source == "obs":
        return f"{datatype}-{source}-{dataset}"
    return f"{datatype}-{dataset}"


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
    source = resolve_source(datatype, args.source)
    if args.source is not None and datatype != "nwp":
        raise SystemExit("--source applies only to --datatype nwp.")
    train_only = not args.test_only
    test_only = args.test_only

    file_profiles = load_profiles_from_files(args.input_profiles, args.target_profiles)
    if file_profiles is not None:
        profiles_data, profiles_target = file_profiles
    else:
        profiles_data, profiles_target = load_profiles(
            datatype,
            train_only=train_only and not test_only,
            test_only=test_only,
            max_stations=args.max_stations,
            lopo_enable=args.lopo_enable,
            use_gt_noise_level=args.use_gt_noise_level,
            source=source,
        )
    if profiles_data is None:
        raise RuntimeError(f"No {datatype.upper()} stations matched the requested split.")

    tag = "train" if train_only and not test_only else ("test" if test_only else "all")
    output_tag = output_tag_for(datatype, source, tag)
    if args.output_profiles is not None:
        out_dir = args.output_profiles.parent
    elif file_profiles is not None:
        out_dir = _profiles_path(args.input_profiles).parent
    else:
        out_dir = RESULT_ROOT / f"{args.model}-{output_tag}"
    work_dir = out_dir / "work"
    training_fs = None

    if args.model == "finetuned":
        model_dir = ensure_finetuned_checkpoint_dir(args.finetuned_model_dir.resolve())
        if args.refinetune:
            ft_data = DEFAULT_FINETUNE_DATA
            if not (ft_data / "train" / "signal.csv").is_file():
                export_finetune_dataset(ft_data, train_only=True)
            train_sr = int(profiles_data.sampling_rate)
            model_dir = run_finetune(
                ft_data,
                init_model_dir=DEFAULT_MODEL_DIR,
                epochs=args.finetune_epochs,
                sampling_rate=train_sr,
            )
            training_fs = float(train_sr)
        else:
            training_fs = read_finetune_training_fs(model_dir)
            if not (model_dir / "training_manifest.json").is_file():
                write_training_manifest(model_dir, training_fs)
        log_checkpoint_resolution(model_dir, "finetuned")
        if weights_match_pretrained(model_dir):
            raise RuntimeError(
                f"Checkpoint in {model_dir} is still identical to pretrained weights."
            )
    else:
        model_dir = resolve_pretrained_model_dir()
        log_checkpoint_resolution(model_dir, "pretrained")
        if weights_match_pretrained(model_dir) is False:
            raise RuntimeError(f"Pretrained checkpoint missing or unreadable under {model_dir}")

    if args.skip_denoise:
        profiles_denoised = import_from_predict_results(profiles_data, work_dir)
    else:
        profiles_denoised = denoise_profiles_tracewise(
            profiles_data,
            work_dir=work_dir,
            model_dir=model_dir,
            training_fs=training_fs,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if file_profiles is None:
        profiles_data.write(str(out_dir / "sections.inp"))
        profiles_target.write(str(out_dir / "sections.gt"))
    profiles_denoised.write(str(args.output_profiles if args.output_profiles is not None else out_dir / "sections.out"))

    # Optional: fragmentize for patch metrics (identity mapping — output is already full section)
    fragment_kwargs = fragment_kwargs_for(datatype)
    ds_gt = profiles_target.fragmentize(**fragment_kwargs)
    ds_inp = profiles_data.fragmentize(**fragment_kwargs)
    ds_out = profiles_denoised.fragmentize(**fragment_kwargs)
    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_inp.rebuild().write(str(out_dir / "rebuild.inp"))
    ds_out.rebuild().write(str(out_dir / "rebuild.out"))
    print(f"Saved DeepDenoiser baseline outputs to {out_dir}")


if __name__ == "__main__":
    main()
