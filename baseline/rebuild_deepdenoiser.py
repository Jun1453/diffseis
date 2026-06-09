"""C1 baseline A: trace-wise DeepDenoiser on OBS input, then save full record sections.

Usage:
  python baseline/rebuild_deepdenoiser.py --model pretrained --train_only
  python baseline/rebuild_deepdenoiser.py --model finetuned --test_only
  python baseline/rebuild_deepdenoiser.py --model finetuned --refinetune --finetune_epochs 10
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np

from baseline.data_noto import FRAGMENT_KWARGS, load_obs_profiles
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
    return p.parse_args()


def main():
    args = parse_args()
    train_only = not args.test_only
    test_only = args.test_only

    profiles_data, profiles_target = load_obs_profiles(
        train_only=train_only and not test_only,
        test_only=test_only,
        max_stations=args.max_stations,
        lopo_enable=args.lopo_enable,
    )
    if profiles_data is None:
        raise RuntimeError("No OBS stations matched the requested split.")

    tag = "train" if train_only and not test_only else ("test" if test_only else "all")
    out_dir = RESULT_ROOT / f"{args.model}-{tag}"
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
    profiles_data.write(str(out_dir / "sections.inp"))
    profiles_target.write(str(out_dir / "sections.gt"))
    profiles_denoised.write(str(out_dir / "sections.out"))

    # Optional: fragmentize for patch metrics (identity mapping — output is already full section)
    ds_gt = profiles_target.fragmentize(**FRAGMENT_KWARGS)
    ds_inp = profiles_data.fragmentize(**FRAGMENT_KWARGS)
    ds_out = profiles_denoised.fragmentize(**FRAGMENT_KWARGS)
    ds_gt.rebuild().write(str(out_dir / "rebuild.gt"))
    ds_inp.rebuild().write(str(out_dir / "rebuild.inp"))
    ds_out.rebuild().write(str(out_dir / "rebuild.out"))
    print(f"Saved DeepDenoiser baseline outputs to {out_dir}")


if __name__ == "__main__":
    main()
