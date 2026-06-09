"""
Prepare DeepDenoiser fine-tuning data from NOTO/OBS training stations (stn_num_to_n value >= 0).

Exports paired trace npz files:
  - signal/ = diversity-stack (clean target)
  - noise/  = raw shot trace (noisy input)

Usage (from repo root, diffseis-baseline env):
  python baseline/prepare_deepdenoiser_finetune.py
  python baseline/prepare_deepdenoiser_finetune.py --holdout 05 27 --max_stations 3
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.deepdenoiser_bridge import FINETUNE_FS, FINETUNE_NT, export_finetune_dataset
from refine_train import stn_num_to_n

DEFAULT_OUTPUT = Path("results/baseline/deepdenoiser/finetune_data")


def parse_args():
    p = argparse.ArgumentParser(description="Export DeepDenoiser fine-tune training data")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output root (creates train/signal, train/noise, csv files)",
    )
    p.add_argument(
        "--holdout",
        nargs="*",
        default=[],
        metavar="STN",
        help="Station keys held out for valid/ split (e.g. 05 27)",
    )
    p.add_argument(
        "--max_stations",
        type=int,
        default=None,
        help="Limit number of training stations (debug / smoke test)",
    )
    p.add_argument("--n_components", type=int, default=3, help="Replicate 1D trace to n channels")
    return p.parse_args()


def main():
    args = parse_args()
    holdout = [k.zfill(2) for k in args.holdout]

    train_keys = [k for k, v in stn_num_to_n.items() if v >= 0 and k not in holdout]
    print(f"Training stations (value>=0, not holdout): {len(train_keys)}")
    if holdout:
        print(f"Validation holdout: {holdout}")

    manifest = export_finetune_dataset(
        args.output,
        train_only=True,
        holdout_keys=holdout,
        max_stations=args.max_stations,
        n_components=args.n_components,
    )
    print(f"Wrote fine-tune data to {args.output.resolve()} ({FINETUNE_NT} samples @ {FINETUNE_FS} Hz)")
    print(f"  train traces: {manifest['stats']['train']}")
    if manifest["stats"].get("valid"):
        print(f"  valid traces: {manifest['stats']['valid']}")
    print("Next: python baseline/train_deepdenoiser.py --data", args.output)


if __name__ == "__main__":
    main()
