"""Fine-tune DeepDenoiser on prepared NOTO/OBS data."""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.deepdenoiser_bridge import DEFAULT_FINETUNED_MODEL, DEFAULT_MODEL_DIR, run_finetune

DEFAULT_DATA = Path("results/baseline/deepdenoiser/finetune_data")


def _gpu_status_hint() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return f"GPU memory now: {out.stdout.strip()}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Check GPU with: nvidia-smi"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune DeepDenoiser on exported NOTO data")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Root from prepare_deepdenoiser_finetune.py")
    p.add_argument(
        "--init_model",
        type=Path,
        default=None,
        help="Checkpoint dir to seed training (default: pretrained). "
        "With --resume, optional source to copy into {data}/finetuned_model/ before continuing.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from checkpoint in {data}/finetuned_model/ "
        f"(default location: {DEFAULT_FINETUNED_MODEL.relative_to(_ROOT)})",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default 8; reduce to 4 if GPU OOM)",
    )
    p.add_argument("--sampling_rate", type=int, default=250, help="OBS sampling rate in finetune NPZ (Hz)")
    p.add_argument(
        "--loss_type",
        default="cross_entropy",
        choices=("cross_entropy", "mean_squared", "IOU", "l1l2"),
        help="DeepDenoiser mask loss (default cross_entropy; matches pretrained model)",
    )
    p.add_argument(
        "--snr_threshold",
        type=float,
        default=2.0,
        help="Min SNR for a trace to enter training queue (default 2 for OBS; upstream default 10)",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Train on CPU (slow but avoids GPU OOM when another job uses the GPU)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.resume:
        init_model_dir = args.init_model
    else:
        init_model_dir = args.init_model or DEFAULT_MODEL_DIR

    try:
        out_dir = run_finetune(
            args.data,
            init_model_dir=init_model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sampling_rate=args.sampling_rate,
            loss_type=args.loss_type,
            snr_threshold=args.snr_threshold,
            resume=args.resume,
            cpu=args.cpu,
        )
    except subprocess.CalledProcessError as exc:
        print(
            "\nDeepDenoiser training failed.\n"
            "If the error was GPU out of memory:\n"
            "  1. Free the GPU (stop other PyTorch/TF jobs), then retry.\n"
            "  2. Lower batch size:  --batch_size 4\n"
            "  3. Train on CPU:      --cpu\n"
            f"  {_gpu_status_hint()}\n",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc
    print(f"Fine-tuned model saved under {out_dir}")


if __name__ == "__main__":
    main()
