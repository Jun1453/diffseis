"""Compare C1 baseline rebuild outputs vs diversity-stack ground truth."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np

from baseline.arrival_metrics import evaluate_arrival_metrics, format_metrics_table


def load_section(path: Path) -> np.ndarray:
    """Load rebuild array only (no filter callables required for metrics)."""
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["data"])


def section_metrics(pred: np.ndarray, gt: np.ndarray, inp: np.ndarray) -> dict:
    """RMS misfit and improvement vs input (lower is better)."""
    mask = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(inp)
    if not np.any(mask):
        return dict(rms_pred=np.nan, rms_inp=np.nan, ratio=np.nan)
    diff_pred = pred[mask] - gt[mask]
    diff_inp = inp[mask] - gt[mask]
    rms_pred = float(np.sqrt(np.mean(diff_pred ** 2)))
    rms_inp = float(np.sqrt(np.mean(diff_inp ** 2)))
    ratio = rms_pred / rms_inp if rms_inp > 0 else np.nan
    return dict(rms_pred=rms_pred, rms_inp=rms_inp, ratio=ratio)


def _rebuild_path(run_dir: Path, stem: str) -> Path | None:
    """Profiles.write() adds .npz; accept stem with or without suffix."""
    for name in (stem, f"{stem}.npz"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _sampling_rate(path: Path, default: float = 250.0) -> float:
    with np.load(path, allow_pickle=True) as z:
        if "sampling_rate" in z.files:
            return float(np.asarray(z["sampling_rate"]).reshape(()))
    return default


def evaluate_run(run_dir: Path, label: str, fs: float | None = None):
    inp_path = _rebuild_path(run_dir, "rebuild.inp")
    out_path = _rebuild_path(run_dir, "rebuild.out")
    gt_path = _rebuild_path(run_dir, "rebuild.gt")
    for stem, p in (
        ("rebuild.inp", inp_path),
        ("rebuild.out", out_path),
        ("rebuild.gt", gt_path),
    ):
        if p is None:
            print(f"[{label}] skip {run_dir}: missing {stem}(.npz)")
            return
    inp = load_section(inp_path)
    out = load_section(out_path)
    gt = load_section(gt_path)
    if fs is None:
        fs = _sampling_rate(gt_path)

    m = section_metrics(out, gt, inp)
    print(
        f"[{label}] {run_dir}\n"
        f"  RMS(output-GT)={m['rms_pred']:.4f}  RMS(input-GT)={m['rms_inp']:.4f}  "
        f"ratio={m['ratio']:.4f}"
    )

    arrival = evaluate_arrival_metrics(out, gt, inp=inp, fs=fs)
    print(format_metrics_table(arrival))


def main():
    p = argparse.ArgumentParser(description="Evaluate C1 baseline rebuild folders")
    p.add_argument("run_dirs", nargs="*", type=Path, help="Directories containing rebuild.{inp,out,gt}")
    p.add_argument("--scan_defaults", action="store_true", help="Scan results/baseline/*")
    p.add_argument("--fs", type=float, default=None, help="Sampling rate Hz (default: from rebuild.gt)")
    args = p.parse_args()

    dirs = list(args.run_dirs)
    if args.scan_defaults:
        root = Path("results/baseline")
        if root.is_dir():
            for sub in sorted(root.rglob("rebuild.out*")):
                if sub.name.startswith("rebuild.out"):
                    dirs.append(sub.parent)

    if not dirs:
        print("No run directories. Pass paths or use --scan_defaults.")
        return

    seen = set()
    for d in dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        evaluate_run(d, d.name, fs=args.fs)


if __name__ == "__main__":
    main()
