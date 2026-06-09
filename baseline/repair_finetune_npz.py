"""
Repair exported DeepDenoiser fine-tune npz files in place.

Fixes layout (nt, nch), channels field, and pads/truncates to FINETUNE_NT.

Usage:
  python baseline/repair_finetune_npz.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.deepdenoiser_bridge import DEFAULT_CHANNELS, FINETUNE_NT, _pad_train_trace

DEFAULT_ROOT = Path("results/baseline/deepdenoiser/finetune_data")


def repair_npz(path: Path, channels: str = DEFAULT_CHANNELS, target_nt: int = FINETUNE_NT) -> bool:
    z = np.load(path, allow_pickle=True)
    data = np.asarray(z["data"], dtype=np.float32)
    changed = False

    if data.ndim == 3 and data.shape[1] == 1:
        data = np.squeeze(data, axis=1)
        changed = True
    elif data.ndim == 1:
        data = data[:, np.newaxis]
        changed = True

    padded = _pad_train_trace(data, target_nt)
    if padded.shape != data.shape:
        data = padded
        changed = True

    itp = int(np.asarray(z["itp"]).reshape(()))
    itp = int(np.clip(itp, 0, data.shape[0] - 1))

    if "channels" in z.files:
        ch = str(np.asarray(z["channels"]).reshape(()))
    else:
        ch = channels
        changed = True

    if not changed:
        return False

    np.savez(path, data=data, itp=np.int64(itp), channels=np.array(ch))
    return True


def repair_tree(root: Path) -> tuple[int, int]:
    n_fixed = n_skip = 0
    for npz_path in sorted(root.glob("**/signal/*.npz")) + sorted(root.glob("**/noise/*.npz")):
        if repair_npz(npz_path):
            n_fixed += 1
        else:
            n_skip += 1
    return n_fixed, n_skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    fixed, skipped = repair_tree(args.root)
    print(
        f"Repaired {fixed} npz files under {args.root.resolve()} "
        f"(nt={FINETUNE_NT}; {skipped} already ok)"
    )


if __name__ == "__main__":
    main()
