"""
Compare DeepDenoiser fine-tune pairs: train/noise (raw) vs train/signal (diversity stack).

Usage (repo root):
  python baseline/inspect_finetune_data.py
  python baseline/inspect_finetune_data.py --station 20 --shot 4 --trace 219
  python baseline/inspect_finetune_data.py --random 6 --out results/baseline/deepdenoiser/finetune_inspect
  python baseline/inspect_finetune_data.py --random 3 --verify_source
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_noto import load_station_pair
from baseline.deepdenoiser_bridge import FINETUNE_FS, FINETUNE_NT

DEFAULT_DATA = Path("results/baseline/deepdenoiser/finetune_data")
FNAME_RE = re.compile(r"^(?P<stn>\d{2})_s(?P<shot>\d)_t(?P<tr>\d{4})\.npz$")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize DeepDenoiser finetune noisy/clean pairs")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Finetune export root")
    p.add_argument("--split", default="train", choices=("train", "valid"))
    p.add_argument("--station", type=str, default=None, help="Station key, e.g. 20")
    p.add_argument("--shot", type=int, default=None, help="Shot index 0-4")
    p.add_argument("--trace", type=int, default=None, help="Trace index")
    p.add_argument("--random", type=int, default=4, help="Number of random pairs to plot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tmax",
        type=int,
        default=6000,
        help="Plot first N samples (6000 = OBS window before zero pad; 9001 = full npz)",
    )
    p.add_argument(
        "--verify_source",
        action="store_true",
        help="Reload OBS via load_station_pair and overlay export vs source diff",
    )
    p.add_argument("--out", type=Path, default=None, help="Save PNGs here (default: show only)")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def _load_pair(root: Path, split: str, fname: str) -> tuple[np.ndarray, np.ndarray, int]:
    sig_path = root / split / "signal" / fname
    noise_path = root / split / "noise" / fname
    sig = np.load(sig_path)
    noise = np.load(noise_path)
    clean = np.asarray(sig["data"][:, 0], dtype=np.float32)
    raw = np.asarray(noise["data"][:, 0], dtype=np.float32)
    itp = int(np.asarray(sig["itp"]).reshape(()))
    return raw, clean, itp


def _parse_fname(fname: str) -> dict | None:
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return {"stn": m.group("stn"), "shot": int(m.group("shot")), "tr": int(m.group("tr"))}


def _select_fnames(csv_path: Path, args) -> list[str]:
    df = pd.read_csv(csv_path)
    if args.station is not None:
        stn = args.station.zfill(2)
        mask = df["fname"].str.startswith(f"{stn}_")
        if args.shot is not None:
            mask &= df["fname"].str.contains(f"_s{args.shot}_")
        if args.trace is not None:
            mask &= df["fname"].str.endswith(f"_t{args.trace:04d}.npz")
        picks = df.loc[mask, "fname"].tolist()
        if not picks:
            raise SystemExit(f"No rows match station={stn} shot={args.shot} trace={args.trace}")
        return picks[: args.random or len(picks)]

    rng = np.random.default_rng(args.seed)
    n = min(args.random, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx]["fname"].tolist()


def _reload_source(meta: dict, tmax: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    pf, clean_for_shot, padded_arrival = load_station_pair(meta["stn"], time_samples=tmax)
    shot, tr = meta["shot"], meta["tr"]
    raw = np.asarray(pf[shot, :tmax, tr], dtype=np.float32)
    clean = np.asarray(clean_for_shot(shot)[:tmax, tr], dtype=np.float32)
    itp = int(padded_arrival[tr])
    return raw, clean, itp, float(pf.sampling_rate)


def _plot_pair(
    fname: str,
    raw: np.ndarray,
    clean: np.ndarray,
    itp: int,
    fs: float,
    tmax: int,
    verify: dict | None,
    out_path: Path | None,
    dpi: int,
) -> None:
    n = min(tmax, len(raw), len(clean))
    t = np.arange(n) / fs
    meta = _parse_fname(fname)

    fig, axes = plt.subplots(3 if verify else 2, 1, figsize=(11, 7 if verify else 5), sharex=True)
    if not verify:
        axes = [axes[0], axes[1]]

    ax0, ax1 = axes[0], axes[1]
    ax0.plot(t, raw[:n], color="C0", lw=0.7, label="noise (raw pass)")
    ax0.axvline(itp / fs, color="k", ls="--", alpha=0.5, label=f"itp={itp}")
    ax0.set_ylabel("Amplitude")
    ax0.set_title(f"{fname} — noisy input")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.plot(t, clean[:n], color="C2", lw=0.7, label="signal (diversity stack)")
    ax1.plot(t, raw[:n], color="C0", lw=0.5, alpha=0.35, label="noisy (overlay)")
    ax1.axvline(itp / fs, color="k", ls="--", alpha=0.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Clean target vs noisy")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    if verify and meta is not None:
        src_raw, src_clean, src_itp, src_fs = verify["source"]
        m = min(n, len(src_raw), len(src_clean))
        ax2 = axes[2]
        ax2.plot(t[:m], raw[:m] - src_raw[:m], label="raw export − source", lw=0.7)
        ax2.plot(t[:m], clean[:m] - src_clean[:m], label="clean export − source", lw=0.7)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel("Δ")
        ax2.set_xlabel("Time (s)")
        ax2.set_title(
            f"Export vs OBS reload (itp export={itp}, source={src_itp}, fs={src_fs:.0f} Hz)"
        )
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
        max_raw = float(np.max(np.abs(raw[:m] - src_raw[:m])))
        max_clean = float(np.max(np.abs(clean[:m] - src_clean[:m])))
        fig.suptitle(
            f"Station {meta['stn']} shot {meta['shot']} trace {meta['tr']} | "
            f"max |Δraw|={max_raw:.2e}, max |Δclean|={max_clean:.2e}",
            fontsize=10,
        )
    else:
        axes[-1].set_xlabel("Time (s)")
        if meta:
            fig.suptitle(f"Station {meta['stn']} shot {meta['shot']} trace {meta['tr']}", fontsize=10)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")
    else:
        plt.show()


def main():
    args = parse_args()
    root = args.data.resolve()
    csv_path = root / args.split / "signal.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")

    manifest_path = root / "manifest.json"
    fs = FINETUNE_FS
    if manifest_path.is_file():
        import json

        manifest = json.loads(manifest_path.read_text())
        fs = float(manifest.get("sampling_rate_hz", FINETUNE_FS))
        print(f"Dataset: {manifest.get('stats', {}).get(args.split, '?')} traces @ {fs} Hz, nt={manifest.get('finetune_nt', FINETUNE_NT)}")

    fnames = _select_fnames(csv_path, args)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for i, fname in enumerate(fnames):
        raw, clean, itp = _load_pair(root, args.split, fname)
        verify = None
        meta = _parse_fname(fname)
        if args.verify_source and meta is not None:
            src = _reload_source(meta, tmax=args.tmax)
            verify = {"source": src}
        out_path = (args.out / f"{Path(fname).stem}.png") if args.out else None
        _plot_pair(fname, raw, clean, itp, fs, args.tmax, verify, out_path, args.dpi)

    if args.out:
        print(f"Saved {len(fnames)} figure(s) under {args.out.resolve()}")


if __name__ == "__main__":
    main()
