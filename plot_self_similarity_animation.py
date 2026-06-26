#!/usr/bin/env python3
"""Animate rebuild outputs from repeated DDPM runs."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from profiledd import Profiles
DEFAULT_RUNS_DIR = Path("results/self_similarity_obs_test/runs")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate rebuild.out snapshots from self-similarity runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing run_XX.rebuild.out.npz files.",
    )
    parser.add_argument("--start", type=int, default=1, help="First run index (inclusive).")
    parser.add_argument("--end", type=int, default=20, help="Last run index (inclusive).")
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        help="Profile index to display (0-based).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="Delay between frames in milliseconds.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path, e.g. results/self_similarity_obs_test/animation.gif",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second when saving to GIF.",
    )
    return parser.parse_args()
def run_paths(runs_dir: Path, start: int, end: int) -> list[Path]:
    paths = [runs_dir / f"run_{idx:02d}.rebuild.out.npz" for idx in range(start, end + 1)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing snapshot(s):\n" + "\n".join(str(p) for p in missing))
    return paths
def load_profile(path: Path, profile_idx: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        kwargs = {key: npz[key] for key in npz.files if key != "data"}
        profiles = Profiles(npz["data"], **kwargs)
    if profile_idx < 0 or profile_idx >= len(profiles):
        raise IndexError(f"{path.name}: profile {profile_idx} out of range (0..{len(profiles) -
1})")
    return np.asarray(profiles[profile_idx])
def main() -> None:
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise ValueError("--start must be >= 1 and --end must be >= --start")
    paths = run_paths(args.runs_dir, args.start, args.end)
    labels = [path.stem.replace(".rebuild.out", "") for path in paths]
    frames = [load_profile(path, args.profile) for path in paths]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(frames[0], aspect=0.15, cmap="seismic", vmin=-1, vmax=1)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time sample")
    title = ax.set_title(f"{labels[0]} | profile {args.profile}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    def update(frame_idx: int):
        im.set_data(frames[frame_idx])
        title.set_text(f"{labels[frame_idx]} | profile {args.profile}")
        return im, title
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=args.interval,
        blit=False,
        repeat=True,
    )
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        anim.save(args.save, writer=PillowWriter(fps=args.fps))
        print(f"Saved {args.save}")
    else:
        plt.show()
if __name__ == "__main__":
    main()
    