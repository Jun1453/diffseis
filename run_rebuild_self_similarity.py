"""Run repeated DDPM rebuilds and measure similarity between outputs.

Defaults to the command requested by the user:

    python rebuild_oop.py --data_type obs --test_only

Each rebuild overwrites ``<result_path>/obs-test/rebuild.out.npz``, so this
driver snapshots that file after every run before calculating pairwise metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_RESULT_PATH = Path("results/baseline/ddpm")
DEFAULT_OUTPUT_DIR = Path("results/self_similarity_obs_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rebuild_oop.py repeatedly and compare rebuild.out outputs."
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of rebuild runs.")
    parser.add_argument(
        "--result_path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Checkpoint/result root passed to rebuild_oop.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-run snapshots and similarity reports are written.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used to launch rebuild_oop.py.",
    )
    parser.add_argument("--data_type", default="obs", choices=("obs", "mcs", "otj", "nwp"))
    parser.add_argument("--batch_size", type=int, default=None, help="Optional rebuild batch size.")
    parser.add_argument("--device", default=None, help="Optional rebuild device, e.g. cuda or cpu.")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument to append to rebuild_oop.py; repeat for multiple args.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove an existing output_dir before writing new snapshots.",
    )
    return parser.parse_args()


def rebuild_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "rebuild_oop.py",
        "--data_type",
        args.data_type,
        "--test_only",
        "--result_path",
        str(args.result_path),
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    cmd.extend(args.extra_arg)
    return cmd


def rebuild_output_path(args: argparse.Namespace) -> Path:
    return args.result_path / f"{args.data_type}-test" / "rebuild.out.npz"


def prepare_output_dir(path: Path, keep_existing: bool) -> None:
    if path.exists() and not keep_existing:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)


def run_rebuilds(args: argparse.Namespace) -> list[Path]:
    cmd = rebuild_command(args)
    output_path = rebuild_output_path(args)
    snapshots: list[Path] = []

    for idx in range(1, args.runs + 1):
        run_name = f"run_{idx:02d}"
        log_path = args.output_dir / "logs" / f"{run_name}.log"
        print(f"[{datetime.now().isoformat(timespec='seconds')}] starting {run_name}/{args.runs}")

        completed = subprocess.run(cmd, text=True, capture_output=True)
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n\n"
            + "STDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{run_name} failed with exit code {completed.returncode}; see {log_path}")
        if not output_path.is_file():
            raise FileNotFoundError(f"{run_name} did not produce expected output: {output_path}")

        snapshot_path = args.output_dir / "runs" / f"{run_name}.rebuild.out.npz"
        shutil.copy2(output_path, snapshot_path)
        snapshots.append(snapshot_path)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] saved {snapshot_path}")

    return snapshots


def load_output(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npzfile:
        if "data" not in npzfile:
            raise KeyError(f"{path} does not contain a 'data' array")
        return np.asarray(npzfile["data"], dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.dot(a, b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return numerator / denominator if denominator else float("nan")


def pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    return cosine_similarity(a_centered, b_centered)


def normalized_rmse(a: np.ndarray, b: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    scale = float(np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(b**2))) / 2.0
    return rmse / scale if scale else float("nan")


def pairwise_metrics(paths: list[Path]) -> dict[str, np.ndarray]:
    arrays = [load_output(path).ravel() for path in paths]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        details = ", ".join(f"{path.name}: {array.shape}" for path, array in zip(paths, arrays))
        raise ValueError(f"Output arrays must have matching shapes; got {details}")

    n = len(arrays)
    cosine = np.eye(n, dtype=float)
    pearson = np.eye(n, dtype=float)
    nrmse = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            cosine[i, j] = cosine[j, i] = cosine_similarity(arrays[i], arrays[j])
            pearson[i, j] = pearson[j, i] = pearson_similarity(arrays[i], arrays[j])
            nrmse[i, j] = nrmse[j, i] = normalized_rmse(arrays[i], arrays[j])

    return {"cosine": cosine, "pearson": pearson, "normalized_rmse": nrmse}


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.10g}" for value in row])


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.array([], dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def metric_summary(matrix: np.ndarray) -> dict[str, float | None]:
    values = off_diagonal_values(matrix)
    if values.size == 0:
        return {"min": None, "mean": None, "median": None, "max": None, "std": None}
    return {
        "min": float(np.nanmin(values)),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "max": float(np.nanmax(values)),
        "std": float(np.nanstd(values)),
    }


def write_reports(output_dir: Path, snapshots: list[Path], metrics: dict[str, np.ndarray]) -> None:
    labels = [path.stem.replace(".rebuild.out", "") for path in snapshots]
    for name, matrix in metrics.items():
        write_matrix_csv(output_dir / f"{name}_matrix.csv", matrix, labels)

    summary = {
        "outputs": [str(path) for path in snapshots],
        "metrics": {name: metric_summary(matrix) for name, matrix in metrics.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\nPairwise self-similarity summary (off-diagonal pairs):")
    for name, values in summary["metrics"].items():
        print(f"  {name}: {values}")


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    prepare_output_dir(args.output_dir, args.keep_existing)
    snapshots = run_rebuilds(args)
    metrics = pairwise_metrics(snapshots)
    write_reports(args.output_dir, snapshots, metrics)


if __name__ == "__main__":
    main()
