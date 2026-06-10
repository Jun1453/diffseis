"""
Bridge Profiles (2D record sections) to AI4EPS DeepDenoiser trace-wise inference.

DeepDenoiser expects per-file npz with variable ``data`` (time, station, channel).
We export one npz per trace, run ``deepdenoiser/predict.py``, and map denoised
time series back onto the original grid.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPDENOISER_ROOT = REPO_ROOT / "external" / "DeepDenoiser"
DEFAULT_MODEL_DIR = DEEPDENOISER_ROOT / "model" / "190614-104802"
DEFAULT_FINETUNED_MODEL = (
    REPO_ROOT / "results" / "baseline" / "deepdenoiser" / "finetune_data" / "finetuned_model"
)
TARGET_FS = 100  # Hz, DeepDenoiser Config.fs (STFT / U-Net domain)
DEEPDENOISER_CWD = DEEPDENOISER_ROOT / "deepdenoiser"
EXPORT_PRETRAINED = "pretrained"
EXPORT_FINETUNE_NATIVE = "finetune_native"  # native OBS length (e.g. 6000 @ 250 Hz), no pad to 9001

# Fine-tune export: native OBS rate, reference length from DeepDenoiser demo (9001 x 3 @ 100 Hz).
FINETUNE_FS = 250
FINETUNE_NT = 9001
FS_MATCH_TOL = 0.5  # Hz — treat OBS 250.0 as 250
_CKPT_SHARDS = ("", ".meta", ".index", ".data-00000-of-00001")


def _abs(path: str | Path) -> Path:
    """Absolute path (predict.py cwd is deepdenoiser/, not repo root)."""
    return Path(path).expanduser().resolve()


def _resample_trace(trace: np.ndarray, orig_fs: float, target_fs: float = TARGET_FS) -> tuple[np.ndarray, int]:
    """Resample 1D trace to target_fs; return (resampled, original_length)."""
    orig_len = len(trace)
    if orig_fs == target_fs:
        return trace.astype(np.float32), orig_len
    t = np.linspace(0.0, 1.0, orig_len)
    new_len = int(np.round(orig_len * target_fs / orig_fs))
    new_len = max(new_len, 2)
    t_new = np.linspace(0.0, 1.0, new_len)
    return interp1d(t, trace, kind="linear")(t_new).astype(np.float32), orig_len


def _pad_train_trace(data: np.ndarray, target_nt: int = FINETUNE_NT) -> np.ndarray:
    """Zero-pad (nt, nch) waveforms so DeepDenoiser STFT windows are long enough."""
    nt = data.shape[0]
    if nt >= target_nt:
        return data[:target_nt].astype(np.float32, copy=False)
    pad = np.zeros((target_nt - nt, data.shape[1]), dtype=np.float32)
    return np.concatenate([data.astype(np.float32), pad], axis=0)


def _resample_back(trace_rs: np.ndarray, orig_len: int, orig_fs: float, model_fs: float = TARGET_FS) -> np.ndarray:
    """Map model output (at model_fs) back to orig_len samples at orig_fs."""
    if orig_fs == model_fs and len(trace_rs) == orig_len:
        return trace_rs
    t = np.linspace(0.0, 1.0, len(trace_rs))
    t_orig = np.linspace(0.0, 1.0, orig_len)
    return interp1d(t, trace_rs, kind="linear")(t_orig).astype(np.float32)


def _fs_match(a: float, b: float, tol: float = FS_MATCH_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def read_finetune_training_fs(model_dir: Path) -> float:
    """
    Hz at which fine-tune NPZ waveforms were stored (DeepDenoiser --sampling_rate).

    Reads ``training_manifest.json`` next to the checkpoint, else ``../manifest.json``.
    """
    model_dir = _abs(model_dir)
    for path in (model_dir / "training_manifest.json", model_dir.parent / "manifest.json"):
        if path.is_file():
            data = json.loads(path.read_text())
            if "sampling_rate_hz" in data:
                return float(data["sampling_rate_hz"])
    raise FileNotFoundError(
        f"No training sampling rate in {model_dir} "
        f"(expected training_manifest.json or {model_dir.parent}/manifest.json)."
    )


def write_training_manifest(
    model_dir: Path,
    sampling_rate_hz: float,
    loss_type: str | None = None,
    snr_threshold: float | None = None,
) -> None:
    model_dir = _abs(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {"sampling_rate_hz": float(sampling_rate_hz)}
    if loss_type is not None:
        payload["loss_type"] = loss_type
    if snr_threshold is not None:
        payload["snr_threshold"] = float(snr_threshold)
    (model_dir / "training_manifest.json").write_text(json.dumps(payload, indent=2))


def _checkpoint_data_file(ckpt_prefix: Path) -> Path | None:
    path = Path(f"{ckpt_prefix}.data-00000-of-00001")
    return path if path.is_file() else None


def _checkpoint_weights_md5(ckpt_prefix: Path) -> str | None:
    data = _checkpoint_data_file(ckpt_prefix)
    if data is None:
        return None
    return hashlib.md5(data.read_bytes()).hexdigest()


def iter_checkpoint_prefixes(root: Path) -> list[tuple[int, float, Path]]:
    """(epoch, mtime, prefix) for each model_*.ckpt under root (recursive)."""
    root = _abs(root)
    if not root.is_dir():
        return []
    found: dict[str, tuple[int, float, Path]] = {}
    for index in root.rglob("model_*.ckpt.index"):
        prefix = index.parent / index.stem
        name = prefix.name
        if not (name.startswith("model_") and name.endswith(".ckpt")):
            continue
        try:
            epoch = int(name[6:-5])
        except ValueError:
            continue
        data = _checkpoint_data_file(prefix)
        if data is None:
            continue
        key = str(prefix)
        mtime = data.stat().st_mtime
        if key not in found or mtime > found[key][1]:
            found[key] = (epoch, mtime, prefix)
    return list(found.values())


def resolve_latest_checkpoint_prefix(model_dir: Path) -> Path:
    model_dir = _abs(model_dir)
    ckpt_file = model_dir / "checkpoint"
    if ckpt_file.is_file():
        for line in ckpt_file.read_text().splitlines():
            if line.startswith("model_checkpoint_path:"):
                name = line.split(":", 1)[1].strip().strip('"')
                prefix = model_dir / name
                if _checkpoint_data_file(prefix):
                    return prefix
    candidates = iter_checkpoint_prefixes(model_dir)
    if not candidates:
        raise FileNotFoundError(f"No TensorFlow checkpoints under {model_dir}")
    return max(candidates, key=lambda x: (x[0], x[1]))[2]


def log_checkpoint_resolution(model_dir: Path, label: str) -> Path:
    """Print which checkpoint predict will load; return checkpoint prefix path."""
    model_dir = _abs(model_dir)
    prefix = resolve_latest_checkpoint_prefix(model_dir)
    digest = _checkpoint_weights_md5(prefix)
    print(f"[{label}] model_dir={model_dir}")
    print(f"[{label}] checkpoint={prefix} (weights md5={digest})")
    return prefix


def weights_match_pretrained(model_dir: Path) -> bool:
    try:
        a = resolve_latest_checkpoint_prefix(model_dir)
        b = resolve_latest_checkpoint_prefix(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return False
    da, db = _checkpoint_weights_md5(a), _checkpoint_weights_md5(b)
    return da is not None and da == db


def promote_checkpoint_prefix(src_prefix: Path, dest_dir: Path) -> Path:
    """Copy one checkpoint shard set to dest_dir root and refresh checkpoint index."""
    dest_dir = _abs(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_prefix = _abs(src_prefix)
    name = src_prefix.name
    for old in dest_dir.glob("model_*.ckpt*"):
        old.unlink()
    for suf in _CKPT_SHARDS:
        src = Path(f"{src_prefix}{suf}")
        if src.is_file():
            shutil.copy2(src, dest_dir / f"{name}{suf}")
    (dest_dir / "checkpoint").write_text(
        f'model_checkpoint_path: "{name}"\nall_model_checkpoint_paths: "{name}"\n'
    )
    return dest_dir / name


def iter_finetuned_checkpoint_candidates(
    search_roots: list[Path],
) -> list[tuple[int, float, Path]]:
    """(epoch, mtime, prefix) for checkpoints that differ from pretrained."""
    pretrained_md5 = _checkpoint_weights_md5(resolve_latest_checkpoint_prefix(DEFAULT_MODEL_DIR))
    candidates: list[tuple[int, float, Path]] = []
    for root in search_roots:
        root = _abs(root)
        for epoch, mtime, prefix in iter_checkpoint_prefixes(root):
            digest = _checkpoint_weights_md5(prefix)
            if digest is None or digest == pretrained_md5:
                continue
            candidates.append((epoch, mtime, prefix))
    return candidates


def resolve_best_finetuned_checkpoint(
    model_dir: Path,
    search_roots: list[Path] | None = None,
) -> Path:
    roots = [_abs(r) for r in (search_roots or [model_dir])]
    candidates = iter_finetuned_checkpoint_candidates(roots)
    if not candidates:
        raise FileNotFoundError(
            f"No fine-tuned checkpoints (distinct from pretrained) under {roots}. "
            "Run baseline/train_deepdenoiser.py without --resume first."
        )
    return max(candidates, key=lambda x: (x[0], x[1]))[2]


def prepare_resume_checkpoint(model_dir: Path) -> Path:
    """
    TensorFlow restore only reads model_dir/checkpoint at the root, not timestamp
    subdirs. Promote the highest-epoch fine-tuned checkpoint there before resuming.
    """
    model_dir = _abs(model_dir)
    best = resolve_best_finetuned_checkpoint(model_dir)
    pretrained_md5 = _checkpoint_weights_md5(resolve_latest_checkpoint_prefix(DEFAULT_MODEL_DIR))
    best_md5 = _checkpoint_weights_md5(best)
    try:
        root = resolve_latest_checkpoint_prefix(model_dir)
        root_md5 = _checkpoint_weights_md5(root)
    except FileNotFoundError:
        root = None
        root_md5 = None

    if (
        root is not None
        and root_md5 == best_md5
        and _abs(root).parent == model_dir
        and root.name == best.name
    ):
        print(f"[finetune-resume] restore checkpoint={root} (weights md5={root_md5})")
        return root

    if root_md5 == pretrained_md5:
        print(
            f"[finetune-resume] Root checkpoint still points at pretrained weights; "
            f"promoting {best.name} from {best.parent}"
        )
    promoted = promote_checkpoint_prefix(best, model_dir)
    digest = _checkpoint_weights_md5(promoted)
    print(f"[finetune-resume] restore checkpoint={promoted} (weights md5={digest})")
    if digest == pretrained_md5:
        raise RuntimeError(
            f"Resume would restore pretrained weights in {model_dir}, not a fine-tuned checkpoint."
        )
    return promoted


def promote_latest_trained_checkpoint(
    dest_dir: Path,
    search_roots: list[Path] | None = None,
) -> Path:
    """
    DeepDenoiser train.py saves to log_dir/<timestamp>/, not model_dir root.
    Copy the highest-epoch checkpoint that differs from pretrained into dest_dir.
    """
    dest_dir = _abs(dest_dir)
    roots = [_abs(r) for r in (search_roots or [dest_dir, DEEPDENOISER_CWD / "log"])]
    candidates = iter_finetuned_checkpoint_candidates(roots)
    if not candidates:
        raise FileNotFoundError(
            f"No fine-tuned checkpoints (distinct from pretrained) under {roots}. "
            "Re-run baseline/train_deepdenoiser.py."
        )
    _, _, best = max(candidates, key=lambda x: (x[0], x[1]))
    if _abs(best).parent == dest_dir and best.name == (dest_dir / best.name).name:
        existing = dest_dir / best.name
        if existing.exists() and _checkpoint_weights_md5(existing) == _checkpoint_weights_md5(best):
            print(f"Fine-tuned checkpoint already in place: {best}")
            return existing
    promoted = promote_checkpoint_prefix(best, dest_dir)
    print(f"Promoted trained checkpoint {best} -> {promoted}")
    return promoted


def ensure_finetuned_checkpoint_dir(model_dir: Path) -> Path:
    """
    Use finetuned weights for inference. If dest_dir still holds copied pretrained
    shards, promote the latest checkpoint from training logs.
    """
    model_dir = _abs(model_dir)
    if not iter_checkpoint_prefixes(model_dir):
        promote_latest_trained_checkpoint(model_dir)
    elif weights_match_pretrained(model_dir):
        print(
            f"Warning: {model_dir} still matches pretrained weights; "
            "searching training logs for a fine-tuned checkpoint."
        )
        promote_latest_trained_checkpoint(model_dir)
        if weights_match_pretrained(model_dir):
            raise RuntimeError(
                f"No fine-tuned weights in {model_dir} (identical to pretrained). "
                "Re-run: python baseline/train_deepdenoiser.py --epochs 20"
            )
    return model_dir


def resolve_pretrained_model_dir() -> Path:
    model_dir = _abs(DEFAULT_MODEL_DIR)
    if not iter_checkpoint_prefixes(model_dir):
        raise FileNotFoundError(f"Missing pretrained checkpoint under {model_dir}")
    return model_dir


def _pack_model_fs_in_native_buffer(
    trace_at_model_fs: np.ndarray, native_len: int
) -> tuple[np.ndarray, int]:
    """Place model-fs samples in a native-length vector; tail stays zero (trimmed in predict)."""
    trace_at_model_fs = np.asarray(trace_at_model_fs, dtype=np.float32).ravel()
    effective_nt = min(len(trace_at_model_fs), native_len)
    buf = np.zeros(native_len, dtype=np.float32)
    buf[:effective_nt] = trace_at_model_fs[:effective_nt]
    return buf, effective_nt


def resolve_predict_export_plan(
    inference_fs: float,
    training_fs: float | None = None,
) -> tuple[str, int]:
    """
    Returns (export_mode, predict_fs).

    finetune_native: one bridge resample to TARGET_FS, packed in native-length NPZ;
    predict --sampling_rate=100 with effective_nt crop (no second 250→100).
    pretrained: bridge resamples to TARGET_FS before predict.
    """
    if training_fs is None:
        return EXPORT_PRETRAINED, int(TARGET_FS)
    return EXPORT_FINETUNE_NATIVE, int(TARGET_FS)


def _write_export_config(work_dir: Path, export_mode: str, predict_fs: int) -> None:
    (work_dir / "export_config.json").write_text(
        json.dumps(
            {"export_mode": export_mode, "predict_fs": predict_fs, "model_fs": TARGET_FS},
            indent=2,
        )
    )


def _read_export_config(work_dir: Path) -> dict:
    path = _abs(work_dir) / "export_config.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def export_traces_to_npz(
    profiles,
    work_dir: Path,
    n_components: int = 3,
    export_mode: str = EXPORT_PRETRAINED,
    predict_fs: int | None = None,
) -> list[dict]:
    """
    Write one npz per trace. Replicate single-component OBS data across channels
    when n_components=3 (pretrained earthquake model convention).

    export_mode:
      - pretrained: resample to TARGET_FS (legacy published-model path)
      - finetune_native: (native_len, 1, 3) NPZ; model-fs waveform in prefix, effective_nt set
    """
    work_dir = Path(work_dir)
    npz_dir = work_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    for stale in npz_dir.glob("*.npz"):
        stale.unlink()
    predict_fs = int(predict_fs if predict_fs is not None else TARGET_FS)

    meta = []
    native_fs = float(profiles.sampling_rate)
    idx = 0
    for p in range(profiles.shape[0]):
        for tr in range(profiles.shape[2]):
            trace = np.asarray(profiles[p, :, tr], dtype=np.float32)
            orig_len = len(trace)
            effective_nt = orig_len
            if export_mode == EXPORT_FINETUNE_NATIVE:
                core, _ = _resample_trace(trace, native_fs, TARGET_FS)
                trace_buf, effective_nt = _pack_model_fs_in_native_buffer(core, orig_len)
                stored_fs = TARGET_FS
            else:
                trace_buf, orig_len = _resample_trace(trace, native_fs, TARGET_FS)
                stored_fs = TARGET_FS
            if n_components == 1:
                data = trace_buf[:, np.newaxis, np.newaxis]
            else:
                data = np.stack([trace_buf] * n_components, axis=-1)[:, np.newaxis, :]
            fname = f"trace_{idx:06d}.npz"
            np.savez(
                npz_dir / fname,
                data=data,
                p_idx=np.array(0),
                effective_nt=np.int32(effective_nt),
                native_fs=np.float32(native_fs),
            )
            meta.append(
                dict(
                    fname=fname,
                    profile=p,
                    trace=tr,
                    orig_len=orig_len,
                    effective_nt=effective_nt,
                    resampled_len=effective_nt,
                    stored_fs=stored_fs,
                    native_fs=native_fs,
                    export_mode=export_mode,
                )
            )
            idx += 1

    csv_path = work_dir / "traces.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fname"])
        for m in meta:
            writer.writerow([m["fname"]])

    (work_dir / "trace_meta.json").write_text(json.dumps(meta))
    _write_export_config(work_dir, export_mode, predict_fs)
    return meta


def load_trace_meta(work_dir: Path) -> list[dict]:
    """Reload trace index map written by export_traces_to_npz."""
    work_dir = _abs(work_dir)
    meta_path = work_dir / "trace_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; run predict export first.")
    return json.loads(meta_path.read_text())


def _map_denoised_to_native(trace_rs: np.ndarray, meta: dict) -> np.ndarray:
    """Inverse of export preprocessing for each trace."""
    trace_rs = np.asarray(trace_rs, dtype=np.float32).ravel()
    orig_len = int(meta["orig_len"])
    native_fs = float(meta.get("native_fs", TARGET_FS))

    return _resample_back(trace_rs, orig_len, native_fs, model_fs=TARGET_FS)


def import_from_predict_results(profiles, work_dir: Path):
    """Map existing DeepDenoiser predict output back onto Profiles (skip re-predict)."""
    work_dir = _abs(work_dir)
    result_dir = work_dir / "dd_output" / "results"
    if not result_dir.is_dir():
        raise FileNotFoundError(f"No prediction results at {result_dir}")
    meta = load_trace_meta(work_dir)
    return import_denoised_profiles(profiles, meta, result_dir)


def run_deepdenoiser_predict(
    work_dir: Path,
    model_dir: Path | None = None,
    sampling_rate: float | None = None,
    batch_size: int = 20,
    finetuned: bool = False,
) -> Path:
    """Run upstream predict.py; return directory with denoised npz results."""
    work_dir = _abs(work_dir)
    if finetuned:
        model_dir = ensure_finetuned_checkpoint_dir(_abs(model_dir or DEFAULT_FINETUNED_MODEL))
        log_checkpoint_resolution(model_dir, "finetuned-predict")
    else:
        model_dir = resolve_pretrained_model_dir()
        log_checkpoint_resolution(model_dir, "pretrained-predict")
    npz_dir = work_dir / "npz"
    traces_csv = work_dir / "traces.csv"
    if not traces_csv.is_file():
        raise FileNotFoundError(
            f"Missing {traces_csv}. export_traces_to_npz may have produced no traces."
        )

    entry_py = REPO_ROOT / "baseline" / "deepdenoiser_entry.py"
    if not entry_py.is_file():
        raise FileNotFoundError(f"DeepDenoiser entry script not found at {entry_py}")

    output_dir = _abs(work_dir / "dd_output")
    cmd = [
        sys.executable,
        str(entry_py),
        "predict",
        "--format=numpy",
        f"--model_dir={model_dir}",
        f"--data_dir={npz_dir}",
        f"--data_list={traces_csv}",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        "--save_signal",
    ]
    if sampling_rate is not None:
        cmd.append(f"--sampling_rate={int(round(sampling_rate))}")

    # Entry script sets cwd/PYTHONPATH; run from repo root so paths stay valid.
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return output_dir / "results"


def import_denoised_profiles(profiles, meta: list[dict], result_dir: Path):
    """Map denoised npz traces back into a Profiles array."""
    from profiledd import Profiles

    out = np.array(profiles, dtype=np.float32, copy=True)
    result_dir = Path(result_dir)
    for m in meta:
        result_path = result_dir / m["fname"]
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing DeepDenoiser output: {result_path}")
        z = np.load(result_path)
        denoised = z["data"]
        if denoised.ndim == 3:
            trace_rs = denoised[:, 0, 0]
        elif denoised.ndim == 2:
            trace_rs = denoised[:, 0]
        else:
            trace_rs = denoised.squeeze()
        trace = _map_denoised_to_native(trace_rs, m)
        out[m["profile"], : len(trace), m["trace"]] = trace
        if len(trace) < out.shape[1]:
            out[m["profile"], len(trace) :, m["trace"]] = 0.0

    return Profiles(
        out,
        sampling_rate=profiles.sampling_rate,
        filter_history=profiles.filter_history,
        reduction_vel=profiles.reduction_vel,
        offsets=profiles.offsets,
        first_arrival_reference=profiles.first_arrival_reference,
    )


def denoise_profiles_tracewise(
    profiles,
    work_dir: str | Path,
    model_dir: str | Path | None = None,
    training_fs: float | None = None,
    n_components: int = 3,
    batch_size: int = 20,
):
    """
    Full trace-wise DeepDenoiser pass on a Profiles object.

    training_fs: native Hz of fine-tune NPZ (from manifest). None => pretrained path.
    Finetuned: one resample to {TARGET_FS} Hz in bridge; NPZ shape (native_len, 1, 3);
    predict crops to effective_nt at {TARGET_FS} Hz (no upstream resample).
    """
    work_dir = _abs(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    inference_fs = float(profiles.sampling_rate)
    export_mode, predict_fs = resolve_predict_export_plan(inference_fs, training_fs)

    if export_mode == EXPORT_FINETUNE_NATIVE:
        print(
            f"Fine-tuned model: NPZ shape ({profiles.shape[1]}, 1, 3) @ {inference_fs:g} Hz native; "
            f"model-fs prefix ({TARGET_FS:g} Hz, effective_nt via npz); "
            f"predict --sampling_rate={predict_fs} (single bridge resample, no predict 250→100)."
        )
    else:
        print(f"Pretrained model — exporting waveforms at {TARGET_FS:g} Hz for predict.")

    meta = export_traces_to_npz(
        profiles,
        work_dir,
        n_components=n_components,
        export_mode=export_mode,
        predict_fs=predict_fs,
    )
    result_dir = run_deepdenoiser_predict(
        work_dir,
        model_dir=model_dir,
        sampling_rate=predict_fs,
        batch_size=batch_size,
        finetuned=training_fs is not None,
    )
    return import_denoised_profiles(profiles, meta, result_dir)


DEFAULT_CHANNELS = "Z_Z_Z"  # matches CSV; DeepDenoiser groups noise by this string


def _trace_to_train_npz(
    trace: np.ndarray,
    itp_orig: int,
    orig_fs: float,
    n_components: int = 3,
    channels: str = DEFAULT_CHANNELS,
    target_nt: int = FINETUNE_NT,
):
    """DeepDenoiser train npz: data (nt, nch), itp, channels — native rate, no resampling."""
    trace = np.asarray(trace, dtype=np.float32).ravel()
    itp = int(np.clip(itp_orig, 0, max(len(trace) - 1, 0)))
    if n_components == 1:
        data = trace[:, np.newaxis]
    else:
        data = np.stack([trace] * n_components, axis=-1)
    data = _pad_train_trace(data, target_nt)
    itp = int(np.clip(itp, 0, data.shape[0] - 1))
    return data, itp, channels


def _save_train_npz(path: Path, data: np.ndarray, itp: int, channels: str = DEFAULT_CHANNELS) -> None:
    np.savez(
        path,
        data=data.astype(np.float32),
        itp=np.int64(itp),
        channels=np.array(channels),
    )


def _write_split_csv(csv_path: Path, rows: list[list]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "itp", "channels"])
        w.writerows(rows)


def export_finetune_dataset(
    output_dir: str | Path,
    train_only: bool = True,
    holdout_keys: list[str] | None = None,
    max_stations: int | None = None,
    n_components: int = 3,
    time_samples: int = FINETUNE_NT,
) -> dict:
    """
    Export NOTO/OBS traces for DeepDenoiser fine-tuning (training stations: value >= 0).

    Layout (per split):
      {split}/signal/*.npz  — diversity-stack (clean) traces
      {split}/noise/*.npz   — raw input (noisy) traces, paired by fname
      {split}/signal.csv, {split}/noise.csv

    holdout_keys: stations reserved for valid/ (default: none; all train keys go to train/).
    """
    from baseline.data_noto import iter_train_station_keys, load_station_pair
    from refine_train import stn_num_to_n

    output_dir = _abs(output_dir)
    holdout_keys = list(holdout_keys or [])
    stats = {"train": 0, "valid": 0, "stations_train": [], "stations_valid": []}

    def export_split(split_name: str, keys: list[str]) -> Path:
        sig_dir = output_dir / split_name / "signal"
        noise_dir = output_dir / split_name / "noise"
        sig_dir.mkdir(parents=True, exist_ok=True)
        noise_dir.mkdir(parents=True, exist_ok=True)
        sig_rows, noise_rows = [], []
        idx = 0
        for key in keys:
            pf, clean_for_shot, padded_arrival = load_station_pair(key, time_samples=time_samples)
            fs = float(pf.sampling_rate)
            for shot in range(pf.shape[0]):
                clean_2d = clean_for_shot(shot)
                for tr in range(pf.shape[2]):
                    fname = f"{key}_s{shot}_t{tr:04d}.npz"
                    itp = int(padded_arrival[tr])
                    ch = DEFAULT_CHANNELS
                    noisy_data, itp_rs, _ = _trace_to_train_npz(
                        np.asarray(pf[shot, :, tr], dtype=np.float32), itp, fs, n_components, ch
                    )
                    clean_data, _, _ = _trace_to_train_npz(
                        np.asarray(clean_2d[:, tr], dtype=np.float32), itp, fs, n_components, ch
                    )
                    _save_train_npz(sig_dir / fname, clean_data, itp_rs, ch)
                    _save_train_npz(noise_dir / fname, noisy_data, itp_rs, ch)
                    sig_rows.append([fname, itp_rs, ch])
                    noise_rows.append([fname, itp_rs, ch])
                    idx += 1
        _write_split_csv(output_dir / split_name / "signal.csv", sig_rows)
        _write_split_csv(output_dir / split_name / "noise.csv", noise_rows)
        stats[split_name] = idx
        return output_dir / split_name

    train_keys = []
    for key in iter_train_station_keys(train_only=train_only, test_only=False, holdout_keys=holdout_keys):
        train_keys.append(key)
        if max_stations is not None and len(train_keys) >= max_stations:
            break

    export_split("train", train_keys)
    stats["stations_train"] = train_keys

    if holdout_keys:
        valid_keys = [k for k in holdout_keys if stn_num_to_n.get(k, -999) >= 0]
        if valid_keys:
            export_split("valid", valid_keys)
        stats["stations_valid"] = valid_keys

    manifest = {
        "sampling_rate_hz": FINETUNE_FS,
        "finetune_nt": FINETUNE_NT,
        "n_components": n_components,
        "stats": stats,
        "holdout_keys": holdout_keys,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_finetune_npz(profiles_noisy, profiles_clean, work_dir: Path, split_name: str = "train") -> Path:
    """Legacy wrapper: export concatenated Profiles (prefer export_finetune_dataset)."""
    work_dir = _abs(work_dir)
    split_dir = work_dir / split_name
    sig_dir = split_dir / "signal"
    noise_dir = split_dir / "noise"
    sig_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    sig_rows, noise_rows = [], []
    fs = float(profiles_noisy.sampling_rate)
    idx = 0
    for p in range(profiles_noisy.shape[0]):
        ntr = min(profiles_noisy.shape[2], profiles_clean.shape[2])
        for tr in range(ntr):
            fname = f"{split_name}_{idx:06d}.npz"
            itp = 0
            ch = DEFAULT_CHANNELS
            noisy_data, itp, _ = _trace_to_train_npz(
                np.asarray(profiles_noisy[p, :, tr], dtype=np.float32), 0, fs, channels=ch
            )
            clean_data, _, _ = _trace_to_train_npz(
                np.asarray(profiles_clean[p, :, tr], dtype=np.float32), 0, fs, channels=ch
            )
            _save_train_npz(sig_dir / fname, clean_data, itp, ch)
            _save_train_npz(noise_dir / fname, noisy_data, itp, ch)
            sig_rows.append([fname, itp, ch])
            noise_rows.append([fname, itp, ch])
            idx += 1
    _write_split_csv(split_dir / "signal.csv", sig_rows)
    _write_split_csv(split_dir / "noise.csv", noise_rows)
    return split_dir / "signal.csv"


def _copy_checkpoint_tree(src_dir: Path, dest_dir: Path, *, replace: bool = True) -> None:
    """Copy TensorFlow checkpoint shards and index from src_dir into dest_dir."""
    src_dir = _abs(src_dir)
    dest_dir = _abs(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if replace:
        for old in dest_dir.glob("model_*.ckpt*"):
            old.unlink()
    for ckpt in src_dir.glob("model_*.ckpt*"):
        shutil.copy2(ckpt, dest_dir / ckpt.name)
    for extra in ("checkpoint", "graph.pbtxt"):
        src = src_dir / extra
        if src.exists():
            shutil.copy2(src, dest_dir / extra)


def run_finetune(
    work_dir: Path,
    init_model_dir: Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    sampling_rate: int | None = FINETUNE_FS,
    loss_type: str = "cross_entropy",
    snr_threshold: float = 2.0,
    resume: bool = False,
    cpu: bool = False,
) -> Path:
    """Run DeepDenoiser train.py on exported NOTO traces."""
    work_dir = _abs(work_dir)
    entry_py = REPO_ROOT / "baseline" / "deepdenoiser_entry.py"
    train_sig_csv = _abs(work_dir / "train" / "signal.csv")
    train_noise_csv = _abs(work_dir / "train" / "noise.csv")
    train_sig_dir = _abs(work_dir / "train" / "signal")
    train_noise_dir = _abs(work_dir / "train" / "noise")
    out_dir = _abs(work_dir / "finetuned_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(entry_py),
        "train",
    ]
    if cpu:
        cmd.append("--cpu")
    cmd.extend(
        [
        "--mode=train",
        f"--train_signal_dir={train_sig_dir}",
        f"--train_signal_list={train_sig_csv}",
        f"--train_noise_dir={train_noise_dir}",
        f"--train_noise_list={train_noise_csv}",
        f"--batch_size={batch_size}",
        f"--epochs={epochs}",
        f"--model_dir={out_dir}",
        f"--log_dir={out_dir}",
        f"--loss_type={loss_type}",
        f"--snr_threshold={snr_threshold}",
        ]
    )
    if sampling_rate is not None:
        cmd.append(f"--sampling_rate={int(sampling_rate)}")
    valid_sig_csv = work_dir / "valid" / "signal.csv"
    if valid_sig_csv.is_file():
        cmd.extend(
            [
                f"--valid_signal_dir={_abs(work_dir / 'valid' / 'signal')}",
                f"--valid_signal_list={_abs(valid_sig_csv)}",
                f"--valid_noise_dir={_abs(work_dir / 'valid' / 'noise')}",
                f"--valid_noise_list={_abs(work_dir / 'valid' / 'noise.csv')}",
            ]
        )
    if resume:
        if init_model_dir is not None:
            _copy_checkpoint_tree(init_model_dir, out_dir, replace=True)
        prepare_resume_checkpoint(out_dir)
    elif init_model_dir is not None:
        _copy_checkpoint_tree(init_model_dir, out_dir, replace=True)

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    promote_latest_trained_checkpoint(out_dir, [out_dir, DEEPDENOISER_CWD / "log"])
    if weights_match_pretrained(out_dir):
        raise RuntimeError(
            f"Fine-tuning finished but weights in {out_dir} still match pretrained. "
            "Check training logs under external/DeepDenoiser/deepdenoiser/log/."
        )
    log_checkpoint_resolution(out_dir, "finetuned-train")
    if sampling_rate is not None:
        write_training_manifest(
            out_dir,
            float(sampling_rate),
            loss_type=loss_type,
            snr_threshold=snr_threshold,
        )
    return out_dir
