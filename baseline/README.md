# C1 baselines (reviewer comment C1)

Run all commands from the repository root (`diffseis/`).

## Environment

- **PyTorch baselines** (deterministic U-Net, DDPM): `conda activate diffseis`.
- **DeepDenoiser**: `**diffseis-baseline`** env only — **not** the main `diffseis` env if that has Python 3.13 + TensorFlow 2.16+ (Keras 3 → `conv2d is not available with Keras 3`).

**Create baseline env (Python 3.11 + TensorFlow 2.15):**

```bash
conda env create -f environment-baseline.yml
conda activate diffseis-baseline
```

**GPU (CUDA):** use the 2.15 CUDA wheel **and** Keras 2. Do **not** set `TF_USE_LEGACY_KERAS=1` on TF 2.15 (that breaks `tf.compat.v1.layers` unless you install `tf-keras`). The entry script clears that flag automatically.

```bash
pip uninstall -y tensorflow keras tf-keras
pip install "tensorflow[and-cuda]==2.15.1" "keras>=2.13,<3"
unset TF_USE_LEGACY_KERAS
python -c "import os; os.environ.pop('TF_USE_LEGACY_KERAS', None); import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU')); print(hasattr(tf.compat.v1.layers,'conv2d'))"
```

**CPU-only:** `pip install -r requirements-baseline.txt` is enough inside `diffseis-baseline`.

DeepDenoiser is launched via `baseline/deepdenoiser_entry.py` (NumPy 2.x aliases + TF version check). Do not run `external/DeepDenoiser/deepdenoiser/predict.py` directly.

**No PyTorch in this env:** `profiledd` loads without `torch`. `rebuild_deepdenoiser.py` only needs NumPy, SciPy, segyio, pandas, and TensorFlow. DDPM / `train_direct.py` still require the `diffseis` env with PyTorch.

## Baseline A — DeepDenoiser (trace-wise)

Pretrained weights ship in `external/DeepDenoiser/model/190614-104802/`.

Run from the **repository root** (`diffseis/`) so data paths and work directories resolve correctly.

```bash
# Pretrained, training stations
python baseline/rebuild_deepdenoiser.py --model pretrained --train_only
```

### Fine-tune data + training

```bash
# Export train/ at 250 Hz, 9001 samples (native OBS rate; no 100 Hz downsample in npz)
python baseline/prepare_deepdenoiser_finetune.py
# Optional: hold out stations for valid/
python baseline/prepare_deepdenoiser_finetune.py --holdout 05 27

# If export predates format fix (channels, shape, or length <6045 samples @100Hz):
python baseline/repair_finetune_npz.py

# Inspect finetune pairs (noisy vs diversity-stack clean)
python baseline/inspect_finetune_data.py --random 6 --out results/baseline/deepdenoiser/finetune_inspect
python baseline/inspect_finetune_data.py --station 20 --shot 4 --trace 219 --verify_source --out results/baseline/deepdenoiser/finetune_inspect

# Fine-tune from pretrained weights (default: cross_entropy, snr_threshold=2, batch_size=8)
python baseline/train_deepdenoiser.py --epochs 20
# Continue from existing finetuned checkpoint (does not overwrite with pretrained)
python baseline/train_deepdenoiser.py --resume --epochs 20
# Training log prints "post-restore loss" on the first batch: ~0.3–0.4 if resume
# worked, ~2.9 if pretrained was loaded by mistake (always use --resume).
# Resume from a checkpoint stored elsewhere
python baseline/train_deepdenoiser.py --resume --init_model path/to/finetuned_model --epochs 10

# GPU OOM: free other jobs, or use smaller batch / CPU
python baseline/train_deepdenoiser.py --epochs 20 --batch_size 4
python baseline/train_deepdenoiser.py --epochs 20 --cpu

# Infer with existing finetuned checkpoint (train or test split)
# NPZ shape (6000, 1, 3): model-fs waveform in prefix (effective_nt); predict --sampling_rate=100
python baseline/rebuild_deepdenoiser.py --model finetuned --train_only
python baseline/rebuild_deepdenoiser.py --model finetuned --test_only
python baseline/rebuild_deepdenoiser.py --model finetuned --mcs --test_only
python baseline/rebuild_deepdenoiser.py --model finetuned --datatype nwp --test_only
python baseline/rebuild_deepdenoiser.py --model finetuned --datatype nwp --source obs --test_only

# Re-train then infer (only if you need a new checkpoint)
python baseline/rebuild_deepdenoiser.py --model finetuned --refinetune --finetune_epochs 10 --train_only
```

Outputs: `results/baseline/deepdenoiser/{pretrained,finetuned}-{train,test}/` for Noto OBS, `{pretrained,finetuned}-mcs-{train,test}/` for Noto MCS, `{pretrained,finetuned}-nwp-{train,test}/` for NWP MCS, and `{pretrained,finetuned}-nwp-obs-{train,test}/` for NWP OBS (`sections.*`, `rebuild.*`).

## Baseline B — Deterministic U-Net (DDPM ablation)

```bash
python train_direct.py
python baseline/rebuild_unet_direct.py --checkpoint results/baseline/unet-direct/model-final.pt
```

## Reference — DDPM OOP rebuild

```bash
python rebuild_oop.py
```

## Metrics

```bash
python baseline/evaluate.py --scan_defaults
```

Reports waveform RMS plus **AIC picking ratio**, **AIC alignment**, **CC delay map**, and **CC-aligned first-arrival** metrics (vs diversity-stack reference). Plot helpers: `baseline/arrival_metrics.py` (`aic_plot`, `cc_plot`).

See `revision_plan.md` § C1 for the comparison table.