"""
Run DeepDenoiser predict.py / train.py with NumPy 2.x + TensorFlow 2.15 compatibility.

Upstream requires tf.compat.v1.layers (Keras 2). TensorFlow >= 2.16 / Keras 3 fails with:
  AttributeError: conv2d is not available with Keras 3

Use conda env diffseis-baseline (Python 3.11, tensorflow==2.15.1), not the PyTorch diffseis env
if that env has TensorFlow 2.16+ on Python 3.13.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DD = _REPO / "external" / "DeepDenoiser" / "deepdenoiser"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _use_cpu_from_argv() -> bool:
    """Parse --cpu before TensorFlow is imported."""
    if "--cpu" in sys.argv[1:]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return True
    return False


def _configure_gpu_memory() -> None:
    """Allow gradual GPU allocation (must run before the first TF session/graph)."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Already initialized — ignore.
            pass


def _numpy2_compat() -> None:
    import numpy as np

    for name, builtin in (("int", int), ("float", float)):
        if not hasattr(np, name):
            setattr(np, name, builtin)


def _configure_keras_for_tensorflow() -> None:
    """
    TF 2.15: do NOT set TF_USE_LEGACY_KERAS=1 — it requires tf_keras and breaks
    tf.compat.v1.layers even when `keras` 2.x is installed.

    TF 2.16+: need TF_USE_LEGACY_KERAS=1 and pip install tf-keras.
    """
    import tensorflow as tf

    parts = tf.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])

    if (major, minor) >= (2, 16):
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        try:
            import tf_keras  # noqa: F401
        except ImportError as exc:
            print(
                f"TensorFlow {tf.__version__} needs legacy Keras for DeepDenoiser.\n"
                "  pip install tf-keras\n"
                "Or use Python 3.11 + tensorflow==2.15.1 (recommended).",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
    else:
        os.environ.pop("TF_USE_LEGACY_KERAS", None)
        os.environ["TF_USE_LEGACY_KERAS"] = "0"


def _verify_v1_layers() -> None:
    """Ensure tf.compat.v1.layers works (DeepDenoiser build depends on it)."""
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    try:
        conv2d = tf.compat.v1.layers.conv2d
    except (ImportError, AttributeError) as exc:
        print(
            "tf.compat.v1.layers is not available (Keras backend not wired).\n"
            "For TensorFlow 2.15.1:\n"
            "  unset TF_USE_LEGACY_KERAS   # must not be 1\n"
            "  pip install 'keras>=2.13,<3'\n"
            "For TensorFlow 2.16+:\n"
            "  pip install tf-keras\n"
            "  export TF_USE_LEGACY_KERAS=1",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Smoke-test on CPU so we do not reserve GPU memory before train.py runs.
    with tf.device("/cpu:0"):
        with tf.compat.v1.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, [None, 4, 4, 1])
            conv2d(x, filters=2, kernel_size=3, padding="same")


def _check_tensorflow() -> None:
    import tensorflow as tf

    parts = tf.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    if (major, minor) >= (2, 16):
        print(
            f"TensorFlow {tf.__version__} uses Keras 3 by default; DeepDenoiser needs legacy stack.\n"
            "Use diffseis-baseline with tensorflow==2.15.1, or install tf-keras with TF_USE_LEGACY_KERAS=1.",
            file=sys.stderr,
        )
        sys.exit(1)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"TensorFlow {tf.__version__}: {len(gpus)} GPU(s) visible", file=sys.stderr)
    else:
        print(
            f"TensorFlow {tf.__version__}: no GPU visible (CPU run). "
            'For GPU: pip install "tensorflow[and-cuda]==2.15.1" "keras>=2.13,<3"',
            file=sys.stderr,
        )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("predict", "train"):
        print(
            "Usage: python baseline/deepdenoiser_entry.py {predict|train} [args...]\n"
            "  Add --cpu to force CPU (avoids GPU OOM if another job holds the card).",
            file=sys.stderr,
        )
        sys.exit(2)

    mode = sys.argv[1]
    script = _DD / ("predict.py" if mode == "predict" else "train.py")
    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        sys.exit(1)

    cpu = _use_cpu_from_argv()
    _numpy2_compat()
    _configure_gpu_memory()
    _configure_keras_for_tensorflow()
    _check_tensorflow()
    _verify_v1_layers()
    os.chdir(_DD)
    sys.path.insert(0, str(_DD))
    # Strip wrapper-only flags before forwarding to DeepDenoiser.
    forwarded = [a for a in sys.argv[2:] if a != "--cpu"]
    sys.argv = [str(script.name)] + forwarded
    if cpu:
        print("DeepDenoiser entry: running on CPU (--cpu)", file=sys.stderr)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
