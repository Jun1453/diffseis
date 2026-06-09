"""
AIC picking and cross-correlation alignment metrics (vs diversity-stack reference).

Mirrors notebook helpers aic_plot / cross_correlation_delay / cc_plot without requiring axes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from refine_train import first_arrival_curve, interp_nan


def _filtered_curve(data: np.ndarray) -> np.ndarray:
    _, filtered = first_arrival_curve(data)
    out = np.empty(len(filtered), dtype=float)
    for i, v in enumerate(filtered):
        out[i] = np.nan if v is None else float(v)
    return out


def _shot_flip(n: int) -> bool:
    """Shots 0 and 4 (per 5-pass station block) use reversed trace order."""
    return (n % 5 == 0) or ((n + 1) % 5 == 0)


def aic_picking_ratio(data: np.ndarray) -> dict[str, float]:
    """Share of traces with a successful (finite) filtered AIC pick."""
    curve = _filtered_curve(data)
    n = curve.size
    if n == 0:
        return {"successful_pick_ratio": np.nan, "failed_pick_ratio": np.nan}
    ok = np.isfinite(curve)
    ratio = float(np.mean(ok))
    return {"successful_pick_ratio": ratio, "failed_pick_ratio": 1.0 - ratio}


def _delay_stats(delays: np.ndarray, fs: float) -> dict[str, float]:
    arr = np.asarray(delays, dtype=float)
    n_total = arr.size
    if n_total == 0:
        return {
            "RMS delay (samples)": np.nan,
            "RMS delay (seconds)": np.nan,
            "Standard deviation (samples)": np.nan,
            "Standard deviation (seconds)": np.nan,
            "Trace ratio with |delay| > 50 samples": np.nan,
        }
    rms = float(np.sqrt(np.nanmean(arr**2)))
    std = float(np.nanstd(arr))
    bad = float(np.sum((np.abs(arr) > 50) | ~np.isfinite(arr)) / n_total)
    return {
        "RMS delay (samples)": rms,
        "RMS delay (seconds)": rms / fs,
        "Standard deviation (samples)": std,
        "Standard deviation (seconds)": std / fs,
        "Trace ratio with |delay| > 50 samples": bad,
    }


def aic_alignment_metrics(
    profiles_pred: np.ndarray,
    profiles_ref: np.ndarray,
    fs: float = 250.0,
) -> dict[str, float]:
    """
    AIC first-arrival delay of pred vs ref (ref curve interpolated over failed picks).
    """
    res = []
    pick_pred = []
    pick_ref = []
    for n in range(profiles_pred.shape[0]):
        curve = _filtered_curve(profiles_pred[n])
        curve_ref = interp_nan(_filtered_curve(profiles_ref[n]))
        delay = curve_ref - curve
        if _shot_flip(n):
            delay = np.flip(delay)
        res.append(delay)
        pick_pred.append(aic_picking_ratio(profiles_pred[n])["successful_pick_ratio"])
        pick_ref.append(aic_picking_ratio(profiles_ref[n])["successful_pick_ratio"])

    out = _delay_stats(np.concatenate(res), fs)
    out["Successful AIC pick ratio (pred)"] = float(np.mean(pick_pred))
    out["Successful AIC pick ratio (ref)"] = float(np.mean(pick_ref))
    return out


def cross_correlation_delays(
    profiles_pred: np.ndarray,
    profiles_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace CC delay (samples) and normalized max correlation."""
    n_k, n_traces = profiles_pred.shape[0], profiles_pred.shape[2]
    delay_times = np.zeros((n_k, n_traces), dtype=float)
    max_corr_values = np.zeros((n_k, n_traces), dtype=float)

    for k in range(n_k):
        for n in range(n_traces):
            trace_tar = profiles_pred[k, :, n]
            trace_ref = profiles_ref[k, :, n]
            correlation = correlate(trace_tar, trace_ref, mode="full")
            max_corr_idx = int(np.argmax(correlation))
            max_corr_raw = correlation[max_corr_idx]
            norm_factor = np.sqrt(np.sum(trace_tar**2) * np.sum(trace_ref**2))
            max_corr_values[k, n] = max_corr_raw / norm_factor if norm_factor > 0 else 0.0
            delay_times[k, n] = max_corr_idx - (len(trace_ref) - 1)
    return delay_times, max_corr_values


def cross_correlation_metrics(
    profiles_pred: np.ndarray,
    profiles_ref: np.ndarray,
    fs: float = 250.0,
) -> dict[str, float]:
    delay_times, max_corr_values = cross_correlation_delays(profiles_pred, profiles_ref)
    out = _delay_stats(delay_times, fs)
    out["Average max correlation"] = float(np.mean(max_corr_values))
    out["Min max correlation"] = float(np.min(max_corr_values))
    out["Max max correlation"] = float(np.max(max_corr_values))
    return out


def cc_alignment_metrics(
    profiles_pred: np.ndarray,
    profiles_ref: np.ndarray,
    fs: float = 250.0,
) -> dict[str, float]:
    """
    CC-shifted first-arrival curve vs reference (same shot-flip convention as aic_plot).
    """
    delay_times, _ = cross_correlation_delays(profiles_pred, profiles_ref)
    res = []
    for n in range(profiles_pred.shape[0]):
        curve_ref = interp_nan(_filtered_curve(profiles_ref[n]))
        curve = delay_times[n, :] + curve_ref
        delay = curve_ref - curve
        if _shot_flip(n):
            delay = np.flip(delay)
        res.append(delay)
    return _delay_stats(np.concatenate(res), fs)


def evaluate_arrival_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    inp: np.ndarray | None = None,
    fs: float = 250.0,
) -> dict[str, dict[str, float]]:
    """Run AIC + CC metrics for output (and optional input) vs reference."""
    results: dict[str, dict[str, float]] = {
        "output_aic": aic_alignment_metrics(pred, ref, fs=fs),
        # "output_cc_map": cross_correlation_metrics(pred, ref, fs=fs),
        "output_cc": cc_alignment_metrics(pred, ref, fs=fs),
    }
    if inp is not None:
        results["input_aic"] = aic_alignment_metrics(inp, ref, fs=fs)
        # results["input_cc_map"] = cross_correlation_metrics(inp, ref, fs=fs)
        results["input_cc"] = cc_alignment_metrics(inp, ref, fs=fs)
    return results


def format_metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    lines = []
    for group, vals in metrics.items():
        lines.append(f"  [{group}]")
        for k, v in vals.items():
            lines.append(f"    {k}: {v:.6g}" if np.isfinite(v) else f"    {k}: nan")
    return "\n".join(lines)


def aic_plot(profiles, profiles_ref, label, ax1, shoot_pass=2, legend=False, fs=250.0):
    """Notebook-style AIC alignment figure (optional)."""
    import matplotlib.pyplot as plt

    y_number = [4, 6, 9, 21, 24, 26, 27, 28]
    y_shift = np.arange(8) + 1
    res = []
    for n in range(profiles.shape[0]):
        curve = _filtered_curve(profiles[n])
        curve_ref = interp_nan(_filtered_curve(profiles_ref[n]))
        delay = curve_ref - curve
        if _shot_flip(n):
            delay = np.flip(delay)
        res.append(delay)
        if n % 5 == shoot_pass:
            ax1.plot(
                np.arange(profiles.shape[2]),
                (-np.nanmean(curve_ref) + curve_ref) / fs + y_shift[int(n / 5)],
                color="black",
                alpha=0.8,
                label="Reference" if n == 1 else "",
            )
            ax1.plot(
                np.arange(profiles.shape[2]),
                (-np.nanmean(curve_ref) + curve) / fs + y_shift[int(n / 5)],
                label=f"OBS {y_number[int(n / 5)]}",
            )
    ax1.set_title(f"AIC first arrival ({label} - Ground truth)")
    if legend:
        ax1.set_xlabel("Trace number")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), borderaxespad=0, ncol=3)
    ax1.set_ylabel("Aligned arrival time (s)")
    ax1.invert_yaxis()
    out = _delay_stats(np.concatenate(res), fs)
    out["Successful AIC pick ratio (pred)"] = float(
        np.mean([aic_picking_ratio(profiles[n])["successful_pick_ratio"] for n in range(profiles.shape[0])])
    )
    out["Successful AIC pick ratio (ref)"] = float(
        np.mean([aic_picking_ratio(profiles_ref[n])["successful_pick_ratio"] for n in range(profiles.shape[0])])
    )
    return out


def cross_correlation_delay(profiles_tar, profiles_ref, label, ax1=None, legend=False, fs=250.0):
    """Notebook-style CC delay heatmap + metrics (ax1 optional)."""
    delay_times, max_corr_values = cross_correlation_delays(profiles_tar, profiles_ref)
    if ax1 is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        im1 = ax1.imshow(delay_times / fs, aspect="auto", cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        ax1.set_ylabel("Station-shot number")
        ax1.set_title(f"Cross-correlation delay ({label} - Ground truth)")
        if legend:
            ax1.set_xlabel("Trace number")
            cax = inset_axes(
                ax1,
                width="100%",
                height="15%",
                loc="upper center",
                borderpad=7.0,
                bbox_to_anchor=(0.0, -0.25, 1.0, 0.4),
                bbox_transform=ax1.transAxes,
            )
            import matplotlib.pyplot as plt

            cbar1 = plt.colorbar(im1, cax=cax, orientation="horizontal")
            cbar1.set_label("Delay time (s)")
    out = _delay_stats(delay_times, fs)
    out["Average max correlation"] = float(np.mean(max_corr_values))
    out["Min max correlation"] = float(np.min(max_corr_values))
    out["Max max correlation"] = float(np.max(max_corr_values))
    return out


def cc_plot(profiles, profiles_ref, label, ax1, legend=False, fs=250.0):
    """Notebook-style CC-aligned first-arrival figure (optional)."""
    import matplotlib.pyplot as plt

    y_number = [4, 6, 9, 21, 24, 26, 27, 28]
    y_shift = np.arange(8) + 1
    delay_times, _ = cross_correlation_delays(profiles, profiles_ref)
    res = []
    for n in range(profiles.shape[0]):
        curve_ref = interp_nan(_filtered_curve(profiles_ref[n]))
        curve = delay_times[n, :] + curve_ref
        delay = curve_ref - curve
        if _shot_flip(n):
            delay = np.flip(delay)
        res.append(delay)
        if (n - 1) % 5 == 0:
            ax1.plot(
                np.arange(profiles.shape[2]),
                (-np.nanmean(curve_ref) + curve_ref) / fs + y_shift[int((n - 1) / 5)],
                color="black",
                alpha=0.8,
                label="Reference" if n == 1 else "",
            )
            ax1.plot(
                np.arange(profiles.shape[2]),
                (-np.nanmean(curve_ref) + curve) / fs + y_shift[int((n - 1) / 5)],
                label=f"OBS {y_number[int((n - 1) / 5)]}",
            )
    ax1.set_title(f"Cross-correlation first arrival ({label} - Ground truth)")
    if legend:
        ax1.set_xlabel("Trace number")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), borderaxespad=0, ncol=3)
    ax1.set_ylabel("Aligned arrival time (s)")
    ax1.invert_yaxis()
    return _delay_stats(np.concatenate(res), fs)
