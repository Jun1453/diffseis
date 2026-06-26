"""NOTO MCS loading for model inference on dense MCS record sections.

MCS SEGY files contain fewer shooting passes than OBS but ~4x denser trace spacing
on the first pass. For denoising inference we:

1. Keep only the first MCS shooting pass (``jamstec_handler`` shot index 0).
2. Load the matching OBS station and take the first-pass source-receiver offsets
   as the reference geometry.
3. For each OBS offset, keep the MCS trace whose source offset is closest.
4. Return MCS waveforms with MCS trace offsets as ``profiles_data``, and the
   diversity-stacked OBS target for the matching pass as ``profiles_target``.
"""
from functools import partial
from typing import Optional

import numpy as np

from data_noto import FRAGMENT_KWARGS, _normalize_passes, _stack_targets, iter_train_station_keys, median_noise_level
from profiledd import Profiles, jamstec_handler, highpass
from refine_train import fit_curves


def _hipass_filter(trace, sample_rate):
    return highpass(trace, 2.0, sample_rate, poles=4)


def mcs_trace_indices_for_obs_offsets(mcs_offsets, obs_offsets):
    """
    Map each OBS offset to the index of the closest MCS trace.

    Parameters
    ----------
    mcs_offsets, obs_offsets : array-like
        Source-receiver offsets in km (1-D, same units as ``Profiles.offsets``).

    Returns
    -------
    np.ndarray
        Integer indices into ``mcs_offsets`` / MCS trace axis, length ``len(obs_offsets)``.
    """
    mcs_offsets = np.asarray(mcs_offsets, dtype=float)
    obs_offsets = np.asarray(obs_offsets, dtype=float)
    return np.abs(mcs_offsets[:, None] - obs_offsets[None, :]).argmin(axis=0)


def _subset_mcs_pass(mcs_pass, trace_indices):
    """Build a single-pass Profiles slice with MCS trace offsets."""
    data = np.asarray(mcs_pass)[0:1, :, trace_indices]
    mcs_offsets = np.asarray(mcs_pass.offsets[0], dtype=float)[trace_indices]
    return Profiles(
        data,
        sampling_rate=mcs_pass.sampling_rate,
        filter_history=mcs_pass.filter_history,
        reduction_vel=mcs_pass.reduction_vel,
        offsets=[mcs_offsets],
    )


def _padded_arrival_for_key(key: str, pf: Profiles, time_samples: int):
    try:
        arrival = fit_curves[f"{key}"] + pf.sampling_rate * 0.5 - 75
        return np.pad(arrival, (0, pf.shape[2] - len(arrival)), mode="edge")
    except KeyError:
        return None


def _preprocess_obs_station(key: str, time_samples: int = 6000, use_gt_noise_level: bool = False):
    """Load and preprocess all OBS passes (matches ``load_obs_profiles``)."""
    obs = Profiles.load(f"noto/OBS/NT24OBS_J{key}C-1.sgy", jamstec_handler)
    if not use_gt_noise_level:
        obs = _normalize_passes(obs)
    obs = obs.filter(partial(_hipass_filter, sample_rate=obs.sampling_rate))
    obs = obs.reduction(6.0)[:, :time_samples, :]
    padded_arrival = _padded_arrival_for_key(key, obs, time_samples)
    return obs, padded_arrival


def _preprocess_mcs_pass(
    pf,
    time_samples: int,
    key: Optional[str] = None,
    noise_level: Optional[float] = None,
):
    """Normalize, filter, and crop one MCS pass."""
    if noise_level is None:
        noise_level = median_noise_level(pf)
    if noise_level > 0:
        pf = pf / noise_level

    pf = pf.filter(partial(_hipass_filter, sample_rate=pf.sampling_rate))
    pf = pf.reduction(6.0)[:, :time_samples, :]

    if key is not None:
        padded_arrival = _padded_arrival_for_key(key, pf, time_samples)
        if padded_arrival is not None:
            pf.first_arrival_reference = [padded_arrival]

    return pf


def load_mcs_station_pair(
    key: str,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
    mcs_pass_idx: int = 0,
    use_gt_noise_level: bool = False,
):
    """
    Load one MCS station and its diversity-stacked OBS target.

    Returns
    -------
    profiles_data : Profiles
        First MCS pass subsampled to the OBS offset grid (MCS trace offsets).
    profiles_target : Profiles
        Diversity-stacked OBS target for the matching reference pass.
    """
    obs, padded_arrival = _preprocess_obs_station(
        key, time_samples=time_samples, use_gt_noise_level=use_gt_noise_level
    )
    obs_pass = obs[obs_pass_idx : obs_pass_idx + 1]
    obs_target = _stack_targets(obs, padded_arrival)[obs_pass_idx : obs_pass_idx + 1]

    mcs = Profiles.load(f"noto/MCS/NT24MCS_J{key}C-1.sgy", jamstec_handler)
    mcs_pass = mcs[mcs_pass_idx : mcs_pass_idx + 1]

    trace_indices = mcs_trace_indices_for_obs_offsets(mcs_pass.offsets[0], obs_pass.offsets[0])
    mcs_aligned = _subset_mcs_pass(mcs_pass, trace_indices)

    if use_gt_noise_level:
        ref_noise = median_noise_level(obs_target)
        mcs_aligned = _preprocess_mcs_pass(
            mcs_aligned, time_samples, key=key, noise_level=ref_noise
        )
        if ref_noise > 0:
            obs_target = obs_target / ref_noise
    else:
        mcs_aligned = _preprocess_mcs_pass(mcs_aligned, time_samples, key=key)

    return mcs_aligned, obs_target


def load_mcs_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
    mcs_pass_idx: int = 0,
    use_gt_noise_level: bool = False,
):
    """
    Load MCS input passes and matching diversity-stacked OBS targets.

    Uses the same train / test split as ``data_noto.load_obs_profiles``
    (``stn_num_to_n``: non-negative = train, negative = test).

    When use_gt_noise_level is True, both MCS input and OBS target are normalized
    by the target pass median noise level (first 50 samples).

    Returns
    -------
    (profiles_data, profiles_target) or (None, None) if no stations match.

    profiles_data
        Concatenated MCS first passes (one per station), subsampled using OBS
        offsets and retaining MCS trace offsets.
    profiles_target
        Concatenated diversity-stacked OBS targets (one per station).
    """
    profiles_data = None
    profiles_target = None
    count = 0

    for key in iter_train_station_keys(train_only=train_only, test_only=test_only):
        mcs_pass, obs_target = load_mcs_station_pair(
            key,
            time_samples=time_samples,
            obs_pass_idx=obs_pass_idx,
            mcs_pass_idx=mcs_pass_idx,
            use_gt_noise_level=use_gt_noise_level,
        )

        if profiles_data is None:
            profiles_data = mcs_pass
            profiles_target = obs_target
        else:
            profiles_data = Profiles.concatenate((profiles_data, mcs_pass))
            profiles_target = Profiles.concatenate((profiles_target, obs_target))

        count += 1
        if max_stations is not None and count >= max_stations:
            break

    return profiles_data, profiles_target


if __name__ == "__main__":
    key = "01"
    mcs, obs_target = load_mcs_station_pair(key)
    obs, _ = _preprocess_obs_station(key)
    mcs_offsets = np.asarray(mcs.offsets[0], dtype=float)
    obs_offsets = np.asarray(obs.offsets[0], dtype=float)
    print(f"station {key}: obs traces={len(obs_offsets)}, mcs aligned traces={mcs.shape[2]}")
    print(f"MCS vs OBS offset max |diff| km: {np.max(np.abs(mcs_offsets - obs_offsets)):.4f}")
    print(f"MCS vs OBS offset mean |diff| km: {np.mean(np.abs(mcs_offsets - obs_offsets)):.6f}")
    print(f"profiles_data shape: {mcs.shape}, profiles_target shape: {obs_target.shape}")

    data, target = load_mcs_profiles(max_stations=3)
    print(f"batch profiles_data shape: {data.shape}, profiles_target shape: {target.shape}")
