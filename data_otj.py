"""Ontong Java OBS/MCS loading (structure aligned with data_noto.py / data_noto_mcs.py)."""
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import segyio

from data_noto import FRAGMENT_KWARGS as _NOTO_FRAGMENT_KWARGS, _normalize_passes, median_noise_level
from data_noto_mcs import _subset_mcs_pass, mcs_trace_indices_for_obs_offsets
from profiledd import Profiles, highpass, ontongjava_handler

OTJ_OBS_DIR = Path("OntongJava/obs")
OTJ_MCS_DIR = Path("OntongJava/mcs")
OTJ_RECEIVER = "kr1005"
OTJ_KEY_MIN = 1
OTJ_KEY_MAX = 100


def _hipass_filter(trace, sample_rate):
    return highpass(trace, 2.0, sample_rate, poles=4)


def otj_obs_segy_path(key: str) -> Path:
    return OTJ_OBS_DIR / f"{OTJ_RECEIVER}_LineNSobs_s{key}-1.sgy"


def otj_mcs_segy_path(key: str) -> Path:
    return OTJ_MCS_DIR / f"{OTJ_RECEIVER}_LineNSmcs_s{key}-1.sgy"


def load_otj_segy(segy_file, data_handle=ontongjava_handler, filter_history=None, reduction_vel=0):
    """Load all shooting passes from one Ontong Java SEGY (ignore_geometry=True)."""
    with segyio.open(segy_file, ignore_geometry=True) as f:
        raws, offsets = data_handle(f)
        sampling_rate = 1e6 / f.header[0][117]
    return Profiles(
        raws,
        sampling_rate=sampling_rate,
        filter_history=filter_history or [],
        reduction_vel=reduction_vel,
        offsets=offsets,
    )


def load_otj_obs(key: str) -> Profiles:
    return load_otj_segy(otj_obs_segy_path(key), ontongjava_handler)


def load_otj_mcs(key: str) -> Profiles:
    return load_otj_segy(otj_mcs_segy_path(key), ontongjava_handler)


def iter_otj_station_keys(train_only=True, test_only=False, holdout_keys=None):
    """Yield shot keys ``001``..``100``."""
    holdout = set(holdout_keys or [])
    for n in range(OTJ_KEY_MIN, OTJ_KEY_MAX + 1):
        key = f"{n:03d}"
        if key in holdout:
            continue
        yield key


def _preprocess_obs_station(key: str, time_samples: int = 6000, use_gt_noise_level: bool = False):
    """Load and preprocess all OBS passes for one shot key."""
    if not otj_obs_segy_path(key).is_file():
        raise FileNotFoundError(otj_obs_segy_path(key))
    obs = load_otj_obs(key)
    if not use_gt_noise_level:
        obs = _normalize_passes(obs)
    obs = obs.filter(partial(_hipass_filter, sample_rate=obs.sampling_rate))
    obs = obs.reduction(6.0)[:, :time_samples, :]
    return obs


def _preprocess_mcs_station(key: str, time_samples: int = 6000, use_gt_noise_level: bool = False):
    """Load and preprocess all MCS passes for one shot key."""
    if not otj_mcs_segy_path(key).is_file():
        raise FileNotFoundError(otj_mcs_segy_path(key))
    mcs = load_otj_mcs(key)
    if not use_gt_noise_level:
        mcs = _normalize_passes(mcs)
    mcs = mcs.filter(partial(_hipass_filter, sample_rate=mcs.sampling_rate))
    mcs = mcs.reduction(6.0)[:, :time_samples, :]
    return mcs


def _mean_offset_match_cost(mcs_offsets, obs_offsets) -> float:
    """Mean |offset error| after mapping each OBS trace to the closest MCS trace."""
    mcs_offsets = np.asarray(mcs_offsets, dtype=float)
    obs_offsets = np.asarray(obs_offsets, dtype=float)
    trace_indices = mcs_trace_indices_for_obs_offsets(mcs_offsets, obs_offsets)
    return float(np.mean(np.abs(mcs_offsets[trace_indices] - obs_offsets)))


def best_mcs_pass_idx(mcs: Profiles, obs_pass: Profiles) -> int:
    """Pick the MCS shooting pass that best matches ``obs_pass`` offsets."""
    obs_offsets = obs_pass.offsets[0]
    costs = [_mean_offset_match_cost(mcs.offsets[j], obs_offsets) for j in range(mcs.shape[0])]
    return int(np.argmin(costs))


def _unique_trace_indices(offsets) -> np.ndarray:
    """Index of the first trace for each unique offset (preserve along-line order)."""
    offsets = np.asarray(offsets, dtype=float)
    _, idx = np.unique(offsets, return_index=True)
    return np.sort(idx)


def _subset_profile_pass(pf_pass: Profiles, trace_indices) -> Profiles:
    """Select traces from a single-profile ``Profiles`` slice."""
    trace_indices = np.asarray(trace_indices, dtype=int)
    data = np.asarray(pf_pass)[0:1, :, trace_indices]
    offsets = [np.asarray(pf_pass.offsets[0], dtype=float)[trace_indices]]
    first_arrival_reference = None
    if pf_pass.first_arrival_reference:
        fa = pf_pass.first_arrival_reference[0]
        if fa is not None:
            first_arrival_reference = [np.asarray(fa, dtype=float)[trace_indices]]
        else:
            first_arrival_reference = [None]
    return Profiles(
        data,
        sampling_rate=pf_pass.sampling_rate,
        filter_history=pf_pass.filter_history,
        reduction_vel=pf_pass.reduction_vel,
        offsets=offsets,
        first_arrival_reference=first_arrival_reference,
    )


def trim_unique_offsets(pf: Profiles) -> Profiles:
    """Drop traces that share an offset with an earlier trace in each profile."""
    if pf.shape[0] == 0 or pf.shape[2] == 0:
        return pf
    passes = [
        _subset_profile_pass(pf[i : i + 1], _unique_trace_indices(pf.offsets[i]))
        for i in range(pf.shape[0])
    ]
    return passes[0] if len(passes) == 1 else Profiles.concatenate(passes)


def _preprocess_mcs_pass(
    pf,
    time_samples: int,
    noise_level: Optional[float] = None,
):
    """Normalize, filter, and crop one MCS pass."""
    if noise_level is None:
        noise_level = median_noise_level(pf)
    if noise_level > 0:
        pf = pf / noise_level
    pf = pf.filter(partial(_hipass_filter, sample_rate=pf.sampling_rate))
    pf = pf.reduction(6.0)[:, :time_samples, :]
    return pf


def load_station_pair(
    key: str,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
    use_gt_noise_level: bool = False,
):
    """Load one Ontong Java OBS station (input and target are the same OBS passes)."""
    pf = trim_unique_offsets(
        _preprocess_obs_station(key, time_samples=time_samples, use_gt_noise_level=use_gt_noise_level)
    )

    def clean_for_shot(shot_idx: int) -> np.ndarray:
        return np.asarray(pf[shot_idx], dtype=np.float32)

    return pf, clean_for_shot, None


def load_mcs_station_pair(
    key: str,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
    use_gt_noise_level: bool = False,
):
    """
    Load one Ontong Java MCS/OBS pair for a single OBS shooting pass.

    The MCS shooting pass with the closest mean offset match to the OBS pass
    is selected over all MCS passes in the file.
    """
    obs = _preprocess_obs_station(
        key, time_samples=time_samples, use_gt_noise_level=use_gt_noise_level
    )
    if obs_pass_idx < 0 or obs_pass_idx >= obs.shape[0]:
        raise IndexError(f"obs_pass_idx {obs_pass_idx} out of range for {obs.shape[0]} OBS passes")
    obs_pass = obs[obs_pass_idx : obs_pass_idx + 1]
    obs_target = obs_pass[:]

    mcs = _preprocess_mcs_station(
        key, time_samples=time_samples, use_gt_noise_level=use_gt_noise_level
    )
    mcs_pass_idx = best_mcs_pass_idx(mcs, obs_pass)
    mcs_pass = mcs[mcs_pass_idx : mcs_pass_idx + 1]

    trace_indices = mcs_trace_indices_for_obs_offsets(mcs_pass.offsets[0], obs_pass.offsets[0])
    mcs_aligned = _subset_mcs_pass(mcs_pass, trace_indices)

    if use_gt_noise_level:
        ref_noise = median_noise_level(obs_target)
        mcs_aligned = _preprocess_mcs_pass(mcs_aligned, time_samples, noise_level=ref_noise)
        if ref_noise > 0:
            obs_target = obs_target / ref_noise
    else:
        mcs_aligned = _preprocess_mcs_pass(mcs_aligned, time_samples)

    unique_idx = _unique_trace_indices(obs_target.offsets[0])
    mcs_aligned = _subset_profile_pass(mcs_aligned, unique_idx)
    obs_target = _subset_profile_pass(obs_target, unique_idx)
    return mcs_aligned, obs_target


def load_otj_obs_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    lopo_enable=False,
    use_gt_noise_level=False,
):
    """
    Load Ontong Java OBS passes (input and target are the same preprocessed OBS data).

    Returns (profiles_data, profiles_target) or (None, None) if no stations match.
    """
    if lopo_enable:
        raise ValueError("--lopo_enable is not supported for Ontong Java OBS data.")

    profiles_data = None
    profiles_target = None
    count = 0

    for key in iter_otj_station_keys(train_only=train_only, test_only=test_only):
        if not otj_obs_segy_path(key).is_file():
            continue
        pf = trim_unique_offsets(_preprocess_obs_station(key, use_gt_noise_level=use_gt_noise_level))

        if profiles_data is None:
            profiles_data = pf
            profiles_target = pf[:]
        else:
            profiles_data = Profiles.concatenate((profiles_data, pf))
            profiles_target = Profiles.concatenate((profiles_target, pf[:]))

        count += 1
        if max_stations is not None and count >= max_stations:
            break

    return profiles_data, profiles_target


def load_otj_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    lopo_enable=False,
    use_gt_noise_level=False,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
):
    """
    Load Ontong Java MCS inputs with matching OBS targets for one OBS shooting pass.

    All OBS and MCS shooting passes are read from SEGY. For each station, the MCS
    pass whose offsets best match the user-specified OBS pass is selected, then
    MCS traces are subsampled onto the OBS offset grid.

    Returns (profiles_data, profiles_target) or (None, None) if no stations match.
    """
    if lopo_enable:
        raise ValueError("--lopo_enable is not supported for Ontong Java MCS data.")

    profiles_data = None
    profiles_target = None
    count = 0

    for key in iter_otj_station_keys(train_only=train_only, test_only=test_only):
        if not otj_obs_segy_path(key).is_file() or not otj_mcs_segy_path(key).is_file():
            continue

        mcs_pass, obs_target = load_mcs_station_pair(
            key,
            time_samples=time_samples,
            obs_pass_idx=obs_pass_idx,
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


def load_otj_mcs_profiles(*args, **kwargs):
    """Backward-compatible alias for ``load_otj_profiles``."""
    return load_otj_profiles(*args, **kwargs)


FRAGMENT_KWARGS = dict(
    **_NOTO_FRAGMENT_KWARGS
)
FRAGMENT_KWARGS['t_interval'] = 7.6
FRAGMENT_KWARGS['vclip'] = 1
