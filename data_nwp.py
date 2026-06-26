"""NW Pacific OBS/MCS loading (structure aligned with data_otj.py / data_noto_mcs.py)."""
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import segyio

from data_noto import FRAGMENT_KWARGS as _NOTO_FRAGMENT_KWARGS, _normalize_passes, median_noise_level
from data_noto_mcs import _subset_mcs_pass, mcs_trace_indices_for_obs_offsets
from profiledd import Profiles, highpass

NWP_DIR = Path("nwp")
NWP_BASE = "kr0906ky0903_A2"
NWP_KEY_MIN = 10
NWP_KEY_MAX = 10
NWP_TRACE_COUNT = 1060


def _hipass_filter(trace, sample_rate):
    return highpass(trace, 2.0, sample_rate, poles=4)


def nwp_obs_segy_path(key: str) -> Path:
    return NWP_DIR / "obs" / f"{NWP_BASE}_obs_s{key}-1.sgy"


def nwp_mcs_segy_path(key: str) -> Path:
    return NWP_DIR / "mcs" / f"{NWP_BASE}_mcs_s{key}-1.sgy"


def nwp_segy_path(key: str) -> Path:
    """Backward-compatible alias for the MCS path."""
    return nwp_mcs_segy_path(key)


def nwp_mcs_handler(f):
    """Stack all traces as (n_traces, n_samples) per segyio convention."""
    raw = np.array([f.trace.raw[i] for i in range(len(f.trace.raw))], dtype=np.float32)
    offsets = np.array([header[37] for header in f.header], dtype=np.float32) / 1000
    return raw, offsets


nwp_obs_handler = nwp_mcs_handler


def _to_profiles_layout(raws, offsets):
    """
    Convert segyio stacks to Profiles layout (n_profiles, n_samples, n_traces).

    Handles (n_traces, n_samples) and (n_traces, n_samples, n_profiles) inputs
    where trace/profile axes are swapped relative to Profiles indexing.
    """
    raw = np.asarray(raws, dtype=np.float32)
    off = np.asarray(offsets, dtype=np.float32)

    if raw.ndim == 2:
        raw = raw.T[np.newaxis, ...]
        if off.ndim == 1:
            off = off[np.newaxis, ...]
    elif raw.ndim == 3:
        if raw.shape[0] > raw.shape[2]:
            raw = np.transpose(raw, (2, 1, 0))
            if off.ndim == 2:
                off = np.transpose(off, (1, 0))
        elif raw.shape[1] < raw.shape[2]:
            raw = np.transpose(raw, (0, 2, 1))
    else:
        raise ValueError(f"unexpected waveform shape for Profiles: {raw.shape}")

    profile_raws = [raw[i] for i in range(raw.shape[0])]
    profile_offsets = [off[i] for i in range(off.shape[0])]
    return profile_raws, profile_offsets


def load_nwp_segy(segy_file, data_handle=nwp_mcs_handler, filter_history=None, reduction_vel=0):
    """Load one NW Pacific SEGY into Profiles (ignore_geometry=True)."""
    with segyio.open(segy_file, ignore_geometry=True) as f:
        raws, offsets = data_handle(f)
        raws, offsets = _to_profiles_layout(raws, offsets)
        sampling_rate = 1e6 / f.header[0][117]
    return Profiles(
        raws,
        sampling_rate=sampling_rate,
        filter_history=filter_history or [],
        reduction_vel=reduction_vel,
        offsets=offsets,
    )


def load_nwp_obs(key: str) -> Profiles:
    return load_nwp_segy(nwp_obs_segy_path(key), nwp_obs_handler)


def load_nwp_mcs(key: str) -> Profiles:
    return load_nwp_segy(nwp_mcs_segy_path(key), nwp_mcs_handler)


def iter_nwp_station_keys(train_only=True, test_only=False, holdout_keys=None):
    """Yield shot keys ``001``..``100``."""
    holdout = set(holdout_keys or [])
    for n in range(NWP_KEY_MIN, NWP_KEY_MAX + 1):
        key = f"{n:03d}"
        if key in holdout:
            continue
        yield key


def _preprocess_obs_station(key: str, time_samples: int = 6000, use_gt_noise_level: bool = False):
    """Load and preprocess OBS passes for one shot key."""
    if not nwp_obs_segy_path(key).is_file():
        raise FileNotFoundError(nwp_obs_segy_path(key))
    obs = load_nwp_obs(key)
    if not use_gt_noise_level:
        obs = _normalize_passes(obs)
    obs = obs.filter(partial(_hipass_filter, sample_rate=obs.sampling_rate))
    obs = obs.reduction(6.0)[:, :time_samples, :]
    return obs


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


def _crop_last_traces(pf, n_traces: int = NWP_TRACE_COUNT):
    """Keep only the last ``n_traces`` along the trace axis."""
    if pf.shape[2] <= n_traces:
        return pf
    return pf[:, :, -n_traces:]


def load_station_pair(key: str, time_samples: int = 6000, use_gt_noise_level: bool = False):
    """Load one NW Pacific OBS station (input and target are the same OBS passes)."""
    pf = _preprocess_obs_station(key, time_samples=time_samples, use_gt_noise_level=use_gt_noise_level)

    def clean_for_shot(shot_idx: int) -> np.ndarray:
        return np.asarray(pf[shot_idx], dtype=np.float32)

    return pf, clean_for_shot, None


def load_mcs_station_pair(
    key: str,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
    mcs_pass_idx: int = 0,
    use_gt_noise_level: bool = False,
):
    """
    Load one NW Pacific MCS station with the matching OBS pass as target.

    MCS traces are subsampled onto the OBS offset grid (closest offset per trace),
    matching ``data_noto_mcs.load_mcs_station_pair``.
    """
    obs = _preprocess_obs_station(
        key, time_samples=time_samples, use_gt_noise_level=use_gt_noise_level
    )
    obs_pass = obs[obs_pass_idx : obs_pass_idx + 1]
    obs_target = obs_pass[:]

    if not nwp_mcs_segy_path(key).is_file():
        raise FileNotFoundError(nwp_mcs_segy_path(key))
    mcs = load_nwp_mcs(key)
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

    mcs_aligned = _crop_last_traces(mcs_aligned)
    obs_target = _crop_last_traces(obs_target)
    return mcs_aligned, obs_target


def load_nwp_obs_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    lopo_enable=False,
    use_gt_noise_level=False,
):
    """
    Load NW Pacific OBS passes (input and target are the same preprocessed OBS data).

    Returns (profiles_data, profiles_target) or (None, None) if no stations match.
    """
    if lopo_enable:
        raise ValueError("--lopo_enable is not supported for NW Pacific OBS data.")

    profiles_data = None
    profiles_target = None
    count = 0

    for key in iter_nwp_station_keys(train_only=train_only, test_only=test_only):
        if not nwp_obs_segy_path(key).is_file():
            continue
        pf = _preprocess_obs_station(key, use_gt_noise_level=use_gt_noise_level)

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


def load_nwp_mcs_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    time_samples: int = 6000,
    obs_pass_idx: int = 0,
    mcs_pass_idx: int = 0,
    use_gt_noise_level: bool = False,
):
    """
    Load NW Pacific MCS inputs with matching OBS passes as targets.

    Returns (profiles_data, profiles_target) or (None, None) if no stations match.
    """
    profiles_data = None
    profiles_target = None
    count = 0

    for key in iter_nwp_station_keys(train_only=train_only, test_only=test_only):
        if not nwp_obs_segy_path(key).is_file() or not nwp_mcs_segy_path(key).is_file():
            continue

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


def load_nwp_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    lopo_enable=False,
    use_gt_noise_level=False,
    **kwargs,
):
    """
    Load NW Pacific MCS inputs with matching OBS targets.

    Alias for ``load_nwp_mcs_profiles`` (``lopo_enable`` is not supported).
    """
    if lopo_enable:
        raise ValueError("--lopo_enable is not supported for NW Pacific MCS data.")
    return load_nwp_mcs_profiles(
        train_only=train_only,
        test_only=test_only,
        max_stations=max_stations,
        use_gt_noise_level=use_gt_noise_level,
        **kwargs,
    )

FRAGMENT_KWARGS = dict(
    **_NOTO_FRAGMENT_KWARGS
)
FRAGMENT_KWARGS['t_interval'] = 8
FRAGMENT_KWARGS['vclip'] = 1