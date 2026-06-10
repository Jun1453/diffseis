"""Shared NOTO OBS loading for C1 baseline scripts (matches train.py / rebuild_oop.py)."""
from functools import partial

import numpy as np
from profiledd import Profiles, jamstec_handler, highpass
from refine_train import stn_num_to_n, fit_curves


def _hipass_filter(trace, sample_rate):
    return highpass(trace, 2.0, sample_rate, poles=4)


def load_station_pair(key: str, time_samples: int = 6000):
    """
    Load one OBS station: noisy input passes (5 shots) and diversity-stack targets (3).

    Shot index mapping matches train.py:
      shot 0,4 -> pf_stack2; shots 1,2,3 -> pf_stack3.
    """
    pf = Profiles.load(f"noto/OBS/NT24OBS_J{key}C-1.sgy", jamstec_handler)
    pf = Profiles.concatenate(
        [pf[n:n + 1] / np.median((np.ravel(np.abs(pf[n, :50, :])))) for n in range(5)]
    )
    pf = pf.filter(partial(_hipass_filter, sample_rate=pf.sampling_rate)).reduction(6.0)[:, :time_samples, :]

    arrival = fit_curves[f"{key}"] + pf.sampling_rate * 0.5 - 75
    padded_arrival = np.pad(arrival, (0, pf.shape[2] - len(arrival)), mode="edge")

    pf_stack3 = pf[1:4].diversity_stack(
        orig_profile_num=True, first_arrival_reference=padded_arrival
    )
    pf_stack2 = pf[0:5:4].diversity_stack(
        orig_profile_num=True, first_arrival_reference=np.flip(padded_arrival)
    )

    def clean_for_shot(shot_idx: int) -> np.ndarray:
        if shot_idx == 0:
            return np.asarray(pf_stack2[0], dtype=np.float32)
        if shot_idx == 4:
            return np.asarray(pf_stack2[1], dtype=np.float32)
        return np.asarray(pf_stack3[shot_idx - 1], dtype=np.float32)

    return pf, clean_for_shot, padded_arrival


def _stack_targets(pf, padded_arrival, lopo_enable=False):
    """
    Build one diversity-stack target per input pass (profile order 0..4).

    Default: shots 1–3 share a 3-pass stack; shots 0 and 4 share a 2-pass stack
    (matches train.py). With lopo_enable, each target stacks all passes except
    the corresponding input pass.
    """
    if lopo_enable:
        stacks = []
        for shot_idx in range(pf.shape[0]):
            others = Profiles.concatenate(
                [pf[n:n + 1] for n in range(pf.shape[0]) if n != shot_idx]
            )
            stack = others.diversity_stack(
                orig_profile_num=False,
                first_arrival_reference=padded_arrival,
            )
            stacks.append(stack[0:1])
        return Profiles.concatenate(stacks)

    pf_stack3 = pf[1:4].diversity_stack(
        orig_profile_num=True, first_arrival_reference=padded_arrival
    )
    pf_stack2 = pf[0:5:4].diversity_stack(
        orig_profile_num=True,
        first_arrival_reference=np.flip(padded_arrival) if padded_arrival is not None else None,
    )
    return Profiles.concatenate((pf_stack2[0:1], pf_stack3, pf_stack2[1:2]))


def iter_train_station_keys(train_only=True, test_only=False, holdout_keys=None):
    """Yield station keys for train / test split (stn_num_to_n)."""
    holdout = set(holdout_keys or [])
    for key, value in stn_num_to_n.items():
        if key in holdout:
            continue
        if test_only:
            if value >= 0:
                continue
        elif train_only:
            if value < 0:
                continue
        yield key


def load_obs_profiles(
    train_only=True,
    test_only=False,
    max_stations=None,
    lopo_enable=False,
):
    """
    Load input passes and diversity-stack targets.

    When lopo_enable is True, each input pass is paired with a diversity stack
    of the remaining passes (leave-one-pass-out).

    Returns (profiles_data, profiles_target) or (None, None) if no stations match.
    """
    profiles_data = None
    profiles_target = None
    count = 0

    for key, value in stn_num_to_n.items():
        if test_only:
            if value >= 0:
                continue
        elif train_only:
            if value < 0:
                continue

        pf = Profiles.load(f'noto/OBS/NT24OBS_J{key}C-1.sgy', jamstec_handler)
        pf = Profiles.concatenate(
            [pf[n:n + 1] / np.median((np.ravel(np.abs(pf[n, :50, :])))) for n in range(5)]
        )
        pf = pf.filter(partial(_hipass_filter, sample_rate=pf.sampling_rate)).reduction(6.0)[:, :6000, :]

        try:
            arrival = fit_curves[f'{key}'] + pf.sampling_rate * 0.5 - 75
            padded_arrival = np.pad(arrival, (0, pf.shape[2] - len(arrival)), mode='edge')
        except:
            padded_arrival = None

        pf_stack_all = _stack_targets(pf, padded_arrival, lopo_enable=lopo_enable)
        norm_pf = pf[:]

        if profiles_data is None:
            profiles_data = norm_pf
            profiles_target = pf_stack_all
        else:
            profiles_data = Profiles.concatenate((profiles_data, norm_pf))
            profiles_target = Profiles.concatenate((profiles_target, pf_stack_all))

        count += 1
        if max_stations is not None and count >= max_stations:
            break

    return profiles_data, profiles_target


# Default fragmentize kwargs aligned with train.py
FRAGMENT_KWARGS = dict(
    vclip=20,
    tmin=0,
    t_interval=3.7,
    x_move_ratio=0.8,
    y_move_ratio=0.8,
)
