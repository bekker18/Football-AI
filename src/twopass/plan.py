"""The gate: turn scored windows into the exact frames to run the ball on.

Pure logic (pandas / numpy only). Windows are selected highest-value first until
the frame budget is exhausted, their frame ranges unioned, and the result handed
to Pass 2 as a sorted, de-duplicated frame set (plus contiguous ranges for
efficient decoding).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TwoPassConfig


def frames_to_ranges(frames: np.ndarray) -> List[Tuple[int, int]]:
    """Collapse sorted unique frame indices into contiguous ``(start, end)`` runs."""
    if len(frames) == 0:
        return []
    frames = np.asarray(sorted(set(int(f) for f in frames)))
    ranges = []
    start = prev = int(frames[0])
    for f in frames[1:]:
        f = int(f)
        if f != prev + 1:
            ranges.append((start, prev))
            start = f
        prev = f
    ranges.append((start, prev))
    return ranges


def plan_ball_frames(
    windows: pd.DataFrame, total_frames: int, cfg: TwoPassConfig
) -> Tuple[np.ndarray, dict]:
    """Select windows within budget and return the frames to run the ball on.

    Greedy by ``peak_value`` (highest first); a window is taken only if its frames
    keep the running total within ``budget_frac * total_frames`` (and under
    ``max_windows``). Returns ``(frames, plan)`` where ``frames`` is a sorted
    unique int array and ``plan`` summarises the decision for the manifest.
    """
    budget = _budget_frames(cfg.budget_frac, total_frames)

    plan_base = dict(
        total_frames=int(total_frames),
        budget_frac=cfg.budget_frac,
        budget_frames=budget,
        n_windows_total=int(len(windows)),
    )
    if windows.empty:
        return np.array([], int), {**plan_base, **_selection_summary([], set(), total_frames)}

    ordered = windows.sort_values("peak_value", ascending=False, kind="stable")
    selected_ids: List[int] = []
    truncated_ids: List[int] = []
    selected: set = set()

    for w in ordered.itertuples():
        if cfg.max_windows is not None and len(selected_ids) >= cfg.max_windows:
            break
        if float(w.peak_value) < cfg.min_peak_value:
            continue
        wframes = _window_frames(int(w.start_frame), int(w.end_frame), total_frames)
        if not wframes:
            continue

        if budget is None:
            selected |= wframes
            selected_ids.append(int(w.window_id))
            continue

        remaining = budget - len(selected)
        if remaining <= 0:
            break
        new = wframes - selected
        if len(new) <= remaining:
            selected |= wframes
            selected_ids.append(int(w.window_id))
        else:
            # window is bigger than the remaining budget: rather than drop the
            # (highest-value) window entirely, spend the budget on a chunk of it
            # centred on its core, so the best moment is never skipped outright.
            chunk = _truncate_center(wframes, remaining, _window_center(w))
            selected |= chunk
            selected_ids.append(int(w.window_id))
            truncated_ids.append(int(w.window_id))
            break  # budget exhausted

    frames = np.array(sorted(selected), dtype=int)
    summary = _selection_summary(selected_ids, selected, total_frames)
    summary["truncated_window_ids"] = truncated_ids
    plan = {**plan_base, **summary}
    return frames, plan


def _budget_frames(budget_frac: Optional[float], total_frames: int) -> Optional[int]:
    if budget_frac is None:
        return None
    return int(budget_frac * total_frames)


def _window_frames(start: int, end: int, total_frames: int) -> set:
    """Clamped frame set for one window; empty if it falls outside the clip."""
    lo = max(0, start)
    hi = min(total_frames - 1, end)
    if hi < lo:
        return set()
    return set(range(lo, hi + 1))


def _window_center(w) -> int:
    """Frame to centre a truncation on: the core midpoint if present, else the
    window midpoint."""
    cs = getattr(w, "core_start_frame", None)
    ce = getattr(w, "core_end_frame", None)
    if cs is not None and ce is not None and not (pd.isna(cs) or pd.isna(ce)):
        return int((int(cs) + int(ce)) // 2)
    return int((int(w.start_frame) + int(w.end_frame)) // 2)


def _truncate_center(wframes: set, k: int, center: int) -> set:
    """Pick ``k`` contiguous frames of a window centred on ``center``."""
    lo, hi = min(wframes), max(wframes)
    k = min(k, hi - lo + 1)
    start = center - k // 2
    start = max(lo, min(start, hi - k + 1))  # keep the window fully inside [lo, hi]
    return set(range(start, start + k))


def _selection_summary(selected_ids: list, selected: set, total_frames: int) -> dict:
    frames = np.array(sorted(selected), dtype=int)
    return dict(
        selected_window_ids=list(selected_ids),
        n_windows_selected=len(selected_ids),
        n_frames=int(len(selected)),
        coverage_frac=round(len(selected) / total_frames, 4) if total_frames else 0.0,
        ranges=frames_to_ranges(frames),
    )
