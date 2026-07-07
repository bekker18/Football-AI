"""Turn the per-frame value stream into high-value windows.

Contiguous runs of high-value frames are bridged across small gaps, filtered by
minimum duration, padded, and scored. Each emitted window is a frame range the
two-pass controller can hand to the expensive ball detector.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import EventConfig

WINDOW_COLS = [
    "window_id", "start_frame", "end_frame", "core_start_frame", "core_end_frame",
    "start_time_s", "end_time_s", "duration_s", "peak_value", "mean_value",
    "attacking_team", "n_high_frames",
]


def windows_from_signals(signals: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """Assemble high-value windows from an :func:`attacking_signals` table.

    Returns one row per window (columns in ``WINDOW_COLS``), sorted by
    ``peak_value`` descending so the highest-value windows lead — the order the
    controller would spend its ball-detection budget in.
    """
    signals = signals.sort_values("frame", kind="stable").reset_index(drop=True)
    frames = signals["frame"].to_numpy()
    values = signals["value_score"].to_numpy(float)
    if len(frames) == 0:
        return pd.DataFrame(columns=WINDOW_COLS)

    fmin, fmax = int(frames[0]), int(frames[-1])
    high_frames = frames[values >= cfg.value_threshold]
    if len(high_frames) == 0:
        return pd.DataFrame(columns=WINDOW_COLS)

    clusters = _bridge(high_frames, cfg.window_merge_gap_frames)

    recs: List[dict] = []
    wid = 0
    for core_start, core_end in clusters:
        if (core_end - core_start + 1) < cfg.window_min_frames:
            continue
        start = max(fmin, core_start - cfg.window_pad_frames)
        end = min(fmax, core_end + cfg.window_pad_frames)
        recs.append(_score_window(wid, signals, start, end, core_start, core_end, cfg))
        wid += 1

    if not recs:
        return pd.DataFrame(columns=WINDOW_COLS)
    out = pd.DataFrame(recs, columns=WINDOW_COLS)
    return out.sort_values("peak_value", ascending=False).reset_index(drop=True)


def _bridge(high_frames: np.ndarray, merge_gap: int) -> List[tuple]:
    """Group sorted frame indices into (start, end) runs, bridging gaps <= merge_gap."""
    clusters = []
    start = prev = int(high_frames[0])
    for f in high_frames[1:]:
        f = int(f)
        if f - prev > merge_gap:
            clusters.append((start, prev))
            start = f
        prev = f
    clusters.append((start, prev))
    return clusters


def _score_window(wid, signals, start, end, core_start, core_end, cfg) -> dict:
    """Summarise one window; value stats are taken over the unpadded core."""
    core = signals[(signals["frame"] >= core_start) & (signals["frame"] <= core_end)]
    high_core = core[core["value_score"] >= cfg.value_threshold]
    team_mode = high_core["attacking_team"].mode()
    attacking_team = float(team_mode.iloc[0]) if len(team_mode) else np.nan

    t_start = float(signals.loc[signals["frame"] == start, "time_s"].iloc[0])
    t_end = float(signals.loc[signals["frame"] == end, "time_s"].iloc[0])
    return dict(
        window_id=wid,
        start_frame=int(start),
        end_frame=int(end),
        core_start_frame=int(core_start),
        core_end_frame=int(core_end),
        start_time_s=round(t_start, 3),
        end_time_s=round(t_end, 3),
        duration_s=round((end - start + 1) / cfg.fps, 3),
        peak_value=round(float(core["value_score"].max()), 3),
        mean_value=round(float(core["value_score"].mean()), 3),
        attacking_team=(None if np.isnan(attacking_team) else int(attacking_team)),
        n_high_frames=int(len(high_core)),
    )
