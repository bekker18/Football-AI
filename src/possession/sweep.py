"""Calibration mode — run the detector across a range of possession radii.

The default ``r_pz_m`` (3.0 m) was measured on a single open-play attacking clip.
That is exactly the sort of number that goes stale on new footage: a congested
box or a set-piece packs many more players inside any given radius, so the duel
rate at 3.0 m there will be nothing like the ~2% seen on the calibration clip.

This mode is how the radius gets **re-validated before it is frozen**. It reruns
the detector at each radius and reports the trade-off curve::

    r_pz_m  coverage%  clean%  duel%  n_segments  median_hold_frames

Read it as: coverage climbs with the radius (good) while clean attribution falls
and the duel rate accelerates (bad). Pick the largest radius that still keeps the
duel rate acceptable on *your* footage.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

import numpy as np
import pandas as pd

from .config import PossessionConfig
from .pipeline import detect_possession

SWEEP_COLUMNS = [
    "r_pz_m", "coverage_pct", "clean_pct", "duel_pct",
    "n_segments", "median_hold_frames",
    "n_attributed_frames", "n_ball_frames",
]


def radius_grid(r_min: float, r_max: float, step: float) -> List[float]:
    """Inclusive radius grid, rounded to avoid float dust in the output."""
    if step <= 0:
        raise ValueError("sweep step must be > 0")
    if r_max < r_min:
        raise ValueError("sweep r_max must be >= r_min")
    n = int(round((r_max - r_min) / step))
    return [round(r_min + i * step, 6) for i in range(n + 1)]


def sweep_radii(
    df: pd.DataFrame, cfg: PossessionConfig, radii: Iterable[float]
) -> pd.DataFrame:
    """Run the detector once per radius; one summary row per radius.

    The distance computation inside each run is vectorized; the only Python loop
    here is over the handful of radii, which keeps the sweep trivially readable.
    """
    rows = []
    for r in radii:
        frames, segments, meta = detect_possession(df, replace(cfg, r_pz_m=float(r)))
        rows.append(
            dict(
                r_pz_m=float(r),
                coverage_pct=meta["coverage_pct"],
                clean_pct=meta["clean_pct"],
                duel_pct=meta["duel_pct"],
                n_segments=meta["n_segments"],
                median_hold_frames=meta["median_hold_frames"],
                n_attributed_frames=meta["n_attributed_frames"],
                n_ball_frames=meta["n_ball_frames"],
            )
        )
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


def format_sweep(sweep: pd.DataFrame) -> str:
    """Render the sweep as a fixed-width table for the terminal."""
    header = (
        f"{'r_pz_m':>7}  {'coverage%':>9}  {'clean%':>7}  {'duel%':>6}  "
        f"{'segments':>8}  {'median_hold_f':>13}"
    )
    lines = [header, "-" * len(header)]
    for r in sweep.itertuples():
        lines.append(
            f"{r.r_pz_m:>7.1f}  {r.coverage_pct:>9.1f}  {r.clean_pct:>7.1f}  "
            f"{r.duel_pct:>6.1f}  {r.n_segments:>8d}  {r.median_hold_frames:>13.1f}"
        )
    return "\n".join(lines)
