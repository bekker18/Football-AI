"""The prerequisite pipeline: chain the five transforms into one call.

Order (see module docstrings for why): stitch ids -> resolve direction ->
smooth ball -> synth dead ball -> rescale coords. Rescale runs last so the
synthetic ball rows added during smoothing are also rescaled; it is otherwise
independent and could run at any point.

Each transform is importable and runnable on its own; this module just composes
them and aggregates their metadata into a single manifest.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .ball import smooth_ball
from .config import PrepConfig
from .deadball import synth_dead_ball
from .direction import resolve_direction
from .rescale import rescale_coords
from .stitch import stitch_ids


def run_prerequisites(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, dict]:
    """Run all five prerequisite transforms in the recommended order.

    Returns ``(prepared_df, meta)`` where ``meta`` collects each transform's
    metadata under its own key.
    """
    meta = {"config": cfg.as_meta(), "steps": {}}

    df, m = stitch_ids(df, cfg)
    meta["steps"]["stitch_ids"] = m

    df, m = resolve_direction(df, cfg)
    meta["steps"]["resolve_direction"] = m

    df, m = smooth_ball(df, cfg)
    meta["steps"]["smooth_ball"] = m

    df, m = synth_dead_ball(df, cfg)
    meta["steps"]["synth_dead_ball"] = m

    df, m = rescale_coords(df, cfg)
    meta["steps"]["rescale_coords"] = m

    return df, meta
