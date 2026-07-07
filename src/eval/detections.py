"""The canonical detection table + coordinate transforms.

Everything the metric engine consumes is a DataFrame with the columns in
``CANON_COLS`` (frame, track_id, x, y, role, team). Format adapters produce it
via :func:`as_detections`; the flip search rewrites coordinates via
:func:`apply_transform`.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .config import CANON_COLS, EvalConfig

# default source-column names when a caller's table already uses ours
_DEFAULT_MAP = {
    "frame": "frame",
    "track_id": "track_id",
    "x": "x",
    "y": "y",
    "role": "role",
    "team": "team",
}


def as_detections(
    df: pd.DataFrame,
    colmap: Optional[Mapping[str, str]] = None,
    *,
    roles: Optional[list] = None,
) -> pd.DataFrame:
    """Normalise an arbitrary table onto the canonical detection schema.

    ``colmap`` maps canonical name -> source column name; unspecified canonical
    names fall back to the identity mapping, and ``role``/``team`` are optional
    (filled with null when absent). Rows without a finite ``x``/``y`` (no pitch
    fix) or without a ``track_id`` are dropped — they cannot be matched. When
    ``roles`` is given, only those roles are kept.
    """
    m = dict(_DEFAULT_MAP)
    if colmap:
        m.update({k: v for k, v in colmap.items() if v is not None})

    out = pd.DataFrame()
    for canon in ("frame", "track_id", "x", "y"):
        src = m[canon]
        if src not in df.columns:
            raise KeyError(f"required column {src!r} (for {canon!r}) not in table")
        out[canon] = df[src]
    for canon in ("role", "team"):
        src = m.get(canon)
        out[canon] = df[src] if (src and src in df.columns) else None

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["x", "y", "track_id"]).copy()
    out["frame"] = out["frame"].astype(int)
    out["track_id"] = out["track_id"].astype(int)

    if roles is not None and out["role"].notna().any():
        out = out[out["role"].isin(list(roles))].copy()

    return out[list(CANON_COLS)].reset_index(drop=True)


def apply_transform(det: pd.DataFrame, transform: str, cfg: EvalConfig) -> pd.DataFrame:
    """Return a copy of ``det`` with x/y rewritten by an orientation transform.

    ``none`` is identity; ``rot180`` rotates the pitch 180 degrees; ``mirror_x``
    / ``mirror_y`` reflect across the halfway lines. Used by the flip search to
    resolve the arbitrary attacking-direction / orientation of our coordinates.
    """
    L, W = cfg.pitch()
    if transform == "none":
        return det.copy()
    out = det.copy()
    if transform in ("rot180", "mirror_x"):
        out["x"] = L - out["x"]
    if transform in ("rot180", "mirror_y"):
        out["y"] = W - out["y"]
    if transform not in ("rot180", "mirror_x", "mirror_y"):
        raise ValueError(f"unknown transform {transform!r}")
    return out
