"""Transform 2 — team normalization + attacking direction (per period).

Each team's attacking direction is resolved from its goalkeeper's median pitch_x:
a GK sitting at *low* x defends the x=0 goal, so the team attacks toward +x
(``attack_dir = +1``); a GK at *high* x means the team attacks toward -x
(``attack_dir = -1``). Fallbacks handle sparse/absent goalkeepers.

Team labels themselves are already stable within a clip (Layer 1 majority-votes
them) but remain arbitrary 0/1 with no inherent direction — that is exactly what
this step supplies. The shared pitch frame is left intact; use
:func:`normalize_to_attack` to get attacking-normalized coordinates on demand.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .common import period_groups
from .config import COL_ATTACK_DIR, PEOPLE_ROLES, PrepConfig

TEAMS = (0.0, 1.0)


def _dir_from_median(median_x: float, center_x: float) -> int:
    """GK below midfield -> team attacks +x (+1); above -> -x (-1)."""
    return 1 if median_x < center_x else -1


def _resolve_period(g: pd.DataFrame, cfg: PrepConfig) -> Tuple[Dict[float, Optional[int]], dict]:
    """Resolve both teams' attacking directions for one period.

    Returns ``({team: dir or None}, info)`` where ``info`` records the method and
    the evidence used, for the manifest.
    """
    center_x = cfg.source_length_m / 2.0
    gk = g[
        (g["role"] == "goalkeeper")
        & g["pitch_valid"].fillna(False).astype(bool)
        & g["team"].isin(TEAMS)
    ]
    gk = gk.dropna(subset=["pitch_x_m"])

    counts = {t: int((gk["team"] == t).sum()) for t in TEAMS}
    medians = {
        t: (float(gk.loc[gk["team"] == t, "pitch_x_m"].median()) if counts[t] else np.nan)
        for t in TEAMS
    }
    enough = {t: counts[t] >= cfg.min_gk_frames for t in TEAMS}

    dirs: Dict[float, Optional[int]] = {t: None for t in TEAMS}

    if enough[0.0] and enough[1.0]:
        # both GKs solid: assign by *relative* position so teams are forced
        # opposite even if homography noise pushes both medians one side.
        if medians[0.0] == medians[1.0]:
            dirs[0.0], dirs[1.0], method = 1, -1, "gk_median_tie"
        else:
            low = 0.0 if medians[0.0] < medians[1.0] else 1.0
            high = 1.0 if low == 0.0 else 0.0
            dirs[low], dirs[high], method = 1, -1, "gk_median"
    elif enough[0.0] or enough[1.0]:
        solid = 0.0 if enough[0.0] else 1.0
        other = 1.0 if solid == 0.0 else 0.0
        dirs[solid] = _dir_from_median(medians[solid], center_x)
        dirs[other] = -dirs[solid]  # opposite of the resolved team
        method = "gk_single"
    else:
        dirs, method = _fallback_defender_centroid(g, cfg, center_x)

    info = dict(
        method=method,
        gk_counts={str(int(t)): counts[t] for t in TEAMS},
        gk_median_x={
            str(int(t)): (None if np.isnan(medians[t]) else round(medians[t], 3))
            for t in TEAMS
        },
        attack_dir={str(int(t)): dirs[t] for t in TEAMS},
    )
    return dirs, info


def _fallback_defender_centroid(
    g: pd.DataFrame, cfg: PrepConfig, center_x: float
) -> Tuple[Dict[float, Optional[int]], str]:
    """Sparse-GK fallback: infer direction from team outfield centroids.

    The team whose players sit deeper (smaller median x) is taken to be defending
    the low-x goal, hence attacking +x. Weak — documented as a last resort; if a
    team has no usable players the direction is left null (not guessed).
    """
    players = g[
        (g["role"] == "player")
        & g["pitch_valid"].fillna(False).astype(bool)
        & g["team"].isin(TEAMS)
    ].dropna(subset=["pitch_x_m"])
    med = {
        t: (float(players.loc[players["team"] == t, "pitch_x_m"].median())
            if (players["team"] == t).any() else np.nan)
        for t in TEAMS
    }
    dirs: Dict[float, Optional[int]] = {t: None for t in TEAMS}
    if not np.isnan(med[0.0]) and not np.isnan(med[1.0]) and med[0.0] != med[1.0]:
        low = 0.0 if med[0.0] < med[1.0] else 1.0
        high = 1.0 if low == 0.0 else 0.0
        dirs[low], dirs[high] = 1, -1
        return dirs, "defender_centroid"
    # single team present -> resolve it vs midfield, mirror the other
    for t in TEAMS:
        if not np.isnan(med[t]):
            dirs[t] = _dir_from_median(med[t], center_x)
            dirs[1.0 - t] = -dirs[t]
            return dirs, "defender_centroid_single"
    return dirs, "unresolved"


def resolve_direction(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, dict]:
    """Add a per-row ``attack_dir`` (+1 / -1 / NaN) column, resolved per period.

    Ball and referee rows keep ``NaN`` (no team, no direction). Returns
    ``(df, meta)`` with the resolved directions and the method used per period.
    """
    df = df.copy()
    df[COL_ATTACK_DIR] = np.nan

    directions_meta: Dict[str, dict] = {}
    unresolved = []
    for key, idx in period_groups(df, cfg.period_col):
        g = df.loc[idx]
        dirs, info = _resolve_period(g, cfg)
        for team, d in dirs.items():
            if d is None:
                continue
            mask = idx.isin(
                g[(g["team"] == team) & g["role"].isin(PEOPLE_ROLES)].index
            )
            df.loc[idx[mask], COL_ATTACK_DIR] = float(d)
        directions_meta[str(key)] = info
        if any(d is None for d in dirs.values()):
            unresolved.append(str(key))

    if unresolved:
        print(
            f"[warn] attacking direction unresolved for period(s) {unresolved}; "
            f"attack_dir left null for the affected team(s)."
        )

    meta = dict(
        params=dict(min_gk_frames=cfg.min_gk_frames),
        periods=directions_meta,
        note=(
            "attack_dir=+1 => team attacks toward +x (target goal at x_max); "
            "-1 => attacks toward -x. team labels remain arbitrary 0/1."
        ),
    )
    return df, meta


def normalize_to_attack(x, attack_dir, pitch_length: float):
    """Flip x so a team's target goal is always at ``x_max`` (== pitch_length).

    Vectorized helper. ``attack_dir == +1`` keeps x; ``-1`` mirrors it to
    ``pitch_length - x``; anything else (NaN) is passed through unchanged. Works
    on scalars, numpy arrays or pandas Series. The shared pitch frame is not
    modified — this returns a *view convention* for a single team.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(attack_dir, dtype=float)
    flipped = pitch_length - x
    out = np.where(d == -1, flipped, x)
    out = np.where(np.isnan(d), x, out)  # unknown direction -> leave as-is
    return out


def attacking_frame(
    df: pd.DataFrame,
    team: float,
    cfg: PrepConfig,
    x_col: str = "pitch_x_m",
    y_col: str = "pitch_y_m",
    pitch_length: Optional[float] = None,
) -> pd.DataFrame:
    """Return ``team``'s rows with attacking-normalized x/y columns added.

    Convenience wrapper over :func:`normalize_to_attack`: picks the team's rows,
    reads their resolved ``attack_dir``, and adds ``<x_col>_att`` / ``<y_col>``
    (y is unchanged). ``pitch_length`` defaults to the source length; pass the
    target length when normalizing rescaled columns.
    """
    length = pitch_length if pitch_length is not None else cfg.source_length_m
    sub = df[df["team"] == team].copy()
    sub[f"{x_col}_att"] = normalize_to_attack(
        sub[x_col].to_numpy(), sub[COL_ATTACK_DIR].to_numpy(), length
    )
    sub[f"{y_col}_att"] = sub[y_col]
    return sub
