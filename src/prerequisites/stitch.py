"""Transform 1 — track id stabilization (motion-based stitching).

ByteTrack fragments a single physical player into several ids whenever it loses
and re-acquires the track. With no appearance features in the Layer 1 output, we
stitch fragments using motion alone: the *end* of track A is linked to the
*start* of track B when

    * the gap between them is small (<= ``stitch_max_gap_frames``),
    * their roles match and their (already majority-voted) teams match, and
    * A's velocity-extrapolated position at B's start is within
      ``stitch_max_dist_m`` metres of B's first position.

Links are chained with a union-find, and every row gains a ``stable_id`` column
(identity for the ball, referees and unlinked tracks). Cross-clip / multi-half
*global* re-identification is intentionally left as a documented interface — see
:func:`global_reid_todo`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import period_groups
from .config import COL_STABLE_ID, PEOPLE_ROLES, PrepConfig


class _UnionFind:
    """Minimal union-find keyed by arbitrary hashables (here: track ids)."""

    def __init__(self) -> None:
        self.parent: Dict[object, object] = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the smaller id as the representative (stable, deterministic)
            lo, hi = (ra, rb) if _rank(ra) <= _rank(rb) else (rb, ra)
            self.parent[hi] = lo


def _rank(x) -> float:
    """Sort key so the numerically smallest track id wins as representative."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("inf")


def _track_summary(g: pd.DataFrame, cfg: PrepConfig) -> Optional[dict]:
    """Summarise one track (rows for a single object_id) for stitching.

    Returns None if the track has no valid pitch positions (cannot be linked).
    """
    valid = g[g["pitch_valid"].fillna(False).astype(bool)] if "pitch_valid" in g else g
    valid = valid.dropna(subset=["pitch_x_m", "pitch_y_m"]).sort_values("frame")
    if valid.empty:
        return None

    roles = g["role"].dropna()
    teams = g["team"].dropna()
    frames = valid["frame"].to_numpy()
    xs = valid["pitch_x_m"].to_numpy(dtype=float)
    ys = valid["pitch_y_m"].to_numpy(dtype=float)

    # End velocity from a linear fit over the last ``vel_window`` valid frames.
    w = min(cfg.stitch_vel_window, len(frames))
    if w >= 2:
        fw = frames[-w:].astype(float)
        vx = float(np.polyfit(fw, xs[-w:], 1)[0]) * cfg.fps  # m per frame -> m/s
        vy = float(np.polyfit(fw, ys[-w:], 1)[0]) * cfg.fps
    else:
        vx = vy = 0.0

    return dict(
        role=roles.mode().iloc[0] if not roles.empty else None,
        team=(float(teams.mode().iloc[0]) if not teams.empty else np.nan),
        start_frame=int(frames[0]),
        end_frame=int(frames[-1]),
        start_pos=np.array([xs[0], ys[0]]),
        end_pos=np.array([xs[-1], ys[-1]]),
        end_vel=np.array([vx, vy]),
    )


def _teams_match(a, b) -> bool:
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na and b_na:
        return True
    if a_na or b_na:
        return False
    return float(a) == float(b)


def stitch_ids(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, dict]:
    """Add a ``stable_id`` column by motion-stitching fragmented person tracks.

    Non-destructive: ``object_id`` is untouched. Ball / referee / unlinked
    tracks get ``stable_id == object_id``. Returns ``(df, meta)`` where ``meta``
    holds the id->stable_id mapping and a summary.
    """
    df = df.copy()
    # default: identity mapping (ball, referees, and anything not linked)
    df[COL_STABLE_ID] = df["object_id"].astype("Int64")

    uf = _UnionFind()
    n_links = 0
    people_ids: set = set()

    for _, idx in period_groups(df, cfg.period_col):
        sub = df.loc[idx]
        people = sub[sub["role"].isin(PEOPLE_ROLES)]
        # only real tracker ids (>= 1); -1 = untracked, 0 = ball sentinel
        summaries: Dict[int, dict] = {}
        for oid, g in people.groupby("object_id"):
            if pd.isna(oid) or int(oid) < 1:
                continue
            people_ids.add(int(oid))
            s = _track_summary(g, cfg)
            if s is not None:
                summaries[int(oid)] = s

        # candidate links: A ends before B starts, within the gap window
        candidates: List[Tuple[float, int, int, int]] = []  # (dist, gap, a, b)
        ids = list(summaries)
        for a in ids:
            sa = summaries[a]
            for b in ids:
                if a == b:
                    continue
                sb = summaries[b]
                gap = sb["start_frame"] - sa["end_frame"]
                if gap < 1 or gap > cfg.stitch_max_gap_frames:
                    continue
                if sa["role"] != sb["role"]:
                    continue
                if not _teams_match(sa["team"], sb["team"]):
                    continue
                pred = sa["end_pos"] + sa["end_vel"] * (gap / cfg.fps)
                dist = float(np.linalg.norm(pred - sb["start_pos"]))
                if dist <= cfg.stitch_max_dist_m:
                    candidates.append((dist, gap, a, b))

        # greedy chaining: best (closest, then shortest gap) first; each track
        # tail links to one successor, each head accepts one predecessor.
        candidates.sort(key=lambda c: (c[0], c[1]))
        tail_used: set = set()
        head_used: set = set()
        for dist, gap, a, b in candidates:
            if a in tail_used or b in head_used:
                continue
            uf.union(a, b)
            tail_used.add(a)
            head_used.add(b)
            n_links += 1

    # resolve each person id to its component representative (smallest id)
    mapping: Dict[int, int] = {}
    for oid in people_ids:
        mapping[oid] = int(uf.find(oid))
    if mapping:
        person_mask = df["role"].isin(PEOPLE_ROLES) & df["object_id"].isin(mapping)
        df.loc[person_mask, COL_STABLE_ID] = (
            df.loc[person_mask, "object_id"].map(mapping).astype("Int64")
        )

    n_stable = len({mapping.get(i, i) for i in people_ids})
    meta = dict(
        params=dict(
            max_gap_frames=cfg.stitch_max_gap_frames,
            max_dist_m=cfg.stitch_max_dist_m,
            vel_window=cfg.stitch_vel_window,
        ),
        n_input_person_tracks=len(people_ids),
        n_stable_ids=n_stable,
        n_links=n_links,
        # only non-identity entries are interesting; keep the map compact
        id_to_stable={str(k): v for k, v in sorted(mapping.items()) if k != v},
        global_reid=global_reid_todo(),
    )
    return df, meta


def global_reid_todo() -> str:
    """Documented interface for cross-clip / multi-half global re-identification.

    NOT implemented here. Motion stitching only links fragments *within* a single
    clip/period. Joining ids across clips or halves needs a stable, appearance- or
    roster-based key (jersey number, embedding, or an external identity service)
    and a mapping ``stable_id -> global_player_id`` applied after this step.
    """
    return (
        "TODO: cross-clip/multi-half global re-ID is out of scope for motion "
        "stitching; wire a stable_id -> global_player_id map here when appearance "
        "or roster keys are available."
    )
