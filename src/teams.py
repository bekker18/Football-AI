"""Team-label logic: goalkeeper assignment and temporal majority vote.

``majority_vote_teams`` is pure (plain dicts) and heavy imports are deferred, so
this module can be imported and tested without the CV stack installed.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np


def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers):
    """Assign each GK to the nearest team centroid. Robust to empty teams."""
    import supervision as sv

    if len(goalkeepers) == 0:
        return np.array([], dtype=int)
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(players) == 0:
        # No outfield players this frame: split GKs by image x (left=0, right=1)
        return (gk_xy[:, 0] > np.median(gk_xy[:, 0])).astype(int)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    out = []
    c0 = pl_xy[players_team_id == 0]
    c1 = pl_xy[players_team_id == 1]
    c0 = c0.mean(axis=0) if len(c0) else None
    c1 = c1.mean(axis=0) if len(c1) else None
    for xy in gk_xy:
        d0 = np.linalg.norm(xy - c0) if c0 is not None else np.inf
        d1 = np.linalg.norm(xy - c1) if c1 is not None else np.inf
        out.append(0 if d0 <= d1 else 1)
    return np.array(out, dtype=int)


def majority_vote_teams(rows: List[dict]) -> Tuple[int, int]:
    """Lock each track to the mode of its per-frame team labels, in place.

    A ByteTrack id is one physical person, so it belongs to one team for its
    whole life; voting over the track removes the frame-to-frame flicker that
    independent per-frame predictions produce. Rows shared with ``frames.jsonl``
    are the same dict objects, so mutating in place updates both outputs.

    Returns (n_tracks, n_labels_corrected).
    """
    votes: dict = {}
    for r in rows:
        oid = r["object_id"]
        if (
            r["role"] in ("player", "goalkeeper")
            and r["team"] is not None
            and isinstance(oid, int)
            and oid >= 0
        ):
            votes.setdefault(oid, Counter())[r["team"]] += 1
    track_team = {oid: c.most_common(1)[0][0] for oid, c in votes.items()}
    n_flips = 0
    for r in rows:
        oid = r["object_id"]
        if oid in track_team and r["team"] != track_team[oid]:
            r["team"] = track_team[oid]
            n_flips += 1
    return len(track_team), n_flips
