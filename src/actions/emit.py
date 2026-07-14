"""Translate :class:`~src.actions.transitions.Transition` records into SPADL rows.

The one place that speaks SPADL. Keeping the mapping here (and the football in
``transitions.py``) is what lets a future shot detector add a *kind* without
touching the serializer, and lets a SPADL version bump be absorbed without
touching the football.

The mapping
-----------
=========================================  ===================================
transition                                 SPADL ``type_name`` / ``result_name``
=========================================  ===================================
pass, same team                            ``pass`` / ``success``
pass from a wide + advanced origin          ``cross`` / ``success``
carry (the ball moved during a touch)      ``dribble`` / ``success``
turnover, ball was in flight               ``pass`` / ``fail``      (loser)
                                           ``interception`` / ``success`` (winner)
turnover, ball came off a settled touch    ``bad_touch`` / ``fail`` (loser)
                                           ``tackle`` / ``success`` (winner)
=========================================  ===================================

A turnover emits **both** rows, per SPADL convention: the losing team's failed
action, then the winning team's defensive one. They carry the two different
``team_id`` / ``player_id`` values, which is what makes possession chains break
in the right place downstream.

Receptions get no row of their own -- a ``pass``/``success`` already means "and
it arrived", with the arrival in ``end_x``/``end_y``.

The extension table
-------------------
``SPADLSchema`` is ``strict``: an actions frame carrying an extra column fails to
validate. So the confidence/occlusion flags -- which the brief asks for, and which
nobody should be without -- go in a **separate provenance table keyed by
``action_id``** rather than as extra columns on the actions table. The actions
parquet stays byte-for-byte loadable by socceraction; ``left join`` on
``action_id`` recovers everything else.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import ActionConfig
from .spadl import (
    add_names,
    bodypart_id,
    clip_x,
    clip_y,
    empty_actions,
    enforce_schema,
    result_id,
    type_id,
)
from .transitions import KIND_CARRY, KIND_PASS, KIND_TURNOVER, Transition

PROVENANCE_COLUMNS = [
    "action_id", "kind", "confidence", "occluded", "low_confidence",
    "duel_candidate", "aerial", "aerial_conf",
    "n_no_ball_frames", "n_gap_frames", "n_span_frames",
    "action_travel_m", "ball_travel_m", "ball_path_m", "path_coherence",
    "heading_deg", "in_flight", "direction", "length_class",
    "dist_to_goal_start_m", "dist_to_goal_end_m",
    "from_touch", "to_touch", "start_frame", "end_frame", "reasons",
]


def _confidence(tr: Transition) -> float:
    """A blunt, honest 0-1 score. Not a probability -- a triage flag.

    Full marks for a clean, fully-observed transition; a fixed penalty per thing
    that went wrong. It exists so a consumer can filter (``confidence >= 0.7``)
    without having to re-derive why an action is shaky, and so the stage meta can
    report what fraction of the stream is trustworthy.
    """
    score = 1.0
    if tr.occluded:
        score -= 0.3
    if "incoherent_ball_path" in tr.reasons:
        score -= 0.2
    if "contested_touch" in tr.reasons:
        score -= 0.2
    if tr.duel_candidate:
        score -= 0.2
    if "endpoint_from_ball" in tr.reasons:
        # A player had no position on his own endpoint frame, so a BALL coordinate
        # set the emitted geometry -- the one route by which mid-flight distortion
        # can still reach an action. Penalised, not refused: the transition plainly
        # happened, and a bounded-but-suspect location beats no location.
        score -= 0.2
    # Being AIRBORNE is not itself a defect: it is a measured property of the pass,
    # and the endpoints are anchored on the players precisely so that a flighted
    # ball no longer damages them. No penalty. (The aerial flag DOES relax the
    # coherence guard, which is why an aerial pass avoids the -0.2 above rather
    # than being docked for a curve the homography invented.)
    return float(max(0.0, round(score, 2)))


def _row(
    cfg: ActionConfig,
    tr: Transition,
    type_name: str,
    result_name: str,
    player_id: int,
    team_id: int,
    time_s: float,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> dict:
    """One SPADL row. Coordinates are clipped into the schema's bounds here.

    ``team_id`` is required, not optional: a transition whose team is unknown is
    refused upstream in :mod:`src.actions.transitions` rather than emitted with a
    sentinel. A silent ``-1`` team would corrupt possession chains and the
    left-to-right normalization xT depends on.
    """
    return dict(
        game_id=cfg.game_id,
        original_event_id=None,
        action_id=-1,  # assigned after the time sort
        period_id=int(cfg.period_id),
        time_seconds=float(max(0.0, time_s)),
        team_id=int(team_id),
        player_id=int(player_id),
        start_x=float(clip_x(start[0])),
        start_y=float(clip_y(start[1])),
        end_x=float(clip_x(end[0])),
        end_y=float(clip_y(end[1])),
        bodypart_id=bodypart_id(cfg.default_bodypart),
        type_id=type_id(type_name),
        result_id=result_id(result_name),
    )


def _rows_for(cfg: ActionConfig, tr: Transition) -> List[dict]:
    """The SPADL row(s) one transition becomes. A turnover becomes two."""
    if tr.kind == KIND_CARRY:
        # SPADL `dribble`: the ball's journey from where the player received it to
        # where they released it. Emitted at the touch's START time so it precedes
        # the action that ends the touch.
        return [
            _row(cfg, tr, "dribble", "success", tr.player_id, tr.team,
                 tr.time_s, tr.path.start, tr.path.end)
        ]

    if tr.kind == KIND_PASS:
        return [
            _row(cfg, tr, "cross" if tr.is_cross else "pass", "success",
                 tr.player_id, tr.team, tr.time_s, tr.path.start, tr.path.end)
        ]

    if tr.kind == KIND_TURNOVER:
        # Losing side. In flight => the pass was attempted and cut out. Settled
        # => the ball was taken off them, which SPADL calls a bad_touch.
        losing_type = "pass" if tr.in_flight else "bad_touch"
        # Winning side. Same discriminator, other half of the pair.
        winning_type = "interception" if tr.in_flight else "tackle"

        rows = [
            _row(cfg, tr, losing_type, "fail", tr.player_id, tr.team,
                 tr.time_s, tr.path.start, tr.path.end),
        ]
        if tr.winner_id is not None:
            # The defensive action happens where the ball was won, so it starts
            # and ends at the reception point.
            rows.append(
                _row(cfg, tr, winning_type, "success", tr.winner_id, tr.winner_team,
                     tr.end_time_s, tr.path.end, tr.path.end)
            )
        return rows

    raise ValueError(f"unknown transition kind {tr.kind!r}")


def _provenance(action_ids: List[int], tr: Transition) -> List[dict]:
    """The extension row(s) for one transition -- one per SPADL row it produced."""
    # A carry has no gap: the frames between its endpoints ARE the carry. Reporting
    # the touch's own span there instead of a meaningless "gap" keeps the column
    # honest -- n_gap_frames means "frames the ball was loose", and for a carry
    # that is zero by construction.
    is_gap = tr.kind != KIND_CARRY
    return [
        dict(
            action_id=int(aid),
            kind=tr.kind,
            confidence=_confidence(tr),
            occluded=bool(tr.occluded),
            low_confidence=bool(tr.low_confidence),
            duel_candidate=bool(tr.duel_candidate),
            # SPADL has no aerial action type, so an aerial pass stays a `pass`
            # (or a `cross`) and its flight is recorded HERE. This is also the
            # column that says why the coherence guard was relaxed on that row.
            aerial=bool(tr.aerial),
            aerial_conf=float(tr.aerial_conf),
            n_no_ball_frames=int(tr.path.n_no_ball),
            n_gap_frames=int(tr.path.n_gap_frames) if is_gap else 0,
            n_span_frames=int(tr.end_frame - tr.frame + 1),
            # The length of the emitted ACTION (player -> player) and, separately,
            # how far the BALL went. They differ, and the difference is the point:
            # the first is the geometry we stand behind, the second is the evidence
            # we judged the gap on. On an aerial pass the second is distorted.
            action_travel_m=float(tr.path.travel_m),
            ball_travel_m=float(tr.path.ball_travel_m),
            ball_path_m=float(tr.path.path_m),
            path_coherence=float(tr.path.coherence),
            heading_deg=float(tr.path.heading_deg),
            in_flight=bool(tr.in_flight),
            direction=tr.direction,
            length_class=tr.length_class,
            dist_to_goal_start_m=float(tr.dist_to_goal_start_m),
            dist_to_goal_end_m=float(tr.dist_to_goal_end_m),
            from_touch=tr.from_touch,
            to_touch=tr.to_touch,
            start_frame=int(tr.frame),
            end_frame=int(tr.end_frame),
            reasons=",".join(tr.reasons),
        )
        for aid in action_ids
    ]


def to_spadl(
    transitions: List[Transition], cfg: ActionConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Serialize transitions into ``(actions, provenance)``.

    ``actions`` is a strict SPADL table: time-ordered, ``action_id`` 0..n-1,
    nothing but the schema's columns. ``provenance`` carries the confidence and
    occlusion flags, keyed by ``action_id``.

    Ordering is by ``(period_id, time_seconds)`` with a **stable** sort, so the
    emission order breaks ties -- which is what keeps a turnover's failed action
    in front of the interception that answers it, and a carry in front of the pass
    that ends its touch, when both land on the same timestamp.
    """
    if not transitions:
        return empty_actions(), pd.DataFrame(columns=PROVENANCE_COLUMNS)

    rows: List[dict] = []
    owner: List[Transition] = []  # rows[i] came from owner[i]
    for tr in transitions:
        for row in _rows_for(cfg, tr):
            rows.append(row)
            owner.append(tr)

    actions = pd.DataFrame(rows)
    actions["_emit"] = np.arange(len(actions))
    actions = actions.sort_values(
        ["period_id", "time_seconds", "_emit"], kind="stable"
    ).reset_index(drop=True)

    order = actions["_emit"].to_numpy()
    actions = actions.drop(columns="_emit")
    actions["action_id"] = np.arange(len(actions), dtype="int64")

    prov: List[dict] = []
    for action_id, emit_idx in enumerate(order):
        prov.extend(_provenance([action_id], owner[int(emit_idx)]))

    actions = enforce_schema(add_names(actions))
    provenance = pd.DataFrame(prov)[PROVENANCE_COLUMNS]
    return actions, provenance
