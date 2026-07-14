"""Compose the event layer: possession stream -> touches -> transitions -> SPADL.

``detect_actions`` is the one call the CLI (and any downstream consumer) needs.

The stage meta is the honest scorecard, so it reports what it *refused* to emit
as prominently as what it emitted. A gap the layer declined to name is a gap
where something happened that we could not see; burying that count would make the
action chain look more complete than it is.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from .aerial import AerialTrack, detect_airborne, summarize_runs
from .config import ActionConfig
from .emit import to_spadl
from .geometry import BallTrack, PlayerTrack
from .source import PossessionSource, no_ball_frames
from .spadl import EMITTED_ACTIONTYPES
from .transitions import coalesce_touches, transitions_from_touches


def _aerial_counts(actions: pd.DataFrame, provenance: pd.DataFrame) -> dict:
    """How many of the emitted actions the ball FLEW across, broken out by type.

    SPADL has no aerial action type, so an aerial pass is indistinguishable from a
    ground one in the actions table itself -- these counts, and the ``aerial``
    column in the provenance table, are the only place the distinction survives.
    Reported prominently for exactly that reason.
    """
    if not len(actions) or "aerial" not in provenance.columns:
        return dict(n_aerial_actions=0, aerial_actions_by_type={},
                    n_aerial_passes=0, n_aerial_crosses=0, mean_aerial_conf=0.0)

    merged = actions[["action_id", "type_name"]].merge(
        provenance[["action_id", "aerial", "aerial_conf"]], on="action_id"
    )
    aerial = merged[merged["aerial"].astype(bool)]
    by_type = aerial["type_name"].value_counts().to_dict()
    return dict(
        n_aerial_actions=int(len(aerial)),
        aerial_actions_by_type={str(k): int(v) for k, v in by_type.items()},
        n_aerial_passes=int(by_type.get("pass", 0)),
        n_aerial_crosses=int(by_type.get("cross", 0)),
        mean_aerial_conf=(
            round(float(aerial["aerial_conf"].mean()), 3) if len(aerial) else 0.0
        ),
    )


def summarize(
    actions: pd.DataFrame,
    provenance: pd.DataFrame,
    touches: list,
    segments: pd.DataFrame,
    skipped: list,
    frames: pd.DataFrame,
    aerial: AerialTrack,
    cfg: ActionConfig,
) -> dict:
    """Event counts, confidence/occlusion rates, and the guards that produced them."""
    n = int(len(actions))

    def pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 2) if den else 0.0

    by_type = (
        actions["type_name"].value_counts().to_dict() if n else {}
    )
    by_result = (
        actions.groupby(["type_name", "result_name"]).size().to_dict() if n else {}
    )
    n_occluded = int(provenance["occluded"].sum()) if n else 0
    n_low = int(provenance["low_confidence"].sum()) if n else 0
    n_duel = int(provenance["duel_candidate"].sum()) if n else 0

    return dict(
        config=cfg.as_meta(),
        # what came in
        n_frames=int(len(frames)),
        n_no_ball_frames=int(len(no_ball_frames(frames))),
        n_segments=int(len(segments)),
        n_touches=len(touches),
        n_segments_merged=int(len(segments) - len(touches)),
        # what went out
        n_actions=n,
        actions_by_type=({str(k): int(v) for k, v in by_type.items()}),
        actions_by_type_result={
            f"{t}/{r}": int(v) for (t, r), v in by_result.items()
        },
        # how much of it we believe
        n_occluded=n_occluded,
        occluded_pct=pct(n_occluded, n),
        n_low_confidence=n_low,
        low_confidence_pct=pct(n_low, n),
        mean_confidence=(
            round(float(provenance["confidence"].mean()), 3) if n else 0.0
        ),
        # EXTENSION POINT (duel resolution): turnovers emitted out of a fleeting
        # contested touch. These are duels the layer saw but did not resolve --
        # a duel resolver should collapse each pair into one won/lost duel.
        n_duel_candidates=n_duel,
        duel_candidate_pct=pct(n_duel, n),
        # aerial passes: the ball was OFF THE GROUND across the gap. SPADL cannot
        # express that, so the actions table cannot either -- this and the
        # provenance `aerial` column are where the distinction lives.
        **_aerial_counts(actions, provenance),
        aerial=summarize_runs(aerial, cfg),
        # what we refused to emit, and why
        n_gaps_skipped=len(skipped),
        skipped_by_reason=dict(Counter(reason for reason, _a, _b in skipped)),
        # the guards, restated so the meta is self-contained
        guards=dict(
            bridge_max_gap_frames=cfg.bridge_max_gap_frames,
            bridge_max_ball_dist_m=cfg.bridge_max_ball_dist_m,
            min_gap_frames=cfg.min_gap_frames,
            min_ball_travel_m=cfg.min_ball_travel_m,
            min_path_coherence=cfg.min_path_coherence,
            aerial_min_path_coherence=cfg.aerial_min_path_coherence,
            max_gap_frames=cfg.max_gap_frames,
            flight_min_travel_m=cfg.flight_min_travel_m,
            min_carry_m=cfg.min_carry_m,
            max_carry_m=cfg.max_carry_m,
            duel_max_touch_frames=cfg.duel_max_touch_frames,
        ),
        scope=dict(
            emitted_actiontypes=list(EMITTED_ACTIONTYPES),
            not_emitted=["shot", "shot_penalty", "shot_freekick", "throw_in",
                         "corner_crossed", "corner_short", "freekick_crossed",
                         "freekick_short", "goalkick", "foul", "take_on",
                         "clearance", "keeper_*"],
            note=(
                "milestone 1: only events that fall directly out of possession "
                "transitions. Shots, set pieces and duel resolution are out of "
                "scope; see the EXTENSION POINT markers in transitions.py."
            ),
        ),
        limitations=dict(
            bodypart=(
                f"no pose data => bodypart is not observable; every action is "
                f"emitted as '{cfg.default_bodypart}' (SPADL has no 'unknown' "
                f"bodypart, and socceraction's own converters default the same "
                f"way). Do NOT read the bodypart columns as measured."
            ),
            occlusion=(
                "no_ball frames are occlusion, never a stoppage. A transition "
                "spanning them is still emitted -- a transition unseen is not a "
                "transition that did not happen -- but it is flagged occluded "
                "and its confidence is reduced."
            ),
            ball_height=(
                "the homography maps the image to the GROUND plane (z=0), so an "
                "airborne ball's pitch coordinates are stretched away from the "
                "camera and there is no height channel to correct them with. Two "
                "consequences. (1) EMITTED GEOMETRY IS ANCHORED ON PLAYERS: a "
                "pass starts where the passer stood on his last controlled frame "
                "and ends where the receiver stood on his first; the ball path is "
                "used only to characterise the gap, never to set start/end. SPADL "
                "actions are start->end, not trajectories, so this keeps "
                "mid-flight distortion out of the event stream and out of "
                "xT/VAEP. (2) AIRBORNE IS FLAGGED, NOT CORRECTED: the `aerial` / "
                "`aerial_conf` columns in the provenance table mark the gaps the "
                "ball flew across (heuristic, from the img_y arc -- NOT a height "
                "measurement), and those gaps bypass the ground-path coherence "
                "guard, which is a straightness test meant for rolling balls. "
                "Ballistic height reconstruction is a documented EXTENSION POINT "
                "in src/actions/aerial.py and is deliberately NOT built."
            ),
        ),
        home_team_id=_home_team_id(cfg),
        note=(
            "coordinates are the target (105x68) frame, which IS SPADL's default "
            "pitch, so no rescale is applied. They are NOT normalized "
            "left-to-right: call socceraction's spadl.play_left_to_right(actions, "
            "home_team_id) with the home_team_id reported here (the team whose "
            "attack_dir is +1)."
        ),
    )


def _home_team_id(cfg: ActionConfig):
    """The team attacking toward +x -- i.e. the one already playing left-to-right.

    socceraction's ``play_left_to_right`` mirrors the *away* team's coordinates so
    that every team attacks toward +x. Handing it the team that already does means
    the flip lands the right way round; guessing would silently invert half the
    xT values.
    """
    for team, d in sorted(cfg.attack_dir.items()):
        if int(d) > 0:
            return int(team)
    return None


@dataclass
class ActionStages:
    """Every intermediate stage of one run: segments -> touches -> actions.

    :func:`detect_actions` returns only what a consumer needs. The review renderer
    needs to *show the derivation* -- which segments got coalesced into which
    touch, and which action came out of which gap -- so it gets the whole chain
    from :func:`run_stages` rather than recomputing it and risking drift from what
    the stage actually emitted.
    """

    frames: pd.DataFrame       # the possession stream, as a table
    segments: pd.DataFrame     # maximal same-possessor runs
    touches: list              # segments after coalescing (list[Touch])
    transitions: list          # the typed events (list[Transition])
    skipped: list              # (reason, from_touch, to_touch) refusals
    actions: pd.DataFrame      # the SPADL table
    provenance: pd.DataFrame   # confidence / occlusion, keyed by action_id
    aerial: AerialTrack        # airborne / aerial_conf, by ball frame
    meta: dict


def run_stages(
    source: PossessionSource,
    tracking: pd.DataFrame,
    cfg: ActionConfig,
) -> ActionStages:
    """Run the event layer and keep every intermediate stage. See :class:`ActionStages`."""
    frames = source.frames()
    segments = source.segments()

    ball = BallTrack.from_prepared(tracking)
    players = PlayerTrack.from_prepared(tracking)
    # Which ball frames were OFF THE GROUND. Computed before the walk, because the
    # answer relaxes a guard the walk applies -- an aerial pass's ground path is a
    # z=0 artefact and must not be judged for straightness like a rolling ball.
    aerial = detect_airborne(tracking, frames, cfg)

    touches = coalesce_touches(segments, ball, players, cfg)
    transitions, skipped = transitions_from_touches(
        touches, ball, players, cfg, aerial
    )
    actions, provenance = to_spadl(transitions, cfg)

    meta = summarize(
        actions, provenance, touches, segments, skipped, frames, aerial, cfg
    )
    return ActionStages(
        frames=frames, segments=segments, touches=touches, transitions=transitions,
        skipped=skipped, actions=actions, provenance=provenance, aerial=aerial,
        meta=meta,
    )


def detect_actions(
    source: PossessionSource,
    tracking: pd.DataFrame,
    cfg: ActionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the event layer.

    ``source`` is the possession stream (any :class:`~src.actions.source.
    PossessionSource`); ``tracking`` is the prepared tracking table, used **only**
    for geometry (ball and player positions, plus the ball's image-space columns
    for the aerial detector). Who is on the ball comes from the source and nowhere
    else.

    Returns ``(actions, provenance, meta)``: the SPADL actions table, the
    confidence/occlusion sidecar keyed by ``action_id``, and the summary. Use
    :func:`run_stages` if you also want the ball's ``airborne`` annotation.
    """
    stages = run_stages(source, tracking, cfg)
    return stages.actions, stages.provenance, stages.meta
