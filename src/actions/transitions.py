"""The transition rules: possession segments in, typed on-ball events out.

**Events are defined by the transitions between possession segments, not by the
segments themselves.** A pass is not a thing that happens to a player; it is the
statement that the ball left player A and arrived at player B. So the whole layer
is a walk over the *gaps*.

Two passes over the stream:

**1. Coalesce segments into touches** (:func:`coalesce_touches`). Consecutive
segments with the *same* possessor separated by a tiny gap are one touch, not
two: at ``r_pz_m = 3 m`` the ball routinely drifts a hair outside the zone and
back while the same player is plainly still carrying it. Merging them first is
what makes the 1-frame blip un-emittable rather than something we have to
recognise and discard later.

**2. Walk the gaps between touches** (:func:`transitions_from_touches`), applying:

===========================  ==================================================
SAME team, DIFFERENT player  ``PASS`` by A's player, ``result=success``, ending
                             at B's reception point. Subtyped ``CROSS`` when it
                             originates from a wide, advanced area.
CROSS team                   ``TURNOVER``. Both sides are emitted, per SPADL
                             convention: a failed action for the losing team
                             (``pass``/``fail`` if the ball was in flight, else
                             ``bad_touch``/``fail``) *and* the winning team's
                             defensive action (``interception`` if the ball was
                             in flight, ``tackle`` if it came off a settled
                             possession).
SAME player                  ``CARRY``, but only if the ball actually moved. See
                             below -- this one is mostly a rule about what NOT to
                             emit.
===========================  ==================================================

Carries are emitted **per touch**, not per gap: the carry is the ball's journey
from where a player received it to where they released it. That is also what
SPADL means by ``dribble``, and it is what keeps the action chain spatially
continuous (a pass ends where the next player's carry starts, and that carry ends
where their pass starts) -- xT reads exactly those deltas. A touch where the ball
sits still is **the ball parked near a player, not a dribble**, and emits nothing.

Receptions are not actions. A successful pass already implies its reception; SPADL
models it as one row with ``result=success`` and the reception as ``end_x/end_y``.

Where the emitted geometry comes from
-------------------------------------
**From the players, never from the mid-flight ball.** The homography maps the
image to the ground plane (z=0), so an airborne ball's pitch coordinates are
stretched away from the camera and wrong by metres. A pass therefore starts where
the PASSER stood on his last controlled frame and ends where the RECEIVER stood on
his first; turnovers and carries anchor the same way. The ball path is used only
to *characterise* the gap -- did the ball travel, how straight, was it in the air.
See :mod:`src.actions.geometry`.

Aerial passes
-------------
:mod:`src.actions.aerial` flags the frames on which the ball was off the ground.
A gap the ball flew across is **subtyped, not retyped**: SPADL has no aerial
action, so the flight is recorded in the provenance table (``aerial`` /
``aerial_conf``) and the action stays a ``pass`` (or a ``cross``, if it also clears
the cross geometry). The flight is also what **relaxes the ground-path coherence
guard** for that gap -- see :func:`_describe`.

Out of scope, deliberately
--------------------------
Shots, set pieces, duel resolution, and ballistic height reconstruction. The
extension points are marked ``EXTENSION POINT`` in the code (here and in
:mod:`src.actions.aerial`) and listed in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from .config import ActionConfig
from .geometry import (
    BallTrack,
    GapPath,
    PlayerTrack,
    carry_path,
    dist_to_goal,
    max_ball_distance,
)

if TYPE_CHECKING:  # pragma: no cover
    from .aerial import AerialTrack

# --- the transition kinds this milestone knows how to name ---------------- #
KIND_PASS = "pass"
KIND_CARRY = "carry"
KIND_TURNOVER = "turnover"

# ...and the reasons a gap can be refused.
SKIP_SPURIOUS = "spurious"        # zone flicker: the ball never went anywhere
SKIP_TOO_LONG = "gap_too_long"    # unobserved for too long to claim one event
SKIP_NO_GEOMETRY = "no_geometry"  # neither ball nor player position available
SKIP_SAME_PLAYER = "same_player_long_gap"  # ball left and returned; unclassifiable
SKIP_STATIC_HOLD = "static_hold"  # a touch where the ball never moved: no carry
SKIP_UNKNOWN_TEAM = "unknown_team"  # cannot tell a pass from a turnover


@dataclass
class Touch:
    """A coalesced possession segment: one player, in control, over a frame range."""

    touch_id: int
    possessor_id: int
    team: Optional[int]
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    n_frames: int
    n_contested: int
    n_segments: int = 1  # how many raw segments were merged into this touch


@dataclass
class Transition:
    """One typed event: either a gap between two touches, or a touch's own carry.

    Deliberately *not* a SPADL row. This is the layer's own vocabulary --
    ``kind``, plus the geometry and the confidence bookkeeping -- and
    :mod:`src.actions.emit` is the only thing that translates it into SPADL. That
    split is what lets a new event type (a shot) be recognised here and mapped
    there without the two concerns tangling.
    """

    kind: str
    path: GapPath
    # who
    player_id: int
    team: Optional[int]
    # the other side of a turnover (the winner); None otherwise
    winner_id: Optional[int] = None
    winner_team: Optional[int] = None
    # when
    frame: int = 0
    time_s: float = 0.0
    end_frame: int = 0
    end_time_s: float = 0.0
    # what
    in_flight: bool = False
    is_cross: bool = False
    direction: str = "lateral"   # progressive | lateral | back
    length_class: str = "short"  # short | long
    #: The ball was OFF THE GROUND across this gap (:mod:`src.actions.aerial`).
    #: SPADL has no aerial action type, so this is a *subtype*, carried in the
    #: provenance table rather than in the action's `type_id`. It is also what
    #: relaxes the ground-path coherence guard -- see :func:`_describe`.
    aerial: bool = False
    aerial_conf: float = 0.0
    # how much we believe it
    occluded: bool = False
    low_confidence: bool = False
    #: EXTENSION POINT (duel resolution). The ball changed hands out of a touch
    #: that was contested and lasted barely any frames -- i.e. two players were
    #: inside the possession zone together and the "possessor" flickered between
    #: them. Milestone 1 emits the turnover it literally sees (and flags it here);
    #: a duel resolver is what should collapse these into a single won/lost duel.
    duel_candidate: bool = False
    reasons: List[str] = field(default_factory=list)
    # provenance
    from_touch: Optional[int] = None
    to_touch: Optional[int] = None
    dist_to_goal_start_m: float = float("nan")
    dist_to_goal_end_m: float = float("nan")


def coalesce_touches(segments: pd.DataFrame, ball: BallTrack, players: PlayerTrack,
                     cfg: ActionConfig) -> List[Touch]:
    """Merge same-possessor segments separated by a blip into single touches.

    A merge requires BOTH that the gap is short (``bridge_max_gap_frames``) AND
    that the ball stayed with the carrier across it (``bridge_max_ball_dist_m``).

    This is where "same player across a short gap -> carry" is implemented, and it
    is implemented by *removing the gap* rather than by classifying it: the two
    segments become one touch, and the touch's own carry (in
    :func:`transitions_from_touches`) then spans the whole thing -- knock, chase
    and all. Doing it this way means the 1-frame zone blip and the 8-frame
    knock-and-chase are the same phenomenon handled by the same rule, and neither
    can be minted as a spurious pass.

    The distance condition is what stops it going too far: a ball that genuinely
    left and came back to the same player is not one touch, and merging it would
    swallow a round trip.
    """
    touches: List[Touch] = []
    for seg in segments.itertuples(index=False):
        team = None if pd.isna(seg.team) else int(seg.team)
        prev = touches[-1] if touches else None

        if prev is not None and prev.possessor_id == int(seg.possessor_id):
            gap = int(seg.start_frame) - prev.end_frame - 1
            stray = max_ball_distance(
                ball, players, prev.end_frame, int(seg.start_frame),
                prev.possessor_id,
            )
            if 0 <= gap <= cfg.bridge_max_gap_frames and stray <= cfg.bridge_max_ball_dist_m:
                prev.end_frame = int(seg.end_frame)
                prev.end_time_s = float(seg.end_time_s)
                prev.n_frames += int(seg.n_frames)
                prev.n_contested += int(seg.n_contested)
                prev.n_segments += 1
                continue

        touches.append(
            Touch(
                touch_id=len(touches),
                possessor_id=int(seg.possessor_id),
                team=team,
                start_frame=int(seg.start_frame),
                end_frame=int(seg.end_frame),
                start_time_s=float(seg.start_time_s),
                end_time_s=float(seg.end_time_s),
                n_frames=int(seg.n_frames),
                n_contested=int(seg.n_contested),
            )
        )
    return touches


def _classify_direction(path: GapPath, team: Optional[int],
                        cfg: ActionConfig) -> str:
    """progressive / lateral / back, measured toward the passer's target goal."""
    if not np.isfinite(path.travel_m):
        return "lateral"
    gained = cfg.attacking_x(path.end[0], team) - cfg.attacking_x(path.start[0], team)
    if gained >= cfg.progressive_min_m:
        return "progressive"
    if gained <= -cfg.progressive_min_m:
        return "back"
    return "lateral"


def _is_cross(path: GapPath, team: Optional[int], cfg: ActionConfig) -> bool:
    """True when the delivery ORIGINATES from a wide, advanced area.

    Origin-only, by design: without pose data we cannot see that a ball was
    clipped in from the flank rather than driven along it, so the honest signal is
    *where it was struck from*. Both thresholds are configurable.

    EXTENSION POINT (aerial crosses). ``path.airborne`` is now available here, and
    *aerial + wide + advanced, landing in the box* is a much stronger cross signal
    than origin alone -- it is close to the definition of one. Deliberately NOT
    wired in yet: it would change which actions come out as ``cross`` rather than
    ``pass``, and that is a recalibration to make against footage with known
    crosses in it, not a change to slip in alongside the aerial flag itself. The
    flag is in the provenance table today, so the two can be correlated before the
    rule is moved. The sample clip contains no crosses to calibrate on.
    """
    if not np.isfinite(path.start[0]) or not np.isfinite(path.start[1]):
        return False
    x_att = cfg.attacking_x(path.start[0], team)
    wide_m = cfg.cross_wide_y_frac * cfg.pitch_width_m
    is_wide = (path.start[1] <= wide_m) or (path.start[1] >= cfg.pitch_width_m - wide_m)
    is_advanced = x_att >= cfg.cross_min_x_frac * cfg.pitch_length_m
    return bool(is_wide and is_advanced)


def _describe(tr: Transition, cfg: ActionConfig) -> Transition:
    """Fill the subclassification + goal geometry shared by passes and turnovers."""
    tr.direction = _classify_direction(tr.path, tr.team, cfg)
    tr.length_class = (
        "long" if np.isfinite(tr.path.travel_m)
        and tr.path.travel_m >= cfg.long_pass_m else "short"
    )
    goal = cfg.goal_xy(tr.team)
    tr.dist_to_goal_start_m = dist_to_goal(tr.path.start, goal)
    tr.dist_to_goal_end_m = dist_to_goal(tr.path.end, goal)

    if tr.path.airborne:
        # Subtyped, not retyped. SPADL has no aerial action, and inventing one
        # would break the `socceraction` contract this layer exists to honour --
        # so the ball's flight is recorded in the provenance table and the action
        # stays a `pass` (or a `cross`, if it also clears the cross geometry).
        tr.aerial = True
        tr.aerial_conf = tr.path.aerial_conf
        tr.reasons.append("aerial")

    if tr.path.occluded:
        tr.occluded = True
        tr.low_confidence = True
        tr.reasons.append("occluded_no_ball")
    if tr.path.player_fallback:
        # A ball coordinate reached the emitted geometry because a player had no
        # position of his own on his endpoint frame. The one remaining route by
        # which mid-flight distortion can touch an action -- so it is flagged.
        tr.low_confidence = True
        tr.reasons.append("endpoint_from_ball")

    # The coherence guard is a STRAIGHTNESS test, and straightness is a property of
    # the ball's GROUND path. An airborne ball has no trustworthy ground path: the
    # z=0 homography stretches its back-projection away from the camera by an
    # amount that grows and shrinks with its height, so a cleanly struck 40 m
    # diagonal comes out bent and scores like an aimless deflection. Applying a
    # rolling-ball test to it would penalise the aerial pass for exactly the
    # distortion this layer has already identified -- so the floor is relaxed
    # (bypassed, by default) for gaps the ball flew across.
    if tr.path.coherence < cfg.coherence_floor(tr.path.airborne):
        tr.low_confidence = True
        tr.reasons.append("incoherent_ball_path")
    return tr


def _credible(path: GapPath, cfg: ActionConfig) -> bool:
    """Is a possessor CHANGE across this gap believable, or is it zone flicker?

    Credible if any of:

    - the ball actually went somewhere (``>= min_ball_travel_m``), or
    - it was loose long enough for a real transfer (``>= min_gap_frames``), or
    - there was no loose phase at all (a 0-frame gap): the ball went hand to hand,
      which is a genuine contact event (a tackle), not noise.

    Everything else is the zone boundary flickering between two players standing
    close together, and inventing a pass out of it is exactly the failure mode the
    guards exist to prevent.

    Note this asks about the BALL's travel, not the action's: the question is
    literally "did the ball go anywhere", and two team-mates standing a metre apart
    while the zone flickers between them would otherwise answer it with the metre
    between their feet.
    """
    if path.n_gap_frames == 0:
        return True
    if path.n_gap_frames >= cfg.min_gap_frames:
        return True
    travel = path.validation_travel_m
    return bool(np.isfinite(travel) and travel >= cfg.min_ball_travel_m)


def transitions_from_touches(
    touches: List[Touch],
    ball: BallTrack,
    players: PlayerTrack,
    cfg: ActionConfig,
    aerial: Optional["AerialTrack"] = None,
) -> tuple:
    """Walk the touches, emitting a :class:`Transition` per carry and per gap.

    Returns ``(transitions, skipped)`` where ``skipped`` is a list of
    ``(reason, from_touch, to_touch)`` -- refusals are counted and reported in the
    stage meta, never swallowed.

    Emission order is the order the events happen in: for each touch, its carry
    (if any), then the action that ends it. That order is preserved as the
    tiebreak when two actions land on the same timestamp.

    ``aerial`` is the airborne-ball annotation. ``None`` means "no aerial
    information", which reads as "nothing was airborne" -- every guard then behaves
    exactly as it did before the detector existed.
    """
    transitions: List[Transition] = []
    skipped: List[tuple] = []

    for i, a in enumerate(touches):
        # A possessor whose team we never resolved poisons everything downstream:
        # `team_id` drives possession chains and the left-to-right normalization
        # xT needs, and without it a gap cannot be told apart from a turnover. Skip
        # the touch outright rather than guess -- and count it.
        if a.team is None:
            skipped.append((SKIP_UNKNOWN_TEAM, a.touch_id, None))
            continue

        # --- 1. the carry WITHIN this touch ------------------------------- #
        cpath = carry_path(
            ball, players, a.start_frame, a.end_frame, a.possessor_id, aerial
        )
        # The carry is EMITTED on the player's own positions (so the chain joins up
        # exactly), but it is GATED on the ball's movement -- because the rule is
        # "the ball went somewhere with him", and a player jogging past a
        # stationary ball must not mint a dribble.
        carry_travel = cpath.validation_travel_m
        if not np.isfinite(cpath.travel_m) or not np.isfinite(carry_travel):
            skipped.append((SKIP_NO_GEOMETRY, a.touch_id, None))
        elif carry_travel < cfg.min_carry_m:
            # The ball did not move: the possessor stood over it, or it sat at
            # their feet. A long touch is not evidence of a dribble -- movement is.
            skipped.append((SKIP_STATIC_HOLD, a.touch_id, None))
        elif carry_travel > cfg.max_carry_m:
            skipped.append((SKIP_TOO_LONG, a.touch_id, None))
        else:
            carry = Transition(
                kind=KIND_CARRY,
                path=cpath,
                player_id=a.possessor_id,
                team=a.team,
                frame=a.start_frame,
                time_s=a.start_time_s,
                end_frame=a.end_frame,
                end_time_s=a.end_time_s,
                from_touch=a.touch_id,
                to_touch=a.touch_id,
            )
            transitions.append(_describe(carry, cfg))

        # --- 2. the gap that ENDS this touch ------------------------------ #
        if i + 1 >= len(touches):
            break  # last touch: the clip ends, nothing terminates it
        b = touches[i + 1]

        if b.team is None:
            # Same reasoning as above: without B's team, "same team => pass,
            # cross-team => turnover" is a coin flip, and the two have opposite
            # meanings for a possession chain.
            skipped.append((SKIP_UNKNOWN_TEAM, a.touch_id, b.touch_id))
            continue

        path = GapPath.from_tracks(
            ball, players, a.end_frame, b.start_frame,
            a.possessor_id, b.possessor_id, aerial,
        )
        if not np.isfinite(path.travel_m):
            skipped.append((SKIP_NO_GEOMETRY, a.touch_id, b.touch_id))
            continue
        if path.n_gap_frames > cfg.max_gap_frames:
            # Too much of the transfer went unobserved to name a single event for
            # it. EXTENSION POINT: a ball-free possession source could bridge this
            # gap and let it be classified after all.
            skipped.append((SKIP_TOO_LONG, a.touch_id, b.touch_id))
            continue
        if a.possessor_id == b.possessor_id:
            # Same player, but too long a gap to coalesce (step 1 already merged
            # the blips). The ball left and came back: we cannot say what happened
            # in between, and each touch's own carry is already emitted.
            skipped.append((SKIP_SAME_PLAYER, a.touch_id, b.touch_id))
            continue
        if not _credible(path, cfg):
            skipped.append((SKIP_SPURIOUS, a.touch_id, b.touch_id))
            continue

        # The ball was IN FLIGHT across the gap iff it actually travelled. This is
        # the interception-vs-tackle discriminator, and the failed-pass-vs-bad-touch
        # one: a tackle takes the ball off a player who has it (the ball barely
        # moves); an interception cuts out a ball that was on its way somewhere.
        # It asks about the BALL, so it measures the ball -- two players a couple of
        # metres apart must not make a tackle look like an interception.
        in_flight = bool(
            path.n_gap_frames > 0
            and path.validation_travel_m >= cfg.flight_min_travel_m
        )

        same_team = int(a.team) == int(b.team)
        tr = Transition(
            kind=KIND_PASS if same_team else KIND_TURNOVER,
            path=path,
            player_id=a.possessor_id,
            team=a.team,
            winner_id=None if same_team else b.possessor_id,
            winner_team=None if same_team else b.team,
            frame=a.end_frame,
            time_s=a.end_time_s,
            end_frame=b.start_frame,
            end_time_s=b.start_time_s,
            in_flight=in_flight,
            from_touch=a.touch_id,
            to_touch=b.touch_id,
        )
        tr = _describe(tr, cfg)
        if same_team:
            tr.is_cross = _is_cross(path, a.team, cfg)
        if a.n_contested or b.n_contested:
            # EXTENSION POINT (duel resolution): the touch on one side of this
            # transition was contested, so which player "had" the ball is exactly
            # the question a duel resolver would re-open.
            tr.low_confidence = True
            tr.reasons.append("contested_touch")
            # ...and if that contested touch was also over in a blink, the ball
            # did not really change hands -- two players are wrestling for it and
            # the zone is flickering between them. Counted, so the noise a duel
            # resolver would remove is measurable rather than invisible.
            fleeting = min(a.n_frames, b.n_frames) <= cfg.duel_max_touch_frames
            if fleeting and not in_flight:
                tr.duel_candidate = True
                tr.reasons.append("duel_candidate")

        transitions.append(tr)

    # EXTENSION POINT (shots): a transition whose ball path ends at/near the goal
    # mouth, or a touch terminated by the ball leaving play near the goal line,
    # is a shot. The geometry it needs (`cfg.goal_xy`, `dist_to_goal_*`) is
    # already computed on every Transition above -- what is missing is a
    # ball-leaves-play signal, not the shape of the code.
    #
    # EXTENSION POINT (set pieces): the prerequisites already emit an `in_play`
    # flag. A gap that spans an out-of-play run restarts the game, and the touch
    # that follows is a throw_in / corner / freekick / goalkick rather than the
    # receiving half of a pass. Milestone 1 treats every gap as open play.
    return transitions, skipped
