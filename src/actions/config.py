"""Configuration for the event (SPADL action) layer.

Every tunable lives here with a documented default and a reason. Pitch/fps
context and the per-team ``attack_dir`` come from the prerequisites'
``prep_meta.json`` via :func:`config_from_prep_meta`.

Coordinate-frame contract
-------------------------
Identical to the possession layer's, and for a stronger reason: this layer reads
the **TARGET** frame (105x68) only --

- players / goalkeepers -> ``pitch_x_t_m`` / ``pitch_y_t_m``
- ball (smoothed AND rescaled) -> ``ball_x_ts_m`` / ``ball_y_ts_m``

105x68 *is* SPADL's default pitch (``socceraction.spadl.config.field_length /
field_width``), so target-frame metres are already SPADL metres and no further
rescale happens anywhere in this layer. Mixing in the source-frame columns
(``pitch_x_m`` / ``ball_x_s_m``, 120x70) would silently emit actions in the
wrong pitch and every xT value downstream would be wrong.

The guard thresholds
--------------------
``loose`` is ~37% of the ball-frames on the test clip. Most of it is real
in-flight passing, but some is the ball merely drifting past the zone radius, so
a naive "every possessor change is an event" walk invents passes. Three guards
stop that, and all three are configurable because they are footage-dependent:

``bridge_max_gap_frames`` / ``bridge_max_ball_dist_m``
    Same-possessor segments separated by at most this many frames, with the ball
    never straying further than this from the carrier, are *coalesced into one
    touch*. This absorbs the 1-frame blip where the ball nudges outside ``r_pz_m``
    and back -- it can no longer split into a spurious pass -- and, with the same
    rule, the several-frame gap where a player knocks the ball ahead and runs onto
    it, which is one carry rather than two touches with a mystery between them.
``min_gap_frames`` / ``min_ball_travel_m``
    A possessor *change* is only credible if the ball actually went somewhere
    (>= ``min_ball_travel_m``), or the ball was loose long enough for a real
    transfer (>= ``min_gap_frames``), or possession changed hand-to-hand with no
    loose phase at all (a 0-frame gap -- a contact/tackle, always credible).
    Anything else is zone flicker between two adjacent players and is skipped.
``min_path_coherence``
    ``straight-line distance / polyline distance`` of the ball through the gap.
    1.0 is a laser-straight pass; a low value is the ball wobbling, ricocheting
    or being scrambled. Below this the transition is still emitted (the ball did
    change hands) but is flagged low-confidence, and it is what separates "a
    pass" from "an aimless deflection" in the subclassification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from .spadl import BODYPARTS, FIELD_LENGTH_M, FIELD_WIDTH_M

# --- prepared-table columns this layer READS (from src.prerequisites) ----- #
BALL_OBJECT_ID = 0
PEOPLE_ROLES = ("player", "goalkeeper")
COL_STABLE_ID = "stable_id"
COL_PITCH_X_T = "pitch_x_t_m"
COL_PITCH_Y_T = "pitch_y_t_m"
COL_BALL_XTS = "ball_x_ts_m"
COL_BALL_YTS = "ball_y_ts_m"

# --- IMAGE-space columns, read ONLY by the aerial detector ---------------- #
# The one deliberate exception to the target-frame contract above. The
# homography is what destroys an airborne ball's position, so the evidence that
# it WAS airborne only survives upstream of it -- in image space. See
# :mod:`src.actions.aerial`. Nothing else in this layer may read these.
COL_IMG_Y = "img_y"
COL_BBOX_Y1 = "bbox_y1"
COL_BBOX_Y2 = "bbox_y2"
COL_BALL_SPEED = "ball_speed_ms"
COL_BALL_OUTLIER = "ball_outlier"

# --- columns the aerial detector ADDS to the ball track ------------------- #
COL_AIRBORNE = "airborne"
COL_AERIAL_CONF = "aerial_conf"

# --- possession states this layer READS (from src.possession) ------------- #
STATE_NO_BALL = "no_ball"
STATE_LOOSE = "loose"
STATE_POSSESSION = "possession"
STATE_CONTESTED = "contested"


@dataclass
class ActionConfig:
    """Parameters for turning possession transitions into SPADL actions."""

    # --- context (from prep_meta.json) --- #
    fps: float = 25.0
    pitch_length_m: float = FIELD_LENGTH_M
    pitch_width_m: float = FIELD_WIDTH_M

    #: team_id -> +1 (attacks toward +x) or -1 (attacks toward -x). Drives every
    #: "toward goal" judgement; without it, progressive/back is a coin flip.
    attack_dir: Dict[int, int] = field(default_factory=dict)

    #: SPADL identifiers. ``game_id`` defaults to the source clip's stem.
    game_id: str = "clip"
    #: The clip is one period unless the prerequisites say otherwise (SPADL
    #: requires 1..5; the prepared table has no period column today).
    period_id: int = 1

    # --- gap guards (see the module docstring) --- #
    #: Same possessor either side of a gap this short => one touch, not two.
    #: 12 frames @25fps = 0.5 s. This has to cover more than the 1-frame blip: at
    #: ``r_pz_m = 3 m`` a player driving with the ball knocks it past the zone and
    #: runs onto it, which shows up as a same-possessor gap of several frames. That
    #: is not two touches with a mystery in between -- it is one carry, and it is
    #: the archetypal carry.
    bridge_max_gap_frames: int = 12
    #: ...and only if the ball stayed WITH the carrier: the furthest the ball got
    #: from them during the gap. A knocked-ahead ball stays close; a ball that
    #: genuinely left and came back does not, and merging that would swallow a
    #: round trip into a single "touch".
    bridge_max_ball_dist_m: float = 10.0

    #: A possessor change needs one of: this many loose frames, ...
    min_gap_frames: int = 2
    #: ...or this much ball travel, ...
    min_ball_travel_m: float = 1.5
    #: ...or a 0-frame gap (hand-to-hand contact), which is always credible.

    #: Below this straight/polyline ratio the ball's path through the gap is
    #: incoherent (a deflection, not a delivery) -> flagged low-confidence.
    #: **Only applied to balls that stayed on the ground** -- see
    #: ``aerial_min_path_coherence``.
    min_path_coherence: float = 0.5

    #: The coherence guard for a gap the ball flew across. It defaults to 0.0 --
    #: i.e. **bypassed** -- and that is the point. Coherence is a *straightness*
    #: test, and it is a test of the ball's GROUND path. An airborne ball has no
    #: trustworthy ground path: the z=0 homography stretches its back-projection
    #: away from the camera by an amount that grows and shrinks with its height,
    #: so a perfectly struck 40 m diagonal comes out as a bent, wandering polyline
    #: and scores terribly. Judging an aerial pass by a straightness test written
    #: for rolling balls penalises it for the distortion this layer already knows
    #: about. Raise it above 0 only if you want aerial gaps re-tested on a curve
    #: you have decided to trust.
    aerial_min_path_coherence: float = 0.0

    #: Beyond this the gap is too long to claim a single event across; the ball
    #: was unobserved for too much of it. 100 frames @25fps = 4 s.
    max_gap_frames: int = 100

    # --- classification --- #
    #: Ball displacement across the gap at/above which the ball was IN FLIGHT.
    #: This is the interception-vs-tackle discriminator: a tackle takes the ball
    #: off a player who has it, so the ball barely moves; an interception cuts
    #: out a ball that was travelling.
    flight_min_travel_m: float = 3.0

    #: EXTENSION POINT (duel resolution). A turnover out of a *contested* touch
    #: this short is flagged ``duel_candidate``: the possessor flickered between
    #: two players in the zone rather than the ball genuinely changing hands.
    #: Milestone 1 still emits the turnover -- it does not resolve duels -- but it
    #: counts them, so the noise a duel resolver would remove is measurable.
    duel_max_touch_frames: int = 2

    #: A touch only earns a ``dribble`` if the ball moved at least this far
    #: between the touch's first and last frame. A long touch where the ball sits
    #: still is the ball parked near a player, NOT a carry.
    min_carry_m: float = 2.0
    #: A carry longer than this is not a carry -- it is a gap in the tracking
    #: dressed up as one. 60 m of "dribble" is a lost ball, not Maradona.
    max_carry_m: float = 60.0

    #: CROSS: the pass ORIGINATES from a wide, advanced area. Wide = within this
    #: fraction of the pitch width of either touchline (0.2 * 68 = 13.6 m).
    cross_wide_y_frac: float = 0.2
    #: Advanced = at/beyond this fraction of the pitch length, measured toward
    #: the passer's target goal (0.66 = the final third).
    cross_min_x_frac: float = 0.66

    #: Progressive / back: metres gained toward the target goal.
    progressive_min_m: float = 5.0
    #: Short / long: straight-line pass length.
    long_pass_m: float = 25.0

    # --- aerial (airborne-ball) detection: see src/actions/aerial.py --- #
    # A HEURISTIC, single-camera detector, NOT height recovery. It answers "was
    # the ball probably off the ground here?" from the shape of img_y, because
    # image space is the only place the ground-plane homography has not already
    # destroyed the evidence. Defaults are calibrated on the aerial pass at frames
    # 123-150 of the sample clip 2e57b9_0 (a ~1.1 s ball: img_y falls 352 -> ~325
    # and back to ~350, apparent ground speed sustained ~25-30 m/s, ball loose
    # throughout, one detection glitch at frame 136).
    #: Master switch. Off => no frame is ever flagged, and every guard behaves
    #: exactly as it did before this feature existed.
    aerial_enabled: bool = True
    #: Usable img_y samples a loose run needs before a parabola means anything.
    #: Three points fit a parabola exactly, so a low bar here is how you get a
    #: perfect R^2 out of noise.
    aerial_min_run_frames: int = 8
    #: ...and above this the run is DROPPED, not truncated. 125 frames = 5 s @25fps.
    #: Over several seconds a camera pan traces an img_y curve of its own, and a
    #: window that long is not evidence about a single ball flight any more.
    aerial_max_run_frames: int = 125
    #: Upward-opening curvature of the img_y parabola, px/frame^2. Image-y
    #: DECREASES as the ball rises, so a ball that went up and came down is a
    #: local MINIMUM in img_y -- a POSITIVE second derivative. The sample clip's
    #: aerial pass sits at ~0.15; this floor is well below it but above the drift
    #: of a ball rolling across a slowly panning frame.
    aerial_min_curvature: float = 0.02
    #: Fit quality. A real flight is close to a clean parabola in img_y (the
    #: sample pass fits at R^2 > 0.99); a ball being scrambled around is not.
    aerial_min_r2: float = 0.80
    #: ...and the arc has to be DEEP enough to be a flight rather than a wobble.
    #: Pixels of img_y climb over the observed span. A ball nudged along the grass
    #: on a jittery camera can trace a few px of upward curvature at a high R^2;
    #: it cannot trace 8 px of it.
    aerial_min_amplitude_px: float = 8.0
    #: Apparent ground speed floor, m/s. The main defence against a camera
    #: pan/tilt reading as an arc: a ball sitting in the grass under a panning
    #: camera is slow, and a flighted ball is not. The speed is measured on the
    #: DISTORTED ground track, which is fine -- we only need "fast", not "how
    #: fast". A run whose speed is unknown (the ground gate nulls exactly the
    #: frames we care about) is neither gated nor corroborated by it.
    aerial_min_speed_ms: float = 12.0
    #: corr(img_y, bbox height) at/above which the box corroborates the arc. The
    #: ball is further from the camera near the apex, so it images smaller there:
    #: img_y and bbox height fall and rise together. Independent of the parabola
    #: fit, which is what makes it worth asking -- a camera tilt moves img_y
    #: without changing the ball's apparent size.
    aerial_bbox_min_corr: float = 0.30
    #: Robust img_y cleaning, in IMAGE space. Deliberately NOT the ``ball_outlier``
    #: flag: that is a ground-speed gate, and it rejects airborne balls *because*
    #: they are airborne -- using it here would delete the arc it is meant to help
    #: find (see the module docstring). Running-median window, ...
    aerial_median_window: int = 5
    #: ...and how many MADs off that median a sample must be to be a glitch.
    aerial_spike_mad: float = 5.0
    #: ...with a floor on the MAD scale, so a perfectly smooth arc (MAD == 0)
    #: does not declare every sample that is not identical to its median an outlier.
    aerial_min_spike_px: float = 3.0

    #: PARTIAL arcs: the ball was already up when the run started, or we only saw
    #: it come down. Rather than force a parabola onto half an arc and invent a
    #: vertex nobody observed, a big one-way img_y ramp on a fast loose ball is
    #: emitted as airborne with LOW, CAPPED confidence. A ramp is also what a pan
    #: looks like, so this branch demands the speed AND bbox evidence too.
    aerial_partial_min_ramp_px: float = 40.0
    #: ...and this fraction of the run's steps must run the same way as the ramp.
    aerial_partial_min_monotone: float = 0.80
    aerial_partial_base_conf: float = 0.20
    #: Half an arc is genuinely weaker evidence than a whole one, and no amount of
    #: corroboration should let it pass for a clean parabola.
    aerial_partial_max_conf: float = 0.45

    # --- known limitation --- #
    #: No pose data => the bodypart is not observable. SPADL has no "unknown"
    #: bodypart, and socceraction's own converters default to ``foot``; we do the
    #: same, and say so loudly in the meta rather than pretending it was measured.
    default_bodypart: str = "foot"

    overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_bodypart not in BODYPARTS:
            raise ValueError(
                f"default_bodypart must be one of {BODYPARTS}, "
                f"got {self.default_bodypart!r}"
            )

    def team_attack_dir(self, team_id: Optional[int]) -> int:
        """``+1`` / ``-1`` for a team; ``+1`` if the direction is unknown.

        Falling back to ``+1`` keeps the geometry defined (nothing crashes on a
        team we never saw a goalkeeper for), and the affected actions are flagged
        low-confidence by the caller rather than silently trusted.
        """
        if team_id is None:
            return 1
        return int(self.attack_dir.get(int(team_id), 1))

    def goal_xy(self, team_id: Optional[int]) -> tuple:
        """Centre of the goal a team is attacking, in target-frame metres.

        Goals sit at ``x=0`` and ``x=pitch_length_m``, mouth centred on
        ``y=pitch_width_m/2``. Milestone 1 needs this only for the progression
        direction -- it is also the hook shot detection will hang off.
        """
        d = self.team_attack_dir(team_id)
        return (self.pitch_length_m if d > 0 else 0.0, self.pitch_width_m / 2.0)

    def attacking_x(self, x: float, team_id: Optional[int]) -> float:
        """x re-expressed as "metres advanced toward the team's target goal"."""
        d = self.team_attack_dir(team_id)
        return float(x) if d > 0 else self.pitch_length_m - float(x)

    def coherence_floor(self, airborne: bool) -> float:
        """The straightness a gap's ball path must clear to be believed.

        Relaxed (bypassed, by default) for a gap the ball FLEW across: coherence
        tests the ball's GROUND path, and an airborne ball's ground path is a
        z=0 back-projection artefact, not a trajectory. Applying the rolling-ball
        threshold to it would flag every well-struck aerial pass as an aimless
        deflection -- for the distortion this layer has already identified.
        """
        return (
            self.aerial_min_path_coherence if airborne else self.min_path_coherence
        )

    def as_meta(self) -> dict:
        """Serializable snapshot of every parameter (for the stage meta json)."""
        return asdict(self)


def config_from_prep_meta(prep_meta: dict, **overrides) -> ActionConfig:
    """Build an :class:`ActionConfig` from a prerequisites ``prep_meta.json``.

    Reads fps, the *target* pitch dims, and the per-team ``attack_dir`` resolved
    by the ``resolve_direction`` transform, then applies any non-None keyword
    overrides.

    ``attack_dir`` is recorded per period in prep_meta. This layer collapses it
    to a single map because the prepared table has no period column yet; when
    periods arrive, this is the one place that has to learn about them.
    """
    cfg = ActionConfig()

    src = prep_meta.get("source_meta", {}) or {}
    cfg.fps = float(src.get("fps", cfg.fps) or cfg.fps)
    source = str(src.get("source") or "")
    if source:
        cfg.game_id = source.rsplit(".", 1)[0]

    steps = prep_meta.get("steps", {}) or {}

    rescale = steps.get("rescale_coords", {}) or {}
    target = rescale.get("target_pitch_m") or rescale.get("target_pitch") or None
    if isinstance(target, (list, tuple)) and len(target) == 2:
        cfg.pitch_length_m, cfg.pitch_width_m = float(target[0]), float(target[1])
    elif isinstance(target, dict):
        cfg.pitch_length_m = float(target.get("length_m", cfg.pitch_length_m))
        cfg.pitch_width_m = float(target.get("width_m", cfg.pitch_width_m))

    direction = steps.get("resolve_direction", {}) or {}
    dirs: Dict[int, int] = {}
    for _period, info in (direction.get("periods", {}) or {}).items():
        for team, d in ((info or {}).get("attack_dir", {}) or {}).items():
            dirs[int(team)] = int(d)
    cfg.attack_dir = dirs

    applied = {}
    for key, value in overrides.items():
        if value is None or not hasattr(cfg, key):
            continue
        setattr(cfg, key, value)
        applied[key] = value
    cfg.overrides = applied
    cfg.__post_init__()
    return cfg
