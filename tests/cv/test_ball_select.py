"""In-play ball selection: the cases the defaults are pinned against.

These are unit tests over constructed candidate streams — they pin the decision
logic. Validation on real footage (spare balls actually visible) is a separate
exercise; see the ball-selection section of the README.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cv.ball_select import (
    BallSelectConfig,
    Candidate,
    associate,
    select_in_play_ball,
)


def _cand(frame, px, py, ix=None, iy=None, conf=0.8, valid=True):
    """A candidate at pitch (px, py); image coords default to a plausible scaling."""
    ix = px * 15.0 if ix is None else ix  # ~15 px per metre, good enough for gating
    iy = py * 15.0 if iy is None else iy
    return Candidate(
        frame=frame,
        img_x=ix,
        img_y=iy,
        pitch_x=px if valid else None,
        pitch_y=py if valid else None,
        pitch_valid=valid,
        conf=conf,
        bbox=(ix - 5, iy - 5, ix + 5, iy + 5),
    )


def _static_ball(n, px, py, jitter=0.0, seed=0):
    """A spare ball: same place every frame, optional homography jitter."""
    rng = np.random.default_rng(seed)
    out = []
    for f in range(n):
        jx, jy = (rng.normal(0, jitter, 2) if jitter else (0.0, 0.0))
        out.append(_cand(f, px + jx, py + jy))
    return out


def _moving_ball(n, x0=20.0, y0=35.0, vx=0.4, vy=0.05):
    """The game ball: sweeps across the pitch (0.4 m/frame = 10 m/s @ 25 fps)."""
    return [_cand(f, x0 + vx * f, y0 + vy * f) for f in range(n)]


def _cfg(**kw):
    return BallSelectConfig(**kw)


# --------------------------------------------------------------------------- #
# association
# --------------------------------------------------------------------------- #
def test_associate_separates_static_and_moving_balls():
    cands = _static_ball(60, 2.0, -3.0) + _moving_ball(60)
    tracks = associate(cands, _cfg())
    # two physical balls -> two tracks, each seen on every frame
    assert len(tracks) == 2
    assert sorted(len(t.cands) for t in tracks.values()) == [60, 60]


def test_associate_bridges_a_short_dropout():
    """A missing detection mid-track must not split the track in two."""
    cands = [c for c in _moving_ball(40) if c.frame not in (10, 11, 12)]
    tracks = associate(cands, _cfg(assoc_gate_px=200.0, assoc_max_gap=8))
    assert len(tracks) == 1


# --------------------------------------------------------------------------- #
# the core failure mode: never lock onto a static sideline ball during play
# --------------------------------------------------------------------------- #
def test_moving_ball_beats_static_spare_ball():
    """The whole point. A spare ball off the touchline must never win."""
    spare = _static_ball(80, 5.0, -3.0)  # 3 m outside the touchline
    game = _moving_ball(80)
    sel, tracks, _ = select_in_play_ball(spare + game, _cfg())

    game_ids = {c.track_id for c in game}
    picked = {s.track_id for s in sel if s.cand is not None}
    assert picked, "nothing was selected at all"
    assert picked <= game_ids, "selection locked onto the static spare ball"


def test_two_static_spares_and_one_game_ball():
    """More static balls than game balls — a support/confidence vote would lose."""
    spare_a = _static_ball(80, 5.0, -3.0)
    spare_b = _static_ball(80, 118.0, 74.0)
    game = _moving_ball(80)
    sel, _, _ = select_in_play_ball(spare_a + spare_b + game, _cfg())

    game_ids = {c.track_id for c in game}
    picked = {s.track_id for s in sel if s.cand is not None}
    assert picked <= game_ids


def test_static_spare_ball_alone_is_not_selected():
    """No game ball in frame => emit NOTHING. Never fall back to a static ball."""
    spare = _static_ball(80, 5.0, -3.0)
    sel, _, _ = select_in_play_ball(spare, _cfg())
    assert all(s.cand is None for s in sel), "fell back to the static spare ball"


def test_jittering_touchline_ball_still_loses():
    """Homography is noisiest exactly where the spare balls sit.

    A static touchline ball whose pitch coords swing ~1 m frame to frame must not
    fake enough motion energy to beat the real ball.
    """
    spare = _static_ball(80, 2.0, -2.0, jitter=1.0, seed=7)
    game = _moving_ball(80)
    sel, _, _ = select_in_play_ball(spare + game, _cfg())

    game_ids = {c.track_id for c in game}
    picked = {s.track_id for s in sel if s.cand is not None}
    assert picked <= game_ids


# --------------------------------------------------------------------------- #
# the case motion energy alone gets wrong: a resting ball, still in play
# --------------------------------------------------------------------------- #
def test_resting_on_pitch_ball_beats_static_off_pitch_ball():
    """Ball stopped for a set-piece: static, but ON the pitch. On-pitch fraction
    is what rescues it — motion energy alone would score both at zero."""
    spare = _static_ball(80, 5.0, -3.0)  # off pitch
    resting = _static_ball(80, 60.0, 35.0)  # centre circle, stationary
    sel, _, _ = select_in_play_ball(spare + resting, _cfg())

    resting_ids = {c.track_id for c in resting}
    picked = {s.track_id for s in sel if s.cand is not None}
    assert picked, "no ball selected — the resting game ball was thrown away"
    assert picked <= resting_ids


# --------------------------------------------------------------------------- #
# the "being played" signals: distance-to-player and trajectory physics
# --------------------------------------------------------------------------- #
def _players(frames, xy, n=6, spread=3.0, seed=1):
    """A cluster of n players around ``xy`` on every frame -> {frame: (n, 2) array}."""
    rng = np.random.default_rng(seed)
    base = np.array(xy, dtype=float)
    return {int(f): base + rng.normal(0, spread, size=(n, 2)) for f in frames}


def test_static_on_pitch_ball_far_from_players_is_rejected():
    """The real-footage failure mode. A static spare resting just INSIDE the goal
    line — on the pitch, barely moving, but ~20 m from anyone — scores well on
    motion + on-pitch and gets locked onto for hundreds of frames. Distance to the
    nearest player is what vetoes it."""
    players = _players(range(80), (60.0, 35.0))         # play in the centre
    spare = _static_ball(80, 117.0, 17.0)               # on pitch, far corner
    game = [_cand(f, 58.0 + 0.2 * f, 35.0) for f in range(80)]  # among the players
    sel, _, _ = select_in_play_ball(spare + game, _cfg(), players)

    game_ids = {c.track_id for c in game}
    picked = {s.track_id for s in sel if s.cand is not None}
    assert picked, "nothing selected"
    assert picked <= game_ids, "locked onto the static on-pitch spare"


def test_moving_inbounds_ball_far_from_players_is_not_in_play():
    """A spare ball kicked along an empty flank is moving AND inbounds, so motion +
    on-pitch rate it highly — but it is nowhere near the play, so it is not the ball
    being played and nothing should be selected."""
    players = _players(range(80), (60.0, 35.0))
    stray = [_cand(f, 10.0, 5.0 + 0.3 * f) for f in range(80)]  # moving, inbounds, far
    sel, _, _ = select_in_play_ball(stray, _cfg(), players)
    assert all(s.cand is None for s in sel), "selected a moving ball far from the play"


def test_distance_signal_is_dropped_when_no_players_given():
    """Backward compatibility: with no player positions the selector must behave as
    the motion + on-pitch predecessor did (the whole existing suite runs this way)."""
    resting = _static_ball(80, 60.0, 35.0)
    sel, _, _ = select_in_play_ball(resting, _cfg())  # no players
    assert any(s.cand is not None for s in sel), "resting on-pitch ball was dropped"


def test_physics_factor_penalises_a_teleporting_track():
    from src.cv.ball_select import _physics_factor

    cfg = _cfg()
    coherent = [_cand(f, 40.0 + 0.3 * f, 35.0) for f in range(20)]
    tele = [_cand(f, 40.0 if f % 2 == 0 else 80.0, 35.0) for f in range(20)]
    assert _physics_factor(coherent, cfg) == 1.0
    assert _physics_factor(tele, cfg) <= cfg.physics_floor + 1e-9


# --------------------------------------------------------------------------- #
# following a fragmented ball across track ids, and bridging short dropouts
# --------------------------------------------------------------------------- #
def test_selection_rides_fragmented_ball_across_track_ids():
    """A fast, occluded game ball is broken into a chain of short tracks. Selection
    must ride the chain as one ball, bridging the frames in between (which here hold
    only a spare) instead of dropping to null or onto the spare."""
    frames_ball = list(range(0, 21)) + list(range(30, 51))  # detector drops 21..29
    ball = [_cand(f, 50.0 + 0.3 * f, 35.0) for f in frames_ball]
    spare = [_cand(f, 117.0, 17.0) for f in range(21, 30)]  # only a spare in the gap
    players = _players(range(51), (55.0, 35.0), spread=6.0)
    sel, tracks, _ = select_in_play_ball(ball + spare, _cfg(), players)

    assert len({c.track_id for c in ball}) >= 2, "expected the ball to fragment"
    by = {s.frame: s for s in sel}
    spare_ids = {c.track_id for c in spare}
    assert not (spare_ids & {s.track_id for s in sel if s.cand is not None}), \
        "selected the spare in the gap"
    for f in range(21, 30):
        assert by[f].bridged, f"gap frame {f} was not bridged"


def test_holds_lock_and_does_not_flicker_between_two_in_play_balls():
    """This footage has several balls on the pitch at once. When the tracked ball
    drops for a few frames, selection must hold (and bridge) its trajectory, not
    teleport onto a second in-play ball elsewhere."""
    players = _players(range(60), (56.0, 37.0), spread=9.0)  # covers both balls
    a = [_cand(f, 50.0, 35.0) for f in range(60) if not (25 <= f <= 29)]  # drops 25..29
    b = [_cand(f, 62.0, 40.0) for f in range(60)]  # a second ball, ~13 m away
    sel, _, _ = select_in_play_ball(a + b, _cfg(), players)

    cfg = _cfg()
    seq = [
        (s.frame,
         s.cand.pitch_x if s.cand is not None else s.pitch_x,
         s.cand.pitch_y if s.cand is not None else s.pitch_y)
        for s in sorted(sel, key=lambda s: s.frame)
        if s.cand is not None or s.bridged
    ]
    for (f0, x0, y0), (f1, x1, y1) in zip(seq, seq[1:]):
        if f1 - f0 == 1:
            reach = cfg.ball_max_speed_ms / cfg.fps + cfg.continuity_slack_m
            assert np.hypot(x1 - x0, y1 - y0) <= reach + 1e-6, \
                f"flickered between two balls at frame {f0}->{f1}"


# --------------------------------------------------------------------------- #
# null output is a legitimate answer
# --------------------------------------------------------------------------- #
def test_occluded_frames_emit_null_not_a_substitute():
    """When the winner has no detection on a frame, emit null — do not silently
    swap in the nearest other candidate (that is how a spare ball gets in)."""
    spare = _static_ball(80, 5.0, -3.0)
    game = [c for c in _moving_ball(80) if not (30 <= c.frame <= 34)]
    sel, _, _ = select_in_play_ball(spare + game, _cfg())

    by_frame = {s.frame: s for s in sel}
    for f in range(30, 35):
        assert by_frame[f].cand is None, f"frame {f} substituted another candidate"


def test_no_candidates_at_all():
    sel, tracks, scores = select_in_play_ball([], _cfg())
    assert sel == [] and tracks == {} and scores == {}


# --------------------------------------------------------------------------- #
# stability
# --------------------------------------------------------------------------- #
def test_selection_does_not_flicker_between_candidates():
    """Two moving balls with close scores must not swap identity every window."""
    a = _moving_ball(120, x0=20.0, y0=30.0, vx=0.40, vy=0.0)
    b = _moving_ball(120, x0=20.0, y0=60.0, vx=0.41, vy=0.0)
    sel, _, _ = select_in_play_ball(a + b, _cfg())

    ids = [s.track_id for s in sel if s.cand is not None]
    switches = sum(1 for i, j in zip(ids[:-1], ids[1:]) if i != j)
    assert switches <= 1, f"identity flickered {switches} times"


# --------------------------------------------------------------------------- #
# online mode
# --------------------------------------------------------------------------- #
def test_trailing_mode_is_causal_and_agrees_with_centered():
    """The causal variant must reach the same verdict (never the spare ball), and
    must not need future frames to do it."""
    spare = _static_ball(120, 5.0, -3.0)
    game = _moving_ball(120)
    cands = spare + game

    sel = select_in_play_ball(cands, _cfg(mode="trailing"))[0]
    game_ids = {c.track_id for c in game}
    picked = {s.track_id for s in sel if s.cand is not None}
    assert picked, "trailing mode selected nothing"
    assert picked <= game_ids


def test_trailing_mode_emits_nothing_during_warmup():
    game = _moving_ball(120)
    sel = select_in_play_ball(game, _cfg(mode="trailing", warmup_frames=20))[0]
    by_frame = {s.frame: s for s in sel}
    assert by_frame[0].cand is None
    assert any(s.cand is not None for s in sel), "never committed after warm-up"


def test_trailing_selection_is_prefix_stable():
    """A causal selector's verdict for frame t may not change when later frames
    arrive — otherwise it cannot run online."""
    spare = _static_ball(120, 5.0, -3.0)
    game = _moving_ball(120)
    cfg = _cfg(mode="trailing")

    full = {s.frame: s.track_id for s in select_in_play_ball(spare + game, cfg)[0]}
    prefix_cands = [c for c in spare + game if c.frame <= 70]
    prefix = {s.frame: s.track_id for s in select_in_play_ball(prefix_cands, cfg)[0]}

    for f, tid in prefix.items():
        assert full[f] == tid, f"frame {f} verdict changed once the future arrived"
