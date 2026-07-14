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
