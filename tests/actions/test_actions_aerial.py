"""The airborne-ball flag, and the two things the event layer does with it.

The root cause, restated: the homography maps the image to the **ground plane
(z=0)**. A ball in the air back-projects to a point stretched away from the camera
-- wrong by metres, with no height channel anywhere to correct it. So:

**CHANGE 1** -- the emitted geometry is anchored on the PLAYERS, never on the
mid-flight ball. A pass starts where the passer stood on his last controlled frame
and ends where the receiver stood on his first. The ball path only *characterises*
the gap. The fixture that pins this is
``test_a_pass_with_corrupted_mid_flight_ball_coords_still_emits_the_player_positions``:
the ball's coordinates through the gap are garbage, the players' are clean, and the
emitted action must not notice.

**CHANGE 2** -- ``airborne`` / ``aerial_conf`` per ball frame, from the arc the
ball traces in ``img_y``. Image-y **decreases** as the ball rises, so a ball that
went up and came down is a **local MINIMUM** in img_y: an upward-opening parabola.

The scenarios:

- an aerial run (img_y arc, elevated smooth speed, loose)   -> airborne=True
- a ground pass (flat img_y, slow)                          -> airborne=False
- a flat run with ONE img_y spike (a detection glitch)      -> airborne=False
- an aerial pass with a bent GROUND projection              -> NOT flagged incoherent
- a pass with corrupted mid-flight ball coords              -> endpoints are the PLAYERS'
"""

import numpy as np
import pytest

from src.actions import ZonePossessionSource, detect_actions, run_stages
from src.actions.aerial import ARC_FULL, detect_airborne
from tests.actions.actions_helpers import (
    FPS,
    aerial_img_y,
    ball_line,
    ball_still,
    config,
    flat_img_y,
    hold,
    loose,
    near,
    stream,
    tracking,
)


def run(frames, ball, players, teams, img=None, **cfg_overrides):
    cfg = config(**cfg_overrides)
    source = ZonePossessionSource(frames)
    track = tracking(ball, players, teams, img=img)
    return detect_actions(source, track, cfg)


def aerial_of(frames, ball, players, teams, img=None, **cfg_overrides):
    """Just the AerialTrack, for the tests that are about the detector itself."""
    cfg = config(**cfg_overrides)
    return detect_airborne(tracking(ball, players, teams, img=img), frames, cfg)


# --------------------------------------------------------------------------- #
# CHANGE 2: the airborne flag
# --------------------------------------------------------------------------- #
def test_an_img_y_arc_on_a_fast_loose_ball_is_airborne():
    """The ball rose and fell: img_y dips to a local MINIMUM and comes back.

    This is the whole signal. It is visible in IMAGE space, which is the one place
    the ground-plane homography has not already destroyed the evidence.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 40), (20.0, 34.0), (60.0, 34.0)),
        **ball_still(range(40, 50), (60.0, 34.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }
    # img_y: 352 -> ~325 -> ~352 across the loose run. An UP-then-DOWN arc.
    img = {
        **flat_img_y(range(0, 10), y=352.0),
        **aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0),
        **flat_img_y(range(40, 50), y=352.0),
    }

    track = aerial_of(frames, ball, players, {1: 0, 2: 0}, img=img)

    assert track.airborne(25) is True          # mid-flight
    assert track.conf(25) > 0.5
    assert len(track.runs) == 1
    run_ = track.runs[0]
    assert run_.kind == ARC_FULL
    assert run_.curvature > 0                  # upward-opening: a MINIMUM in img_y
    assert run_.r2 > 0.9
    assert 10 <= run_.vertex_frame <= 39       # the apex was OBSERVED, not extrapolated
    # ...and the touch frames either side are NOT in flight: a possessor is within
    # the possession radius of the ball, which is what made him the possessor.
    assert track.airborne(5) is False
    assert track.airborne(45) is False


def test_a_flat_slow_ground_pass_is_not_airborne():
    """img_y barely moves and the ball is slow. Nothing went up."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 30)),
        hold(2, team=0, frames=range(30, 40)),
    )
    # 6 m over 20 frames = 7.5 m/s: a rolled pass, under the speed floor.
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 30), (30.0, 34.0), (36.0, 34.0)),
        **ball_still(range(30, 40), (36.0, 34.0)),
    }
    players = {
        1: {f: (30.0, 34.5) for f in range(0, 40)},
        2: {f: (36.0, 34.5) for f in range(0, 40)},
    }
    img = flat_img_y(range(0, 40), y=400.0, jitter=1.0)

    track = aerial_of(frames, ball, players, {1: 0, 2: 0}, img=img)

    assert track.airborne(20) is False
    assert track.conf(20) == 0.0
    assert len(track) == 0
    assert track.runs[0].airborne is False


def test_one_img_y_spike_in_a_flat_run_is_rejected_as_a_glitch():
    """A single bad detection must not be able to invent an arc.

    Frame 136 of the real clip reads ``img_y=449`` in the middle of a run sitting
    around 325 -- the detector briefly latched onto something else. A least-squares
    parabola is not robust, so one such sample left in the fit would drag the vertex
    and manufacture curvature out of nothing. The robust image-space filter (median
    + MAD) throws it out BEFORE the fit sees it.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 30)),
        hold(2, team=0, frames=range(30, 40)),
    )
    # Fast and loose -- so the ONLY thing standing between this and an "airborne"
    # verdict is the spike rejection. If it regressed, this test goes red.
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 30), (20.0, 34.0), (45.0, 34.0)),
        **ball_still(range(30, 40), (45.0, 34.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 40)},
        2: {f: (45.0, 34.5) for f in range(0, 40)},
    }
    img = {
        **flat_img_y(range(0, 40), y=350.0),
        20: 480.0,          # the glitch: 130 px out, for exactly one frame
    }

    track = aerial_of(frames, ball, players, {1: 0, 2: 0}, img=img)

    assert track.airborne(20) is False
    run_ = track.runs[0]
    assert run_.airborne is False
    assert run_.n_rejected == 1          # ...and we can SAY it was thrown out


def test_the_spike_does_not_break_a_real_arc_either():
    """The glitch-rejection has to keep the arc it is defending.

    The real clip has both at once: a clean parabola AND one wild sample sitting in
    the middle of it. Rejecting the sample must not cost us the flight.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 40), (20.0, 34.0), (60.0, 34.0)),
        **ball_still(range(40, 50), (60.0, 34.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }
    img = {
        **flat_img_y(range(0, 10), y=352.0),
        **aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0),
        **flat_img_y(range(40, 50), y=352.0),
        23: 449.0,          # the frame-136 glitch, transplanted
    }

    track = aerial_of(frames, ball, players, {1: 0, 2: 0}, img=img)

    assert track.airborne(25) is True
    assert track.runs[0].n_rejected == 1
    assert track.runs[0].r2 > 0.9          # the fit survived intact


def test_a_ball_that_is_ATTRIBUTED_is_never_airborne():
    """Airborne is computed over LOOSE runs only -- the ball has to be nobody's.

    A ball at a player's feet is on the ground by definition, and requiring the ball
    to be unattributed is a large part of what stops a camera pan (which also moves
    img_y) from being read as a flight.
    """
    # The same arc, but the ball never leaves player 1's possession.
    frames = stream(hold(1, team=0, frames=range(0, 50)))
    ball = ball_line(range(0, 50), (20.0, 34.0), (60.0, 34.0))
    players = {1: near(ball, range(0, 50))}
    img = aerial_img_y(range(0, 50), y_ground=352.0, apex_rise_px=27.0)

    track = aerial_of(frames, ball, players, {1: 0}, img=img)

    assert len(track) == 0
    assert track.runs == []          # there was no loose run to even look at


def test_a_run_with_no_image_columns_degrades_to_not_airborne():
    """No img_y => no evidence => not airborne. Never a guess.

    A ball-free possession source, or an older prepared table, must still run. The
    safe default is "not airborne", which changes nothing -- rather than relaxing a
    guard on the strength of data we do not have.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = ball_line(range(0, 50), (20.0, 34.0), (60.0, 34.0))
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }

    track = aerial_of(frames, ball, players, {1: 0, 2: 0}, img=None)
    assert len(track) == 0
    assert track.airborne(25) is False


def test_the_detector_can_be_switched_off_entirely():
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = ball_line(range(0, 50), (20.0, 34.0), (60.0, 34.0))
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }
    img = aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0)

    track = aerial_of(
        frames, ball, players, {1: 0, 2: 0}, img=img, aerial_enabled=False
    )
    assert len(track) == 0


def test_the_thresholds_are_configurable():
    """Raise the curvature floor above the arc and the same flight stops qualifying."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 40), (20.0, 34.0), (60.0, 34.0)),
        **ball_still(range(40, 50), (60.0, 34.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }
    img = {
        **flat_img_y(range(0, 10), y=352.0),
        **aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0),
        **flat_img_y(range(40, 50), y=352.0),
    }

    assert aerial_of(frames, ball, players, {1: 0, 2: 0}, img=img).airborne(25) is True
    # ...and the same clip, judged against an absurd curvature floor, is not.
    strict = aerial_of(
        frames, ball, players, {1: 0, 2: 0}, img=img, aerial_min_curvature=5.0
    )
    assert strict.airborne(25) is False
    # ...nor against a speed floor no real pass could clear.
    slow = aerial_of(
        frames, ball, players, {1: 0, 2: 0}, img=img, aerial_min_speed_ms=99.0
    )
    assert slow.airborne(25) is False


# --------------------------------------------------------------------------- #
# CHANGE 1: the emitted geometry comes from the PLAYERS
# --------------------------------------------------------------------------- #
def test_a_pass_with_corrupted_mid_flight_ball_coords_still_emits_the_player_positions():
    """**The fixture this whole change exists for.**

    The ball's coordinates through the gap are garbage -- which is exactly what an
    airborne ball's z=0 back-projection IS: stretched away from the camera, off by
    tens of metres, curving through places the ball never went. The passer's and the
    receiver's positions are clean, because they are standing on the grass, on the
    plane the homography is actually valid for.

    The emitted action must take its endpoints from the PLAYERS and must not notice
    the corruption at all.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 30)),
        hold(2, team=0, frames=range(30, 40)),
    )
    ball = {
        # At the release frame the ball is ALREADY off by 12 m in y -- the moment of
        # the strike is the moment the distortion begins.
        **ball_still(range(0, 10), (28.0, 46.0)),
        # ...and mid-flight it wanders off to the far touchline and back.
        **ball_line(range(10, 20), (28.0, 46.0), (70.0, 4.0)),
        **ball_line(range(20, 30), (70.0, 4.0), (44.0, 60.0)),
        **ball_still(range(30, 40), (44.0, 60.0)),
    }
    # The players, meanwhile, are exactly where they are.
    players = {
        1: {f: (30.0, 34.0) for f in range(0, 40)},     # the passer
        2: {f: (55.0, 40.0) for f in range(0, 40)},     # the receiver
    }

    actions, prov, _meta = run(frames, ball, players, {1: 0, 2: 0})

    the_pass = actions[actions["type_name"] == "pass"]
    assert len(the_pass) == 1
    row = the_pass.iloc[0]

    # the PASSER's position on his last controlled frame...
    assert row["start_x"] == pytest.approx(30.0)
    assert row["start_y"] == pytest.approx(34.0)
    # ...and the RECEIVER's on his first. NOT the ball's (28,46) -> (44,60).
    assert row["end_x"] == pytest.approx(55.0)
    assert row["end_y"] == pytest.approx(40.0)

    # The ball path is still MEASURED -- it is what told us a transfer happened --
    # it just does not get to say where the action was.
    p = prov[prov["kind"] == "pass"].iloc[0]
    assert p["ball_travel_m"] > 0          # the (garbage) ball did "travel"
    assert p["action_travel_m"] == pytest.approx(
        np.hypot(55.0 - 30.0, 40.0 - 34.0)
    )


def test_a_turnover_anchors_on_the_loser_and_the_winner():
    """Both sides of a turnover take their geometry from their own player."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(9, team=1, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (33.0, 20.0)),        # distorted at release
        **ball_line(range(10, 20), (33.0, 20.0), (66.0, 55.0)),
        **ball_still(range(20, 30), (66.0, 55.0)),
    }
    players = {
        1: {f: (30.0, 34.0) for f in range(0, 30)},      # loses it here
        9: {f: (58.0, 30.0) for f in range(0, 30)},      # wins it here
    }

    actions, prov, _meta = run(frames, ball, players, {1: 0, 9: 1})

    assert list(actions["type_name"]) == ["pass", "interception"]
    failed, won = actions.iloc[0], actions.iloc[1]

    # the failed pass starts at the LOSER and ends at the WINNER
    assert (failed["start_x"], failed["start_y"]) == pytest.approx((30.0, 34.0))
    assert (failed["end_x"], failed["end_y"]) == pytest.approx((58.0, 30.0))
    # the interception is won on the spot, at the WINNER's own position
    assert (won["start_x"], won["start_y"]) == pytest.approx((58.0, 30.0))
    assert (won["end_x"], won["end_y"]) == pytest.approx((58.0, 30.0))
    # ...and it is still an interception, because the BALL was in flight. The
    # discriminator reads the ball, not the players.
    assert bool(prov[prov["kind"] == "turnover"].iloc[0]["in_flight"]) is True


def test_a_static_hold_is_still_not_a_dribble_even_though_the_player_moved():
    """The carry is EMITTED on the player but GATED on the ball. Both halves matter.

    If the gate moved onto the player along with the geometry, a player jogging 20 m
    past a ball parked in the grass would mint a dribble. The rule is, and stays,
    "the ball went somewhere with him".
    """
    frames = stream(hold(1, team=0, frames=range(0, 60)))
    ball = ball_still(range(0, 60), (30.0, 34.0))               # never moves
    players = {1: {f: (30.0 + 0.3 * f, 34.0) for f in range(0, 60)}}  # runs 18 m

    actions, _prov, meta = run(frames, ball, players, {1: 0})

    assert len(actions) == 0
    assert meta["skipped_by_reason"]["static_hold"] == 1


def test_the_chain_joins_up_exactly_because_both_ends_are_the_same_player_frame():
    """A pass ends at the receiver's position on frame f; his carry starts there too.

    Player-anchoring both makes the chain *exactly* continuous rather than
    approximately -- and xT reads precisely those start->end deltas.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (50.0, 34.0)),
        **ball_line(range(20, 50), (50.0, 34.0), (70.0, 34.0)),
    }
    players = {
        1: {f: (29.0, 33.0) for f in range(0, 50)},
        2: near(ball, range(0, 50), offset=(-1.0, 1.0)),
    }

    actions, prov, _meta = run(frames, ball, players, {1: 0, 2: 0})
    merged = actions.merge(prov[["action_id", "kind"]], on="action_id")

    the_pass = merged[merged["kind"] == "pass"].iloc[0]
    the_carry = merged[merged["kind"] == "carry"].iloc[0]
    # not "within 3 m" -- the SAME POINT.
    assert the_carry["start_x"] == pytest.approx(the_pass["end_x"])
    assert the_carry["start_y"] == pytest.approx(the_pass["end_y"])


# --------------------------------------------------------------------------- #
# the two meeting: an aerial pass must survive the coherence guard
# --------------------------------------------------------------------------- #
def test_an_airborne_pass_with_a_bent_ground_path_is_not_flagged_incoherent():
    """The guard it must survive, and the reason it must.

    ``min_path_coherence`` is a STRAIGHTNESS test, and straightness is a property of
    the ball's GROUND path. An airborne ball has no trustworthy ground path: the z=0
    homography stretches its back-projection away from the camera by an amount that
    grows and shrinks with its height, so a cleanly struck diagonal comes out bent
    and scores like an aimless deflection.

    So: a pass whose ground projection wanders badly, but which we can SEE was
    airborne, must not be docked for a curve the homography invented.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    # the classic z=0 signature: the ball appears to bulge away from the camera
    # mid-flight and come back. Straight-line/polyline is far below 0.5.
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 25), (20.0, 34.0), (35.0, 4.0)),
        **ball_line(range(25, 40), (35.0, 4.0), (45.0, 36.0)),
        **ball_still(range(40, 50), (45.0, 36.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (45.0, 35.5) for f in range(0, 50)},
    }
    img = {
        **flat_img_y(range(0, 10), y=352.0),
        **aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0),
        **flat_img_y(range(40, 50), y=352.0),
    }

    actions, prov, meta = run(frames, ball, players, {1: 0, 2: 0}, img=img)

    assert list(actions["type_name"]) == ["pass"]
    row = prov[prov["kind"] == "pass"].iloc[0]

    # the ground path really IS incoherent...
    assert row["path_coherence"] < 0.5
    # ...and the guard was relaxed anyway, because we know why it is bent.
    assert bool(row["aerial"]) is True
    assert row["aerial_conf"] > 0.0
    assert "incoherent_ball_path" not in row["reasons"]
    assert "aerial" in row["reasons"]

    # and the meta counts it as the aerial pass it is
    assert meta["n_aerial_passes"] == 1
    assert meta["aerial"]["n_aerial_runs"] == 1


def test_the_same_bent_path_on_a_GROUND_ball_is_still_flagged_incoherent():
    """The relaxation is bought by the aerial evidence, not given away.

    Same wandering ball path, but flat img_y and a slow ball -- so nothing says it
    flew. The coherence guard applies in full, exactly as it did before.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 25), (30.0, 34.0), (34.0, 44.0)),
        **ball_line(range(25, 40), (34.0, 44.0), (36.0, 36.0)),
        **ball_still(range(40, 50), (36.0, 36.0)),
    }
    players = {
        1: {f: (30.0, 34.5) for f in range(0, 50)},
        2: {f: (36.0, 36.5) for f in range(0, 50)},
    }
    img = flat_img_y(range(0, 50), y=400.0, jitter=1.0)

    _actions, prov, meta = run(frames, ball, players, {1: 0, 2: 0}, img=img)
    row = prov[prov["kind"] == "pass"].iloc[0]

    assert row["path_coherence"] < 0.5
    assert bool(row["aerial"]) is False
    assert "incoherent_ball_path" in row["reasons"]
    assert bool(row["low_confidence"]) is True
    assert meta["n_aerial_passes"] == 0


def test_an_aerial_pass_keeps_its_spadl_type_and_is_subtyped_in_provenance():
    """SPADL has no aerial action. So the type stays `pass`; the flight goes in prov.

    Inventing an action type would break the very `socceraction` contract this layer
    exists to honour -- ``SPADLSchema`` is strict, and ``EMITTED_ACTIONTYPES`` would
    reject it. The flight is a *subtype*, and it lives in the sidecar.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 40), (20.0, 34.0), (60.0, 34.0)),
        **ball_still(range(40, 50), (60.0, 34.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }
    img = {
        **flat_img_y(range(0, 10), y=352.0),
        **aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0),
        **flat_img_y(range(40, 50), y=352.0),
    }

    actions, prov, meta = run(frames, ball, players, {1: 0, 2: 0}, img=img)

    assert list(actions["type_name"]) == ["pass"]      # NOT "aerial_pass"
    assert bool(prov[prov["kind"] == "pass"].iloc[0]["aerial"]) is True
    assert meta["aerial"]["thresholds"]["aerial_min_curvature"] == 0.02
    assert "HEURISTIC" in meta["aerial"]["note"]
    assert "z=0" in meta["limitations"]["ball_height"]


def test_the_ball_annotation_covers_every_ball_frame():
    """`airborne` / `aerial_conf` are columns on the BALL TRACK, not just on gaps."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (20.0, 34.0)),
        **ball_line(range(10, 40), (20.0, 34.0), (60.0, 34.0)),
        **ball_still(range(40, 50), (60.0, 34.0)),
    }
    players = {
        1: {f: (20.0, 34.5) for f in range(0, 50)},
        2: {f: (60.0, 34.5) for f in range(0, 50)},
    }
    img = {
        **flat_img_y(range(0, 10), y=352.0),
        **aerial_img_y(range(10, 40), y_ground=352.0, apex_rise_px=27.0),
        **flat_img_y(range(40, 50), y=352.0),
    }

    cfg = config()
    stages = run_stages(
        ZonePossessionSource(frames),
        tracking(ball, players, {1: 0, 2: 0}, img=img),
        cfg,
    )
    ann = stages.aerial.to_frame()

    assert list(ann.columns) == ["frame", "airborne", "aerial_conf"]
    assert len(ann) == 50                       # one row per BALL frame, not per gap
    assert ann["airborne"].dtype == bool
    assert bool(ann.loc[ann["frame"] == 25, "airborne"].iloc[0]) is True
    assert bool(ann.loc[ann["frame"] == 5, "airborne"].iloc[0]) is False
    assert ((ann["aerial_conf"] >= 0.0) & (ann["aerial_conf"] <= 1.0)).all()
