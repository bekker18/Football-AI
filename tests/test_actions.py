"""The transition rules, one scenario at a time.

Each fixture is a hand-built possession stream plus the ball/player geometry that
makes the scenario what it is, so a regression names the rule it broke rather
than just moving a count.

The scenarios are the ones the rules actually turn on:

- a clean same-team pass                -> ``pass`` / ``success``
- a cross-team turnover, ball in flight -> ``pass``/``fail`` + ``interception``
- a cross-team turnover, ball settled   -> ``bad_touch``/``fail`` + ``tackle``
- a same-player carry that progressed   -> ``dribble`` / ``success``
- a same-player STATIC hold             -> **nothing**
- a gap containing a ``no_ball`` run    -> emitted, but flagged occluded
- a wide + advanced delivery            -> ``cross``
- a 1-frame zone blip                   -> **not** a pass
"""

import pytest

from src.actions import ZonePossessionSource, detect_actions, segments_from_stream
from src.actions.spadl import EMITTED_ACTIONTYPES, SPADL_COLUMNS
from tests.actions_helpers import (
    ball_line,
    ball_still,
    config,
    contested,
    hold,
    loose,
    near,
    no_ball,
    stream,
    tracking,
)


def run(frames, ball, players, teams, **cfg_overrides):
    """Drive the whole layer on a synthetic clip; return ``(actions, prov, meta)``."""
    cfg = config(**cfg_overrides)
    source = ZonePossessionSource(frames)
    track = tracking(ball, players, teams)
    return detect_actions(source, track, cfg)


def types_of(actions):
    return list(actions["type_name"])


# --------------------------------------------------------------------------- #
# a clean same-team pass
# --------------------------------------------------------------------------- #
def test_same_team_different_player_is_a_successful_pass():
    """Player 1 holds, the ball travels, player 2 receives. One pass, no reception row."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    # The ball sits with player 1, flies 20 m, then sits with player 2.
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (50.0, 34.0)),
        **ball_still(range(20, 30), (50.0, 34.0)),
    }
    players = {
        1: near(ball, range(0, 30)),   # shadows the ball; only frames 0-9 matter
        2: {f: (50.0, 34.5) for f in range(0, 30)},
    }

    actions, prov, meta = run(frames, ball, players, {1: 0, 2: 0})

    passes = actions[actions["type_name"] == "pass"]
    assert len(passes) == 1
    row = passes.iloc[0]
    assert row["result_name"] == "success"
    assert row["player_id"] == 1          # the PASSER owns the action
    assert row["team_id"] == 0
    # the pass ENDS at the reception point -- the reception is not its own row
    assert row["end_x"] == pytest.approx(50.0, abs=1.0)
    assert row["end_y"] == pytest.approx(34.0, abs=1.0)
    assert "interception" not in types_of(actions)
    # a reception is implied by result=success, never emitted
    assert len(actions) == len(actions[actions["player_id"].isin([1, 2])])
    assert meta["actions_by_type"]["pass"] == 1


def test_pass_subclassification_uses_attack_dir():
    """"Progressive" is measured toward the passer's OWN target goal, not toward +x.

    Team 1 attacks toward -x, so the same -x ball movement that would be a
    backward pass for team 0 is a progressive one for them. Getting this wrong is
    silent: the coordinates look fine and every xT number is subtly wrong.
    """
    frames = stream(
        hold(1, team=1, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=1, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (60.0, 34.0)),
        **ball_line(range(10, 20), (60.0, 34.0), (40.0, 34.0)),  # toward -x
        **ball_still(range(20, 30), (40.0, 34.0)),
    }
    players = {1: near(ball, range(0, 30)), 2: {f: (40.0, 34.5) for f in range(0, 30)}}

    _actions, prov, _meta = run(frames, ball, players, {1: 1, 2: 1})
    the_pass = prov[prov["kind"] == "pass"].iloc[0]
    assert the_pass["direction"] == "progressive"


# --------------------------------------------------------------------------- #
# a cross-team turnover: interception (ball in flight) and tackle (ball settled)
# --------------------------------------------------------------------------- #
def test_cross_team_ball_in_flight_is_a_failed_pass_plus_an_interception():
    """Both sides of the turnover are emitted, per SPADL convention."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(9, team=1, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (55.0, 34.0)),  # 25 m: in flight
        **ball_still(range(20, 30), (55.0, 34.0)),
    }
    players = {
        1: near(ball, range(0, 30)),
        9: {f: (55.0, 34.5) for f in range(0, 30)},
    }

    actions, prov, meta = run(frames, ball, players, {1: 0, 9: 1})

    assert types_of(actions) == ["pass", "interception"]

    failed, won = actions.iloc[0], actions.iloc[1]
    assert failed["result_name"] == "fail"      # the loser's attempted pass
    assert failed["player_id"] == 1 and failed["team_id"] == 0
    assert won["result_name"] == "success"      # the winner's defensive action
    assert won["player_id"] == 9 and won["team_id"] == 1
    # the failed action comes FIRST, and the interception happens where it was won
    assert failed["time_seconds"] <= won["time_seconds"]
    assert won["start_x"] == pytest.approx(failed["end_x"])

    assert prov[prov["kind"] == "turnover"]["in_flight"].all()


def test_cross_team_settled_ball_is_a_bad_touch_plus_a_tackle():
    """The ball barely moves and changes hands: taken off him, not cut out.

    This is the interception/tackle discriminator. A tackle takes the ball off a
    player who has it (so the ball hardly moves); an interception cuts out a ball
    that was already travelling.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        hold(9, team=1, frames=range(10, 20)),  # no loose phase: hand to hand
    )
    ball = ball_still(range(0, 20), (30.0, 34.0))
    players = {
        1: {f: (30.4, 34.4) for f in range(0, 20)},
        9: {f: (30.6, 34.6) for f in range(0, 20)},
    }

    actions, prov, _meta = run(frames, ball, players, {1: 0, 9: 1})

    assert types_of(actions) == ["bad_touch", "tackle"]
    assert list(actions["result_name"]) == ["fail", "success"]
    assert actions.iloc[0]["player_id"] == 1   # dispossessed
    assert actions.iloc[1]["player_id"] == 9   # won it
    assert not prov[prov["kind"] == "turnover"]["in_flight"].any()


# --------------------------------------------------------------------------- #
# carries: movement is the evidence, not duration
# --------------------------------------------------------------------------- #
def test_same_player_carry_that_progressed_emits_a_dribble():
    """One player, the ball moves 20 m with him -> a dribble."""
    frames = stream(hold(1, team=0, frames=range(0, 40)))
    ball = ball_line(range(0, 40), (30.0, 34.0), (50.0, 34.0))
    players = {1: near(ball, range(0, 40))}

    actions, prov, _meta = run(frames, ball, players, {1: 0})

    assert types_of(actions) == ["dribble"]
    row = actions.iloc[0]
    assert row["result_name"] == "success"
    assert row["player_id"] == 1
    assert row["start_x"] == pytest.approx(30.0, abs=0.6)
    assert row["end_x"] == pytest.approx(50.0, abs=0.6)
    assert prov.iloc[0]["kind"] == "carry"
    # a carry has no loose gap: the frames between its endpoints ARE the carry
    assert prov.iloc[0]["n_gap_frames"] == 0


def test_same_player_static_hold_emits_no_carry():
    """The ball is PARKED next to a player for 4 seconds. That is not a dribble.

    The rule is movement, not duration -- this is the fixture that stops "he had
    it for ages" being mistaken for "he carried it".
    """
    frames = stream(hold(1, team=0, frames=range(0, 100)))
    ball = ball_still(range(0, 100), (30.0, 34.0))
    players = {1: {f: (30.5, 34.5) for f in range(0, 100)}}

    actions, _prov, meta = run(frames, ball, players, {1: 0})

    assert len(actions) == 0
    assert "dribble" not in meta["actions_by_type"]
    assert meta["skipped_by_reason"]["static_hold"] == 1


def test_carry_across_a_short_same_player_gap_is_one_dribble_not_two():
    """Knock the ball ahead, run onto it: one carry, not two touches and a mystery.

    At r_pz = 3 m a driving run puts the ball outside the zone for a few frames.
    Those segments are coalesced into one touch, so this cannot be minted as a
    pass, and the carry spans the whole run.

    Note the emitted geometry is the CARRIER's, not the ball's: he is running 2 m
    behind the ball throughout (that is what a knock-and-chase looks like), so the
    dribble reads 28 -> 46, not 30 -> 48. All emitted geometry is player-anchored
    -- see the module docstring of ``src.actions.geometry`` -- and for a dribble
    that is also what socceraction means by one: the connector between the action
    that fed him and the action he plays next, both of which are at HIS feet.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 18)),                    # 8-frame knock-and-chase
        hold(1, team=0, frames=range(18, 30)),
    )
    ball = ball_line(range(0, 30), (30.0, 34.0), (48.0, 34.0))
    players = {1: near(ball, range(0, 30), offset=(-2.0, 0.5))}

    actions, prov, meta = run(frames, ball, players, {1: 0})

    assert types_of(actions) == ["dribble"]
    assert meta["n_segments"] == 2 and meta["n_touches"] == 1  # coalesced
    assert actions.iloc[0]["start_x"] == pytest.approx(28.0, abs=0.6)  # the PLAYER
    assert actions.iloc[0]["end_x"] == pytest.approx(46.0, abs=0.6)
    # ...but the DRIBBLE was earned by the ball moving, not by the player running.
    assert prov.iloc[0]["ball_travel_m"] == pytest.approx(18.0, abs=0.6)


# --------------------------------------------------------------------------- #
# occlusion: emit, but never pretend we saw it
# --------------------------------------------------------------------------- #
def test_gap_with_a_no_ball_run_is_emitted_but_tagged_occluded():
    """The ball vanishes mid-pass. The pass still happened; we just did not see it.

    ``no_ball`` is occlusion, never a stoppage -- so the transition is emitted on
    the strength of the segments either side, and flagged rather than trusted.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 13)),
        no_ball(range(13, 18)),                  # ball occluded mid-flight
        loose(range(18, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 13), (30.0, 34.0), (36.0, 34.0)),
        **{f: None for f in range(13, 18)},      # ball row present, coords null
        **ball_line(range(18, 20), (46.0, 34.0), (50.0, 34.0)),
        **ball_still(range(20, 30), (50.0, 34.0)),
    }
    players = {
        1: {f: (30.5, 34.5) for f in range(0, 30)},
        2: {f: (50.0, 34.5) for f in range(0, 30)},
    }

    actions, prov, meta = run(frames, ball, players, {1: 0, 2: 0})

    the_pass = prov[prov["kind"] == "pass"]
    assert len(the_pass) == 1
    assert bool(the_pass.iloc[0]["occluded"]) is True
    assert bool(the_pass.iloc[0]["low_confidence"]) is True
    assert the_pass.iloc[0]["n_no_ball_frames"] == 5
    assert the_pass.iloc[0]["confidence"] < 1.0
    assert "occluded_no_ball" in the_pass.iloc[0]["reasons"]
    # ...and it IS still a pass. Occlusion downgrades confidence, not existence.
    assert types_of(actions)[0] == "pass"
    assert meta["n_occluded"] >= 1


def test_endpoint_falls_back_to_the_possessor_when_the_ball_is_occluded():
    """No ball at the release frame -> use the possessor's own position.

    A possessor is within the possession radius of the ball by definition, so the
    fallback's error is bounded by that radius rather than unbounded.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 9), (30.0, 34.0)),
        9: None,                                  # the release frame is occluded
        **ball_line(range(10, 20), (30.0, 34.0), (50.0, 34.0)),
        **ball_still(range(20, 30), (50.0, 34.0)),
    }
    players = {
        1: {f: (29.0, 33.0) for f in range(0, 30)},
        2: {f: (50.0, 34.5) for f in range(0, 30)},
    }

    actions, prov, _meta = run(frames, ball, players, {1: 0, 2: 0})
    row = actions[actions["type_name"] == "pass"].iloc[0]
    assert row["start_x"] == pytest.approx(29.0)   # the PLAYER's position
    assert row["start_y"] == pytest.approx(33.0)
    assert bool(prov[prov["kind"] == "pass"].iloc[0]["occluded"]) is True


# --------------------------------------------------------------------------- #
# crosses
# --------------------------------------------------------------------------- #
def test_delivery_from_a_wide_advanced_area_is_a_cross():
    """Wide + advanced origin -> SPADL ``cross`` rather than ``pass``."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    # team 0 attacks +x: x=95 is deep, y=5 is against the touchline
    ball = {
        **ball_still(range(0, 10), (95.0, 5.0)),
        **ball_line(range(10, 20), (95.0, 5.0), (95.0, 34.0)),
        **ball_still(range(20, 30), (95.0, 34.0)),
    }
    players = {
        1: {f: (95.0, 5.5) for f in range(0, 30)},
        2: {f: (95.0, 34.5) for f in range(0, 30)},
    }

    actions, _prov, _meta = run(frames, ball, players, {1: 0, 2: 0})
    assert types_of(actions) == ["cross"]
    assert actions.iloc[0]["result_name"] == "success"


def test_central_delivery_from_the_same_depth_is_a_plain_pass():
    """The cross rule is about being WIDE, not just advanced."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (95.0, 34.0)),   # advanced but central
        **ball_line(range(10, 20), (95.0, 34.0), (95.0, 50.0)),
        **ball_still(range(20, 30), (95.0, 50.0)),
    }
    players = {
        1: {f: (95.0, 34.5) for f in range(0, 30)},
        2: {f: (95.0, 50.5) for f in range(0, 30)},
    }
    actions, _prov, _meta = run(frames, ball, players, {1: 0, 2: 0})
    assert types_of(actions) == ["pass"]


# --------------------------------------------------------------------------- #
# the gap guards
# --------------------------------------------------------------------------- #
def test_a_one_frame_zone_blip_does_not_become_a_pass():
    """The ball nudges past the zone radius and back. Nothing happened.

    Without the guards this is the failure mode that mints phantom passes out of
    the 37% of ball-frames that are ``loose``.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose([10]),                              # one frame, ball goes nowhere
        hold(1, team=0, frames=range(11, 20)),
    )
    ball = ball_still(range(0, 20), (30.0, 34.0))
    players = {1: {f: (30.5, 34.5) for f in range(0, 20)}}

    actions, _prov, meta = run(frames, ball, players, {1: 0})

    assert len(actions) == 0                      # not a pass, and not a carry
    assert meta["n_segments"] == 2
    assert meta["n_touches"] == 1                 # the blip was coalesced away


def test_a_possessor_flicker_between_two_close_players_is_refused():
    """Two team-mates stand together, the zone flickers, the ball never moves.

    Credible transitions need the ball to have gone somewhere, OR a long enough
    loose phase, OR hand-to-hand contact. This has none of the three.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose([10]),                              # 1 frame < min_gap_frames (2)
        hold(2, team=0, frames=range(11, 20)),    # ...and the ball is still there
    )
    ball = ball_still(range(0, 20), (30.0, 34.0))
    players = {
        1: {f: (30.4, 34.0) for f in range(0, 20)},
        2: {f: (30.6, 34.0) for f in range(0, 20)},
    }

    actions, _prov, meta = run(frames, ball, players, {1: 0, 2: 0})

    assert len(actions) == 0
    assert meta["skipped_by_reason"]["spurious"] == 1


def test_guards_are_configurable():
    """The same flicker becomes a pass once the guards are relaxed to allow it."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose([10]),
        hold(2, team=0, frames=range(11, 20)),
    )
    ball = ball_still(range(0, 20), (30.0, 34.0))
    players = {
        1: {f: (30.4, 34.0) for f in range(0, 20)},
        2: {f: (30.6, 34.0) for f in range(0, 20)},
    }

    actions, _prov, _meta = run(
        frames, ball, players, {1: 0, 2: 0},
        min_gap_frames=1, min_ball_travel_m=0.0,
    )
    assert types_of(actions) == ["pass"]


def test_an_incoherent_ball_path_is_flagged_low_confidence():
    """A ball that ricochets rather than travels is a deflection, not a delivery."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    # out 15 m and back again: long polyline, short displacement -> low coherence
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 15), (30.0, 34.0), (30.0, 49.0)),
        **ball_line(range(15, 20), (30.0, 49.0), (33.0, 36.0)),
        **ball_still(range(20, 30), (33.0, 36.0)),
    }
    players = {
        1: {f: (30.5, 34.5) for f in range(0, 30)},
        2: {f: (33.0, 36.5) for f in range(0, 30)},
    }

    _actions, prov, _meta = run(frames, ball, players, {1: 0, 2: 0})
    row = prov[prov["kind"] == "pass"].iloc[0]
    assert row["path_coherence"] < 0.5
    assert bool(row["low_confidence"]) is True
    assert "incoherent_ball_path" in row["reasons"]


# --------------------------------------------------------------------------- #
# duels are seen, flagged, and NOT resolved (that is a later milestone)
# --------------------------------------------------------------------------- #
def test_a_fleeting_contested_touch_is_flagged_as_an_unresolved_duel():
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        contested(9, team=1, frames=[10]),        # 1 contested frame
        hold(1, team=0, frames=range(11, 20)),
    )
    ball = ball_still(range(0, 20), (30.0, 34.0))
    players = {
        1: {f: (30.4, 34.0) for f in range(0, 20)},
        9: {f: (30.6, 34.0) for f in range(0, 20)},
    }

    _actions, prov, meta = run(frames, ball, players, {1: 0, 9: 1})

    assert meta["n_duel_candidates"] > 0
    duels = prov[prov["duel_candidate"]]
    assert set(duels["kind"]) == {"turnover"}
    assert (duels["confidence"] < 1.0).all()


# --------------------------------------------------------------------------- #
# degenerate inputs
# --------------------------------------------------------------------------- #
def test_empty_possession_stream_emits_an_empty_but_valid_table():
    frames = stream(loose(range(0, 20)))
    ball = ball_still(range(0, 20), (30.0, 34.0))

    actions, prov, meta = run(frames, ball, {}, {})

    assert len(actions) == 0
    assert list(actions.columns) == SPADL_COLUMNS   # still the right SHAPE
    assert len(prov) == 0
    assert meta["n_actions"] == 0 and meta["n_touches"] == 0


def test_single_segment_clip_emits_at_most_its_own_carry():
    """One touch has no gap to terminate it, so nothing but its carry can exist."""
    frames = stream(hold(1, team=0, frames=range(0, 40)))
    ball = ball_line(range(0, 40), (30.0, 34.0), (45.0, 34.0))
    players = {1: near(ball, range(0, 40))}

    actions, _prov, _meta = run(frames, ball, players, {1: 0})
    assert types_of(actions) == ["dribble"]


def test_a_possessor_with_an_unknown_team_is_refused_not_guessed():
    """Without B's team, "pass or turnover?" is a coin flip. Refuse, don't guess.

    The two answers have opposite meanings for a possession chain, and ``team_id``
    is what xT's left-to-right normalization keys off -- so a guess here does not
    degrade the output, it silently corrupts it.
    """
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=None, frames=range(20, 30)),   # team never resolved
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (50.0, 34.0)),
        **ball_still(range(20, 30), (50.0, 34.0)),
    }
    players = {
        1: near(ball, range(0, 30)),
        2: {f: (50.0, 34.5) for f in range(0, 30)},
    }

    actions, _prov, meta = run(frames, ball, players, {1: 0, 2: 0})

    assert "pass" not in meta["actions_by_type"]
    assert meta["skipped_by_reason"]["unknown_team"] >= 1
    # ...and no action is ever emitted with a sentinel team
    assert (actions["team_id"] >= 0).all()


def test_a_possessor_with_no_geometry_anywhere_is_refused_not_crashed():
    """Null coords everywhere: refuse the transition, do not invent one."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    ball = {f: None for f in range(0, 30)}         # ball never usable
    players = {}                                   # ...and no player positions

    actions, _prov, meta = run(frames, ball, players, {})
    assert len(actions) == 0
    assert meta["skipped_by_reason"]["no_geometry"] >= 1


# --------------------------------------------------------------------------- #
# scope + output contract
# --------------------------------------------------------------------------- #
def test_only_the_milestone_1_action_types_are_ever_emitted():
    """Shots, set pieces and duels must not appear. The pipeline enforces it."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(9, team=1, frames=range(20, 30)),
        loose(range(30, 40)),
        hold(2, team=0, frames=range(40, 50)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (55.0, 34.0)),
        **ball_still(range(20, 30), (55.0, 34.0)),
        **ball_line(range(30, 40), (55.0, 34.0), (75.0, 20.0)),
        **ball_still(range(40, 50), (75.0, 20.0)),
    }
    players = {
        1: near(ball, range(0, 50)),
        9: {f: (55.0, 34.5) for f in range(0, 50)},
        2: {f: (75.0, 20.5) for f in range(0, 50)},
    }

    actions, _prov, meta = run(frames, ball, players, {1: 0, 9: 1, 2: 0})

    assert set(actions["type_name"]) <= set(EMITTED_ACTIONTYPES)
    for banned in ("shot", "throw_in", "corner_short", "foul", "take_on"):
        assert banned not in meta["actions_by_type"]


def test_actions_are_time_ordered_with_contiguous_ids():
    frames = stream(
        hold(1, team=0, frames=range(0, 20)),
        loose(range(20, 30)),
        hold(9, team=1, frames=range(30, 50)),
    )
    ball = {
        **ball_line(range(0, 20), (20.0, 34.0), (35.0, 34.0)),
        **ball_line(range(20, 30), (35.0, 34.0), (60.0, 34.0)),
        **ball_still(range(30, 50), (60.0, 34.0)),
    }
    players = {
        1: near(ball, range(0, 50)),
        9: {f: (60.0, 34.5) for f in range(0, 50)},
    }

    actions, prov, _meta = run(frames, ball, players, {1: 0, 9: 1})

    assert actions["time_seconds"].is_monotonic_increasing
    assert list(actions["action_id"]) == list(range(len(actions)))
    assert list(prov["action_id"]) == list(actions["action_id"])
    # the carry precedes the pass that ends its touch
    assert types_of(actions)[0] == "dribble"


def test_coordinates_are_clipped_into_the_spadl_pitch():
    """The homography puts players off the pitch; SPADL forbids it. Clip, don't crash."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, -2.5)),     # 2.5 m off the touchline
        **ball_line(range(10, 20), (30.0, -2.5), (50.0, 70.9)),
        **ball_still(range(20, 30), (50.0, 70.9)),    # 2.9 m off the other one
    }
    players = {
        1: {f: (30.0, -2.0) for f in range(0, 30)},
        2: {f: (50.0, 70.0) for f in range(0, 30)},
    }

    actions, _prov, _meta = run(frames, ball, players, {1: 0, 2: 0})
    row = actions.iloc[0]
    assert row["start_y"] == 0.0            # clipped up to the touchline
    assert row["end_y"] == 68.0             # clipped down to the other one
    assert 0.0 <= row["start_x"] <= 105.0


def test_bodypart_defaults_to_foot_and_is_not_a_measurement():
    """No pose data => the bodypart is not observable. Documented, not invented."""
    frames = stream(
        hold(1, team=0, frames=range(0, 10)),
        loose(range(10, 20)),
        hold(2, team=0, frames=range(20, 30)),
    )
    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (50.0, 34.0)),
        **ball_still(range(20, 30), (50.0, 34.0)),
    }
    players = {
        1: near(ball, range(0, 30)),
        2: {f: (50.0, 34.5) for f in range(0, 30)},
    }

    actions, _prov, meta = run(frames, ball, players, {1: 0, 2: 0})
    assert set(actions["bodypart_name"]) == {"foot"}
    assert "not observable" in meta["limitations"]["bodypart"]


# --------------------------------------------------------------------------- #
# the possession source is an INTERFACE, not the zone detector
# --------------------------------------------------------------------------- #
def test_a_custom_possession_source_drives_the_layer_without_the_zone_detector():
    """A source only has to yield ``(frame, time, possessor, team, state)``.

    This is the seam a ball-free (PathCRF-style) possession model plugs into. If
    this test ever needs to know about ``possession_frames.parquet``, the layer
    has been coupled to the zone detector and the abstraction has failed.
    """
    from src.actions.source import PossessionFrame, PossessionSource

    class FakeSource(PossessionSource):
        """A possession model that owes nothing to the zone detector."""

        def stream(self):
            for f in range(0, 10):
                yield PossessionFrame(f, f / 25.0, 1, 0, "possession")
            for f in range(10, 20):
                yield PossessionFrame(f, f / 25.0, None, None, "loose")
            for f in range(20, 30):
                yield PossessionFrame(f, f / 25.0, 2, 0, "possession")

    ball = {
        **ball_still(range(0, 10), (30.0, 34.0)),
        **ball_line(range(10, 20), (30.0, 34.0), (50.0, 34.0)),
        **ball_still(range(20, 30), (50.0, 34.0)),
    }
    players = {
        1: near(ball, range(0, 30)),
        2: {f: (50.0, 34.5) for f in range(0, 30)},
    }

    actions, _prov, meta = detect_actions(
        FakeSource(), tracking(ball, players, {1: 0, 2: 0}), config()
    )
    assert types_of(actions) == ["pass"]
    assert meta["n_segments"] == 2


def test_segments_are_derived_from_the_stream_not_delegated():
    """Segments come from the stream, so every source gets the same definition."""
    from src.actions.source import PossessionFrame

    frames = [
        PossessionFrame(0, 0.0, 1, 0, "possession"),
        PossessionFrame(1, 0.04, 1, 0, "contested"),
        PossessionFrame(2, 0.08, None, None, "loose"),    # breaks the run
        PossessionFrame(3, 0.12, 1, 0, "possession"),     # ...so this is a NEW segment
        PossessionFrame(5, 0.20, 1, 0, "possession"),     # frame jump: another one
    ]
    segs = segments_from_stream(frames)

    assert len(segs) == 3
    assert list(segs["start_frame"]) == [0, 3, 5]
    assert list(segs["n_frames"]) == [2, 1, 1]
    assert segs.iloc[0]["n_contested"] == 1
