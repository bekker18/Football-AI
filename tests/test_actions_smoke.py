"""Smoke test: run the event layer on the real prepared clip.

Skips cleanly if the sample possession outputs or the tabular deps aren't
present, so the pure-logic tests still run in a minimal environment.

The assertions are deliberately STRUCTURAL, not calibration numbers. Clip
2e57b9_0 currently yields 23 segments -> 18 touches -> 33 actions; pinning those
exactly would make this a change-detector for the upstream possession stage
rather than a check on this one. What must hold regardless of what the footage
does are the contracts:

- the table loads through socceraction's own ``SPADLSchema`` without modification
- it is time-ordered, with contiguous action ids
- it contains ONLY milestone-1 action types (no shots, no set pieces)
- the chain is spatially coherent: an action ends where the next one starts,
  except exactly where we refused to name a transition
- socceraction's xT ingests it, and rates exactly the actions it should

Note xT cannot be *fitted* on this clip: its value surface comes from shots, and
milestone 1 emits none. That is asserted explicitly rather than worked around --
see ``test_xt_cannot_be_FITTED_on_a_shotless_clip_and_that_is_expected``.
"""

import json
import os

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from src.actions import ZonePossessionSource, config_from_prep_meta, detect_actions  # noqa: E402
from src.actions.spadl import EMITTED_ACTIONTYPES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMESTATE = os.path.join(REPO, "data", "gamestate")
PREPARED = os.path.join(GAMESTATE, "tracking_prepared.parquet")
POSSESSION = os.path.join(GAMESTATE, "possession_frames.parquet")
PREP_META = os.path.join(GAMESTATE, "prep_meta.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(PREPARED) and os.path.exists(POSSESSION)),
    reason="sample clip not present; run the prerequisites + possession stages first",
)


@pytest.fixture(scope="module")
def clip():
    """``(actions, provenance, meta)`` for the real clip."""
    tracking = pd.read_parquet(PREPARED)
    source = ZonePossessionSource.from_dir(GAMESTATE)
    with open(PREP_META, "r", encoding="utf-8") as f:
        prep_meta = json.load(f)
    cfg = config_from_prep_meta(prep_meta)
    return detect_actions(source, tracking, cfg)


@pytest.fixture(scope="module")
def stages():
    """Every intermediate stage for the real clip -- incl. the aerial annotation."""
    from src.actions import run_stages

    tracking = pd.read_parquet(PREPARED)
    source = ZonePossessionSource.from_dir(GAMESTATE)
    with open(PREP_META, "r", encoding="utf-8") as f:
        prep_meta = json.load(f)
    return run_stages(source, tracking, config_from_prep_meta(prep_meta))


# --------------------------------------------------------------------------- #
# the aerial pass at frames 123-150 -- the grounded case this feature was built on
# --------------------------------------------------------------------------- #
#
# What is in the footage, measured: the ball is hoofed at ~frame 102 and comes down
# at ~frame 166. Across it, img_y falls 457 -> ~325 and climbs back to ~449 (image-y
# DECREASES as the ball rises, so the flight is a local MINIMUM in img_y), the
# apparent ground speed sits at 25-30 m/s, and the ball is attributed to nobody
# throughout. Frame 136 carries one bad detection (img_y=449 amid ~325).
#
# The detail that proves the root cause rather than merely asserting it: frames
# 104-122 and 150-165 are ALL `ball_outlier=True` in the prepared table. The
# prerequisites' speed gate threw them out as physically impossible -- and it was
# right to, given what it could see. Their ground coordinates ARE impossible,
# *because the ball was in the air* and the z=0 homography stretched its
# back-projection away from the camera. The outlier flag is a symptom of the height
# problem, which is exactly why it cannot be used to diagnose it.
AERIAL_PASS = range(123, 151)   # the brief's frames 123-150, inclusive


def test_the_aerial_pass_at_frames_123_150_is_flagged_airborne(stages):
    """The grounded real-clip case. If this goes red, the detector is not working."""
    ann = stages.aerial.to_frame().set_index("frame")

    flagged = ann.loc[ann.index.isin(AERIAL_PASS), "airborne"]
    assert len(flagged) == len(AERIAL_PASS)   # every frame has a ball row
    assert bool(flagged.all()), "frames 123-150 are a 1.1 s aerial pass"

    conf = ann.loc[ann.index.isin(AERIAL_PASS), "aerial_conf"]
    assert (conf > 0.5).all()                 # a full arc, not a partial one

    run = next(
        r for r in stages.aerial.runs
        if r.airborne and r.start_frame <= 123 and r.end_frame >= 150
    )
    assert run.kind == "arc"
    assert run.curvature > 0                  # upward-opening: a MINIMUM in img_y
    assert run.r2 > 0.9
    assert 123 <= run.vertex_frame <= 150     # the apex was OBSERVED
    assert run.median_speed_ms > 20           # elevated, and smooth
    assert run.n_rejected >= 1                # the frame-136 glitch was thrown out
    # ...and the ground gate had already binned most of this pass as "impossible".
    # It is not a contradiction; it is the same fact, seen from the ground plane.
    assert run.n_ball_outlier > 0


def test_the_pass_over_the_aerial_gap_takes_its_endpoints_from_the_players(stages):
    """CHANGE 1, on the real clip: no mid-flight ball coordinate sets the geometry.

    The ball's own pitch position through this gap is a z=0 artefact -- it is the
    reason frames 104-122 and 150-165 were rejected as impossible in the first
    place. The emitted pass must be anchored on the passer and the receiver.
    """
    from src.actions.config import COL_PITCH_X_T, COL_PITCH_Y_T, COL_STABLE_ID

    merged = stages.actions.merge(stages.provenance, on="action_id")
    spanning = merged[
        (merged["start_frame"] <= 123) & (merged["end_frame"] >= 150)
    ]
    assert len(spanning) == 1, "one action spans the aerial pass"
    row = spanning.iloc[0]

    assert row["type_name"] == "pass"      # subtyped, not RE-typed: SPADL has no aerial
    assert bool(row["aerial"]) is True
    assert row["aerial_conf"] > 0.5

    # the emitted endpoints ARE the two players' own positions, to the metre
    tracking = pd.read_parquet(PREPARED)
    people = tracking.dropna(subset=[COL_STABLE_ID, COL_PITCH_X_T, COL_PITCH_Y_T])

    def at(stable_id, frame):
        hit = people[
            (people[COL_STABLE_ID] == stable_id) & (people["frame"] == frame)
        ]
        return float(hit.iloc[0][COL_PITCH_X_T]), float(hit.iloc[0][COL_PITCH_Y_T])

    passer = at(row["player_id"], int(row["start_frame"]))
    assert row["start_x"] == pytest.approx(passer[0], abs=0.01)
    assert row["start_y"] == pytest.approx(passer[1], abs=0.01)

    # ...and the receiver's carry begins at EXACTLY the point the pass ended at.
    nxt = merged[merged["action_id"] == row["action_id"] + 1].iloc[0]
    assert nxt["start_x"] == pytest.approx(row["end_x"])
    assert nxt["start_y"] == pytest.approx(row["end_y"])


def test_the_aerial_pass_is_not_dropped_by_the_ground_path_coherence_guard(stages):
    """Whatever its ground path looks like, the flight must not be held against it."""
    merged = stages.actions.merge(stages.provenance, on="action_id")
    aerial = merged[merged["aerial"].astype(bool)]
    assert len(aerial) > 0

    # the guard is bypassed for these gaps, so none of them may carry its reason --
    # a bent ground projection is the DISTORTION, not evidence of a bad pass.
    for row in aerial.itertuples():
        assert "incoherent_ball_path" not in row.reasons
        assert "aerial" in row.reasons

    # ...and they are emitted, not refused. An aerial gap is a ball in flight, so
    # the only things it can come out as are a delivery (pass / cross) or a
    # delivery that got cut out -- which SPADL splits into the failed pass AND the
    # interception that answers it, and BOTH rows are aerial, because both describe
    # the same flighted ball.
    assert set(aerial["type_name"]) <= {"pass", "cross", "interception"}
    assert "bad_touch" not in set(aerial["type_name"])   # a settled ball, by definition
    assert "tackle" not in set(aerial["type_name"])


def test_the_meta_reports_the_aerial_counts_and_the_thresholds(stages):
    """The scorecard has to show the aerial passes and the knobs that found them."""
    meta = stages.meta
    aer = meta["aerial"]

    assert aer["enabled"] is True
    assert aer["n_aerial_runs"] >= 1
    assert aer["n_full_arcs"] >= 1
    assert meta["n_aerial_passes"] >= 1
    assert meta["n_aerial_actions"] == (
        meta["n_aerial_passes"] + meta["n_aerial_crosses"]
        + sum(
            v for k, v in meta["aerial_actions_by_type"].items()
            if k not in ("pass", "cross")
        )
    )
    for knob in ("aerial_min_curvature", "aerial_min_r2", "aerial_min_run_frames",
                 "aerial_min_speed_ms"):
        assert knob in aer["thresholds"]
    # the honesty clause: this is a heuristic, and the meta says so out loud
    assert "HEURISTIC" in aer["note"]
    assert "z=0" in meta["limitations"]["ball_height"]
    assert "EXTENSION POINT" in meta["limitations"]["ball_height"]


def test_the_upstream_outputs_were_not_touched(stages):
    """This layer ADDS a ball annotation. It rewrites nothing.

    The airborne flag lives in its own sidecar (``ball_aerial.parquet``, joinable on
    ``frame``) precisely so the prerequisite and possession stages' outputs stay
    exactly as they were -- see the EXTENSION POINT in ``src/actions/cli.py`` for
    why feeding it back into the ball smoother is a pipeline change, not a feature.
    """
    tracking = pd.read_parquet(PREPARED)
    assert "airborne" not in tracking.columns
    assert "aerial_conf" not in tracking.columns

    ann = stages.aerial.to_frame()
    assert list(ann.columns) == ["frame", "airborne", "aerial_conf"]
    # one row per ball frame, and it joins cleanly onto the prepared ball track
    ball_frames = set(tracking.loc[tracking["object_id"] == 0, "frame"].astype(int))
    assert set(ann["frame"]) == ball_frames


def test_the_clip_produces_a_coherent_chain(clip):
    actions, prov, meta = clip

    # ~23 possession segments in, a real chain out. Loose bounds on purpose.
    assert 15 <= meta["n_segments"] <= 30
    assert meta["n_touches"] <= meta["n_segments"]      # coalescing never invents
    assert 15 <= len(actions) <= 60

    assert actions["time_seconds"].is_monotonic_increasing
    assert list(actions["action_id"]) == list(range(len(actions)))
    assert len(prov) == len(actions)
    assert list(prov["action_id"]) == list(actions["action_id"])

    # a real chain has all three of the milestone's event families in it
    kinds = set(prov["kind"])
    assert {"pass", "carry", "turnover"} <= kinds


def test_only_milestone_1_action_types_are_emitted(clip):
    """Shots, set pieces and duel resolution are OUT OF SCOPE and must not appear."""
    actions, _prov, _meta = clip
    assert set(actions["type_name"]) <= set(EMITTED_ACTIONTYPES)


def test_every_turnover_emits_both_sides(clip):
    """SPADL convention: the loser's failed action AND the winner's defensive one."""
    actions, prov, _meta = clip
    merged = actions.merge(prov[["action_id", "kind"]], on="action_id")
    turnovers = merged[merged["kind"] == "turnover"]

    losers = turnovers[turnovers["result_name"] == "fail"]
    winners = turnovers[turnovers["result_name"] == "success"]
    assert len(losers) == len(winners) > 0
    assert set(losers["type_name"]) <= {"pass", "bad_touch"}
    assert set(winners["type_name"]) <= {"interception", "tackle"}

    # the two sides of a turnover belong to DIFFERENT teams -- that is the point
    for (l, w) in zip(losers.itertuples(), winners.itertuples()):
        assert l.team_id != w.team_id


def test_the_chain_is_spatially_continuous_where_we_did_not_refuse_a_gap(clip):
    """A pass ends where the receiver's carry starts, and so on down the chain.

    xT reads exactly those start->end deltas, so a chain full of teleports would
    produce numbers that look fine and mean nothing. The tolerance is generous:
    the possession radius is 3 m, so endpoints legitimately differ by that much
    when a touch was static (no carry emitted to bridge it).
    """
    actions, prov, _meta = clip
    merged = actions.merge(prov[["action_id", "kind"]], on="action_id")

    jumps = []
    for a, b in zip(merged.itertuples(), merged.iloc[1:].itertuples()):
        gap = ((b.start_x - a.end_x) ** 2 + (b.start_y - a.end_y) ** 2) ** 0.5
        jumps.append(gap)

    # the vast majority of consecutive actions are joined end-to-start
    joined = sum(1 for j in jumps if j <= 3.0)
    assert joined / len(jumps) >= 0.75


def test_occlusion_is_flagged_rather_than_hidden(clip):
    """The clip has ~71 no_ball frames. Actions spanning them must say so."""
    _actions, prov, meta = clip
    assert meta["n_no_ball_frames"] > 0
    # nothing is silently trusted: every occluded action is also low-confidence
    occluded = prov[prov["occluded"]]
    assert bool(occluded["low_confidence"].all())
    assert (occluded["confidence"] < 1.0).all()
    assert 0.0 <= meta["occluded_pct"] <= 100.0


def test_the_meta_reports_the_guards_and_the_refusals(clip):
    """The scorecard has to show what was NOT emitted, or it is not a scorecard."""
    _actions, _prov, meta = clip
    for guard in ("bridge_max_gap_frames", "min_gap_frames", "min_ball_travel_m",
                  "min_path_coherence", "flight_min_travel_m", "min_carry_m"):
        assert guard in meta["guards"]
    assert "n_gaps_skipped" in meta
    assert isinstance(meta["skipped_by_reason"], dict)
    assert meta["home_team_id"] is not None      # needed for play_left_to_right


def test_segments_agree_with_the_upstream_possession_stage(clip):
    """Our stream-derived segments match ``possession_segments.parquet``.

    We derive segments from the possessor STREAM rather than reading the upstream
    segments table (that is what keeps the possession source swappable). This
    test is what makes that safe: the two definitions must not have drifted apart.
    """
    upstream_path = os.path.join(GAMESTATE, "possession_segments.parquet")
    if not os.path.exists(upstream_path):
        pytest.skip("possession_segments.parquet not present")

    upstream = pd.read_parquet(upstream_path)
    ours = ZonePossessionSource.from_dir(GAMESTATE).segments()

    assert len(ours) == len(upstream)
    assert list(ours["start_frame"]) == list(upstream["start_frame"])
    assert list(ours["end_frame"]) == list(upstream["end_frame"])
    assert list(ours["possessor_id"]) == list(upstream["possessor_id"])


# --------------------------------------------------------------------------- #
# the real contract: socceraction has to be able to load and value this
# --------------------------------------------------------------------------- #
def test_socceraction_loads_the_actions_without_modification(clip):
    """``SPADLSchema.validate`` on the emitted table, unmodified. This is the DoD."""
    pytest.importorskip(
        "socceraction", reason='pip install -e ".[spadl]"'
    )
    from socceraction.spadl.schema import SPADLSchema

    actions, _prov, _meta = clip
    validated = SPADLSchema.validate(actions)   # strict=True: extra columns fail
    assert len(validated) == len(actions)


def test_socceractions_xt_consumes_the_stream(clip):
    """xT's ``fit`` ingests the actions and builds non-degenerate move matrices.

    This is the end-to-end well-formedness check. ``ExpectedThreat.fit`` validates
    the frame against ``SPADLSchema``, bins every action into its grid cell, and
    builds a move-transition matrix out of the successful moves it recognises. If
    our action types, results, or coordinates were wrong, those matrices would come
    out empty -- so a non-empty one IS the evidence that socceraction understood
    the stream as passes, crosses and dribbles on a 105x68 pitch.
    """
    pytest.importorskip("socceraction", reason='pip install -e ".[spadl]"')
    import numpy as np
    import socceraction.spadl as spadl_lib
    import socceraction.xthreat as xthreat

    actions, _prov, meta = clip
    ltr = spadl_lib.play_left_to_right(actions, meta["home_team_id"])

    model = xthreat.ExpectedThreat(l=16, w=12)
    model.fit(ltr)

    assert np.nansum(model.move_prob_matrix) > 0
    assert np.nansum(model.transition_matrix) > 0


def test_xt_cannot_be_FITTED_on_a_shotless_clip_and_that_is_expected(clip):
    """Fitting xT on our own clip yields an all-zero grid. This is not a bug.

    xT's value surface is ``P(score | cell)``, and socceraction estimates it from
    the SHOTS in the stream. Milestone 1 emits no shots *by design*, so the scoring
    probability is 0 everywhere and the resulting grid is degenerate -- ``rate()``
    on it raises ``NotFittedError``.

    Pinned deliberately, so that when the shot detector lands and this test starts
    failing, that failure is the signal that xT has become self-fittable -- rather
    than someone rediscovering the zero grid from scratch and assuming the action
    stream is broken.
    """
    pytest.importorskip("socceraction", reason='pip install -e ".[spadl]"')
    import numpy as np
    import socceraction.spadl as spadl_lib
    import socceraction.xthreat as xthreat

    actions, _prov, meta = clip
    ltr = spadl_lib.play_left_to_right(actions, meta["home_team_id"])
    assert (ltr["type_name"] == "shot").sum() == 0     # the cause

    model = xthreat.ExpectedThreat(l=16, w=12)
    model.fit(ltr)
    assert not np.any(model.xT)                        # ...and the effect


def test_xt_rate_values_the_stream_when_given_a_real_value_surface(clip):
    """``rate()`` runs end to end and values exactly the actions it should.

    Since the clip cannot fit its own value surface (see the test above), we hand
    xT a stand-in grid -- a plain "closer to the opponent's goal is worth more"
    surface. The GRID is a stand-in; the RATING PATH is socceraction's real one:
    it re-validates the schema, maps every start/end coordinate into a grid cell,
    and picks out the actions it considers ball-progressing.

    The assertion that matters is the NaN split. xT rates successful passes,
    crosses and dribbles and nothing else, so if socceraction agrees with us about
    which of our rows those are, our type/result vocabulary is correct.
    """
    pytest.importorskip("socceraction", reason='pip install -e ".[spadl]"')
    import numpy as np
    import socceraction.spadl as spadl_lib
    import socceraction.xthreat as xthreat

    actions, _prov, meta = clip
    ltr = spadl_lib.play_left_to_right(actions, meta["home_team_id"])

    model = xthreat.ExpectedThreat(l=16, w=12)
    model.fit(ltr)
    # Stand-in surface: value rises toward the attacking goal (grid col 15).
    model.xT = np.tile(np.linspace(0.0, 0.3, 16), (12, 1))

    ratings = model.rate(ltr)
    assert len(ratings) == len(actions)

    moves = ltr["type_name"].isin(["pass", "cross", "dribble"]) & (
        ltr["result_name"] == "success"
    )
    rated = pd.Series(ratings).notna()
    assert (rated == moves).all()       # socceraction agrees on what a move is
    assert rated.sum() > 0
    assert np.isfinite(ratings[rated.to_numpy()]).all()
