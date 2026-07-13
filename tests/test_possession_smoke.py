"""Smoke test: run the possession detector on the real prepared clip.

Skips cleanly if the sample prepared file or the tabular deps aren't present, so
the pure-logic tests still run in a minimal environment.

The assertions are deliberately loose RANGES, not the exact calibration numbers.
Clip 2e57b9_0 at R_pz=3.0 m measures ~63% coverage / ~98% clean / ~2% duel / 23
segments; pinning those exactly would make the test a change-detector for the
upstream prerequisite stage rather than a check on this one. What must hold no
matter what the footage does are the invariants: no possessor on a loose or
no_ball frame, and coverage that cannot exceed ball presence.
"""

import os

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from src.possession import (  # noqa: E402
    PossessionConfig,
    config_from_prep_meta,
    detect_possession,
    radius_grid,
    sweep_radii,
)
from src.possession.config import (  # noqa: E402
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMESTATE = os.path.join(REPO, "data", "gamestate")
PREPARED = os.path.join(GAMESTATE, "tracking_prepared.parquet")


@pytest.fixture(scope="module")
def detected():
    if not os.path.exists(PREPARED):
        pytest.skip("no sample tracking_prepared.parquet in data/gamestate")
    df = pd.read_parquet(PREPARED)
    import json
    prep_meta = {}
    meta_path = os.path.join(GAMESTATE, "prep_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            prep_meta = json.load(f)
    cfg = config_from_prep_meta(prep_meta, r_pz_m=3.0)
    frames, segments, meta = detect_possession(df, cfg)
    return df, frames, segments, meta, cfg


def test_config_picks_up_the_target_pitch_frame(detected):
    *_, cfg = detected
    assert cfg.r_pz_m == 3.0
    assert (cfg.pitch_length_m, cfg.pitch_width_m) == (105.0, 68.0)


def test_one_row_per_frame(detected):
    df, frames, *_ = detected
    assert len(frames) == df["frame"].nunique()
    assert frames["frame"].is_unique


def test_summary_lands_in_sane_ranges(detected):
    *_, meta, _ = detected
    # ~63% coverage / ~98% clean / ~2% duel on this clip at 3.0 m
    assert 50.0 <= meta["coverage_pct"] <= 75.0
    assert meta["clean_pct"] >= 90.0
    assert meta["duel_pct"] <= 10.0
    assert meta["clean_pct"] + meta["duel_pct"] == pytest.approx(100.0, abs=0.05)
    # ~23 touches in this 30 s attacking phase
    assert 10 <= meta["n_segments"] <= 50
    assert meta["median_hold_frames"] > 0


def test_coverage_cannot_exceed_ball_presence(detected):
    *_, meta, _ = detected
    # the ceiling is ball presence (~90.5% of frames on this clip), not 100%
    assert meta["n_attributed_frames"] <= meta["n_ball_frames"] <= meta["n_frames"]
    assert meta["coverage_all_frames_pct"] <= meta["ball_presence_pct"]
    assert 85.0 <= meta["ball_presence_pct"] <= 95.0


def test_no_possessor_on_loose_or_no_ball_frames(detected):
    _, frames, *_ = detected
    unattributed = frames[frames["state"].isin([STATE_LOOSE, STATE_NO_BALL])]
    assert len(unattributed) > 0                      # the clip really has them
    assert unattributed["possessor_id"].isna().all()  # ...and none is fabricated
    assert unattributed["possessor_team"].isna().all()


def test_every_attributed_frame_has_a_possessor(detected):
    _, frames, *_ = detected
    attributed = frames[frames["state"].isin([STATE_POSSESSION, STATE_CONTESTED])]
    assert attributed["possessor_id"].notna().all()
    assert attributed["dist_m"].notna().all()
    # the zone is what it says it is: every possessor is inside R_pz
    assert (attributed["dist_m"] <= 3.0 + 1e-9).all()
    assert (frames.loc[frames["state"] == STATE_POSSESSION, "n_in_zone"] == 1).all()
    assert (frames.loc[frames["state"] == STATE_CONTESTED, "n_in_zone"] >= 2).all()


def test_no_ball_frames_have_no_distance(detected):
    _, frames, *_ = detected
    no_ball = frames[frames["state"] == STATE_NO_BALL]
    assert no_ball["dist_m"].isna().all()
    assert (no_ball["n_in_zone"] == 0).all()


def test_possessors_are_stable_ids_of_real_candidates(detected):
    df, frames, *_ = detected
    valid = set(
        df[df["role"].isin(("player", "goalkeeper"))]["stable_id"].dropna().tolist()
    )
    got = set(frames["possessor_id"].dropna().tolist())
    assert got <= valid          # never a referee, never the ball, never invented
    assert len(got) > 1          # the clip changes hands


def test_segments_are_contiguous_and_consistent(detected):
    _, frames, segments, *_ = detected
    assert len(segments) > 0
    assert (segments["n_frames"] ==
            segments["end_frame"] - segments["start_frame"] + 1).all()
    assert (segments["n_frames"] > 0).all()
    assert (segments["n_contested"] <= segments["n_frames"]).all()
    # segments partition exactly the attributed frames, and never overlap
    covered = sum(int(n) for n in segments["n_frames"])
    attributed = int(frames["possessor_id"].notna().sum())
    assert covered == attributed
    ordered = segments.sort_values("start_frame")
    assert (ordered["start_frame"].to_numpy()[1:]
            > ordered["end_frame"].to_numpy()[:-1]).all()


def test_sweep_reproduces_the_calibration_trend(detected):
    df, *_, cfg = detected
    sweep = sweep_radii(df, cfg, radius_grid(1.0, 5.0, 1.0)).set_index("r_pz_m")

    # coverage rises with the radius; duels accelerate
    cov = sweep["coverage_pct"].tolist()
    assert cov == sorted(cov)
    assert sweep.loc[1.0, "duel_pct"] <= sweep.loc[3.0, "duel_pct"]
    assert sweep.loc[3.0, "duel_pct"] < sweep.loc[5.0, "duel_pct"]
    # the headline calibration claim: tight at 3 m, ugly at 5 m
    assert sweep.loc[3.0, "duel_pct"] < 5.0
    assert sweep.loc[5.0, "duel_pct"] > 8.0
    assert sweep.loc[5.0, "clean_pct"] < sweep.loc[3.0, "clean_pct"]


def test_prerequisite_outputs_are_not_modified(detected):
    """This stage is non-destructive: it must not write to the prepared table."""
    df, frames, *_ = detected
    fresh = pd.read_parquet(PREPARED)
    assert list(fresh.columns) == list(df.columns)
    assert len(fresh) == len(df)
    # none of our output column names leaked back into the prepared table
    assert not {"state", "possessor_id", "n_in_zone"} & set(fresh.columns)
