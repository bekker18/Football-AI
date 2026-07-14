"""Format adapters end-to-end: GSR ground truth + our tracking -> evaluate."""

import json

import pandas as pd

from src.eval import EvalConfig, evaluate
from src.eval.adapters import load_soccernet_gsr, load_tracking


def _write_gsr(tmp_path):
    """A minimal Labels-GameState.json with two players in one frame.

    Pitch coords are centimetres relative to the pitch centre (GSR convention):
    (0,0) cm -> centre, (1000,0) cm -> 10 m toward +x.
    """
    doc = dict(
        images=[dict(image_id="img1", file_name="000001.jpg", frame_index=1)],
        annotations=[
            dict(image_id="img1", track_id=7,
                 attributes=dict(role="player", team="left"),
                 bbox_pitch=dict(x_bottom_middle=0.0, y_bottom_middle=0.0)),
            dict(image_id="img1", track_id=8,
                 attributes=dict(role="player", team="right"),
                 bbox_pitch=dict(x_bottom_middle=1000.0, y_bottom_middle=0.0)),
        ],
    )
    p = tmp_path / "Labels-GameState.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _write_tracking(tmp_path):
    """Our tracking.parquet with the same two players at the matching metres."""
    df = pd.DataFrame(
        dict(
            frame=[1, 1],
            object_id=[101, 102],
            role=["player", "player"],
            team=[0, 1],
            pitch_x_m=[52.5, 62.5],   # centre and +10 m in a 105 m frame
            pitch_y_m=[34.0, 34.0],
            pitch_valid=[True, True],
        )
    )
    p = tmp_path / "tracking.parquet"
    df.to_parquet(p, index=False)
    return tmp_path


def test_gsr_and_tracking_adapters_align(tmp_path):
    cfg = EvalConfig(pitch_length_m=105.0, pitch_width_m=68.0, flip_candidates=("none",))
    _write_gsr(tmp_path)
    gt = load_soccernet_gsr(tmp_path, cfg)  # dir containing the labels json

    assert len(gt) == 2
    # centre-origin cm -> corner-origin metres
    assert abs(gt.iloc[0]["x"] - 52.5) < 1e-6 and abs(gt.iloc[0]["y"] - 34.0) < 1e-6
    assert abs(gt.iloc[1]["x"] - 62.5) < 1e-6

    pred_dir = _write_tracking(tmp_path)
    pred = load_tracking(pred_dir, cfg)
    assert len(pred) == 2

    report = evaluate(gt, pred, cfg)
    assert report["detection"]["f1"] == 1.0
    assert report["localization"]["mean_m"] == 0.0
