#!/usr/bin/env python3
"""
Layer 1 — CV extraction: broadcast/tactical video -> structured game state.

Pipeline (all from roboflow/sports + ultralytics + supervision):
  pitch keypoints  -> homography (image px -> pitch metres)
  player detection -> ByteTrack -> SigLIP+UMAP+KMeans team assignment
  ball detection   -> tiled slicer + simple proximity tracker   [optional]

Outputs (into --out-dir):
  tracking.parquet / tracking.csv  one row per (frame, object)
  frames.jsonl                     one JSON line per frame (objects nested)
  meta.json                        fps, resolution, pitch dims, versions, args

Pitch coordinate system: x in [0, 120] m (length), y in [0, 70] m (width),
origin at the top-left pitch corner. Matches SoccerPitchConfiguration (cm/100).
"""

from __future__ import annotations

import argparse
import json
import os
from importlib import metadata
from typing import List, Optional

import numpy as np
import pandas as pd
import supervision as sv
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from tqdm import tqdm
from ultralytics import YOLO

# Class ids in the player-detection model
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(CONFIG.vertices)  # (N, 2) in cm
PITCH_LEN_M = CONFIG.length / 100.0
PITCH_WID_M = CONFIG.width / 100.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def jersey_crop(frame: np.ndarray, xyxy) -> np.ndarray:
    """Upper-torso ('jersey') crop, to focus the team classifier on kit colour
    rather than grass / shorts / skin. Falls back to the full box when tiny."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:  # too small to carve up reliably
        return sv.crop_image(frame, xyxy)
    # skip the head, stop before the shorts, and trim side background
    band = np.array(
        [x1 + 0.15 * w, y1 + 0.12 * h, x2 - 0.15 * w, y1 + 0.55 * h],
        dtype=np.float32,
    )
    return sv.crop_image(frame, band)


def get_team_crops(
    frame: np.ndarray, detections: sv.Detections, jersey: bool
) -> List[np.ndarray]:
    """Crops fed to the TeamClassifier — jersey band by default, full box if off."""
    if not jersey:
        return get_crops(frame, detections)
    return [jersey_crop(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections
) -> np.ndarray:
    """Assign each GK to the nearest team centroid. Robust to empty teams."""
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(players) == 0:
        # No outfield players this frame: split GKs by image x (left=0, right=1)
        return (gk_xy[:, 0] > np.median(gk_xy[:, 0])).astype(int)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    out = []
    c0 = pl_xy[players_team_id == 0]
    c1 = pl_xy[players_team_id == 1]
    c0 = c0.mean(axis=0) if len(c0) else None
    c1 = c1.mean(axis=0) if len(c1) else None
    for xy in gk_xy:
        d0 = np.linalg.norm(xy - c0) if c0 is not None else np.inf
        d1 = np.linalg.norm(xy - c1) if c1 is not None else np.inf
        out.append(0 if d0 <= d1 else 1)
    return np.array(out, dtype=int)


def build_transformer(pitch_result) -> Optional[ViewTransformer]:
    """Homography from detected pitch keypoints; None if <4 reliable points."""
    kp = sv.KeyPoints.from_ultralytics(pitch_result)
    if kp.xy is None or len(kp.xy) == 0:
        return None
    pts = kp.xy[0]
    mask = (pts[:, 0] > 1) & (pts[:, 1] > 1)
    if int(mask.sum()) < 4:
        return None
    try:
        return ViewTransformer(
            source=pts[mask].astype(np.float32),
            target=PITCH_VERTICES[mask].astype(np.float32),
        )
    except Exception:
        return None


def to_pitch_m(transformer: Optional[ViewTransformer], img_x: float, img_y: float):
    """Image bottom-center -> pitch (x_m, y_m). Returns (None, None, False) if no homography."""
    if transformer is None:
        return None, None, False
    t = transformer.transform_points(np.array([[img_x, img_y]], dtype=np.float32))[0]
    return float(t[0] / 100.0), float(t[1] / 100.0), True


def pkg_versions() -> dict:
    out = {}
    for p in (
        "torch",
        "ultralytics",
        "supervision",
        "transformers",
        "umap-learn",
        "scikit-learn",
        "opencv-python-headless",
        "numpy",
    ):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract soccer game state from video.")
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--out-dir", default="/output", help="directory for outputs")
    ap.add_argument(
        "--model-dir", default="/data/models", help="dir holding the .pt checkpoints"
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument(
        "--imgsz", type=int, default=1280, help="player-model inference size"
    )
    ap.add_argument("--ball-imgsz", type=int, default=640, help="ball-model tile size")
    ap.add_argument(
        "--stride-fit",
        type=int,
        default=60,
        help="frame stride for collecting crops to fit the team classifier",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="0 = whole video (else cap)"
    )
    ap.add_argument(
        "--ball",
        action="store_true",
        help="enable ball detection (slow on CPU; off by default)",
    )
    ap.add_argument(
        "--save-video",
        default=None,
        help="optional path to also write an annotated mp4 (headless)",
    )
    ap.add_argument(
        "--full-crop",
        action="store_true",
        help="classify teams on the full player box instead of the jersey band",
    )
    ap.add_argument(
        "--no-team-vote",
        action="store_true",
        help="disable the per-track majority vote that stabilises team labels",
    )
    args = ap.parse_args()

    jersey = not args.full_crop
    os.makedirs(args.out_dir, exist_ok=True)
    player_pt = os.path.join(args.model_dir, "football-player-detection.pt")
    pitch_pt = os.path.join(args.model_dir, "football-pitch-detection.pt")
    ball_pt = os.path.join(args.model_dir, "football-ball-detection.pt")
    for p in (player_pt, pitch_pt) + ((ball_pt,) if args.ball else ()):
        if not os.path.exists(p):
            raise SystemExit(f"Missing checkpoint: {p}\nRun the `download` step first.")

    info = sv.VideoInfo.from_video_path(args.source)
    fps = info.fps or 25
    total = info.total_frames or 0
    if args.max_frames:
        total = min(total, args.max_frames) if total else args.max_frames
    print(
        f"[info] {args.source}  {info.width}x{info.height} @ {fps:.3f}fps  "
        f"frames={total or 'unknown'}  device={args.device}  ball={args.ball}"
    )

    player_model = YOLO(player_pt).to(args.device)
    pitch_model = YOLO(pitch_pt).to(args.device)

    # --- phase 1: fit team classifier on player crops sampled across the clip ---
    print("[phase 1] collecting crops for team classification ...")
    crops: List[np.ndarray] = []
    for frame in tqdm(
        sv.get_video_frames_generator(args.source, stride=args.stride_fit), desc="crops"
    ):
        res = player_model(frame, imgsz=args.imgsz, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        crops += get_team_crops(frame, det[det.class_id == PLAYER_CLASS_ID], jersey)
    if len(crops) < 8:
        raise SystemExit(
            f"Only {len(crops)} player crops found — clip too short/empty "
            f"or models mislocated. Lower --stride-fit or check the video."
        )
    print(f"[phase 1] fitting TeamClassifier on {len(crops)} crops ...")
    team_classifier = TeamClassifier(device=args.device)
    team_classifier.fit(crops)

    # --- optional ball detection setup ---
    ball_model = ball_tracker = ball_slicer = None
    if args.ball:
        from sports.common.ball import BallTracker

        ball_model = YOLO(ball_pt).to(args.device)
        ball_tracker = BallTracker(buffer_size=20)

        def _ball_cb(image_slice: np.ndarray) -> sv.Detections:
            r = ball_model(image_slice, imgsz=args.ball_imgsz, verbose=False)[0]
            return sv.Detections.from_ultralytics(r)

        import inspect

        slicer_kwargs = dict(
            callback=_ball_cb,
            slice_wh=(args.ball_imgsz, args.ball_imgsz),
        )
        slicer_params = inspect.signature(sv.InferenceSlicer.__init__).parameters
        if "overlap_filter" in slicer_params:
            slicer_kwargs["overlap_filter"] = sv.OverlapFilter.NONE
        elif "overlap_filter_strategy" in slicer_params:
            slicer_kwargs["overlap_filter_strategy"] = sv.OverlapFilter.NONE
        ball_slicer = sv.InferenceSlicer(**slicer_kwargs)

    # --- optional annotated-video setup (headless, no imshow) ---
    sink = annot = None
    if args.save_video:
        from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

        colors = ["#FF1493", "#00BFFF", "#FFD700"]  # team0, team1, referee
        ellipse = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(colors), thickness=2
        )
        sink = sv.VideoSink(args.save_video, info)
        sink.__enter__()

        def _radar(merged, lookup, transformer):
            radar = draw_pitch(config=CONFIG)
            if transformer is None or len(merged) == 0:
                return radar
            xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            txy = transformer.transform_points(xy)
            for i, hexc in enumerate(colors):
                radar = draw_points_on_pitch(
                    config=CONFIG,
                    xy=txy[lookup == i],
                    face_color=sv.Color.from_hex(hexc),
                    radius=20,
                    pitch=radar,
                )
            return radar

    # --- phase 2: per-frame extraction ---
    print("[phase 2] extracting game state ...")
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    rows: List[dict] = []
    frames_jsonl: List[dict] = []

    gen = sv.get_video_frames_generator(args.source)
    bar = tqdm(total=(total or None), desc="frames")
    for fidx, frame in enumerate(gen):
        if args.max_frames and fidx >= args.max_frames:
            break
        time_s = fidx / fps

        transformer = build_transformer(pitch_model(frame, verbose=False)[0])

        det = sv.Detections.from_ultralytics(
            player_model(frame, imgsz=args.imgsz, verbose=False)[0]
        )
        det = tracker.update_with_detections(det)

        players = det[det.class_id == PLAYER_CLASS_ID]
        goalkeepers = det[det.class_id == GOALKEEPER_CLASS_ID]
        referees = det[det.class_id == REFEREE_CLASS_ID]

        players_team = (
            team_classifier.predict(get_team_crops(frame, players, jersey))
            if len(players)
            else np.array([], dtype=int)
        )
        gk_team = resolve_goalkeepers_team_id(players, players_team, goalkeepers)

        frame_objs = []

        def emit(role, team, oid, xyxy):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            ix, iy = (x1 + x2) / 2.0, y2  # bottom-center anchor
            px, py, ok = to_pitch_m(transformer, ix, iy)
            row = dict(
                frame=fidx,
                time_s=round(time_s, 4),
                object_id=oid,
                role=role,
                team=(int(team) if team is not None else None),
                img_x=round(ix, 2),
                img_y=round(iy, 2),
                pitch_x_m=(round(px, 3) if ok else None),
                pitch_y_m=(round(py, 3) if ok else None),
                pitch_valid=ok,
                bbox_x1=round(x1, 1),
                bbox_y1=round(y1, 1),
                bbox_x2=round(x2, 1),
                bbox_y2=round(y2, 1),
            )
            rows.append(row)
            frame_objs.append(row)

        for i in range(len(players)):
            tid = int(players.tracker_id[i]) if players.tracker_id is not None else -1
            emit(
                "player",
                players_team[i] if i < len(players_team) else None,
                tid,
                players.xyxy[i],
            )
        for i in range(len(goalkeepers)):
            tid = (
                int(goalkeepers.tracker_id[i])
                if goalkeepers.tracker_id is not None
                else -1
            )
            emit(
                "goalkeeper",
                gk_team[i] if i < len(gk_team) else None,
                tid,
                goalkeepers.xyxy[i],
            )
        for i in range(len(referees)):
            tid = int(referees.tracker_id[i]) if referees.tracker_id is not None else -1
            emit("referee", None, tid, referees.xyxy[i])

        if args.ball:
            bdet = ball_slicer(frame).with_nms(threshold=0.1)
            bdet = ball_tracker.update(bdet)
            for i in range(len(bdet)):
                emit("ball", None, None, bdet.xyxy[i])  # ball has no track id

        frames_jsonl.append(
            dict(
                frame=fidx,
                time_s=round(time_s, 4),
                pitch_valid=transformer is not None,
                objects=frame_objs,
            )
        )

        if args.save_video:
            merged = sv.Detections.merge([players, goalkeepers, referees])
            lookup = np.array(
                players_team.tolist() + gk_team.tolist() + [2] * len(referees)
            )  # 2 == referee color
            annotated = frame.copy()
            if len(merged):
                annotated = ellipse.annotate(
                    annotated, merged, custom_color_lookup=lookup
                )
            radar = _radar(merged, lookup, transformer)
            radar = sv.resize_image(radar, (info.width // 2, info.height // 2))
            rh, rw, _ = radar.shape
            rect = sv.Rect(
                x=info.width // 2 - rw // 2, y=info.height - rh, width=rw, height=rh
            )
            annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
            sink.write_frame(annotated)

        bar.update(1)
    bar.close()
    if sink is not None:
        sink.__exit__(None, None, None)

    # --- stabilise team labels: one team per track (temporal majority vote) ---
    # A ByteTrack id is one physical person, so it belongs to one team for its
    # whole life. Voting over the track kills the frame-to-frame flicker that
    # independent per-frame KMeans predictions produce. Rows in `frames_jsonl`
    # are the same dict objects, so mutating in place updates both outputs.
    if not args.no_team_vote:
        from collections import Counter

        votes: dict = {}
        for r in rows:
            oid = r["object_id"]
            if (
                r["role"] in ("player", "goalkeeper")
                and r["team"] is not None
                and isinstance(oid, int)
                and oid >= 0
            ):
                votes.setdefault(oid, Counter())[r["team"]] += 1
        track_team = {oid: c.most_common(1)[0][0] for oid, c in votes.items()}
        n_flips = 0
        for r in rows:
            oid = r["object_id"]
            if oid in track_team and r["team"] != track_team[oid]:
                r["team"] = track_team[oid]
                n_flips += 1
        print(
            f"[team] majority-vote stabilised {len(track_team)} tracks; "
            f"corrected {n_flips} frame labels"
        )

    # --- write outputs ---
    df = pd.DataFrame(rows)
    if "object_id" in df.columns:
        # Single-typed column so arrow/parquet can serialise it: tracker ids stay
        # integers, the ball (which has no track) becomes <NA>.
        df["object_id"] = df["object_id"].astype("Int64")
    pq = os.path.join(args.out_dir, "tracking.parquet")
    csv = os.path.join(args.out_dir, "tracking.csv")
    jsonl = os.path.join(args.out_dir, "frames.jsonl")
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    with open(jsonl, "w") as f:
        for rec in frames_jsonl:
            f.write(json.dumps(rec) + "\n")

    meta = dict(
        source=os.path.basename(args.source),
        fps=fps,
        width=info.width,
        height=info.height,
        n_frames_processed=len(frames_jsonl),
        n_object_rows=len(rows),
        pitch=dict(
            length_m=PITCH_LEN_M,
            width_m=PITCH_WID_M,
            origin="top-left",
            units="metres",
            note="x=length 0..120, y=width 0..70; from SoccerPitchConfiguration",
        ),
        ball_enabled=args.ball,
        team_assignment=dict(jersey_crop=jersey, majority_vote=not args.no_team_vote),
        team_label_note="team 0/1 are arbitrary KMeans clusters, NOT stable across clips",
        args=vars(args),
        package_versions=pkg_versions(),
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    valid = df["pitch_valid"].mean() if len(df) else 0.0
    print(f"\n[done] {len(rows)} object rows across {len(frames_jsonl)} frames")
    print(f"[done] homography valid on {valid * 100:.1f}% of object rows")
    print(
        f"[done] wrote:\n  {pq}\n  {csv}\n  {jsonl}\n  "
        f"{os.path.join(args.out_dir, 'meta.json')}"
        + (f"\n  {args.save_video}" if args.save_video else "")
    )


if __name__ == "__main__":
    main()
