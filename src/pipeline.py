"""End-to-end extraction: video -> per-frame game-state rows -> output files.

Pipeline (all from roboflow/sports + ultralytics + supervision):
  pitch keypoints  -> homography (image px -> pitch metres)
  player detection -> ByteTrack -> SigLIP+UMAP+KMeans team assignment
  ball detection   -> tiled slicer + simple proximity tracker   [optional]
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import supervision as sv
from sports.common.team import TeamClassifier
from tqdm import tqdm
from ultralytics import YOLO

from . import config
from .annotate import VideoAnnotator
from .ball import BallDetector
from .detection import get_team_crops
from .geometry import build_transformer, to_pitch_m
from .outputs import pkg_versions, write_outputs
from .teams import majority_vote_teams, resolve_goalkeepers_team_id


def run(args) -> None:
    """Run the full extraction for one video given parsed CLI ``args``."""
    jersey = not args.full_crop
    use_half = args.device == "cuda"  # FP16 inference on GPU (~1.5-2x)
    args.team_stride = max(1, args.team_stride)  # avoid modulo-by-zero
    args.pitch_stride = max(1, args.pitch_stride)

    player_pt, pitch_pt, ball_pt = _resolve_checkpoints(args)

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

    team_classifier = _fit_team_classifier(player_model, args, jersey, use_half)

    ball_detector = (
        BallDetector(ball_pt, args.device, args.ball_imgsz, use_half)
        if args.ball
        else None
    )
    annotator = VideoAnnotator(args.save_video, info) if args.save_video else None

    # --- phase 2: per-frame extraction ---
    print("[phase 2] extracting game state ...")
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    rows: List[dict] = []
    frames_jsonl: List[dict] = []
    transformer = None  # reused between pitch-detection frames
    team_cache: dict = {}  # track_id -> last predicted team, carried forward

    bar = tqdm(total=(total or None), desc="frames")
    for fidx, frame in enumerate(sv.get_video_frames_generator(args.source)):
        if args.max_frames and fidx >= args.max_frames:
            break
        time_s = fidx / fps

        # Homography changes slowly relative to fps: recompute on a stride and
        # reuse in between (also retry whenever we don't yet have a valid one).
        if fidx % args.pitch_stride == 0 or transformer is None:
            transformer = build_transformer(
                pitch_model(frame, verbose=False, half=use_half)[0]
            )

        det = sv.Detections.from_ultralytics(
            player_model(frame, imgsz=args.imgsz, verbose=False, half=use_half)[0]
        )
        det = tracker.update_with_detections(det)
        players = det[det.class_id == config.PLAYER_CLASS_ID]
        goalkeepers = det[det.class_id == config.GOALKEEPER_CLASS_ID]
        referees = det[det.class_id == config.REFEREE_CLASS_ID]

        players_team = _predict_teams(
            team_classifier, frame, players, jersey, fidx, args.team_stride, team_cache
        )
        gk_team = resolve_goalkeepers_team_id(players, players_team, goalkeepers)

        frame_objs: List[dict] = []

        def add(role, team, oid, xyxy):
            row = _build_row(fidx, time_s, oid, role, team, xyxy, transformer)
            rows.append(row)
            frame_objs.append(row)

        for i in range(len(players)):
            tid = int(players.tracker_id[i]) if players.tracker_id is not None else -1
            team = players_team[i] if i < len(players_team) else None
            add("player", team, tid, players.xyxy[i])
        for i in range(len(goalkeepers)):
            tid = (
                int(goalkeepers.tracker_id[i])
                if goalkeepers.tracker_id is not None
                else -1
            )
            team = gk_team[i] if i < len(gk_team) else None
            add("goalkeeper", team, tid, goalkeepers.xyxy[i])
        for i in range(len(referees)):
            tid = int(referees.tracker_id[i]) if referees.tracker_id is not None else -1
            add("referee", None, tid, referees.xyxy[i])

        bdet = None
        if ball_detector is not None:
            bdet = ball_detector.detect(frame)
            for i in range(len(bdet)):
                add("ball", None, config.BALL_OBJECT_ID, bdet.xyxy[i])

        frames_jsonl.append(
            dict(
                frame=fidx,
                time_s=round(time_s, 4),
                pitch_valid=transformer is not None,
                objects=frame_objs,
            )
        )

        if annotator is not None:
            annotator.write(
                frame, players, goalkeepers, referees, players_team, gk_team,
                transformer, bdet,
            )

        bar.update(1)
    bar.close()
    if annotator is not None:
        annotator.close()

    # --- stabilise team labels: one team per track (temporal majority vote) ---
    if not args.no_team_vote:
        n_tracks, n_flips = majority_vote_teams(rows)
        print(
            f"[team] majority-vote stabilised {n_tracks} tracks; "
            f"corrected {n_flips} frame labels"
        )

    meta = _build_meta(args, info, fps, rows, frames_jsonl, jersey, use_half)
    df, paths = write_outputs(args.out_dir, rows, frames_jsonl, meta)

    valid = df["pitch_valid"].mean() if len(df) else 0.0
    print(f"\n[done] {len(rows)} object rows across {len(frames_jsonl)} frames")
    print(f"[done] homography valid on {valid * 100:.1f}% of object rows")
    tail = f"\n  {args.save_video}" if args.save_video else ""
    print(
        f"[done] wrote:\n  {paths['parquet']}\n  {paths['csv']}\n  "
        f"{paths['jsonl']}\n  {paths['meta']}{tail}"
    )


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _resolve_checkpoints(args):
    """Locate the three .pt checkpoints; fail early if any required one is missing."""
    player_pt = os.path.join(args.model_dir, "football-player-detection.pt")
    pitch_pt = os.path.join(args.model_dir, "football-pitch-detection.pt")
    ball_pt = os.path.join(args.model_dir, "football-ball-detection.pt")
    for p in (player_pt, pitch_pt) + ((ball_pt,) if args.ball else ()):
        if not os.path.exists(p):
            raise SystemExit(f"Missing checkpoint: {p}\nRun the `download` step first.")
    return player_pt, pitch_pt, ball_pt


def _fit_team_classifier(player_model, args, jersey, use_half) -> TeamClassifier:
    """Phase 1: fit the team classifier on player crops sampled across the clip."""
    print("[phase 1] collecting crops for team classification ...")
    crops: List[np.ndarray] = []
    for frame in tqdm(
        sv.get_video_frames_generator(args.source, stride=args.stride_fit),
        desc="crops",
    ):
        res = player_model(frame, imgsz=args.imgsz, verbose=False, half=use_half)[0]
        det = sv.Detections.from_ultralytics(res)
        crops += get_team_crops(frame, det[det.class_id == config.PLAYER_CLASS_ID], jersey)
    if len(crops) < 8:
        raise SystemExit(
            f"Only {len(crops)} player crops found — clip too short/empty or "
            f"models mislocated. Lower --stride-fit or check the video."
        )
    print(f"[phase 1] fitting TeamClassifier on {len(crops)} crops ...")
    clf = TeamClassifier(device=args.device)
    clf.fit(crops)
    return clf


def _predict_teams(clf, frame, players, jersey, fidx, team_stride, team_cache):
    """Team ids for this frame's players.

    Prediction (SigLIP + UMAP) is expensive, and the final labels are chosen by a
    per-track majority vote anyway, so we only predict every ``team_stride``
    frames and carry the last label forward per track id in between. ``-1`` marks
    a track not yet classified (rendered null / neutral downstream).
    """
    if not len(players):
        return np.array([], dtype=int)
    tids = (
        players.tracker_id
        if players.tracker_id is not None
        else np.full(len(players), -1, dtype=int)
    )
    if fidx % team_stride == 0:
        pred = clf.predict(get_team_crops(frame, players, jersey))
        for tid, t in zip(tids, pred):
            if int(tid) >= 0:
                team_cache[int(tid)] = int(t)
        return pred
    return np.array([team_cache.get(int(tid), -1) for tid in tids], dtype=int)


def _build_row(fidx, time_s, oid, role, team, xyxy, transformer) -> dict:
    """Build one output row for a person/ball in a frame.

    Uses the bottom-center (feet) anchor for the pitch projection. A negative
    ``team`` is the "not yet predicted" sentinel and is stored as null.
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    ix, iy = (x1 + x2) / 2.0, y2  # bottom-center anchor
    px, py, ok = to_pitch_m(transformer, ix, iy)
    return dict(
        frame=fidx,
        time_s=round(time_s, 4),
        object_id=oid,
        role=role,
        team=(int(team) if team is not None and int(team) >= 0 else None),
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


def _build_meta(args, info, fps, rows, frames_jsonl, jersey, use_half) -> dict:
    """Assemble the meta.json manifest (context for interpreting the outputs)."""
    return dict(
        source=os.path.basename(args.source),
        fps=fps,
        width=info.width,
        height=info.height,
        n_frames_processed=len(frames_jsonl),
        n_object_rows=len(rows),
        pitch=dict(
            length_m=config.PITCH_LEN_M,
            width_m=config.PITCH_WID_M,
            origin="top-left",
            units="metres",
            note="x=length 0..120, y=width 0..70; from SoccerPitchConfiguration",
        ),
        ball_enabled=args.ball,
        team_assignment=dict(
            jersey_crop=jersey,
            majority_vote=not args.no_team_vote,
            team_stride=args.team_stride,
        ),
        perf=dict(half=use_half, pitch_stride=args.pitch_stride),
        team_label_note="team 0/1 are arbitrary KMeans clusters, NOT stable across clips",
        args=vars(args),
        package_versions=pkg_versions(),
    )
