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
        BallDetector(
            ball_pt,
            args.device,
            args.ball_imgsz,
            use_half,
            legacy=getattr(args, "ball_legacy_tracker", False),
        )
        if args.ball
        else None
    )
    annotator = VideoAnnotator(args.save_video, info) if args.save_video else None

    # --- phase 2: per-frame extraction ---
    print("[phase 2] extracting game state ...")
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    rows: List[dict] = []
    frames_jsonl: List[dict] = []
    ball_cands: List = []  # every ball-class detection; resolved after the loop
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
            # Every ball-class detection is kept as a *candidate*. Which one is the
            # in-play ball is decided after the loop, by ball_select — it needs a
            # window of motion to tell the game ball from a static spare ball, and
            # that evidence does not exist yet on this frame. Candidates are held
            # out of `rows` so tracking.parquet keeps its one-ball-per-frame
            # contract (six downstream modules select on object_id == 0).
            for i in range(len(bdet)):
                ball_cands.append(
                    _build_candidate(fidx, bdet, i, transformer)
                )

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

    # --- resolve the single in-play ball out of the candidates ---
    ball_debug, ball_meta = _select_ball(args, fps, ball_cands, rows, frames_jsonl)

    meta = _build_meta(args, info, fps, rows, frames_jsonl, jersey, use_half)
    if ball_meta:
        meta["ball_selection"] = ball_meta
    df, paths = write_outputs(
        args.out_dir, rows, frames_jsonl, meta, ball_debug=ball_debug
    )

    valid = df["pitch_valid"].mean() if len(df) else 0.0
    print(f"\n[done] {len(rows)} object rows across {len(frames_jsonl)} frames")
    print(f"[done] homography valid on {valid * 100:.1f}% of object rows")
    tail = f"\n  {args.save_video}" if args.save_video else ""
    print(
        f"[done] wrote:\n  {paths['parquet']}\n  {paths['csv']}\n  "
        f"{paths['jsonl']}\n  {paths['meta']}{tail}"
    )

    if getattr(args, "prepare", False):
        _run_prerequisites(args.out_dir)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _run_prerequisites(out_dir: str) -> None:
    """Optional post-step: turn the just-written raw game state into event-ready
    game state (see ``src/prerequisites``).

    Reloads the artifacts we just wrote (so this runs the exact same pipeline as
    the standalone ``python -m src.prerequisites`` CLI, with canonical dtypes)
    and writes ``*_prepared.*`` alongside them. Imported lazily so a plain
    extraction never pays for it.
    """
    from .prerequisites import (
        config_from_meta,
        load_gamestate,
        run_prerequisites,
        write_prepared,
    )

    print("[prepare] running prerequisites on the extracted game state ...")
    df, meta = load_gamestate(out_dir)
    cfg = config_from_meta(meta)
    prepared, prep_meta = run_prerequisites(df, cfg)
    prep_meta["source_meta"] = {
        k: meta.get(k) for k in ("source", "fps", "pitch", "perf") if k in meta
    }
    paths = write_prepared(out_dir, prepared, prep_meta)
    print(
        f"[prepare] wrote:\n  {paths['parquet']}\n  {paths['jsonl']}\n  {paths['meta']}"
    )


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


def _build_candidate(fidx, bdet, i, transformer):
    """One ball-class detection -> a ball_select.Candidate.

    Anchored at the bbox CENTRE, not the bottom edge. For people the bottom edge
    is the feet — where they actually touch the pitch — but a ball is a sphere in
    flight, and projecting its bbox bottom through the homography places an
    airborne ball metres from where it is. The centre is the honest anchor for
    deciding *which* ball this is. (The emitted row still carries the pipeline's
    usual bottom-centre anchor as well, so downstream geometry is unchanged.)
    """
    from .ball_select import Candidate

    x1, y1, x2, y2 = (float(v) for v in bdet.xyxy[i])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    px, py, ok = to_pitch_m(transformer, cx, cy)
    conf = (
        float(bdet.confidence[i])
        if getattr(bdet, "confidence", None) is not None
        else float("nan")
    )
    return Candidate(
        frame=fidx,
        img_x=cx,
        img_y=cy,
        pitch_x=px,
        pitch_y=py,
        pitch_valid=ok,
        conf=conf,
        bbox=(x1, y1, x2, y2),
    )


def _select_ball(args, fps, ball_cands, rows, frames_jsonl):
    """Resolve one in-play ball per frame and append it to ``rows``.

    Returns ``(debug_records, meta)``. ``debug_records`` is every candidate with
    its track id, its score and whether it won — the side output for inspecting
    exactly what was rejected and why.
    """
    if not ball_cands:
        return [], {}

    from .ball_select import BallSelectConfig, select_in_play_ball

    # The window is a duration, not a frame count: ~2 s of context is what
    # separates a static spare ball from a game ball that happens to be moving
    # slowly. Pinning it to frames would silently halve the context on 60 fps
    # footage. Odd so it can be centred.
    window = args.ball_window
    if window <= 0:
        window = max(5, int(round(config.BALL_WINDOW_SECONDS * fps)) | 1)

    cfg = BallSelectConfig(
        fps=fps,
        mode=args.ball_window_mode,
        window_frames=window,
        window_hop=max(1, window // 5),
        min_score=args.ball_min_score,
        pitch_len_m=config.PITCH_LEN_M,
        pitch_wid_m=config.PITCH_WID_M,
    )
    # Player positions per frame drive the distance-to-player ("being played")
    # signal. Built here from the already-projected person rows — outfielders and
    # keepers, referees excluded (a ball resting by the far-side official is not in
    # play). Referees also loiter off the touchline where spare balls sit, so
    # including them would blunt the very signal that rejects those spares.
    player_pos_by_frame: dict = {}
    for r in rows:
        if r["role"] in ("player", "goalkeeper") and r["pitch_valid"]:
            player_pos_by_frame.setdefault(r["frame"], []).append(
                (r["pitch_x_m"], r["pitch_y_m"])
            )
    player_pos_by_frame = {
        f: np.asarray(v, dtype=float) for f, v in player_pos_by_frame.items()
    }
    selections, tracks, scores = select_in_play_ball(
        ball_cands, cfg, player_pos_by_frame
    )

    objs_by_frame = {rec["frame"]: rec["objects"] for rec in frames_jsonl}
    n_sel = 0
    n_bridged = 0
    for s in selections:
        if s.cand is None and not s.bridged:
            continue  # no in-play ball this frame: null is the correct answer
        if s.cand is not None:
            c = s.cand
            bbox = c.bbox
            img_x, img_y = c.img_x, c.img_y
            px, py, pv = c.pitch_x, c.pitch_y, c.pitch_valid
            bridged = False
        else:
            # bridged frame: no detection, position linearly interpolated across a
            # short same-ball dropout. Synthesise a nominal bbox around the point so
            # the row (and the review overlay) has something to draw.
            img_x, img_y = s.img_x, s.img_y
            px, py, pv = s.pitch_x, s.pitch_y, True
            bbox = (img_x - 8.0, img_y - 8.0, img_x + 8.0, img_y + 8.0)
            bridged = True
        row = _build_row(
            s.frame,
            round(s.frame / fps, 4),
            config.BALL_OBJECT_ID,
            "ball",
            None,
            bbox,
            None,  # pitch coords are overwritten below from the centre anchor
        )
        # Keep the centre-anchored position from selection, image AND pitch, so the
        # row is internally consistent (img_x/img_y is the point that projects to
        # pitch_x_m/pitch_y_m). _build_row's bottom-edge anchor is right for people
        # — that is where their feet meet the pitch — but wrong for a ball, which
        # spends much of the match off the ground.
        row["img_x"] = round(img_x, 2)
        row["img_y"] = round(img_y, 2)
        row["pitch_x_m"] = round(px, 3) if pv else None
        row["pitch_y_m"] = round(py, 3) if pv else None
        row["pitch_valid"] = bool(pv)
        row["ball_sel_score"] = round(float(s.score), 4)
        row["ball_sel_margin"] = round(float(s.margin), 4)
        row["ball_track_id"] = int(s.track_id)
        row["ball_bridged"] = bridged
        rows.append(row)
        if s.frame in objs_by_frame:
            objs_by_frame[s.frame].append(row)
        if bridged:
            n_bridged += 1
        else:
            n_sel += 1

    sel_key = {(s.track_id, s.frame) for s in selections if s.cand is not None}
    debug = [
        dict(
            frame=c.frame,
            ball_track_id=int(c.track_id),
            conf=float(c.conf),
            img_x=round(c.img_x, 2),
            img_y=round(c.img_y, 2),
            pitch_x_m=(round(c.pitch_x, 3) if c.pitch_valid else None),
            pitch_y_m=(round(c.pitch_y, 3) if c.pitch_valid else None),
            pitch_valid=bool(c.pitch_valid),
            bbox_x1=round(c.bbox[0], 1),
            bbox_y1=round(c.bbox[1], 1),
            bbox_x2=round(c.bbox[2], 1),
            bbox_y2=round(c.bbox[3], 1),
            selected=bool((c.track_id, c.frame) in sel_key),
            track_score=(
                round(float(scores[c.track_id].score), 4)
                if c.track_id in scores
                else None
            ),
            track_motion=(
                round(float(scores[c.track_id].motion), 4)
                if c.track_id in scores
                else None
            ),
            track_onpitch=(
                round(float(scores[c.track_id].onpitch), 4)
                if c.track_id in scores
                else None
            ),
            track_gyration_m=(
                round(float(scores[c.track_id].gyration_m), 3)
                if c.track_id in scores
                else None
            ),
            track_dist=(
                round(float(scores[c.track_id].dist), 4)
                if c.track_id in scores
                else None
            ),
            track_physics=(
                round(float(scores[c.track_id].physics), 4)
                if c.track_id in scores
                else None
            ),
            track_player_dist_m=(
                round(float(scores[c.track_id].player_dist_m), 3)
                if c.track_id in scores
                and np.isfinite(scores[c.track_id].player_dist_m)
                else None
            ),
        )
        for c in ball_cands
    ]

    n_frames_with_cands = len({c.frame for c in ball_cands})
    meta = dict(
        method=(
            "multi-candidate track scoring (motion + on-pitch base, "
            "* distance-to-player being-played factor * trajectory physics factor), "
            "emitted by positional continuity across fragments"
        ),
        mode=cfg.mode,
        window_frames=cfg.window_frames,
        min_score=cfg.min_score,
        n_candidates=len(ball_cands),
        n_candidate_tracks=len(tracks),
        n_frames_with_candidates=n_frames_with_cands,
        n_frames_ball_selected=n_sel,
        n_frames_ball_bridged=n_bridged,
        note=(
            "one in-play ball per frame in tracking.parquet; every candidate "
            "(selected or not) in ball_candidates.parquet with its scoring signals. "
            "The in-play ball follows the nearest continuous detection across "
            "fragmented tracks; short same-ball dropouts are bridged (ball_bridged), "
            "and genuine absences (only spare/off-pitch candidates, or the ball "
            "off-screen) emit NO ball row — null feeds the gap handling in "
            "prerequisites.ball."
        ),
    )
    print(
        f"[ball] {len(ball_cands)} candidates -> {len(tracks)} tracks; "
        f"in-play ball on {n_sel}/{n_frames_with_cands} candidate frames "
        f"(+{n_bridged} bridged)"
    )
    return debug, meta


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
