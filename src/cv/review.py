"""Review the in-play ball selection: the metrics, and an overlay to eyeball.

Reads ``ball_candidates.parquet`` (every ball-class detection with its scoring
signals) and, by default, **re-runs selection live** against those candidates plus
the player positions in ``tracking.parquet``. Re-running is what lets you validate
a change to :mod:`src.cv.ball_select` without re-extracting the clip (Layer 1 needs
a GPU): the parquet already holds every candidate and every player, so the whole
decision reproduces on CPU in a second. Pass ``--no-recompute`` to instead read the
``selected`` column baked into the parquet by the extractor that wrote it.

What it reports:

* **coverage** — frames given an in-play ball (detected + bridged), against the
  frames that even had a candidate, and against the ceiling of frames where any
  in-play candidate exists at all;
* **false-lock rate** — how often the selected ball is a *spare*: a track that is
  static AND far from every player over its life. Target ~0. (Distance to players,
  not just off-pitch, because the worst spare in real footage is a ball resting
  just *inside* the goal line — on the pitch, but 20 m from anyone.)
* the **no-ball (a)/(b) split** — of frames with no ball emitted, how many are
  genuine absence (only spare/off-pitch candidates, or the ball off-screen) vs a
  short dropout that was bridged back;
* **positional flicker** — consecutive emitted frames whose ball position jumps
  further than the ball could travel: the sign of the selection hopping between
  two different balls.

With ``--source`` / ``--out-video`` it also renders an overlay: every candidate
boxed, the selected one highlighted, bridged frames marked, spares flagged.

    python -m src.cv.review --in data/gamestate \
        --source data/raw/clip.mp4 --out-video data/gamestate/ball_review.mp4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

PITCH_LEN_M = 120.0
PITCH_WID_M = 70.0
MARGIN_M = 1.0


# --------------------------------------------------------------------------- #
# live re-run of selection (candidates + players -> per-frame decision)
# --------------------------------------------------------------------------- #
def _players_by_frame(tracking_path: str):
    """Map frame -> (N,2) on-pitch player+keeper positions from tracking.parquet."""
    if not os.path.exists(tracking_path):
        return {}
    trk = pd.read_parquet(tracking_path)
    who = trk["role"].isin(["player", "goalkeeper"]) if "role" in trk else slice(None)
    pl = trk[who & trk["pitch_valid"]] if "role" in trk else trk[trk["pitch_valid"]]
    return {
        int(f): g[["pitch_x_m", "pitch_y_m"]].to_numpy(dtype=float)
        for f, g in pl.groupby("frame")
    }


def _candidates(cand_df: pd.DataFrame):
    """Reconstruct ball_select.Candidate objects from the debug parquet."""
    from .ball_select import Candidate

    out = []
    for r in cand_df.itertuples(index=False):
        pv = bool(r.pitch_valid)
        out.append(
            Candidate(
                frame=int(r.frame),
                img_x=float(r.img_x),
                img_y=float(r.img_y),
                pitch_x=(float(r.pitch_x_m) if pv and pd.notna(r.pitch_x_m) else None),
                pitch_y=(float(r.pitch_y_m) if pv and pd.notna(r.pitch_y_m) else None),
                pitch_valid=pv,
                conf=float(r.conf),
                bbox=(r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2),
            )
        )
    return out


def recompute(cand_df: pd.DataFrame, in_dir: str) -> tuple[pd.DataFrame, list]:
    """Re-run ball selection live; return (annotated cand_df, selections).

    Rewrites ``selected`` and adds ``bridged`` on ``cand_df`` to reflect the current
    :mod:`src.cv.ball_select`, using the same window params the extractor used
    (read from meta.json when present).
    """
    from .ball_select import BallSelectConfig, select_in_play_ball

    meta_path = os.path.join(in_dir, "meta.json")
    fps, window, mode, min_score = 25.0, 0, "centered", 0.20
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        fps = float(meta.get("fps", fps))
        bs = meta.get("ball_selection", {})
        window = int(bs.get("window_frames", 0)) or 0
        mode = bs.get("mode", mode)
        min_score = float(bs.get("min_score", min_score))
    if window <= 0:
        window = max(5, int(round(2.0 * fps)) | 1)

    cfg = BallSelectConfig(
        fps=fps, mode=mode, window_frames=window, window_hop=max(1, window // 5),
        min_score=min_score, pitch_len_m=PITCH_LEN_M, pitch_wid_m=PITCH_WID_M,
    )
    players = _players_by_frame(os.path.join(in_dir, "tracking.parquet"))
    if not players:
        print("[warn] no tracking.parquet player positions found — the "
              "distance-to-player signal is inactive; pass --no-recompute to read "
              "the extractor's baked-in selection instead.")
    cands = _candidates(cand_df)
    selections, _tracks, _scores = select_in_play_ball(cands, cfg, players)

    sel_key = {(s.track_id, s.frame) for s in selections if s.cand is not None}
    df = cand_df.copy()
    df["selected"] = [
        (int(r.ball_track_id), int(r.frame)) in sel_key for r in df.itertuples(index=False)
    ]
    df.attrs["selections"] = selections
    df.attrs["players"] = players
    df.attrs["fps"] = fps
    return df, selections


# --------------------------------------------------------------------------- #
# track classification
# --------------------------------------------------------------------------- #
def classify_tracks(df: pd.DataFrame, static_gyr_m: float,
                    players: dict | None = None) -> pd.DataFrame:
    """Per-track summary: lifetime spread, on-pitch fraction, spare-ball flag.

    A track is a *spare ball* when it is static AND not in the play — the latter now
    measured by distance to the nearest player (when player positions are available),
    which catches an on-pitch spare that an off-pitch test would miss.
    """
    recs = []
    for tid, g in df.groupby("ball_track_id"):
        pv = g[g["pitch_valid"]]
        pts = pv[["pitch_x_m", "pitch_y_m"]].to_numpy(dtype=float)
        if len(pts) >= 2:
            centre = np.median(pts, axis=0)
            gyr = float(np.median(np.linalg.norm(pts - centre, axis=1)))
        else:
            gyr = 0.0
        if len(pts):
            inside = (
                (pts[:, 0] >= -MARGIN_M)
                & (pts[:, 0] <= PITCH_LEN_M + MARGIN_M)
                & (pts[:, 1] >= -MARGIN_M)
                & (pts[:, 1] <= PITCH_WID_M + MARGIN_M)
            )
            onpitch = float(inside.mean())
        else:
            onpitch = 0.0
        npd = np.nan
        if players:
            dists = []
            for r in pv.itertuples(index=False):
                P = players.get(int(r.frame))
                if P is not None and len(P):
                    dists.append(float(np.min(np.hypot(P[:, 0] - r.pitch_x_m,
                                                       P[:, 1] - r.pitch_y_m))))
            if dists:
                npd = float(np.median(dists))
        recs.append(
            dict(
                ball_track_id=int(tid),
                n_frames=len(g),
                first=int(g["frame"].min()),
                last=int(g["frame"].max()),
                gyration_m=gyr,
                onpitch_frac=onpitch,
                player_dist_m=npd,
                n_selected=int(g["selected"].sum()),
                mean_conf=float(g["conf"].mean()),
                is_static=gyr < static_gyr_m,
                is_offpitch=onpitch < 0.5,
                is_far=(npd > 8.0) if np.isfinite(npd) else False,
            )
        )
    out = pd.DataFrame(recs).sort_values("n_frames", ascending=False)
    # spare = static AND not in the play. "Not in the play" is far-from-players when
    # we know player positions, else the older off-pitch proxy.
    not_in_play = out["is_far"] if (players and out["player_dist_m"].notna().any()) \
        else out["is_offpitch"]
    out["is_spare_ball"] = out["is_static"] & not_in_play
    return out


def _runs(frames):
    """Contiguous runs in a sorted frame list -> [(first, last), ...]."""
    if not frames:
        return []
    runs, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f != prev + 1:
            runs.append((start, prev))
            start = f
        prev = f
    runs.append((start, prev))
    return runs


def report(df: pd.DataFrame, tracks: pd.DataFrame, selections=None,
           players=None, fps: float = 25.0, baseline_selected=None) -> None:
    spare_ids = set(tracks.loc[tracks["is_spare_ball"], "ball_track_id"])
    sel = df[df["selected"]]
    n_cand_frames = df["frame"].nunique()
    n_sel_frames = sel["frame"].nunique()
    n_false_lock = int(sel["ball_track_id"].isin(spare_ids).sum())

    # bridged frames + emitted positions come from the live selections
    bridged = [s for s in (selections or []) if getattr(s, "bridged", False)]
    n_bridged = len(bridged)

    print("=" * 78)
    print("CANDIDATE TRACKS")
    print("=" * 78)
    cols = [
        "ball_track_id", "n_frames", "first", "last", "gyration_m",
        "onpitch_frac", "player_dist_m", "n_selected", "mean_conf", "is_spare_ball",
    ]
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(tracks[cols].head(40).to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"))

    print()
    print("=" * 78)
    print("METRICS")
    print("=" * 78)
    print(f"frames with >=1 ball candidate  : {n_cand_frames}")
    if baseline_selected is not None:
        print(f"frames selected — BASELINE      : {baseline_selected} "
              f"({100.0 * baseline_selected / max(1, n_cand_frames):.1f}%)")
    print(
        f"frames with a DETECTED ball     : {n_sel_frames} "
        f"({100.0 * n_sel_frames / max(1, n_cand_frames):.1f}%)"
    )
    print(f"frames BRIDGED (short dropout)  : {n_bridged}")
    covered = n_sel_frames + n_bridged
    print(f"frames with an in-play ball     : {covered} "
          f"({100.0 * covered / max(1, n_cand_frames):.1f}% of candidate frames)")

    # ceiling: frames where at least one in-play candidate exists at all
    if selections is not None:
        by_frame = {s.frame: s for s in selections}
        emitted = {f for f, s in by_frame.items()
                   if s.cand is not None or getattr(s, "bridged", False)}
        print()
        print(f"spare-ball tracks (static + not-in-play): {len(spare_ids)} {sorted(spare_ids)}")
        print(
            f"FALSE-LOCK frames onto a spare ball    : {n_false_lock} "
            f"({100.0 * n_false_lock / max(1, n_sel_frames):.2f}% of detected)   [target ~0]"
        )

        # (a)/(b) split of no-ball frames
        null_frames = sorted(set(by_frame) - emitted)
        a_genuine, b_recoverable = _ab_split(null_frames, df, players)
        print()
        print(f"NO-BALL frames (of candidate frames)   : {len(null_frames)}")
        print(f"  (a) genuine — no candidate near the play (spare/off-pitch/off-screen): "
              f"{len(a_genuine)}")
        print(f"  (b) a near-player candidate present, held (multi-ball / sub-threshold): "
              f"{len(b_recoverable)}")
        print(f"  bridged back (short same-ball dropouts)       : {n_bridged}")

        # positional flicker — only frame-to-frame (small-gap) counts as flicker;
        # a jump after a genuine multi-frame gap is a re-acquisition, not a flicker.
        jumps = _flicker(selections, fps, max_gap=3)
        reacq = _flicker(selections, fps, max_gap=10 ** 9)
        print()
        print(f"POSITIONAL flicker (frame-to-frame jump beyond reachable): {len(jumps)}")
        for a, b, d, reach in jumps[:12]:
            print(f"    frame {a}->{b}: {d:.1f} m  (reachable {reach:.1f} m)")
        print(f"  (post-gap re-acquisitions, not flicker: {len(reacq) - len(jumps)})")

    onp = sel.dropna(subset=["pitch_x_m"])
    if len(onp):
        inside = (
            (onp["pitch_x_m"] >= -MARGIN_M)
            & (onp["pitch_x_m"] <= PITCH_LEN_M + MARGIN_M)
            & (onp["pitch_y_m"] >= -MARGIN_M)
            & (onp["pitch_y_m"] <= PITCH_WID_M + MARGIN_M)
        )
        print()
        print(
            f"selected ball ON-PITCH fraction        : {inside.mean():.3f} "
            f"({int(inside.sum())}/{len(onp)} pitch-valid detected frames)"
        )

    # what the naive rules this stage replaces would have done
    top_conf = df.loc[df.groupby("frame")["conf"].idxmax()]
    naive = int(top_conf["ball_track_id"].isin(spare_ids).sum())
    print()
    print(
        f"[contrast] a top-CONFIDENCE vote picks a spare ball on {naive}/{len(top_conf)} "
        f"frames ({100.0 * naive / max(1, len(top_conf)):.1f}%)"
    )


def _ab_split(null_frames, df, players, near_m: float = 6.0):
    """Split no-ball frames into (a) genuine absence and (b) a near-player candidate
    was present but held.

    (b) is when some candidate on the frame sits within ``near_m`` of a player —
    a plausibly-in-play ball that the emitter did not take (a second ball while it
    held its lock, or a fragment too short to clear the support floor). Everything
    else — only spare/off-pitch candidates, or no candidate at all — is (a) genuine.
    """
    if not players:
        # no player positions: fall back to "any pitch-valid candidate present"
        with_cand = set(df.loc[df["pitch_valid"], "frame"].tolist())
        a = [f for f in null_frames if f not in with_cand]
        return a, [f for f in null_frames if f in with_cand]
    by_frame = {int(f): g for f, g in df[df["pitch_valid"]].groupby("frame")}
    a_genuine, b_recoverable = [], []
    for f in null_frames:
        g = by_frame.get(f)
        P = players.get(f)
        near = False
        if g is not None and P is not None and len(P):
            for r in g.itertuples(index=False):
                if np.min(np.hypot(P[:, 0] - r.pitch_x_m, P[:, 1] - r.pitch_y_m)) <= near_m:
                    near = True
                    break
        (b_recoverable if near else a_genuine).append(f)
    return a_genuine, b_recoverable


def _flicker(selections, fps: float, cap_ms: float = 36.0, slack_m: float = 6.0,
             max_gap: int = 3):
    """Emitted frames within ``max_gap`` of each other whose ball position jumps
    beyond a reachable step — the signature of hopping between two different balls.
    A jump after a longer gap is a re-acquisition, not a flicker, so it is excluded
    by ``max_gap``."""
    def pos(s):
        if s.cand is not None:
            return (s.cand.pitch_x, s.cand.pitch_y, s.frame)
        if getattr(s, "bridged", False):
            return (s.pitch_x, s.pitch_y, s.frame)
        return None

    seq = [p for p in (pos(s) for s in sorted(selections, key=lambda s: s.frame))
           if p and p[0] is not None]
    jumps = []
    for a, b in zip(seq[:-1], seq[1:]):
        if b[2] - a[2] > max_gap:
            continue
        reach = cap_ms * (b[2] - a[2]) / fps + slack_m
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if d > reach:
            jumps.append((a[2], b[2], d, reach))
    return jumps


def render(df: pd.DataFrame, tracks: pd.DataFrame, source: str, out_video: str,
           selections=None, max_frames: int = 0) -> None:
    import cv2

    spare_ids = set(tracks.loc[tracks["is_spare_ball"], "ball_track_id"])
    bridged_by_frame = {}
    if selections:
        for s in selections:
            if getattr(s, "bridged", False):
                bridged_by_frame[int(s.frame)] = (s.img_x, s.img_y)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    by_frame = {int(f): g for f, g in df.groupby("frame")}
    n = max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fidx in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        g = by_frame.get(fidx)
        any_sel = False
        if g is not None:
            for _, r in g.iterrows():
                x1, y1 = int(r["bbox_x1"]), int(r["bbox_y1"])
                x2, y2 = int(r["bbox_x2"]), int(r["bbox_y2"])
                tid = int(r["ball_track_id"])
                if r["selected"]:
                    col, lab = (0, 255, 0), f"IN-PLAY t{tid}"
                    any_sel = True
                elif tid in spare_ids:
                    col, lab = (0, 0, 255), f"spare t{tid}"
                else:
                    col, lab = (0, 165, 255), f"rej t{tid}"
                pad = 8
                cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 2)
                sc = r.get("track_score")
                txt = lab if pd.isna(sc) else f"{lab} s={float(sc):.2f}"
                cv2.putText(frame, txt, (x1 - pad, y1 - pad - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if fidx in bridged_by_frame and not any_sel:
            bx, by = bridged_by_frame[fidx]
            cv2.circle(frame, (int(bx), int(by)), 12, (0, 255, 255), 2)
            cv2.putText(frame, "BRIDGED", (int(bx) - 20, int(by) - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            any_sel = True
        if not any_sel:
            cv2.putText(frame, "NO IN-PLAY BALL", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"f{fidx}", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(frame)
    cap.release()
    vw.release()
    print(f"\n[overlay] wrote {out_video}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Review in-play ball selection.")
    ap.add_argument("--in", dest="in_dir", default="data/gamestate",
                    help="dir holding ball_candidates.parquet + tracking.parquet")
    ap.add_argument("--source", default=None, help="video, for the overlay")
    ap.add_argument("--out-video", default=None, help="overlay mp4 to write")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--static-gyr-m", type=float, default=1.0,
                    help="lifetime pitch spread (m) below which a track counts as static")
    ap.add_argument("--no-recompute", action="store_true",
                    help="use the parquet's baked-in `selected` instead of re-running")
    args = ap.parse_args(argv)

    path = os.path.join(args.in_dir, "ball_candidates.parquet")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found — re-run the extractor with --ball (it writes every "
            f"ball candidate there alongside tracking.parquet)."
        )
    df = pd.read_parquet(path)
    baseline_selected = int(df["selected"].sum()) if "selected" in df else None
    baseline_frames = df.loc[df["selected"], "frame"].nunique() if "selected" in df else None

    selections, players, fps = None, None, 25.0
    if not args.no_recompute:
        df, selections = recompute(df, args.in_dir)
        players = df.attrs.get("players")
        fps = df.attrs.get("fps", 25.0)

    tracks = classify_tracks(df, args.static_gyr_m, players)
    report(df, tracks, selections, players, fps, baseline_frames)
    if args.source and args.out_video:
        render(df, tracks, args.source, args.out_video, selections, args.max_frames)


if __name__ == "__main__":
    main()
