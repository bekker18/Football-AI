"""Review the in-play ball selection: the metrics, and an overlay to eyeball.

Reads ``ball_candidates.parquet`` (every ball-class detection, with its candidate
track, its scores, and whether it won) and reports what you need to know:

* **false-lock rate** — how often the selected ball is a *static off-pitch* track,
  i.e. a spare ball. Target ~0.
* **on-pitch coverage** of the selected ball, to compare against the previous run.
* **null runs** — the frames where no in-play ball was emitted, so you can check
  they line up with occlusions and out-of-play phases rather than dropouts.
* a **contrast**: what a naive top-confidence vote would have picked instead.

With ``--source`` / ``--out-video`` it also renders an overlay: every candidate
boxed, the selected one highlighted, spare balls marked, so you can watch the
decision frame by frame.

A track counts as a "spare ball" from the data, not by hand: low robust spread in
pitch metres over its lifetime (it never moves) AND mostly off-pitch.

    python -m src.cv.review --in data/gamestate \
        --source data/raw/clip.mp4 --out-video data/gamestate/ball_review.mp4
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PITCH_LEN_M = 120.0
PITCH_WID_M = 70.0
MARGIN_M = 1.0


def classify_tracks(df: pd.DataFrame, static_gyr_m: float) -> pd.DataFrame:
    """Per-track summary: lifetime spread, on-pitch fraction, spare-ball flag."""
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
        recs.append(
            dict(
                ball_track_id=int(tid),
                n_frames=len(g),
                first=int(g["frame"].min()),
                last=int(g["frame"].max()),
                gyration_m=gyr,
                onpitch_frac=onpitch,
                n_selected=int(g["selected"].sum()),
                mean_conf=float(g["conf"].mean()),
                is_static=gyr < static_gyr_m,
                is_offpitch=onpitch < 0.5,
            )
        )
    out = pd.DataFrame(recs).sort_values("n_frames", ascending=False)
    out["is_spare_ball"] = out["is_static"] & out["is_offpitch"]
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


def report(df: pd.DataFrame, tracks: pd.DataFrame) -> None:
    spare_ids = set(tracks.loc[tracks["is_spare_ball"], "ball_track_id"])
    sel = df[df["selected"]]
    n_cand_frames = df["frame"].nunique()
    n_sel_frames = sel["frame"].nunique()
    n_false_lock = int(sel["ball_track_id"].isin(spare_ids).sum())

    print("=" * 78)
    print("CANDIDATE TRACKS")
    print("=" * 78)
    cols = [
        "ball_track_id", "n_frames", "first", "last", "gyration_m",
        "onpitch_frac", "n_selected", "mean_conf", "is_spare_ball",
    ]
    with pd.option_context("display.width", 200, "display.max_rows", 80):
        print(tracks[cols].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print()
    print("=" * 78)
    print("METRICS")
    print("=" * 78)
    print(f"frames with >=1 ball candidate  : {n_cand_frames}")
    print(
        f"frames with an in-play ball     : {n_sel_frames} "
        f"({100.0 * n_sel_frames / max(1, n_cand_frames):.1f}%)"
    )
    print(f"frames emitting NULL (no ball)  : {n_cand_frames - n_sel_frames}")
    print()
    print(f"spare-ball tracks (static + off-pitch) : {len(spare_ids)} {sorted(spare_ids)}")
    print(
        f"FALSE-LOCK frames onto a spare ball    : {n_false_lock} "
        f"({100.0 * n_false_lock / max(1, n_sel_frames):.2f}% of selected)   [target ~0]"
    )

    onp = sel.dropna(subset=["pitch_x_m"])
    if len(onp):
        inside = (
            (onp["pitch_x_m"] >= -MARGIN_M)
            & (onp["pitch_x_m"] <= PITCH_LEN_M + MARGIN_M)
            & (onp["pitch_y_m"] >= -MARGIN_M)
            & (onp["pitch_y_m"] <= PITCH_WID_M + MARGIN_M)
        )
        print(
            f"selected ball ON-PITCH fraction        : {inside.mean():.3f} "
            f"({int(inside.sum())}/{len(onp)} pitch-valid selected frames)"
        )

    # what the naive rules this stage replaces would have done
    top_conf = df.loc[df.groupby("frame")["conf"].idxmax()]
    naive = int(top_conf["ball_track_id"].isin(spare_ids).sum())
    print()
    print(
        f"[contrast] a top-CONFIDENCE vote picks a spare ball on {naive}/{len(top_conf)} "
        f"frames ({100.0 * naive / max(1, len(top_conf)):.1f}%)"
    )

    null_f = sorted(set(df["frame"]) - set(sel["frame"]))
    runs = _runs(null_f)
    if runs:
        print()
        print(f"NULL runs (no in-play ball) — {len(runs)} run(s), longest first:")
        for a, b in sorted(runs, key=lambda r: -(r[1] - r[0]))[:15]:
            print(f"    frames {a}-{b}  ({b - a + 1} frames)")


def render(df: pd.DataFrame, tracks: pd.DataFrame, source: str, out_video: str,
           max_frames: int = 0) -> None:
    import cv2

    spare_ids = set(tracks.loc[tracks["is_spare_ball"], "ball_track_id"])
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
        if g is not None:
            for _, r in g.iterrows():
                x1, y1 = int(r["bbox_x1"]), int(r["bbox_y1"])
                x2, y2 = int(r["bbox_x2"]), int(r["bbox_y2"])
                tid = int(r["ball_track_id"])
                if r["selected"]:
                    col, lab = (0, 255, 0), f"IN-PLAY t{tid}"
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
            if not g["selected"].any():
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
                    help="dir holding ball_candidates.parquet")
    ap.add_argument("--source", default=None, help="video, for the overlay")
    ap.add_argument("--out-video", default=None, help="overlay mp4 to write")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--static-gyr-m", type=float, default=0.75,
                    help="lifetime pitch spread below which a track counts as static")
    args = ap.parse_args(argv)

    path = os.path.join(args.in_dir, "ball_candidates.parquet")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found — re-run the extractor with --ball (it writes every "
            f"ball candidate there alongside tracking.parquet)."
        )
    df = pd.read_parquet(path)
    tracks = classify_tracks(df, args.static_gyr_m)
    report(df, tracks)
    if args.source and args.out_video:
        render(df, tracks, args.source, args.out_video, args.max_frames)


if __name__ == "__main__":
    main()
