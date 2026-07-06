"""Transform 4 — ball smoothing + outlier rejection (stride-boundary aware).

The raw ball track is noisy: ~11% of consecutive-frame steps imply impossible
speed (ID spikes / homography error). This transform, in order:

1. flags physically impossible **isolated** points via a speed gate
   (``ball_outlier``) — distinguishing single-frame spikes (rejected) from
   sustained jumps (a genuine fast move / relocation, kept);
2. linearly interpolates only **short** gaps (<= ``ball_max_interp_gap``),
   emitting synthetic ball rows flagged ``ball_interp``; long gaps stay missing;
3. Savitzky-Golay smooths each contiguous segment (order 2, window 7 @ 25 fps by
   default) into ``ball_x_s_m`` / ``ball_y_s_m``;
4. recomputes velocity / speed / acceleration **from the smoothed track**.

If ``pitch_stride > 1`` (homography reused across frames), the known small steps
at stride boundaries are excluded from spike detection so they don't register as
events. Nothing is deleted — originals are preserved and only flagged.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import (
    BALL_OBJECT_ID,
    COL_BALL_ACCEL,
    COL_BALL_INTERP,
    COL_BALL_OUTLIER,
    COL_BALL_SPEED,
    COL_BALL_VX,
    COL_BALL_VY,
    COL_BALL_XS,
    COL_BALL_YS,
    COL_SYNTHETIC,
    PrepConfig,
)

_BALL_COLS = (
    COL_BALL_XS,
    COL_BALL_YS,
    COL_BALL_VX,
    COL_BALL_VY,
    COL_BALL_SPEED,
    COL_BALL_ACCEL,
)


def savgol_smooth(y: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay smoothing with a numpy fallback if scipy is unavailable.

    Prefers ``scipy.signal.savgol_filter`` (mode='interp'); if scipy is missing,
    uses an equivalent numpy implementation (interior via convolution with the
    S-G coefficients, edges via a local least-squares polynomial fit). Segments
    shorter than the window fall back to a single polynomial fit.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 3 or win <= order:
        # too short to smooth meaningfully: fit one low-order polynomial
        deg = min(order, max(0, n - 1))
        if n == 0:
            return y
        x = np.arange(n)
        return np.polyval(np.polyfit(x, y, deg), x)
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(y, win, order, mode="interp")
    except Exception:
        return _savgol_numpy(y, win, order)


def _savgol_numpy(y: np.ndarray, window: int, order: int) -> np.ndarray:
    """S-G smoothing without scipy (mode='interp'-like edge handling)."""
    half = window // 2
    z = np.arange(-half, half + 1, dtype=float)
    A = np.vander(z, order + 1, increasing=True)  # (window, order+1)
    coeffs = np.linalg.pinv(A)[0]  # value at the window centre
    out = np.convolve(y, coeffs[::-1], mode="same")
    # fix the two edges with a local polynomial fit over the boundary window
    xw = np.arange(window)
    cl = np.polyfit(xw, y[:window], order)
    out[:half] = np.polyval(cl, xw[:half])
    cr = np.polyfit(xw, y[-window:], order)
    out[-half:] = np.polyval(cr, xw[window - half:])
    return out


def _flag_outliers(frames, xs, ys, cfg) -> np.ndarray:
    """Return a boolean mask of isolated impossible-speed ball points.

    A point is an outlier when the speed to *both* its present neighbours exceeds
    the gate but the direct neighbour-to-neighbour speed does not — i.e. removing
    just this one point makes the trajectory plausible (a single-frame spike). A
    sustained jump (consecutive far points) is left unflagged.
    """
    n = len(frames)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    fps = cfg.fps
    vmax = cfg.ball_max_speed_ms

    def speed(i, j):
        dt = (frames[j] - frames[i]) / fps
        if dt <= 0:
            return np.inf
        return float(np.hypot(xs[j] - xs[i], ys[j] - ys[i]) / dt)

    for i in range(1, n - 1):
        if speed(i - 1, i) > vmax and speed(i, i + 1) > vmax and speed(i - 1, i + 1) <= vmax:
            out[i] = True
    # endpoints: odd-one-out if the neighbouring step is plausible
    if n >= 3:
        if speed(0, 1) > vmax and speed(1, 2) <= vmax:
            out[0] = True
        if speed(n - 2, n - 1) > vmax and speed(n - 3, n - 2) <= vmax:
            out[n - 1] = True

    # stride-boundary awareness: don't let known homography steps read as spikes
    if cfg.pitch_stride > 1:
        boundary = (frames % cfg.pitch_stride) == 0
        out &= ~boundary
    return out


def _segments(frames: np.ndarray):
    """Yield (start, stop) index slices of runs of consecutive frames (step 1)."""
    if len(frames) == 0:
        return
    breaks = np.where(np.diff(frames) != 1)[0]
    start = 0
    for b in breaks:
        yield start, b + 1
        start = b + 1
    yield start, len(frames)


def _ensure_ball_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ball/smoothing columns with NA defaults if not already present."""
    if COL_BALL_OUTLIER not in df.columns:
        df[COL_BALL_OUTLIER] = pd.array([pd.NA] * len(df), dtype="boolean")
    if COL_BALL_INTERP not in df.columns:
        df[COL_BALL_INTERP] = pd.array([pd.NA] * len(df), dtype="boolean")
    if COL_SYNTHETIC not in df.columns:
        df[COL_SYNTHETIC] = False
    for c in _BALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def smooth_ball(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, dict]:
    """Flag ball outliers, interpolate short gaps, S-G smooth, recompute kinematics.

    Non-destructive: raw ``pitch_x_m`` / ``pitch_y_m`` on real ball rows are kept;
    a smoothed track lands in ``ball_x_s_m`` / ``ball_y_s_m`` with velocity/speed/
    acceleration alongside. Synthetic rows are appended for interpolated frames
    (flagged ``ball_interp`` and ``synthetic``). Returns ``(df, meta)``.
    """
    df = _ensure_ball_columns(df.copy())

    ball = df[df["object_id"] == BALL_OBJECT_ID].sort_values("frame")
    present = ball.dropna(subset=["pitch_x_m", "pitch_y_m"])
    # one ball position per frame anchors the trajectory (defensive: Layer 1
    # emits a single ball row per frame, but never assume it).
    present = present.drop_duplicates("frame", keep="first")
    if len(present) < 2:
        meta = dict(params=_ball_params(cfg), n_ball_frames=int(len(present)),
                    n_outliers=0, n_interpolated=0, note="too few ball points to smooth")
        return df, meta

    frames = present["frame"].to_numpy(dtype=int)
    xs = present["pitch_x_m"].to_numpy(dtype=float)
    ys = present["pitch_y_m"].to_numpy(dtype=float)

    outlier_mask = _flag_outliers(frames, xs, ys, cfg)
    outlier_frames = set(frames[outlier_mask].tolist())

    # clean = present minus rejected spikes; these anchor interpolation/smoothing
    clean_f = frames[~outlier_mask]
    clean_x = xs[~outlier_mask]
    clean_y = ys[~outlier_mask]

    # build the "known" track: measured clean frames + interpolated short gaps
    known_f, known_x, known_y = [], [], []
    interp_frames = set()
    for i in range(len(clean_f)):
        known_f.append(int(clean_f[i]))
        known_x.append(float(clean_x[i]))
        known_y.append(float(clean_y[i]))
        if i + 1 < len(clean_f):
            gap = int(clean_f[i + 1] - clean_f[i]) - 1  # missing frames between
            if 1 <= gap <= cfg.ball_max_interp_gap:
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    fr = int(clean_f[i]) + k
                    known_f.append(fr)
                    known_x.append(float(clean_x[i] + frac * (clean_x[i + 1] - clean_x[i])))
                    known_y.append(float(clean_y[i] + frac * (clean_y[i + 1] - clean_y[i])))
                    interp_frames.add(fr)

    known_f = np.array(known_f, dtype=int)
    known_x = np.array(known_x, dtype=float)
    known_y = np.array(known_y, dtype=float)
    order = np.argsort(known_f)
    known_f, known_x, known_y = known_f[order], known_x[order], known_y[order]

    # per-frame smoothed results across each contiguous segment
    results = {}  # frame -> (xs, ys, vx, vy, speed, accel)
    dt = 1.0 / cfg.fps
    for a, b in _segments(known_f):
        seg_f = known_f[a:b]
        sx = savgol_smooth(known_x[a:b], cfg.ball_savgol_window, cfg.ball_savgol_order)
        sy = savgol_smooth(known_y[a:b], cfg.ball_savgol_window, cfg.ball_savgol_order)
        if len(seg_f) >= 2:
            vx = np.gradient(sx, dt)
            vy = np.gradient(sy, dt)
        else:
            vx = np.zeros_like(sx)
            vy = np.zeros_like(sy)
        speed = np.hypot(vx, vy)
        accel = np.gradient(speed, dt) if len(seg_f) >= 2 else np.zeros_like(speed)
        for j, fr in enumerate(seg_f):
            results[int(fr)] = (sx[j], sy[j], vx[j], vy[j], speed[j], accel[j])

    # write onto existing ball rows
    ball_idx = df.index[df["object_id"] == BALL_OBJECT_ID]
    existing_frames = set(df.loc[ball_idx, "frame"].astype(int).tolist())
    for idx in ball_idx:
        fr = int(df.at[idx, "frame"])
        df.at[idx, COL_BALL_OUTLIER] = fr in outlier_frames
        df.at[idx, COL_BALL_INTERP] = False
        if fr in results:
            sx, sy, vx, vy, sp, ac = results[fr]
            df.at[idx, COL_BALL_XS] = sx
            df.at[idx, COL_BALL_YS] = sy
            df.at[idx, COL_BALL_VX] = vx
            df.at[idx, COL_BALL_VY] = vy
            df.at[idx, COL_BALL_SPEED] = sp
            df.at[idx, COL_BALL_ACCEL] = ac

    # append synthetic rows for interpolated frames that had no measurement
    new_rows = []
    known_pos = {int(f): (float(x), float(y)) for f, x, y in zip(known_f, known_x, known_y)}
    for fr in sorted(interp_frames):
        if fr in existing_frames:
            continue  # a real (outlier) row already exists for this frame
        rx, ry = known_pos[fr]
        sx, sy, vx, vy, sp, ac = results.get(fr, (rx, ry, 0.0, 0.0, 0.0, 0.0))
        row = {c: np.nan for c in df.columns}
        row.update(
            frame=fr,
            time_s=round(fr / cfg.fps, 4),
            object_id=BALL_OBJECT_ID,
            role="ball",
            team=np.nan,
            pitch_x_m=rx,
            pitch_y_m=ry,
            pitch_valid=False,
            **{
                COL_SYNTHETIC: True,
                COL_BALL_INTERP: True,
                COL_BALL_OUTLIER: False,
                COL_BALL_XS: sx,
                COL_BALL_YS: sy,
                COL_BALL_VX: vx,
                COL_BALL_VY: vy,
                COL_BALL_SPEED: sp,
                COL_BALL_ACCEL: ac,
            },
        )
        if "stable_id" in df.columns:
            row["stable_id"] = BALL_OBJECT_ID
        new_rows.append(row)

    if new_rows:
        add = pd.DataFrame(new_rows)
        df = pd.concat([df, add], ignore_index=True)
        # restore dtypes that a concat with fresh rows can disturb
        df["object_id"] = df["object_id"].astype("Int64")
        if "stable_id" in df.columns:
            df["stable_id"] = df["stable_id"].astype("Int64")
        df[COL_BALL_OUTLIER] = df[COL_BALL_OUTLIER].astype("boolean")
        df[COL_BALL_INTERP] = df[COL_BALL_INTERP].astype("boolean")
        df[COL_SYNTHETIC] = df[COL_SYNTHETIC].astype(bool)

    meta = dict(
        params=_ball_params(cfg),
        n_ball_frames=int(len(present)),
        n_outliers=int(outlier_mask.sum()),
        n_interpolated=int(len(interp_frames)),
        n_synthetic_rows=len(new_rows),
        note=(
            "ball_outlier flags isolated impossible-speed spikes; ball_x_s_m/"
            "ball_y_s_m are the smoothed track; velocity/speed/accel derived from it."
        ),
    )
    return df, meta


def _ball_params(cfg: PrepConfig) -> dict:
    return dict(
        max_speed_ms=cfg.ball_max_speed_ms,
        max_interp_gap=cfg.ball_max_interp_gap,
        savgol_window=cfg.ball_savgol_window,
        savgol_order=cfg.ball_savgol_order,
        pitch_stride=cfg.pitch_stride,
    )
