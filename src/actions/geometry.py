"""Ball and player geometry: where the ball went between two touches.

The transition rules need three geometric questions answered, all in the target
(105x68) frame:

1. **Where were the two players** at the moment a touch ended and the next began?
   Those are the pass's release and reception points -- the ``start_x/y`` and
   ``end_x/y`` of the SPADL row. See "Endpoints are anchored on players" below;
   this is *not* the ball's position, and that is deliberate.
2. **How did the ball travel** through the gap? A straight 30 m path is a pass; a
   3 m wobble is a deflection or a tackle. :class:`GapPath` carries the features
   that decide this, and the ball path is used for **nothing else**.
3. **Was the ball airborne** across the gap? If so its ground path is a z=0
   artefact and the straightness guard must not be applied to it.

Endpoints are anchored on PLAYERS, not on the ball
--------------------------------------------------
The homography maps the image to the **ground plane (z=0)**, so a ball in the air
back-projects to a point stretched away from the camera -- wrong by metres, with
no height channel anywhere to correct it. An aerial pass's mid-flight ball
coordinates are therefore fiction, and they used to *set the emitted geometry* of
the very actions xT and VAEP are computed from.

So the emitted endpoints come from the **players**:

- a pass **starts** where the PASSER was on the last frame he controlled the ball
  (the outgoing touch's final frame),
- and **ends** where the RECEIVER was on the first frame he controlled it (the
  incoming touch's first frame) -- the reception point.
- Turnovers anchor the same way: the loser's position at the loss, the winner's
  at the win.
- Carries too, which additionally makes the chain **exactly** continuous: a pass
  ends at the receiver's position on frame *f*, and the receiver's carry starts at
  his position on frame *f*. The same point, not a point 3 m away. socceraction's
  own converters treat ``dribble`` as precisely this connector.

A player standing over the ball is on the ground, and his position is measured on
the plane the homography is actually valid for. His error is bounded by the
possession radius (a possessor is within ``r_pz_m`` of the ball by definition);
the airborne ball's error is bounded by nothing.

The ball path still decides **what the action was** -- whether the ball travelled
at all (a tackle takes it off a settled player; an interception cuts out a ball in
flight), how straight it flew, whether the gap is credible. It just never again
says *where the action happened*. Since SPADL actions are ``start -> end`` and not
trajectories, this alone keeps every distorted mid-flight coordinate out of the
event stream, and out of xT and VAEP.

Occlusion policy
----------------
``no_ball`` frames are occlusion, never a stoppage. A transition spanning them is
still emitted and flagged :attr:`GapPath.occluded` -- a transition unseen is not a
transition that did not happen. Note the endpoints no longer *depend* on the ball
being visible, so occlusion now degrades only the ball-path *characterisation*
(travel, coherence), not the geometry itself.

The ball-free hook
------------------
:meth:`BallTrack.xy` returning ``None`` for every frame is a legitimate state: a
future ball-free possession source would produce exactly that. Player-anchored
endpoints mean such a source now produces a fully-specified action chain rather
than a degraded one -- only the gap *characterisation* is lost. That is the seam a
PathCRF-style source plugs into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BALL_OBJECT_ID,
    COL_BALL_XTS,
    COL_BALL_YTS,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    COL_STABLE_ID,
    PEOPLE_ROLES,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle: aerial imports config, not us
    from .aerial import AerialTrack

Point = Tuple[float, float]


class BallTrack:
    """Smoothed ball position by frame, in the target frame. Missing => ``None``."""

    def __init__(self, xy: Dict[int, Point]):
        self._xy = xy

    @classmethod
    def from_prepared(cls, df: pd.DataFrame) -> "BallTrack":
        """Read ``ball_x_ts_m`` / ``ball_y_ts_m`` off the ball rows.

        Rows whose smoothed position was rejected as an outlier and not
        interpolated are null, and are simply absent here -- those frames are the
        ``no_ball`` ones.
        """
        ball = df[df["object_id"] == BALL_OBJECT_ID].dropna(
            subset=[COL_BALL_XTS, COL_BALL_YTS]
        )
        xy = {
            int(f): (float(x), float(y))
            for f, x, y in zip(
                ball["frame"], ball[COL_BALL_XTS], ball[COL_BALL_YTS]
            )
        }
        return cls(xy)

    def xy(self, frame: int) -> Optional[Point]:
        """Ball position on a frame, or ``None`` if it was not usable there."""
        return self._xy.get(int(frame))

    def polyline(self, first: int, last: int) -> np.ndarray:
        """Every usable ball position in ``[first, last]``, as an ``(n, 2)`` array."""
        pts = [self._xy[f] for f in range(int(first), int(last) + 1) if f in self._xy]
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self._xy)


class PlayerTrack:
    """Player position by ``(stable_id, frame)``, in the target frame."""

    def __init__(self, xy: Dict[Tuple[int, int], Point]):
        self._xy = xy

    @classmethod
    def from_prepared(cls, df: pd.DataFrame) -> "PlayerTrack":
        people = df[df["role"].isin(PEOPLE_ROLES)].dropna(
            subset=[COL_STABLE_ID, COL_PITCH_X_T, COL_PITCH_Y_T]
        )
        xy = {
            (int(pid), int(f)): (float(x), float(y))
            for pid, f, x, y in zip(
                people[COL_STABLE_ID], people["frame"],
                people[COL_PITCH_X_T], people[COL_PITCH_Y_T],
            )
        }
        return cls(xy)

    def xy(self, player_id: Optional[int], frame: int) -> Optional[Point]:
        if player_id is None:
            return None
        return self._xy.get((int(player_id), int(frame)))


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def _known(*points: Point) -> bool:
    """True when every point has a usable (non-null) coordinate pair."""
    return all(np.isfinite(p[0]) and np.isfinite(p[1]) for p in points)


@dataclass(frozen=True)
class GapPath:
    """The journey between the end of one touch and the start of the next.

    Two coordinate stories live here and must not be confused. ``start`` / ``end``
    are the **emitted geometry** and come from the *players*. Everything named
    ``ball_*`` / ``path_*`` is the **ball's characterisation** of the gap, used to
    decide what kind of action it was, and never to say where it happened.

    ``start`` / ``end``
        The release and reception points: the PASSER's position on the last frame
        he controlled the ball, and the RECEIVER's on the first frame he did.
        These become the SPADL ``start_x/y`` and ``end_x/y``. The ball is used only
        if the player has no position at all on that frame (see :meth:`_endpoint`).
    ``travel_m``
        Straight-line ``start`` -> ``end`` distance: the length of the ACTION.
        Drives short/long and the progressive/back call.
    ``ball_travel_m``
        Straight-line distance the BALL covered across the gap. This is the
        interception-vs-tackle discriminator (a tackle takes the ball off a player,
        so the ball hardly moves; an interception cuts out a ball in flight) and
        the "did anything actually happen here" credibility test. ``NaN`` when the
        ball was never seen -- callers fall back to :attr:`validation_travel_m`.
    ``path_m``
        Length of the ball's actual polyline through the gap.
    ``coherence``
        ``ball_travel_m / path_m`` in ``[0, 1]``. 1.0 is a laser-straight delivery;
        a low value is a ball wobbling, ricocheting or being scrambled -- i.e. an
        aimless deflection rather than an intentional pass. **Meaningless when
        :attr:`airborne`**: a flighted ball's ground path is a z=0 back-projection
        artefact, so it bends and wanders no matter how cleanly it was struck. The
        guard is relaxed for those gaps -- see ``ActionConfig.coherence_floor``.
    ``heading_deg``
        Direction of the action, degrees, ``atan2(dy, dx)`` in pitch coordinates.
    ``airborne`` / ``aerial_conf``
        The ball was off the ground somewhere in this gap, and how sure we are
        (0-1). From :mod:`src.actions.aerial` -- a heuristic image-space detector,
        not a height measurement.
    ``occluded``
        The ball was missing (``no_ball``) somewhere in the span. The action is
        still emitted -- a transition unseen is not a transition that did not
        happen -- but it is flagged, never silently trusted. Note this no longer
        threatens the *geometry* (which is player-anchored), only the ball-path
        characterisation.
    ``player_fallback``
        We had no position for a player at his own endpoint frame and had to use
        the ball's instead. Rare, and flagged: this is the one path by which a
        ball coordinate can still reach the emitted geometry.
    ``n_no_ball``
        How many frames of the span had no usable ball position.
    """

    start: Point
    end: Point
    travel_m: float
    ball_travel_m: float
    path_m: float
    coherence: float
    heading_deg: float
    airborne: bool
    aerial_conf: float
    occluded: bool
    player_fallback: bool
    n_no_ball: int
    n_gap_frames: int

    @property
    def validation_travel_m(self) -> float:
        """How far the ball went, for the guards that ask "did anything happen?".

        The ball's own travel, falling back to the action's when the ball was never
        seen at either endpoint. The fallback matters for a ball-free source, where
        the player positions are the only evidence there is -- refusing every gap
        for want of a ball would be the wrong answer, not a safe one.
        """
        if np.isfinite(self.ball_travel_m):
            return float(self.ball_travel_m)
        return float(self.travel_m)

    @staticmethod
    def _endpoint(
        ball: BallTrack, players: PlayerTrack, frame: int, player_id: Optional[int]
    ) -> Tuple[Point, bool]:
        """The PLAYER's position on ``frame``; the ball's only if he has none.

        The priority is the whole point of CHANGE 1 and it is the reverse of what
        it used to be. On the frame a pass is released the ball may well be in the
        air already (it is, by the next frame), and its z=0 back-projection is
        stretched away from the camera by metres. The player who struck it is
        standing on the grass -- the exact plane the homography is valid for.

        The second element is ``True`` when the ball fallback was used, which is
        the only remaining route by which a ball coordinate can set emitted
        geometry. It is flagged rather than silently taken.
        """
        pt = players.xy(player_id, frame)
        if pt is not None:
            return pt, False
        pt = ball.xy(frame)
        if pt is not None:
            return pt, True
        return (np.nan, np.nan), True

    @classmethod
    def from_tracks(
        cls,
        ball: BallTrack,
        players: PlayerTrack,
        end_frame_a: int,
        start_frame_b: int,
        player_a: Optional[int],
        player_b: Optional[int],
        aerial: Optional["AerialTrack"] = None,
    ) -> "GapPath":
        """Build the path for the gap between touch A (ends) and touch B (begins).

        The gap frames themselves are ``end_frame_a + 1 .. start_frame_b - 1`` and
        may be empty (adjacent touches -- a hand-to-hand change of possession).
        The ball *path* is measured over the closed interval ``[end_frame_a,
        start_frame_b]``, so that it includes the ball leaving A and arriving at B;
        the *airborne* question is asked of the gap's interior only, because a ball
        sitting at a player's feet on a touch frame is by definition not in flight.
        """
        start, start_fb = cls._endpoint(ball, players, end_frame_a, player_a)
        end, end_fb = cls._endpoint(ball, players, start_frame_b, player_b)

        n_gap = max(0, int(start_frame_b) - int(end_frame_a) - 1)
        seen = ball.polyline(end_frame_a, start_frame_b)
        n_no_ball = (int(start_frame_b) - int(end_frame_a) + 1) - len(seen)

        travel = _dist(start, end) if _known(start, end) else np.nan

        # The BALL's own displacement across the gap -- what the guards mean by
        # "did the ball actually go anywhere". Measured on the ball, because the
        # question is about the ball; the players have already told us where the
        # action was.
        ball_a, ball_b = ball.xy(end_frame_a), ball.xy(start_frame_b)
        ball_travel = (
            _dist(ball_a, ball_b)
            if ball_a is not None and ball_b is not None
            else np.nan
        )

        # Polyline length over the ball positions we actually have. With fewer
        # than two of them there is no path to measure, so the straight line is
        # the best (and only) statement we can make -- coherence is then 1.0 by
        # convention, and `occluded` is what tells you not to trust it.
        if len(seen) >= 2:
            steps = np.hypot(np.diff(seen[:, 0]), np.diff(seen[:, 1]))
            path = float(steps.sum())
        elif np.isfinite(ball_travel):
            path = float(ball_travel)
        else:
            path = float(travel) if np.isfinite(travel) else 0.0

        ref = ball_travel if np.isfinite(ball_travel) else travel
        coherence = (
            1.0 if path <= 1e-9
            else float(np.clip((ref if np.isfinite(ref) else path) / path, 0.0, 1.0))
        )
        heading = (
            float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))
            if np.isfinite(travel) else np.nan
        )

        # Was the ball in the air on its way across? Only the gap's interior can
        # be: on the touch frames themselves a player is within the possession
        # radius of it, which is what made him the possessor.
        if aerial is not None and n_gap > 0:
            airborne, aerial_conf = aerial.over(
                int(end_frame_a) + 1, int(start_frame_b) - 1
            )
        else:
            airborne, aerial_conf = False, 0.0

        return cls(
            start=start,
            end=end,
            travel_m=float(travel),
            ball_travel_m=float(ball_travel),
            path_m=path,
            coherence=coherence,
            heading_deg=heading,
            airborne=bool(airborne),
            aerial_conf=float(aerial_conf),
            occluded=bool(n_no_ball > 0),
            player_fallback=bool(start_fb or end_fb),
            n_no_ball=int(n_no_ball),
            n_gap_frames=n_gap,
        )


def max_ball_distance(
    ball: BallTrack,
    players: PlayerTrack,
    first: int,
    last: int,
    player_id: Optional[int],
) -> float:
    """Furthest the ball got from one player over ``[first, last]``, in metres.

    The test for "did the ball stay with this carrier?". Returns ``0.0`` when
    there is nothing to compare (no ball or no player position anywhere in the
    range) -- absence of evidence that the ball left is not evidence that it did,
    and the caller flags such spans as occluded anyway.
    """
    dists = [
        _dist(b, p)
        for f in range(int(first), int(last) + 1)
        if (b := ball.xy(f)) is not None and (p := players.xy(player_id, f)) is not None
    ]
    return max(dists) if dists else 0.0


def carry_path(
    ball: BallTrack,
    players: PlayerTrack,
    start_frame: int,
    end_frame: int,
    player_id: Optional[int],
    aerial: Optional["AerialTrack"] = None,
) -> GapPath:
    """The journey *within* one touch: the carry from reception to release.

    Same geometry as a gap, measured over the touch's own frames -- and with the
    same split of duties. ``start`` / ``end`` are the CARRIER's own positions on
    the touch's first and last frames, which is what makes the chain exactly
    continuous (the pass that fed him ends on the same frame, at the same point)
    and is what socceraction means by a ``dribble``: the connector between the
    action before and the action after.

    But ``ball_travel_m`` -- the BALL's displacement across the touch -- is what
    decides whether the touch earns a dribble at all. That distinction is the
    whole rule: **a touch where the ball sits still is the ball parked near a
    player, not a dribble.** Anchoring the gate on the player instead would mint a
    dribble every time someone jogged past a stationary ball.
    """
    return GapPath.from_tracks(
        ball, players, start_frame, end_frame, player_id, player_id, aerial
    )


def dist_to_goal(point: Point, goal: Point) -> float:
    """Straight-line distance from a point to the centre of a goal mouth.

    Milestone 1 uses this only to describe progression. It is also the primitive
    a shot detector will need, which is why goal geometry is derived here rather
    than inlined into the pass classifier.
    """
    if not _known(point):
        return float("nan")
    return _dist(point, goal)
