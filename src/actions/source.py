"""The possession source: the ONLY way this layer learns who is on the ball.

The event layer must never read the possession-zone detector's internals. It
consumes a **stream of** ``(frame, time_s, possessor_id, team, state)`` records
and nothing else, so a completely different possession model -- notably a
ball-free, PathCRF-style one that infers the possessor from player motion alone
-- can be dropped in by implementing one method.

The contract
------------
Implement :meth:`PossessionSource.stream`. That is the whole interface. Segments
("touches") are *derived from the stream* by :func:`segments_from_stream`, so a
new source does **not** have to produce a segments table, and cannot disagree
with one: there is a single definition of a segment in this codebase and it
lives here.

    class PathCRFPossessionSource(PossessionSource):
        def stream(self):
            for f, pid, team in self.model.decode():
                yield PossessionFrame(f, f / self.fps, pid, team, STATE_POSSESSION)

:class:`ZonePossessionSource` is the concrete source for milestone 1: it wraps
``possession_frames.parquet`` from ``src.possession``. Note it reads only the
*frames* table -- not ``possession_segments.parquet`` -- precisely so that the
coupling stays at the level of the stream.

A segment is a maximal run of consecutive frames with the same possessor. This
matches ``src.possession.segments`` by construction (same rule, applied to the
same stream); the smoke test asserts the two agree on the real clip, which is
what makes swapping the source a *safe* change rather than a hopeful one.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd

from .config import STATE_CONTESTED, STATE_NO_BALL

SEGMENT_COLUMNS = [
    "segment_id", "possessor_id", "team", "start_frame", "end_frame",
    "n_frames", "start_time_s", "end_time_s", "n_contested",
]


@dataclass(frozen=True)
class PossessionFrame:
    """Who held the ball on one frame.

    ``possessor_id`` and ``team`` are ``None`` on any frame with no possessor
    (``loose`` / ``no_ball``). A source must never fabricate a possessor: the
    event layer relies on the absence of one to know a gap exists.
    """

    frame: int
    time_s: float
    possessor_id: Optional[int]
    team: Optional[int]
    state: str


class PossessionSource(ABC):
    """A per-frame possessor stream. Implement :meth:`stream`; that is all."""

    @abstractmethod
    def stream(self) -> Iterator[PossessionFrame]:
        """Yield one :class:`PossessionFrame` per frame, in ascending frame order."""

    def segments(self) -> pd.DataFrame:
        """Collapse the stream into possession segments (touches).

        Derived, never delegated -- see the module docstring.
        """
        return segments_from_stream(self.stream())

    def frames(self) -> pd.DataFrame:
        """The stream as a table (used for the occlusion/no_ball bookkeeping)."""
        rows = [
            (p.frame, p.time_s, p.possessor_id, p.team, p.state) for p in self.stream()
        ]
        return pd.DataFrame(
            rows, columns=["frame", "time_s", "possessor_id", "team", "state"]
        )


class ZonePossessionSource(PossessionSource):
    """The possession-zone detector as a source (``possession_frames.parquet``)."""

    def __init__(self, frames: pd.DataFrame):
        missing = {"frame", "state", "possessor_id"} - set(frames.columns)
        if missing:
            raise KeyError(
                f"possession frames table is missing {sorted(missing)}; run "
                f"`python -m src.possession detect_possession` first."
            )
        self._frames = frames.sort_values("frame", kind="stable")

    @classmethod
    def from_dir(cls, in_dir: str) -> "ZonePossessionSource":
        """Load ``<in_dir>/possession_frames.parquet``."""
        path = os.path.join(in_dir, "possession_frames.parquet")
        if not os.path.exists(path):
            raise SystemExit(
                f"no possession_frames.parquet in {in_dir!r}; run "
                f"`python -m src.possession detect_possession` first."
            )
        return cls(pd.read_parquet(path))

    def stream(self) -> Iterator[PossessionFrame]:
        for row in self._frames.itertuples(index=False):
            pid = getattr(row, "possessor_id")
            team = getattr(row, "possessor_team", None)
            yield PossessionFrame(
                frame=int(row.frame),
                time_s=float(getattr(row, "time_s", np.nan)),
                possessor_id=None if pd.isna(pid) else int(pid),
                team=None if team is None or pd.isna(team) else int(team),
                state=str(row.state),
            )


def segments_from_stream(stream: Iterable[PossessionFrame]) -> pd.DataFrame:
    """Collapse a possessor stream into maximal same-possessor runs.

    A run is broken by a change of possessor, by a frame with no possessor
    (``loose`` / ``no_ball`` -- the gaps the event layer is made of), and by a
    jump in frame number. Returns one row per segment::

        segment_id, possessor_id, team, start_frame, end_frame, n_frames,
        start_time_s, end_time_s, n_contested

    ``n_contested`` is carried through because a transition out of a heavily
    contested touch is exactly what a future duel-resolution step will want to
    revisit; milestone 1 only uses it to flag confidence.
    """
    rows: List[dict] = []
    current: Optional[dict] = None

    for p in stream:
        if p.possessor_id is None:
            current = None
            continue

        contiguous = (
            current is not None
            and current["possessor_id"] == p.possessor_id
            and p.frame == current["end_frame"] + 1
        )
        if not contiguous:
            current = dict(
                segment_id=len(rows),
                possessor_id=int(p.possessor_id),
                team=None if p.team is None else int(p.team),
                start_frame=p.frame,
                end_frame=p.frame,
                n_frames=0,
                start_time_s=p.time_s,
                end_time_s=p.time_s,
                n_contested=0,
            )
            rows.append(current)

        current["end_frame"] = p.frame
        current["end_time_s"] = p.time_s
        current["n_frames"] += 1
        current["n_contested"] += int(p.state == STATE_CONTESTED)

    if not rows:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    return pd.DataFrame(rows)[SEGMENT_COLUMNS]


def no_ball_frames(frames: pd.DataFrame) -> np.ndarray:
    """The frame numbers with no usable ball position (occlusion, not stoppage)."""
    return frames.loc[frames["state"] == STATE_NO_BALL, "frame"].to_numpy()
