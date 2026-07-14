"""Optional annotated-video overlay (headless): ellipses + top-down radar.

This is a sanity-check overlay only. Note it renders the *pre-vote* team labels,
because frames are written during the pass while the majority vote runs after —
trust ``tracking.parquet`` over the video for label quality.
"""

from __future__ import annotations

import numpy as np


class VideoAnnotator:
    # palette index: 0 team0, 1 team1, 2 referee, 3 unknown (not yet predicted)
    COLORS = ["#FF1493", "#00BFFF", "#FFD700", "#808080"]

    def __init__(self, path: str, info):
        import supervision as sv
        from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

        from .config import PITCH

        self._sv = sv
        self._draw_pitch = draw_pitch
        self._draw_points = draw_points_on_pitch
        self._pitch = PITCH
        self.info = info
        self.ellipse = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(self.COLORS), thickness=2
        )
        # The ball is tiny on 1080p; a white triangle above it reads far better
        # than a ~10px box. Swap for sv.BoxAnnotator(...) if you want a box.
        self.ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.WHITE, base=25, height=21, outline_thickness=1
        )
        self.sink = sv.VideoSink(path, info)
        self.sink.__enter__()

    def _radar(self, merged, lookup, transformer, ball=None):
        sv = self._sv
        radar = self._draw_pitch(config=self._pitch)
        if transformer is None:
            return radar
        if len(merged):
            xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            txy = transformer.transform_points(xy)
            for i, hexc in enumerate(self.COLORS):
                radar = self._draw_points(
                    config=self._pitch,
                    xy=txy[lookup == i],
                    face_color=sv.Color.from_hex(hexc),
                    radius=20,
                    pitch=radar,
                )
        if ball is not None and len(ball):
            bxy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            radar = self._draw_points(
                config=self._pitch,
                xy=transformer.transform_points(bxy),
                face_color=sv.Color.WHITE,
                radius=20,
                pitch=radar,
            )
        return radar

    def write(
        self, frame, players, goalkeepers, referees, players_team, gk_team,
        transformer, ball=None,
    ):
        sv = self._sv
        from .config import UNKNOWN_COLOR_IDX

        merged = sv.Detections.merge([players, goalkeepers, referees])
        lookup = np.array(
            players_team.tolist() + gk_team.tolist() + [2] * len(referees)
        )  # 2 == referee colour
        # players with team not yet predicted (-1) -> neutral colour, so the
        # annotator never receives a negative index (newer supervision errors).
        if len(lookup):
            lookup[lookup < 0] = UNKNOWN_COLOR_IDX

        annotated = frame.copy()
        if len(merged):
            annotated = self.ellipse.annotate(
                annotated, merged, custom_color_lookup=lookup
            )
        if ball is not None and len(ball):
            annotated = self.ball_annotator.annotate(annotated, ball)
        radar = self._radar(merged, lookup, transformer, ball)
        radar = sv.resize_image(radar, (self.info.width // 2, self.info.height // 2))
        rh, rw, _ = radar.shape
        rect = sv.Rect(
            x=self.info.width // 2 - rw // 2,
            y=self.info.height - rh,
            width=rw,
            height=rh,
        )
        annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
        self.sink.write_frame(annotated)

    def close(self):
        self.sink.__exit__(None, None, None)
