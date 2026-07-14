"""The actions-review overlay's pure geometry (no cv2 needed).

The rendering itself can only be judged by watching it, but the parts that decide
*what* gets drawn on a given frame are ordinary logic and are pinned here: the
frame -> segment/touch/action lookup, and the strip colouring that makes the
SEGMENTS -> TOUCHES -> ACTIONS derivation legible.

``import src.actions.review`` must not require the CV stack -- that is asserted
too, since a lazy import is easy to break by accident.
"""

import numpy as np

from src.actions.review import (
    ACTION_BGR,
    IDLE_BGR,
    action_color,
    run_strip,
    segment_color,
    spans_to_lookup,
)


def test_importing_the_review_module_does_not_require_cv2():
    """cv2 is imported lazily inside render_review, never at module scope.

    Easy to break by accident (one top-level ``import cv2``), and the breakage is
    invisible until someone without the CV stack tries to import the package.
    """
    import importlib

    module = importlib.import_module("src.actions.review")
    assert hasattr(module, "render_review")


def test_spans_map_each_frame_to_the_run_containing_it():
    frames = np.arange(0, 10)
    spans = [(0, 2, "a"), (5, 7, "b")]

    got = spans_to_lookup(spans, frames)

    assert got[0] == "a" and got[2] == "a"
    assert got[3] is None and got[4] is None      # the gap between the runs
    assert got[5] == "b" and got[7] == "b"
    assert got[8] is None and got[9] is None


def test_a_later_span_wins_on_overlap():
    """A turnover's two rows share a timestamp; the LAST one explains what follows."""
    frames = np.arange(0, 5)
    spans = [(0, 3, "carry"), (3, 4, "pass")]

    got = spans_to_lookup(spans, frames)

    assert got[2] == "carry"
    assert got[3] == "pass"     # the action that ENDS the touch wins the frame


def test_spans_outside_the_frame_range_are_ignored_not_crashed():
    frames = np.array([10, 11, 12])
    got = spans_to_lookup([(0, 3, "x"), (11, 11, "y")], frames)
    assert got == [None, "y", None]


def test_segment_colour_alternates_so_boundaries_are_visible():
    """Consecutive runs of the SAME team must not blend into one block.

    Two touches by two different players on the same team, rendered the same
    colour, would look like one touch -- and hide exactly what the strip exists to
    show.
    """
    a = segment_color((0, 0.0, 7))
    b = segment_color((1, 0.0, 9))    # same team, next run
    assert a != b
    assert segment_color(None) == IDLE_BGR


def test_action_colour_is_keyed_by_spadl_type():
    assert action_color((0, "pass")) == ACTION_BGR["pass"]
    assert action_color((1, "interception")) == ACTION_BGR["interception"]
    assert action_color(None) == IDLE_BGR


def test_every_emitted_action_type_has_a_colour():
    """A type with no colour would render as 'unknown' and quietly mislead."""
    from src.actions.spadl import EMITTED_ACTIONTYPES

    assert set(EMITTED_ACTIONTYPES) <= set(ACTION_BGR)


def test_the_strip_resamples_the_whole_clip_onto_its_width():
    """750 frames onto a 400 px strip: every column coloured, correct shape."""
    payloads = [(0, "pass")] * 375 + [(1, "dribble")] * 375
    strip = run_strip(payloads, width=400, height=12, color_of=action_color)

    assert strip.shape == (12, 400, 3)
    assert tuple(strip[0, 0]) == ACTION_BGR["pass"]        # first half
    assert tuple(strip[0, 399]) == ACTION_BGR["dribble"]   # second half


def test_an_empty_clip_yields_a_blank_strip_rather_than_an_error():
    strip = run_strip([], width=100, height=10, color_of=action_color)
    assert strip.shape == (10, 100, 3)
    assert not strip.any()
