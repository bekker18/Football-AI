"""Layer 1 — CV extraction: soccer video -> structured per-frame game state.

The heavy pieces (torch / ultralytics / supervision / roboflow-sports) are only
imported when you actually run the pipeline, so ``import src`` stays cheap. Use
the submodules directly, or the lazy attributes below:

    from src import run, main   # run(args) / CLI entry point
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name):  # PEP 562: keep top-level import free of heavy deps
    if name == "main":
        from .cli import main

        return main
    if name == "run":
        from .pipeline import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
