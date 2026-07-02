#!/usr/bin/env python3
"""Entry point for the extractor.

The implementation lives in the ``src`` package (see ``src/pipeline.py``); this
launcher just calls it, so ``python main.py --source ...`` works without an
install (e.g. the Kaggle notebook workflow). It mirrors the installed
``football-ai`` console entry point and ``python -m src.cli``.
"""

from src.cli import main

if __name__ == "__main__":
    main()
