"""Small shared helpers for the prerequisites transforms."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


def period_groups(
    df: pd.DataFrame, period_col: Optional[str]
) -> Iterator[Tuple[object, pd.Index]]:
    """Yield ``(period_key, row_index)`` for each period.

    When ``period_col`` is None or absent, the whole frame is one period
    (key ``None``). Designed so callers can operate per period without caring
    whether the data is single- or multi-half.
    """
    if period_col and period_col in df.columns:
        for key, idx in df.groupby(period_col).groups.items():
            yield key, idx
    else:
        yield None, df.index
