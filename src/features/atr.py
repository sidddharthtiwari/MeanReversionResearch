"""Average True Range (ATR) feature computation for quantitative research.

ATR is the rolling mean of True Range. This module orchestrates existing True
Range and rolling feature helpers and does not load data, validate datasets,
generate signals, plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling import compute_rolling_mean
from src.features.true_range import (
    DEFAULT_CLOSE_COLUMN,
    DEFAULT_HIGH_COLUMN,
    DEFAULT_LOW_COLUMN,
    compute_true_range,
)

__all__ = [
    "compute_atr",
]

ATR_PREFIX = "atr"
_TEMP_TRUE_RANGE_COLUMN = "__true_range_temp__"


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the ATR output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``atr_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{ATR_PREFIX}_{window}"


def _compute_atr(
    df: pd.DataFrame,
    window: int,
    high_column: str,
    low_column: str,
    close_column: str,
    output_column: str | None,
    min_periods: int | None,
) -> pd.DataFrame:
    """Compute ATR by composing True Range and rolling mean helpers.

    A temporary True Range column is used internally and is not exposed in the
    returned DataFrame.

    Args:
        df: Input OHLC DataFrame containing ``symbol`` and price columns.
        window: Rolling window size for the ATR mean.
        high_column: High price column.
        low_column: Low price column.
        close_column: Close price column used for True Range gaps.
        output_column: Explicit ATR column name, or ``None`` for default.
        min_periods: Minimum observations required for the rolling window.

    Returns:
        A new DataFrame containing the original columns plus the ATR column.
        The caller's DataFrame is never modified.
    """
    name = _generate_output_column_name(window, output_column)

    with_true_range = compute_true_range(
        df,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
        output_column=_TEMP_TRUE_RANGE_COLUMN,
    )
    with_atr = compute_rolling_mean(
        with_true_range,
        window=window,
        column=_TEMP_TRUE_RANGE_COLUMN,
        output_column=name,
        min_periods=min_periods,
    )

    result = df.copy()
    result[name] = with_atr[name]
    return result


def compute_atr(
    df: pd.DataFrame,
    window: int,
    high_column: str = DEFAULT_HIGH_COLUMN,
    low_column: str = DEFAULT_LOW_COLUMN,
    close_column: str = DEFAULT_CLOSE_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute Average True Range (ATR).

    ATR is the per-symbol rolling mean of True Range over ``window``.

    Args:
        df: OHLC DataFrame containing ``symbol`` and price columns.
        window: Rolling window size. Must be a positive integer.
        high_column: High price column.
        low_column: Low price column.
        close_column: Close price column used for True Range gaps.
        output_column: Optional output column name. Defaults to
            ``atr_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the ATR column appended. Intermediate True Range
        columns are not exposed. The input is unchanged.
    """
    return _compute_atr(
        df=df,
        window=window,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
        output_column=output_column,
        min_periods=min_periods,
    )
