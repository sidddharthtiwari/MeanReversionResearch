"""Rolling z-score feature computation for quantitative research.

Z-score is defined as ``(value - rolling_mean) / rolling_std``. This module
orchestrates existing rolling feature helpers and does not load data, validate
datasets, generate signals, plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling import compute_rolling_mean, compute_rolling_std

__all__ = [
    "compute_zscore",
]

ZSCORE_PREFIX = "zscore"
_TEMP_ROLLING_MEAN_COLUMN = "__rolling_mean_temp__"
_TEMP_ROLLING_STD_COLUMN = "__rolling_std_temp__"


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the z-score output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``zscore_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{ZSCORE_PREFIX}_{window}"


def _compute_zscore(
    df: pd.DataFrame,
    window: int,
    column: str,
    output_column: str | None,
    min_periods: int | None,
) -> pd.DataFrame:
    """Compute a rolling z-score by composing rolling mean and std helpers.

    Temporary rolling mean and standard deviation columns are used internally
    and are not exposed in the returned DataFrame.

    Args:
        df: Input DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size.
        column: Numeric column to standardize.
        output_column: Explicit z-score column name, or ``None`` for default.
        min_periods: Minimum observations required for the rolling window.

    Returns:
        A new DataFrame containing the original columns plus the z-score
        column. The caller's DataFrame is never modified.
    """
    name = _generate_output_column_name(window, output_column)

    with_mean = compute_rolling_mean(
        df,
        window=window,
        column=column,
        output_column=_TEMP_ROLLING_MEAN_COLUMN,
        min_periods=min_periods,
    )
    with_std = compute_rolling_std(
        df,
        window=window,
        column=column,
        output_column=_TEMP_ROLLING_STD_COLUMN,
        min_periods=min_periods,
    )

    rolling_mean = with_mean[_TEMP_ROLLING_MEAN_COLUMN]
    rolling_std = with_std[_TEMP_ROLLING_STD_COLUMN]

    result = df.copy()
    values = result[column]
    result[name] = (values - rolling_mean) / rolling_std
    return result


def compute_zscore(
    df: pd.DataFrame,
    window: int,
    column: str,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the rolling z-score of a numeric column.

    Z-score is ``(value - rolling_mean) / rolling_std`` over ``window``.

    Args:
        df: DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size. Must be a positive integer.
        column: Numeric column to standardize.
        output_column: Optional output column name. Defaults to
            ``zscore_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the z-score column appended. Intermediate rolling
        mean and standard deviation columns are not exposed. The input is
        unchanged.
    """
    return _compute_zscore(
        df=df,
        window=window,
        column=column,
        output_column=output_column,
        min_periods=min_periods,
    )
