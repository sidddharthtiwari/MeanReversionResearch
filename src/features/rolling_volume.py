"""Rolling average volume feature computation for quantitative research.

Rolling volume is the per-symbol rolling mean of a volume column. This module
orchestrates the existing rolling mean helper and does not load data, validate
datasets, generate signals, plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling import compute_rolling_mean

__all__ = [
    "compute_rolling_volume",
]

DEFAULT_VOLUME_COLUMN = "volume"
ROLLING_VOLUME_PREFIX = "rolling_volume"


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the rolling-volume output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``rolling_volume_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{ROLLING_VOLUME_PREFIX}_{window}"


def _compute_rolling_volume(
    df: pd.DataFrame,
    window: int,
    volume_column: str,
    output_column: str | None,
    min_periods: int | None,
) -> pd.DataFrame:
    """Compute rolling average volume by composing the rolling mean helper.

    Args:
        df: Input DataFrame containing ``symbol`` and ``volume_column``.
        window: Rolling window size.
        volume_column: Numeric volume column to average.
        output_column: Explicit output column name, or ``None`` for default.
        min_periods: Minimum observations required for the rolling window.

    Returns:
        A new DataFrame containing the original columns plus the rolling
        volume column. The caller's DataFrame is never modified.
    """
    name = _generate_output_column_name(window, output_column)
    return compute_rolling_mean(
        df,
        window=window,
        column=volume_column,
        output_column=name,
        min_periods=min_periods,
    )


def compute_rolling_volume(
    df: pd.DataFrame,
    window: int,
    volume_column: str = DEFAULT_VOLUME_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling average of volume.

    Args:
        df: DataFrame containing ``symbol`` and a volume column.
        window: Rolling window size. Must be a positive integer.
        volume_column: Numeric volume column to average.
        output_column: Optional output column name. Defaults to
            ``rolling_volume_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling-volume column appended. The input is
        unchanged.
    """
    return _compute_rolling_volume(
        df=df,
        window=window,
        volume_column=volume_column,
        output_column=output_column,
        min_periods=min_periods,
    )
