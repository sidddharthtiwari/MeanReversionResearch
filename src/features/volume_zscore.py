"""Rolling volume z-score feature computation for quantitative research.

Volume z-score is ``(volume - rolling_mean(volume)) / rolling_std(volume)``.
This module orchestrates existing rolling feature helpers and does not load
data, validate datasets, generate signals, plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling import compute_rolling_mean, compute_rolling_std

__all__ = [
    "compute_volume_zscore",
]

DEFAULT_VOLUME_COLUMN = "volume"
VOLUME_ZSCORE_PREFIX = "volume_zscore"
_TEMP_ROLLING_MEAN_COLUMN = "__rolling_mean_temp__"
_TEMP_ROLLING_STD_COLUMN = "__rolling_std_temp__"


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the volume z-score output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``volume_zscore_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{VOLUME_ZSCORE_PREFIX}_{window}"


def _compute_volume_zscore(
    df: pd.DataFrame,
    window: int,
    volume_column: str,
    output_column: str | None,
    min_periods: int | None,
) -> pd.DataFrame:
    """Compute volume z-score by composing rolling mean and std helpers.

    Temporary rolling mean and standard deviation columns are used internally
    and are not exposed in the returned DataFrame.

    Args:
        df: Input DataFrame containing ``symbol`` and ``volume_column``.
        window: Rolling window size.
        volume_column: Numeric volume column to standardize.
        output_column: Explicit z-score column name, or ``None`` for default.
        min_periods: Minimum observations required for the rolling window.

    Returns:
        A new DataFrame containing the original columns plus the volume
        z-score column. The caller's DataFrame is never modified.
    """
    name = _generate_output_column_name(window, output_column)

    with_mean = compute_rolling_mean(
        df,
        window=window,
        column=volume_column,
        output_column=_TEMP_ROLLING_MEAN_COLUMN,
        min_periods=min_periods,
    )
    with_std = compute_rolling_std(
        df,
        window=window,
        column=volume_column,
        output_column=_TEMP_ROLLING_STD_COLUMN,
        min_periods=min_periods,
    )

    rolling_mean = with_mean[_TEMP_ROLLING_MEAN_COLUMN]
    rolling_std = with_std[_TEMP_ROLLING_STD_COLUMN]

    result = df.copy()
    values = result[volume_column]
    result[name] = (values - rolling_mean) / rolling_std
    return result


def compute_volume_zscore(
    df: pd.DataFrame,
    window: int,
    volume_column: str = DEFAULT_VOLUME_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling z-score of volume.

    Volume z-score is ``(volume - rolling_mean) / rolling_std`` over
    ``window``.

    Args:
        df: DataFrame containing ``symbol`` and a volume column.
        window: Rolling window size. Must be a positive integer.
        volume_column: Numeric volume column to standardize.
        output_column: Optional output column name. Defaults to
            ``volume_zscore_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the volume z-score column appended. Intermediate
        rolling mean and standard deviation columns are not exposed. The
        input is unchanged.
    """
    return _compute_volume_zscore(
        df=df,
        window=window,
        volume_column=volume_column,
        output_column=output_column,
        min_periods=min_periods,
    )
