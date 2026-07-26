"""Per-symbol rolling beta feature computation for quantitative research.

Rolling beta is defined as rolling covariance of two columns divided by the
rolling variance of the right column. This module orchestrates existing
rolling feature helpers and does not load data, validate datasets, generate
signals, plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling_covariance import compute_rolling_covariance

__all__ = [
    "compute_rolling_beta",
]

ROLLING_BETA_PREFIX = "rolling_beta"
_TEMP_ROLLING_COVARIANCE_COLUMN = "__rolling_covariance_temp__"
_TEMP_ROLLING_VARIANCE_COLUMN = "__rolling_variance_temp__"


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the rolling-beta output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``rolling_beta_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{ROLLING_BETA_PREFIX}_{window}"


def _compute_rolling_beta(
    df: pd.DataFrame,
    window: int,
    left_column: str,
    right_column: str,
    output_column: str | None,
    min_periods: int | None,
) -> pd.DataFrame:
    """Compute rolling beta by composing rolling covariance helpers.

    Temporary covariance and variance columns are used internally and are not
    exposed in the returned DataFrame. Variance of the right column is obtained
    as its rolling covariance with itself.

    Args:
        df: Input DataFrame containing ``symbol`` and both numeric columns.
        window: Rolling window size.
        left_column: Dependent numeric column (numerator covariance leg).
        right_column: Independent numeric column (denominator variance leg).
        output_column: Explicit beta column name, or ``None`` for default.
        min_periods: Minimum observations required for the rolling window.

    Returns:
        A new DataFrame containing the original columns plus the beta column.
        The caller's DataFrame is never modified.
    """
    name = _generate_output_column_name(window, output_column)

    with_covariance = compute_rolling_covariance(
        df,
        window=window,
        left_column=left_column,
        right_column=right_column,
        output_column=_TEMP_ROLLING_COVARIANCE_COLUMN,
        min_periods=min_periods,
    )
    with_variance = compute_rolling_covariance(
        df,
        window=window,
        left_column=right_column,
        right_column=right_column,
        output_column=_TEMP_ROLLING_VARIANCE_COLUMN,
        min_periods=min_periods,
    )

    rolling_covariance = with_covariance[_TEMP_ROLLING_COVARIANCE_COLUMN]
    rolling_variance = with_variance[_TEMP_ROLLING_VARIANCE_COLUMN]

    result = df.copy()
    result[name] = rolling_covariance / rolling_variance
    return result


def compute_rolling_beta(
    df: pd.DataFrame,
    window: int,
    left_column: str,
    right_column: str,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute per-symbol rolling beta between two numeric columns.

    Rolling beta is ``cov(left, right) / var(right)`` over ``window``.
    Windows never cross symbol boundaries.

    Args:
        df: DataFrame containing ``symbol`` and both numeric columns.
        window: Rolling window size. Must be a positive integer.
        left_column: Dependent numeric column (numerator covariance leg).
        right_column: Independent numeric column (denominator variance leg).
        output_column: Optional output column name. Defaults to
            ``rolling_beta_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling-beta column appended. Intermediate
        covariance and variance columns are not exposed. The input is
        unchanged.
    """
    return _compute_rolling_beta(
        df=df,
        window=window,
        left_column=left_column,
        right_column=right_column,
        output_column=output_column,
        min_periods=min_periods,
    )
