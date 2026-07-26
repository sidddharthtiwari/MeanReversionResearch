"""Per-symbol rolling covariance for quantitative research.

This module computes rolling covariance between two numeric columns within
each symbol. It does not load data, validate datasets, generate signals, plot,
or backtest.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "compute_rolling_covariance",
]

ROLLING_COVARIANCE_PREFIX = "rolling_covariance"


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Validate that ``column`` exists and has a numeric dtype.

    Args:
        df: Input DataFrame containing the column to validate.
        column: Name of the numeric column to validate.

    Raises:
        KeyError: If ``column`` is not present in ``df``.
        TypeError: If ``column`` is not numeric.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' must be numeric, got dtype '{df[column].dtype}'."
        )


def _validate_window(window: int) -> None:
    """Validate that ``window`` is a positive integer.

    Args:
        window: Rolling window size.

    Raises:
        TypeError: If ``window`` is not an ``int``.
        ValueError: If ``window`` is not greater than zero.
    """
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(f"window must be an int, got {type(window).__name__}.")
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}.")


def _validate_min_periods(min_periods: int | None, window: int) -> None:
    """Validate ``min_periods`` relative to ``window``.

    ``None`` is allowed and later resolved to ``window``.

    Args:
        min_periods: Minimum observations required in a window, or ``None``.
        window: Validated rolling window size.

    Raises:
        TypeError: If ``min_periods`` is not ``None`` and not an ``int``.
        ValueError: If ``min_periods`` is outside ``[1, window]``.
    """
    if min_periods is None:
        return
    if isinstance(min_periods, bool) or not isinstance(min_periods, int):
        raise TypeError(
            f"min_periods must be an int or None, got {type(min_periods).__name__}."
        )
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}.")
    if min_periods > window:
        raise ValueError(
            f"min_periods must be <= window ({window}), got {min_periods}."
        )


def _group_by_symbol(df: pd.DataFrame):
    """Group rows by symbol without reordering groups.

    Args:
        df: Input DataFrame containing a ``symbol`` column.

    Returns:
        A DataFrameGroupBy object grouped by ``symbol`` with ``sort=False``.
    """
    return df.groupby("symbol", sort=False)


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the rolling-covariance output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``rolling_covariance_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{ROLLING_COVARIANCE_PREFIX}_{window}"


def _compute_rolling_covariance(
    df: pd.DataFrame,
    window: int,
    left_column: str,
    right_column: str,
    output_column: str,
    min_periods: int | None,
) -> pd.DataFrame:
    """Append per-symbol rolling covariance to a copied DataFrame.

    Args:
        df: Input DataFrame containing ``symbol`` and both numeric columns.
        window: Rolling window size.
        left_column: First numeric column in the covariance pair.
        right_column: Second numeric column in the covariance pair.
        output_column: Final resolved name for the covariance column.
        min_periods: Minimum observations required; ``None`` uses ``window``.

    Returns:
        A new DataFrame with the rolling-covariance column appended. The
        caller's DataFrame is never modified.
    """
    if min_periods is None:
        min_periods = window

    result = df.copy()
    rolling_covariances: list[pd.Series] = []
    for _, symbol_df in _group_by_symbol(result):
        group_cov = (
            symbol_df[left_column]
            .rolling(window=window, min_periods=min_periods)
            .cov(symbol_df[right_column])
        )
        rolling_covariances.append(group_cov)

    rolling_cov = pd.concat(rolling_covariances).sort_index()
    result[output_column] = rolling_cov
    return result


def compute_rolling_covariance(
    df: pd.DataFrame,
    window: int,
    left_column: str,
    right_column: str,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute per-symbol rolling covariance between two columns.

    Covariance windows never cross symbol boundaries.

    Args:
        df: DataFrame containing ``symbol`` and both numeric columns.
        window: Rolling window size. Must be a positive integer.
        left_column: First numeric column in the covariance pair.
        right_column: Second numeric column in the covariance pair.
        output_column: Optional output column name. Defaults to
            ``rolling_covariance_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling-covariance column appended. The
        input is unchanged.
    """
    _validate_numeric_column(df, left_column)
    _validate_numeric_column(df, right_column)
    _validate_window(window)
    _validate_min_periods(min_periods, window)
    name = _generate_output_column_name(window, output_column)
    return _compute_rolling_covariance(
        df=df,
        window=window,
        left_column=left_column,
        right_column=right_column,
        output_column=name,
        min_periods=min_periods,
    )
