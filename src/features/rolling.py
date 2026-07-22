"""Rolling statistical feature computation for quantitative research.

This module computes per-symbol rolling mean, standard deviation, min, max,
and median statistics. It does not load, validate datasets, generate signals,
plot, or backtest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pandas as pd

__all__ = [
    "compute_rolling_mean",
    "compute_rolling_std",
    "compute_rolling_min",
    "compute_rolling_max",
    "compute_rolling_median",
]

DEFAULT_PRICE_COLUMN = "close"
ROLLING_MEAN_PREFIX = "rolling_mean"
ROLLING_STD_PREFIX = "rolling_std"
ROLLING_MIN_PREFIX = "rolling_min"
ROLLING_MAX_PREFIX = "rolling_max"
ROLLING_MEDIAN_PREFIX = "rolling_median"

RollingOperation = Callable[..., pd.Series]
RollingOperationName = Literal["mean", "std", "min", "max", "median"]

_ROLLING_OPERATIONS: dict[RollingOperationName, RollingOperation] = {
    "mean": lambda rolling: rolling.mean(),
    "std": lambda rolling: rolling.std(),
    "min": lambda rolling: rolling.min(),
    "max": lambda rolling: rolling.max(),
    "median": lambda rolling: rolling.median(),
}


def _validate_symbol_column(df: pd.DataFrame) -> None:
    """Validate that the required ``symbol`` column exists.

    Args:
        df: Input DataFrame expected to contain a ``symbol`` column.

    Raises:
        KeyError: If ``symbol`` is not present in ``df``.
    """
    if "symbol" not in df.columns:
        raise KeyError("Required column 'symbol' not found in DataFrame.")


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
    prefix: str,
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the output column name for a rolling statistic.

    Args:
        prefix: Statistic-specific column prefix constant.
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``{prefix}_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{prefix}_{window}"


def _apply_rolling(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None,
    operation: RollingOperationName,
) -> pd.Series:
    """Apply a per-symbol rolling operation and return an aligned Series.

    Args:
        df: DataFrame containing ``symbol`` and the numeric ``column``.
        column: Numeric column to roll over.
        window: Rolling window size.
        min_periods: Minimum observations required; ``None`` uses ``window``.
        operation: Rolling operation key (``mean``, ``std``, ``min``, ``max``,
            or ``median``).

    Returns:
        A Series of rolling values aligned to ``df``'s original index order.

    Raises:
        ValueError: If ``operation`` is not a supported rolling operation.
    """
    if min_periods is None:
        min_periods = window

    if operation not in _ROLLING_OPERATIONS:
        raise ValueError(f"Unsupported rolling operation '{operation}'.")

    rolling = _group_by_symbol(df)[column].rolling(
        window=window,
        min_periods=min_periods,
    )
    rolled = _ROLLING_OPERATIONS[operation](rolling)
    return rolled.reset_index(level=0, drop=True)


def _compute_rolling_feature(
    df: pd.DataFrame,
    window: int,
    column: str,
    output_column: str | None,
    min_periods: int | None,
    prefix: str,
    operation: RollingOperationName,
) -> pd.DataFrame:
    """Validate inputs and append one rolling feature column.

    Args:
        df: Input OHLC DataFrame.
        window: Rolling window size.
        column: Numeric column to roll over.
        output_column: Explicit output column name, or ``None`` for default.
        min_periods: Minimum observations required, or ``None`` for ``window``.
        prefix: Default output-column prefix constant.
        operation: Rolling operation to apply.

    Returns:
        A new DataFrame with the rolling feature column appended.
    """
    _validate_symbol_column(df)
    _validate_numeric_column(df, column)
    _validate_window(window)
    _validate_min_periods(min_periods, window)

    result = df.copy()
    name = _generate_output_column_name(prefix, window, output_column)
    result[name] = _apply_rolling(result, column, window, min_periods, operation)
    return result


def compute_rolling_mean(
    df: pd.DataFrame,
    window: int,
    column: str = DEFAULT_PRICE_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling mean of a numeric column.

    Args:
        df: OHLC DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size. Must be a positive integer.
        column: Numeric column to roll over.
        output_column: Optional output column name. Defaults to
            ``rolling_mean_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling mean column appended.
    """
    return _compute_rolling_feature(
        df=df,
        window=window,
        column=column,
        output_column=output_column,
        min_periods=min_periods,
        prefix=ROLLING_MEAN_PREFIX,
        operation="mean",
    )


def compute_rolling_std(
    df: pd.DataFrame,
    window: int,
    column: str = DEFAULT_PRICE_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling standard deviation of a numeric column.

    Args:
        df: OHLC DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size. Must be a positive integer.
        column: Numeric column to roll over.
        output_column: Optional output column name. Defaults to
            ``rolling_std_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling standard deviation column appended.
    """
    return _compute_rolling_feature(
        df=df,
        window=window,
        column=column,
        output_column=output_column,
        min_periods=min_periods,
        prefix=ROLLING_STD_PREFIX,
        operation="std",
    )


def compute_rolling_min(
    df: pd.DataFrame,
    window: int,
    column: str = DEFAULT_PRICE_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling minimum of a numeric column.

    Args:
        df: OHLC DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size. Must be a positive integer.
        column: Numeric column to roll over.
        output_column: Optional output column name. Defaults to
            ``rolling_min_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling minimum column appended.
    """
    return _compute_rolling_feature(
        df=df,
        window=window,
        column=column,
        output_column=output_column,
        min_periods=min_periods,
        prefix=ROLLING_MIN_PREFIX,
        operation="min",
    )


def compute_rolling_max(
    df: pd.DataFrame,
    window: int,
    column: str = DEFAULT_PRICE_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling maximum of a numeric column.

    Args:
        df: OHLC DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size. Must be a positive integer.
        column: Numeric column to roll over.
        output_column: Optional output column name. Defaults to
            ``rolling_max_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling maximum column appended.
    """
    return _compute_rolling_feature(
        df=df,
        window=window,
        column=column,
        output_column=output_column,
        min_periods=min_periods,
        prefix=ROLLING_MAX_PREFIX,
        operation="max",
    )


def compute_rolling_median(
    df: pd.DataFrame,
    window: int,
    column: str = DEFAULT_PRICE_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the per-symbol rolling median of a numeric column.

    Args:
        df: OHLC DataFrame containing ``symbol`` and ``column``.
        window: Rolling window size. Must be a positive integer.
        column: Numeric column to roll over.
        output_column: Optional output column name. Defaults to
            ``rolling_median_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the rolling median column appended.
    """
    return _compute_rolling_feature(
        df=df,
        window=window,
        column=column,
        output_column=output_column,
        min_periods=min_periods,
        prefix=ROLLING_MEDIAN_PREFIX,
        operation="median",
    )