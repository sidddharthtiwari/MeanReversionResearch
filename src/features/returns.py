"""Return-based feature computation for quantitative research.

This module computes simple, log, and forward returns from OHLC price data.
It does not load, validate, signal, plot, or backtest data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PRICE_COLUMN = "close"
DEFAULT_SIMPLE_RETURN = "simple_return"
DEFAULT_LOG_RETURN = "log_return"


def _validate_symbol_column(df: pd.DataFrame) -> None:
    """Validate that the required ``symbol`` column exists.

    Args:
        df: Input DataFrame expected to contain a ``symbol`` column.

    Raises:
        KeyError: If ``symbol`` is not present in ``df``.
    """
    if "symbol" not in df.columns:
        raise KeyError("Required column 'symbol' not found in DataFrame.")


def _validate_price_column(df: pd.DataFrame, price_column: str) -> None:
    """Validate that ``price_column`` exists and has a numeric dtype.

    Args:
        df: Input DataFrame containing price data.
        price_column: Name of the price column to validate.

    Raises:
        KeyError: If ``price_column`` is not present in ``df``.
        TypeError: If ``price_column`` is not numeric.
    """
    if price_column not in df.columns:
        raise KeyError(f"Price column '{price_column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[price_column]):
        raise TypeError(
            f"Price column '{price_column}' must be numeric, "
            f"got dtype '{df[price_column].dtype}'."
        )


def _group_by_symbol(df: pd.DataFrame):
    """Group rows by symbol without reordering groups.

    Args:
        df: Input DataFrame containing a ``symbol`` column.

    Returns:
        A DataFrameGroupBy object grouped by ``symbol`` with ``sort=False``.
    """
    return df.groupby("symbol", sort=False)


def compute_simple_returns(
    df: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    output_column: str = DEFAULT_SIMPLE_RETURN,
) -> pd.DataFrame:
    """Compute per-symbol simple returns.

    Simple return is defined as ``(price_t / price_{t-1}) - 1``.

    Args:
        df: OHLC DataFrame containing ``symbol`` and a price column.
        price_column: Column used as the price series.
        output_column: Name of the column written with simple returns.

    Returns:
        A new DataFrame with ``output_column`` appended. The input is unchanged.
    """
    _validate_symbol_column(df)
    _validate_price_column(df, price_column)
    result = df.copy()
    result[output_column] = _group_by_symbol(result)[price_column].pct_change()
    return result


def compute_log_returns(
    df: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    output_column: str = DEFAULT_LOG_RETURN,
) -> pd.DataFrame:
    """Compute per-symbol log returns.

    Log return is defined as ``ln(price_t / price_{t-1})``.

    Args:
        df: OHLC DataFrame containing ``symbol`` and a price column.
        price_column: Column used as the price series.
        output_column: Name of the column written with log returns.

    Returns:
        A new DataFrame with ``output_column`` appended. The input is unchanged.
    """
    _validate_symbol_column(df)
    _validate_price_column(df, price_column)
    result = df.copy()
    previous_price = _group_by_symbol(result)[price_column].shift(1)
    result[output_column] = np.log(result[price_column] / previous_price)
    return result


def compute_forward_returns(
    df: pd.DataFrame,
    periods: int = 1,
    price_column: str = DEFAULT_PRICE_COLUMN,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute per-symbol forward returns.

    Forward return is defined as ``(price_{t+n} / price_t) - 1``.

    Args:
        df: OHLC DataFrame containing ``symbol`` and a price column.
        periods: Forward horizon ``n`` in rows. Must be greater than zero.
        price_column: Column used as the price series.
        output_column: Name of the forward-return column. If ``None``, uses
            ``forward_return_{periods}d``.

    Returns:
        A new DataFrame with the forward-return column appended. The input is
        unchanged.

    Raises:
        ValueError: If ``periods`` is less than or equal to zero.
    """
    if periods <= 0:
        raise ValueError(f"periods must be > 0, got {periods}.")

    _validate_symbol_column(df)
    _validate_price_column(df, price_column)

    if output_column is None:
        output_column = f"forward_return_{periods}d"

    result = df.copy()
    future_price = _group_by_symbol(result)[price_column].shift(-periods)
    result[output_column] = future_price / result[price_column] - 1.0
    return result
