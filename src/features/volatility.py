"""Rolling volatility feature computation for quantitative research.

Volatility is defined as the rolling standard deviation of simple returns.
This module orchestrates existing return and rolling feature helpers and does
not load data, validate datasets, generate signals, plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

from src.features.returns import compute_simple_returns
from src.features.rolling import compute_rolling_std

__all__ = [
    "compute_volatility",
]

DEFAULT_PRICE_COLUMN = "close"
DEFAULT_RETURN_COLUMN = "simple_return"
VOLATILITY_PREFIX = "volatility"


def _generate_output_column_name(
    window: int,
    output_column: str | None,
) -> str:
    """Resolve the volatility output column name.

    Args:
        window: Rolling window size used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``volatility_{window}``.
    """
    if output_column is not None:
        return output_column
    return f"{VOLATILITY_PREFIX}_{window}"


def _compute_volatility(
    df: pd.DataFrame,
    window: int,
    price_column: str,
    return_column: str,
    output_column: str | None,
    min_periods: int | None,
) -> pd.DataFrame:
    """Compute rolling volatility by composing return and rolling helpers.

    Uses an existing return column when present; otherwise computes simple
    returns first. Rolling standard deviation is then applied to the return
    series.

    Args:
        df: Input OHLC DataFrame.
        window: Rolling window size.
        price_column: Price column used when simple returns must be computed.
        return_column: Simple-return column to reuse or create.
        output_column: Explicit volatility column name, or ``None`` for default.
        min_periods: Minimum observations required for the rolling window.

    Returns:
        A new DataFrame containing the volatility column. The caller's
        DataFrame is never modified.
    """
    name = _generate_output_column_name(window, output_column)

    working = (
        df
        if return_column in df.columns
        else compute_simple_returns(
            df,
            price_column=price_column,
            output_column=return_column,
        )
    )

    return compute_rolling_std(
        working,
        window=window,
        column=return_column,
        output_column=name,
        min_periods=min_periods,
    )


def compute_volatility(
    df: pd.DataFrame,
    window: int,
    price_column: str = DEFAULT_PRICE_COLUMN,
    return_column: str = DEFAULT_RETURN_COLUMN,
    output_column: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute rolling volatility of simple returns.

    Volatility is the rolling standard deviation of ``return_column``. If that
    column is absent, simple returns are computed from ``price_column`` first.

    Args:
        df: OHLC DataFrame containing ``symbol`` and price data.
        window: Rolling window size. Must be a positive integer.
        price_column: Price column used when returns must be computed.
        return_column: Simple-return column to reuse or create.
        output_column: Optional output column name. Defaults to
            ``volatility_{window}``.
        min_periods: Minimum observations required. Defaults to ``window``.

    Returns:
        A new DataFrame with the volatility column appended. The input is
        unchanged.
    """
    return _compute_volatility(
        df=df,
        window=window,
        price_column=price_column,
        return_column=return_column,
        output_column=output_column,
        min_periods=min_periods,
    )
