"""Strategy return, cumulative return, and equity-curve computation.

Computes portfolio strategy returns from positions and asset returns, then
derives cumulative returns and an equity curve. This module does not generate
positions, manage exposure, or perform backtesting.
"""

from __future__ import annotations

import pandas as pd

from src.portfolio.constants import (
    CUMULATIVE_RETURN_SUFFIX,
    EQUITY_CURVE_SUFFIX,
    STRATEGY_RETURN_SUFFIX,
)
from src.portfolio.validation import (
    _validate_position_column,
    _validate_return_column,
)

__all__ = [
    "generate_strategy_return_column_name",
    "generate_strategy_returns",
]


def generate_strategy_return_column_name(
    position_column: str,
    strategy_return_column: str | None = None,
) -> str:
    """Generate the strategy-return output column name.

    Args:
        position_column: Source position column used in the default name.
        strategy_return_column: Explicit output name, or ``None`` for default.

    Returns:
        ``strategy_return_column`` when provided, otherwise
        ``{position_column}{STRATEGY_RETURN_SUFFIX}``.
    """
    if strategy_return_column is not None:
        return strategy_return_column
    return f"{position_column}{STRATEGY_RETURN_SUFFIX}"


def _generate_cumulative_return_column_name(
    position_column: str,
    cumulative_return_column: str | None,
) -> str:
    """Resolve the cumulative-return output column name.

    Args:
        position_column: Source position column used in the default name.
        cumulative_return_column: Explicit output name, or ``None`` for default.

    Returns:
        ``cumulative_return_column`` when provided, otherwise
        ``{position_column}{CUMULATIVE_RETURN_SUFFIX}``.
    """
    if cumulative_return_column is not None:
        return cumulative_return_column
    return f"{position_column}{CUMULATIVE_RETURN_SUFFIX}"


def _generate_equity_curve_column_name(
    position_column: str,
    equity_curve_column: str | None,
) -> str:
    """Resolve the equity-curve output column name.

    Args:
        position_column: Source position column used in the default name.
        equity_curve_column: Explicit output name, or ``None`` for default.

    Returns:
        ``equity_curve_column`` when provided, otherwise
        ``{position_column}{EQUITY_CURVE_SUFFIX}``.
    """
    if equity_curve_column is not None:
        return equity_curve_column
    return f"{position_column}{EQUITY_CURVE_SUFFIX}"


def _validate_return_inputs(
    df: pd.DataFrame,
    position_column: str,
    asset_return_column: str,
) -> None:
    """Validate inputs required to compute strategy returns.

    Args:
        df: Input DataFrame containing position and asset-return columns.
        position_column: Discrete portfolio position column.
        asset_return_column: Numeric asset-return column.

    Raises:
        KeyError: If either column is missing.
        TypeError: If either column is not numeric.
        ValueError: If ``position_column`` contains invalid position values.
    """
    _validate_position_column(df, position_column)
    _validate_return_column(df, asset_return_column)


def _compute_strategy_returns(
    positions: pd.Series,
    asset_returns: pd.Series,
) -> pd.Series:
    """Compute period strategy returns as position times asset return.

    Args:
        positions: Discrete portfolio position series.
        asset_returns: Asset return series aligned to ``positions``.

    Returns:
        Strategy return series aligned to ``positions.index``.
    """
    return positions * asset_returns


def _compute_cumulative_returns(strategy_returns: pd.Series) -> pd.Series:
    """Compute cumulative strategy returns.

    Cumulative return is ``(1 + strategy_return).cumprod() - 1``.

    Args:
        strategy_returns: Period strategy return series.

    Returns:
        Cumulative return series aligned to ``strategy_returns.index``.
    """
    return (1.0 + strategy_returns).cumprod() - 1.0


def _compute_equity_curve(cumulative_returns: pd.Series) -> pd.Series:
    """Compute the equity curve from cumulative returns.

    Equity curve is ``1 + cumulative_return``.

    Args:
        cumulative_returns: Cumulative strategy return series.

    Returns:
        Equity curve series aligned to ``cumulative_returns.index``.
    """
    return 1.0 + cumulative_returns


def generate_strategy_returns(
    df: pd.DataFrame,
    position_column: str,
    asset_return_column: str,
    strategy_return_column: str | None = None,
    cumulative_return_column: str | None = None,
    equity_curve_column: str | None = None,
) -> pd.DataFrame:
    """Compute strategy returns, cumulative returns, and equity curve.

    Strategy return is ``position * asset_return``. Missing asset-return
    values are treated as ``0.0`` so leading return gaps do not poison the
    cumulative product. Cumulative return is
    ``(1 + strategy_return).cumprod() - 1``. Equity curve is
    ``1 + cumulative_return``.

    Args:
        df: Input DataFrame containing ``position_column`` and
            ``asset_return_column``.
        position_column: Discrete portfolio position column.
        asset_return_column: Numeric asset-return column.
        strategy_return_column: Optional strategy-return column name.
            Defaults to ``{position_column}_strategy_return``.
        cumulative_return_column: Optional cumulative-return column name.
            Defaults to ``{position_column}_cumulative_return``.
        equity_curve_column: Optional equity-curve column name. Defaults to
            ``{position_column}_equity_curve``.

    Returns:
        A new DataFrame with strategy-return, cumulative-return, and equity
        curve columns appended. Missing values in ``asset_return_column`` are
        filled with ``0.0`` in the returned frame. The input is unchanged.

    Raises:
        KeyError: If either required column is missing.
        TypeError: If either required column is not numeric.
        ValueError: If ``position_column`` contains invalid position values.
    """
    _validate_return_inputs(df, position_column, asset_return_column)

    strategy_name = generate_strategy_return_column_name(
        position_column,
        strategy_return_column,
    )
    cumulative_name = _generate_cumulative_return_column_name(
        position_column,
        cumulative_return_column,
    )
    equity_name = _generate_equity_curve_column_name(
        position_column,
        equity_curve_column,
    )

    result = df.copy()
    result[asset_return_column] = result[asset_return_column].fillna(0.0)
    strategy_returns = _compute_strategy_returns(
        positions=result[position_column],
        asset_returns=result[asset_return_column],
    )
    cumulative_returns = _compute_cumulative_returns(strategy_returns)

    result[strategy_name] = strategy_returns
    result[cumulative_name] = cumulative_returns
    result[equity_name] = _compute_equity_curve(cumulative_returns)
    return result
