"""Slippage application for quantitative backtesting.

Deducts execution slippage from an existing strategy-return column based on
changes in the portfolio position. This module does not generate positions,
compute strategy returns, apply transaction costs, or compute analytics.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.constants import (
    DEFAULT_SLIPPAGE,
    NET_RETURN_SUFFIX,
    SLIPPAGE_SUFFIX,
)
from src.backtest.validation import (
    _validate_slippage_inputs as _validate_shared_slippage_inputs,
)
from src.portfolio.validation import _validate_position_column

__all__ = [
    "apply_slippage",
]


def _generate_slippage_column_name(
    strategy_return_column: str,
    slippage_column: str | None,
) -> str:
    """Resolve the slippage output column name.

    Args:
        strategy_return_column: Source return column used in the default name.
        slippage_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``slippage_column`` when provided, otherwise
        ``{strategy_return_column}{SLIPPAGE_SUFFIX}``.
    """
    if slippage_column is not None:
        return slippage_column
    return f"{strategy_return_column}{SLIPPAGE_SUFFIX}"


def _generate_output_column_name(
    strategy_return_column: str,
    output_column: str | None,
) -> str:
    """Resolve the net-return output column name.

    Args:
        strategy_return_column: Source return column used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``{strategy_return_column}{NET_RETURN_SUFFIX}``.
    """
    if output_column is not None:
        return output_column
    return f"{strategy_return_column}{NET_RETURN_SUFFIX}"


def _validate_slippage_inputs(
    df: pd.DataFrame,
    strategy_return_column: str,
    position_column: str,
    slippage: int | float,
) -> None:
    """Validate inputs required to apply slippage.

    Args:
        df: Input DataFrame containing return and position columns.
        strategy_return_column: Numeric strategy-return column.
        position_column: Discrete portfolio position column.
        slippage: Non-negative slippage parameter.

    Raises:
        KeyError: If either required column is missing.
        TypeError: If a required column is not numeric, or if ``slippage``
            is not an ``int`` or ``float``.
        ValueError: If ``position_column`` contains invalid values, or if
            ``slippage`` is negative.
    """
    _validate_shared_slippage_inputs(
        df=df,
        strategy_return_column=strategy_return_column,
        slippage=slippage,
    )
    _validate_position_column(df, position_column)


def _compute_trade_size(positions: pd.Series) -> pd.Series:
    """Compute absolute trade size from consecutive position changes.

    The first observation uses ``abs(position)`` so an initial long or short
    position is treated as one trade.

    Args:
        positions: Discrete portfolio position series.

    Returns:
        Absolute trade-size series aligned to ``positions.index``.
    """
    return positions.diff().abs().fillna(positions.abs())


def _compute_slippage(
    trade_size: pd.Series,
    slippage: int | float,
) -> pd.Series:
    """Compute period slippage costs from trade size.

    Args:
        trade_size: Absolute trade-size series.
        slippage: Slippage charged per unit of trade size.

    Returns:
        Slippage-cost series aligned to ``trade_size.index``.
    """
    return trade_size * slippage


def _compute_net_returns(
    strategy_returns: pd.Series,
    slippage_costs: pd.Series,
) -> pd.Series:
    """Compute net returns after deducting slippage costs.

    Args:
        strategy_returns: Gross strategy-return series.
        slippage_costs: Slippage-cost series to deduct.

    Returns:
        Net-return series aligned to ``strategy_returns.index``.
    """
    return strategy_returns - slippage_costs


def apply_slippage(
    df: pd.DataFrame,
    strategy_return_column: str,
    position_column: str,
    slippage: int | float = DEFAULT_SLIPPAGE,
    slippage_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Apply execution slippage to a strategy-return column.

    Trade size is the absolute change in position, with the first row filled
    by ``abs(position)``. Slippage costs are ``trade_size * slippage`` and are
    deducted from strategy returns.

    Args:
        df: Input DataFrame containing return and position columns.
        strategy_return_column: Numeric strategy-return column.
        position_column: Discrete portfolio position column.
        slippage: Slippage charged per unit of trade size.
        slippage_column: Optional slippage column name. Defaults to
            ``{strategy_return_column}_slippage``.
        output_column: Optional net-return column name. Defaults to
            ``{strategy_return_column}_net_return``.

    Returns:
        A new DataFrame with slippage and net-return columns appended. The
        input is unchanged.

    Raises:
        KeyError: If either required column is missing.
        TypeError: If a required column is not numeric, or if ``slippage``
            is not an ``int`` or ``float``.
        ValueError: If ``position_column`` contains invalid values, or if
            ``slippage`` is negative.
    """
    _validate_slippage_inputs(
        df=df,
        strategy_return_column=strategy_return_column,
        position_column=position_column,
        slippage=slippage,
    )

    slippage_name = _generate_slippage_column_name(
        strategy_return_column,
        slippage_column,
    )
    net_name = _generate_output_column_name(
        strategy_return_column,
        output_column,
    )

    result = df.copy()
    positions = result[position_column]
    trade_size = _compute_trade_size(positions)
    slippage_costs = _compute_slippage(trade_size, slippage)
    net_returns = _compute_net_returns(
        strategy_returns=result[strategy_return_column],
        slippage_costs=slippage_costs,
    )

    result[slippage_name] = slippage_costs
    result[net_name] = net_returns
    return result
