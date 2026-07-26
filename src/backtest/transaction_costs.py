"""Transaction-cost application for quantitative backtesting.

Deducts transaction costs from an existing strategy-return column based on
changes in the portfolio position. This module does not generate positions,
compute strategy returns, apply slippage, or compute analytics.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.constants import (
    DEFAULT_TRANSACTION_COST,
    NET_RETURN_SUFFIX,
    TRANSACTION_COST_SUFFIX,
)
from src.backtest.validation import _validate_transaction_inputs
from src.portfolio.validation import _validate_position_column

__all__ = [
    "apply_transaction_costs",
]


def _generate_transaction_cost_column_name(
    strategy_return_column: str,
    transaction_cost_column: str | None,
) -> str:
    """Resolve the transaction-cost output column name.

    Args:
        strategy_return_column: Source return column used in the default name.
        transaction_cost_column: Explicit output name, or ``None`` for default.

    Returns:
        ``transaction_cost_column`` when provided, otherwise
        ``{strategy_return_column}{TRANSACTION_COST_SUFFIX}``.
    """
    if transaction_cost_column is not None:
        return transaction_cost_column
    return f"{strategy_return_column}{TRANSACTION_COST_SUFFIX}"


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


def _validate_transaction_cost_inputs(
    df: pd.DataFrame,
    strategy_return_column: str,
    position_column: str,
    transaction_cost: int | float,
) -> None:
    """Validate inputs required to apply transaction costs.

    Args:
        df: Input DataFrame containing return and position columns.
        strategy_return_column: Numeric strategy-return column.
        position_column: Discrete portfolio position column.
        transaction_cost: Non-negative transaction-cost parameter.

    Raises:
        KeyError: If either required column is missing.
        TypeError: If a required column is not numeric, or if
            ``transaction_cost`` is not an ``int`` or ``float``.
        ValueError: If ``position_column`` contains invalid values, or if
            ``transaction_cost`` is negative.
    """
    _validate_transaction_inputs(
        df=df,
        strategy_return_column=strategy_return_column,
        transaction_cost=transaction_cost,
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


def _compute_transaction_cost(
    trade_size: pd.Series,
    transaction_cost: int | float,
) -> pd.Series:
    """Compute period transaction costs from trade size.

    Args:
        trade_size: Absolute trade-size series.
        transaction_cost: Cost charged per unit of trade size.

    Returns:
        Transaction-cost series aligned to ``trade_size.index``.
    """
    return trade_size * transaction_cost


def _compute_net_returns(
    strategy_returns: pd.Series,
    transaction_costs: pd.Series,
) -> pd.Series:
    """Compute net returns after deducting transaction costs.

    Args:
        strategy_returns: Gross strategy-return series.
        transaction_costs: Transaction-cost series to deduct.

    Returns:
        Net-return series aligned to ``strategy_returns.index``.
    """
    return strategy_returns - transaction_costs


def apply_transaction_costs(
    df: pd.DataFrame,
    strategy_return_column: str,
    position_column: str,
    transaction_cost: int | float = DEFAULT_TRANSACTION_COST,
    transaction_cost_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Apply transaction costs to a strategy-return column.

    Trade size is the absolute change in position, with the first row filled
    by ``abs(position)``. Transaction costs are ``trade_size * transaction_cost``
    and are deducted from strategy returns.

    Args:
        df: Input DataFrame containing return and position columns.
        strategy_return_column: Numeric strategy-return column.
        position_column: Discrete portfolio position column.
        transaction_cost: Cost charged per unit of trade size.
        transaction_cost_column: Optional transaction-cost column name.
            Defaults to ``{strategy_return_column}_transaction_cost``.
        output_column: Optional net-return column name. Defaults to
            ``{strategy_return_column}_net_return``.

    Returns:
        A new DataFrame with transaction-cost and net-return columns
        appended. The input is unchanged.

    Raises:
        KeyError: If either required column is missing.
        TypeError: If a required column is not numeric, or if
            ``transaction_cost`` is not an ``int`` or ``float``.
        ValueError: If ``position_column`` contains invalid values, or if
            ``transaction_cost`` is negative.
    """
    _validate_transaction_cost_inputs(
        df=df,
        strategy_return_column=strategy_return_column,
        position_column=position_column,
        transaction_cost=transaction_cost,
    )

    cost_name = _generate_transaction_cost_column_name(
        strategy_return_column,
        transaction_cost_column,
    )
    net_name = _generate_output_column_name(
        strategy_return_column,
        output_column,
    )

    result = df.copy()
    positions = result[position_column]
    trade_size = _compute_trade_size(positions)
    transaction_costs = _compute_transaction_cost(trade_size, transaction_cost)
    net_returns = _compute_net_returns(
        strategy_returns=result[strategy_return_column],
        transaction_costs=transaction_costs,
    )

    result[cost_name] = transaction_costs
    result[net_name] = net_returns
    return result
