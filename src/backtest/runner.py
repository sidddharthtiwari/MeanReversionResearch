"""Backtest-stage orchestration for quantitative research.

Coordinates transaction-cost and slippage modules without implementing their
business logic. This module does not compute returns, signals, or positions.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.constants import (
    DEFAULT_SLIPPAGE,
    DEFAULT_TRANSACTION_COST,
    NET_RETURN_SUFFIX,
)
from src.backtest.slippage import apply_slippage
from src.backtest.transaction_costs import apply_transaction_costs

__all__ = [
    "resolve_backtest_return_column",
    "run_backtest",
]


def resolve_backtest_return_column(
    strategy_return_column: str,
    transaction_cost: int | float = DEFAULT_TRANSACTION_COST,
    slippage: int | float = DEFAULT_SLIPPAGE,
    output_column: str | None = None,
) -> str:
    """Resolve the final return column produced by ``run_backtest``.

    Mirrors the default net-return naming used when transaction costs and/or
    slippage stages are applied.

    Args:
        strategy_return_column: Gross strategy-return column used as the start.
        transaction_cost: Cost charged per unit of trade size.
        slippage: Slippage charged per unit of trade size.
        output_column: Optional explicit net-return column name for each
            applied stage. Defaults to ``{input_return_column}_net_return``.

    Returns:
        Name of the final return column available after ``run_backtest``.
    """
    current_return_column = strategy_return_column

    if transaction_cost > 0:
        current_return_column = (
            output_column
            if output_column is not None
            else f"{current_return_column}{NET_RETURN_SUFFIX}"
        )

    if slippage > 0:
        current_return_column = (
            output_column
            if output_column is not None
            else f"{current_return_column}{NET_RETURN_SUFFIX}"
        )

    return current_return_column


def run_backtest(
    df: pd.DataFrame,
    strategy_return_column: str,
    position_column: str,
    transaction_cost: int | float = DEFAULT_TRANSACTION_COST,
    slippage: int | float = DEFAULT_SLIPPAGE,
    transaction_cost_column: str | None = None,
    slippage_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Run the backtest execution-cost stage.

    Optionally applies transaction costs and then slippage to
    ``strategy_return_column``, chaining each stage's net-return column into
    the next. Stages with a zero cost parameter are skipped.

    Args:
        df: Input DataFrame containing return and position columns.
        strategy_return_column: Gross strategy-return column to start from.
        position_column: Discrete portfolio position column.
        transaction_cost: Cost charged per unit of trade size. Skipped when
            not greater than zero.
        slippage: Slippage charged per unit of trade size. Skipped when not
            greater than zero.
        transaction_cost_column: Optional transaction-cost column name.
        slippage_column: Optional slippage column name.
        output_column: Optional net-return column name for each applied stage.
            Defaults to ``{input_return_column}_net_return`` per stage.

    Returns:
        The DataFrame returned by the last applied execution-cost stage, or
        ``df`` unchanged when both cost parameters are zero.

    Raises:
        ValueError: If both execution-cost stages are enabled and
            ``output_column`` is set, which would overwrite the intermediate
            net-return column.
    """
    if transaction_cost > 0 and slippage > 0 and output_column is not None:
        raise ValueError(
            "output_column cannot be set when both transaction_cost and "
            "slippage are greater than zero, because a shared net-return "
            "column would overwrite the intermediate stage result. Leave "
            "output_column=None, or apply the execution-cost modules "
            "individually with custom output names."
        )

    result = df
    current_return_column = strategy_return_column

    if transaction_cost > 0:
        result = apply_transaction_costs(
            result,
            strategy_return_column=current_return_column,
            position_column=position_column,
            transaction_cost=transaction_cost,
            transaction_cost_column=transaction_cost_column,
            output_column=output_column,
        )
        current_return_column = resolve_backtest_return_column(
            strategy_return_column=current_return_column,
            transaction_cost=transaction_cost,
            slippage=0,
            output_column=output_column,
        )

    if slippage > 0:
        result = apply_slippage(
            result,
            strategy_return_column=current_return_column,
            position_column=position_column,
            slippage=slippage,
            slippage_column=slippage_column,
            output_column=output_column,
        )

    return result
