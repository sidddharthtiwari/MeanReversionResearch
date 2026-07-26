"""Shared validation helpers for the backtest package.

This module validates return columns and execution-cost parameters used by
backtest modules. It contains validation logic only and exposes no public API.
"""

from __future__ import annotations

import pandas as pd

from src.portfolio.validation import _validate_return_column


def _validate_transaction_cost(transaction_cost: int | float) -> None:
    """Validate that ``transaction_cost`` is a non-negative number.

    Args:
        transaction_cost: Transaction-cost parameter to validate.

    Raises:
        TypeError: If ``transaction_cost`` is not an ``int`` or ``float``.
        ValueError: If ``transaction_cost`` is negative.
    """
    if isinstance(transaction_cost, bool) or not isinstance(
        transaction_cost, (int, float)
    ):
        raise TypeError(
            "transaction_cost must be an int or float, "
            f"got {type(transaction_cost).__name__}."
        )
    if transaction_cost < 0:
        raise ValueError(
            f"transaction_cost must be greater than or equal to 0, "
            f"got {transaction_cost}."
        )


def _validate_slippage(slippage: int | float) -> None:
    """Validate that ``slippage`` is a non-negative number.

    Args:
        slippage: Slippage parameter to validate.

    Raises:
        TypeError: If ``slippage`` is not an ``int`` or ``float``.
        ValueError: If ``slippage`` is negative.
    """
    if isinstance(slippage, bool) or not isinstance(slippage, (int, float)):
        raise TypeError(
            f"slippage must be an int or float, got {type(slippage).__name__}."
        )
    if slippage < 0:
        raise ValueError(
            f"slippage must be greater than or equal to 0, got {slippage}."
        )


def _validate_return_inputs(
    df: pd.DataFrame,
    strategy_return_column: str,
) -> None:
    """Validate the strategy-return column used by backtest modules.

    Args:
        df: Input DataFrame containing the strategy-return column.
        strategy_return_column: Numeric strategy-return column to validate.

    Raises:
        KeyError: If ``strategy_return_column`` is missing.
        TypeError: If ``strategy_return_column`` is not numeric.
    """
    _validate_return_column(df, strategy_return_column)


def _validate_transaction_inputs(
    df: pd.DataFrame,
    strategy_return_column: str,
    transaction_cost: int | float,
) -> None:
    """Validate inputs required to apply transaction costs.

    Args:
        df: Input DataFrame containing the strategy-return column.
        strategy_return_column: Numeric strategy-return column to validate.
        transaction_cost: Non-negative transaction-cost parameter.

    Raises:
        KeyError: If ``strategy_return_column`` is missing.
        TypeError: If ``strategy_return_column`` is not numeric, or if
            ``transaction_cost`` is not an ``int`` or ``float``.
        ValueError: If ``transaction_cost`` is negative.
    """
    _validate_return_inputs(df, strategy_return_column)
    _validate_transaction_cost(transaction_cost)


def _validate_slippage_inputs(
    df: pd.DataFrame,
    strategy_return_column: str,
    slippage: int | float,
) -> None:
    """Validate inputs required to apply slippage.

    Args:
        df: Input DataFrame containing the strategy-return column.
        strategy_return_column: Numeric strategy-return column to validate.
        slippage: Non-negative slippage parameter.

    Raises:
        KeyError: If ``strategy_return_column`` is missing.
        TypeError: If ``strategy_return_column`` is not numeric, or if
            ``slippage`` is not an ``int`` or ``float``.
        ValueError: If ``slippage`` is negative.
    """
    _validate_return_inputs(df, strategy_return_column)
    _validate_slippage(slippage)
