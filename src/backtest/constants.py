"""Shared default parameters and naming conventions for the backtest package.

This module defines default transaction-cost and slippage values, and suffix
constants used when deriving default column names in backtest modules. It
contains constants only.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_TRANSACTION_COST",
    "DEFAULT_SLIPPAGE",
    "TRANSACTION_COST_SUFFIX",
    "SLIPPAGE_SUFFIX",
    "NET_RETURN_SUFFIX",
]

# Default execution-cost parameters when the caller does not override them.
DEFAULT_TRANSACTION_COST = 0.0
DEFAULT_SLIPPAGE = 0.0

# Appended when deriving default backtest-related column names.
TRANSACTION_COST_SUFFIX = "_transaction_cost"
SLIPPAGE_SUFFIX = "_slippage"
NET_RETURN_SUFFIX = "_net_return"
