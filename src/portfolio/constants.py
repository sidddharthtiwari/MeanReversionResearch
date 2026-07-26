"""Shared naming conventions for the portfolio package.

This module defines suffix constants used when deriving default column names
in portfolio modules. It contains constants only.
"""

from __future__ import annotations

__all__ = [
    "POSITION_SUFFIX",
    "STRATEGY_RETURN_SUFFIX",
    "CUMULATIVE_RETURN_SUFFIX",
    "EQUITY_CURVE_SUFFIX",
]

# Appended when deriving default portfolio-related column names.
POSITION_SUFFIX = "_position"
STRATEGY_RETURN_SUFFIX = "_strategy_return"
CUMULATIVE_RETURN_SUFFIX = "_cumulative_return"
EQUITY_CURVE_SUFFIX = "_equity_curve"
