"""Shared constants for the analytics package.

This module defines annualisation factors, default risk-free rate parameters,
and column-name suffixes reused across analytics modules. It contains
constants only.
"""

from __future__ import annotations

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "TRADING_WEEKS_PER_YEAR",
    "TRADING_MONTHS_PER_YEAR",
    "PERIODS_PER_YEAR_BY_FREQUENCY",
    "DEFAULT_RISK_FREE_RATE",
    "CUMULATIVE_RETURN_SUFFIX",
    "DRAWDOWN_SUFFIX",
]

# ---------------------------------------------------------------------------
# Annualisation factors
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52
TRADING_MONTHS_PER_YEAR = 12

PERIODS_PER_YEAR_BY_FREQUENCY = {
    "D": TRADING_DAYS_PER_YEAR,
    "W": TRADING_WEEKS_PER_YEAR,
    "M": TRADING_MONTHS_PER_YEAR,
}

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_RISK_FREE_RATE = 0.0

# ---------------------------------------------------------------------------
# Column name suffixes
# ---------------------------------------------------------------------------

CUMULATIVE_RETURN_SUFFIX = "_cumulative_return"
DRAWDOWN_SUFFIX = "_drawdown"
