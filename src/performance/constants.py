"""Shared constants for the performance package.

This module defines canonical framework column names for derived performance
time-series. It contains constants only.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_EQUITY_COLUMN",
    "DEFAULT_DRAWDOWN_COLUMN",
]

# ---------------------------------------------------------------------------
# Canonical performance column names
# ---------------------------------------------------------------------------

DEFAULT_EQUITY_COLUMN = "equity"
DEFAULT_DRAWDOWN_COLUMN = "drawdown"
