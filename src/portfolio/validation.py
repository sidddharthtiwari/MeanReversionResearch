"""Shared validation helpers for the portfolio package.

This module contains validation logic only. It does not generate positions,
compute returns, or perform exposure calculations.
"""

from __future__ import annotations

import pandas as pd

from src.signals.signal_utils import _validate_signal_column
from src.signals.validation import _validate_numeric_column


def _validate_position_column(df: pd.DataFrame, position_column: str) -> None:
    """Validate that a position column exists, is numeric, and is discrete.

    Positions must contain only ``LONG_SIGNAL``, ``FLAT_SIGNAL``, and
    ``SHORT_SIGNAL``. Validation is delegated to the signal-layer helper.

    Args:
        df: Input DataFrame containing the position column.
        position_column: Name of the position column to validate.

    Raises:
        KeyError: If ``position_column`` is not present in ``df``.
        TypeError: If ``position_column`` is not numeric.
        ValueError: If any unique value is outside the valid signal set.
    """
    _validate_signal_column(df, position_column)


def _validate_return_column(df: pd.DataFrame, return_column: str) -> None:
    """Validate that a return column exists and is numeric.

    Args:
        df: Input DataFrame containing the return column.
        return_column: Name of the return column to validate.

    Raises:
        KeyError: If ``return_column`` is not present in ``df``.
        TypeError: If ``return_column`` is not numeric.
    """
    _validate_numeric_column(df, return_column)
