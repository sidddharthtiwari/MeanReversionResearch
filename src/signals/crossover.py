"""Crossover event signal generation for quantitative research.

Detects discrete crossover events between two numeric feature columns.
This module does not compute features, generate positions or trades, or
perform backtesting. It signals events only, not above/below state.
"""

from __future__ import annotations

import pandas as pd

from src.signals.constants import (
    CROSSOVER_SIGNAL_SUFFIX,
    FLAT_SIGNAL,
    LONG_SIGNAL,
    SHORT_SIGNAL,
)
from src.signals.validation import _validate_two_numeric_columns

__all__ = [
    "generate_crossover_signal",
]


def _generate_output_column_name(
    fast_column: str,
    output_column: str | None,
) -> str:
    """Resolve the crossover signal output column name.

    Args:
        fast_column: Fast feature column used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``{fast_column}_cross_signal``.
    """
    if output_column is not None:
        return output_column
    return f"{fast_column}{CROSSOVER_SIGNAL_SUFFIX}"


def _apply_crossover_logic(
    fast: pd.Series,
    slow: pd.Series,
) -> pd.Series:
    """Map fast/slow series into discrete crossover event signals.

    Rules:
        - Previous ``fast <= slow`` and current ``fast > slow`` ->
          ``LONG_SIGNAL``
        - Previous ``fast >= slow`` and current ``fast < slow`` ->
          ``SHORT_SIGNAL``
        - Otherwise -> ``FLAT_SIGNAL``

    The first row is always ``FLAT_SIGNAL`` because no previous observation
    exists.

    Args:
        fast: Fast numeric feature series.
        slow: Slow numeric feature series.

    Returns:
        Integer signal series aligned to ``fast.index``.
    """
    previous_fast = fast.shift(1)
    previous_slow = slow.shift(1)

    long_cross = (previous_fast <= previous_slow) & (fast > slow)
    short_cross = (previous_fast >= previous_slow) & (fast < slow)

    signal = pd.Series(FLAT_SIGNAL, index=fast.index, dtype="int64")
    signal = signal.mask(long_cross, LONG_SIGNAL)
    signal = signal.mask(short_cross, SHORT_SIGNAL)
    return signal


def generate_crossover_signal(
    df: pd.DataFrame,
    fast_column: str,
    slow_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Generate discrete signals from fast/slow crossover events.

    A long signal is emitted on the bar where ``fast`` crosses above
    ``slow``. A short signal is emitted where ``fast`` crosses below
    ``slow``. All other bars are flat, including the first row.

    Args:
        df: Input DataFrame containing ``fast_column`` and ``slow_column``.
        fast_column: Numeric fast feature column.
        slow_column: Numeric slow feature column.
        output_column: Optional output column name. Defaults to
            ``{fast_column}_cross_signal``.

    Returns:
        A new DataFrame with the signal column appended. The input is
        unchanged.

    Raises:
        KeyError: If either feature column is missing.
        TypeError: If either feature column is not numeric.
    """
    _validate_two_numeric_columns(df, fast_column, slow_column)

    name = _generate_output_column_name(fast_column, output_column)
    result = df.copy()
    result[name] = _apply_crossover_logic(
        fast=result[fast_column],
        slow=result[slow_column],
    )
    return result
