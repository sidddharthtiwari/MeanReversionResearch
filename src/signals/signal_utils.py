"""Utilities for validating, transforming, and summarizing signal columns.

This module works with existing signal columns only. It does not generate
signals, manage positions, implement strategy logic, or perform backtesting.
"""

from __future__ import annotations

import pandas as pd

from src.signals.constants import (
    FLAT_SIGNAL,
    LONG_SIGNAL,
    SHORT_SIGNAL,
    VALID_SIGNALS,
)
from src.signals.validation import _validate_numeric_column

__all__ = [
    "validate_signal_column",
    "invert_signal",
    "count_signal_changes",
    "summarize_signal",
]


def _validate_signal_column(df: pd.DataFrame, signal_column: str) -> None:
    """Validate that ``signal_column`` exists, is numeric, and is discrete.

    Args:
        df: Input DataFrame containing the signal column.
        signal_column: Name of the signal column to validate.

    Raises:
        KeyError: If ``signal_column`` is not present in ``df``.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If any unique value is outside ``VALID_SIGNALS``.
    """
    _validate_numeric_column(df, signal_column)

    unique_values = set(df[signal_column].unique().tolist())
    if not unique_values.issubset(VALID_SIGNALS):
        invalid_values = sorted(unique_values - VALID_SIGNALS)
        raise ValueError(
            f"Column '{signal_column}' contains invalid signal values: "
            f"{invalid_values}. Expected values from {set(VALID_SIGNALS)}."
        )


def validate_signal_column(df: pd.DataFrame, signal_column: str) -> None:
    """Validate that a signal column contains only discrete signal values.

    Args:
        df: Input DataFrame containing the signal column.
        signal_column: Name of the signal column to validate.

    Returns:
        ``None``.

    Raises:
        KeyError: If ``signal_column`` is not present in ``df``.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If any unique value is outside ``VALID_SIGNALS``.
    """
    _validate_signal_column(df, signal_column)


def invert_signal(
    df: pd.DataFrame,
    signal_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Invert long and short signals while leaving flat unchanged.

    Inversion is ``signal * -1``, so ``LONG_SIGNAL`` becomes ``SHORT_SIGNAL``,
    ``SHORT_SIGNAL`` becomes ``LONG_SIGNAL``, and ``FLAT_SIGNAL`` remains
    ``FLAT_SIGNAL``.

    Args:
        df: Input DataFrame containing ``signal_column``.
        signal_column: Existing discrete signal column to invert.
        output_column: Optional output column name. Defaults to
            ``{signal_column}_inverted``.

    Returns:
        A new DataFrame with the inverted signal column appended. The input
        is unchanged.

    Raises:
        KeyError: If ``signal_column`` is missing.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If ``signal_column`` contains invalid signal values.
    """
    _validate_signal_column(df, signal_column)

    name = output_column if output_column is not None else f"{signal_column}_inverted"
    result = df.copy()
    result[name] = result[signal_column] * -1
    return result


def count_signal_changes(df: pd.DataFrame, signal_column: str) -> int:
    """Count how many times consecutive signal values differ.

    The first row is excluded because it has no previous observation.

    Args:
        df: Input DataFrame containing ``signal_column``.
        signal_column: Existing discrete signal column to inspect.

    Returns:
        The number of consecutive signal changes as an ``int``.

    Raises:
        KeyError: If ``signal_column`` is missing.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If ``signal_column`` contains invalid signal values.
    """
    _validate_signal_column(df, signal_column)

    signal = df[signal_column]
    return int(signal.ne(signal.shift()).sum() - 1)


def summarize_signal(df: pd.DataFrame, signal_column: str) -> pd.Series:
    """Summarize counts of long, flat, and short signals.

    Missing signal categories are reported as zero.

    Args:
        df: Input DataFrame containing ``signal_column``.
        signal_column: Existing discrete signal column to summarize.

    Returns:
        A ``Series`` indexed by ``long``, ``flat``, and ``short`` with
        integer counts for each category.

    Raises:
        KeyError: If ``signal_column`` is missing.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If ``signal_column`` contains invalid signal values.
    """
    _validate_signal_column(df, signal_column)

    counts = (
        df[signal_column]
        .value_counts()
        .reindex([LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL], fill_value=0)
        .astype("int64")
    )
    counts.index = pd.Index(["long", "flat", "short"])
    return counts
