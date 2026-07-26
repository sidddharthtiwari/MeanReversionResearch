"""Mean-reversion signal generation for quantitative research.

Converts an existing numeric feature column into discrete mean-reversion
trading signals using a symmetric entry threshold. This module does not
compute features, manage positions, generate exits, or perform backtesting.
"""

from __future__ import annotations

import pandas as pd

from src.signals.constants import (
    FLAT_SIGNAL,
    LONG_SIGNAL,
    SHORT_SIGNAL,
    SIGNAL_SUFFIX,
)
from src.signals.validation import _validate_numeric_column

__all__ = [
    "generate_mean_reversion_signal",
]


def _generate_output_column_name(
    feature_column: str,
    output_column: str | None,
) -> str:
    """Resolve the mean-reversion signal output column name.

    Args:
        feature_column: Source feature column used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``{feature_column}{SIGNAL_SUFFIX}``.
    """
    if output_column is not None:
        return output_column
    return f"{feature_column}{SIGNAL_SUFFIX}"


def _validate_entry_threshold(entry_threshold: float) -> None:
    """Validate that ``entry_threshold`` is strictly positive.

    Args:
        entry_threshold: Symmetric entry threshold magnitude.

    Raises:
        ValueError: If ``entry_threshold`` is less than or equal to zero.
    """
    if entry_threshold <= 0:
        raise ValueError(
            f"entry_threshold must be greater than 0, got {entry_threshold}."
        )


def _apply_mean_reversion_logic(
    values: pd.Series,
    entry_threshold: float,
) -> pd.Series:
    """Map feature values to discrete mean-reversion signals.

    Rules:
        - ``value <= -entry_threshold`` -> ``LONG_SIGNAL``
        - ``value >= entry_threshold`` -> ``SHORT_SIGNAL``
        - otherwise -> ``FLAT_SIGNAL``

    Args:
        values: Numeric feature series to convert.
        entry_threshold: Positive threshold magnitude for long/short entries.

    Returns:
        Integer signal series aligned to ``values.index``.
    """
    signal = pd.Series(FLAT_SIGNAL, index=values.index, dtype="int64")
    signal = signal.mask(values <= -entry_threshold, LONG_SIGNAL)
    signal = signal.mask(values >= entry_threshold, SHORT_SIGNAL)
    return signal


def generate_mean_reversion_signal(
    df: pd.DataFrame,
    feature_column: str,
    entry_threshold: float,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Generate discrete mean-reversion signals from a numeric feature.

    Values at or below ``-entry_threshold`` map to ``LONG_SIGNAL``. Values at
    or above ``entry_threshold`` map to ``SHORT_SIGNAL``. All other values
    map to ``FLAT_SIGNAL``.

    Args:
        df: Input DataFrame containing ``feature_column``.
        feature_column: Numeric feature used to generate signals.
        entry_threshold: Positive threshold magnitude. Must be greater than 0.
        output_column: Optional output column name. Defaults to
            ``{feature_column}_signal``.

    Returns:
        A new DataFrame with the signal column appended. The input is
        unchanged.

    Raises:
        KeyError: If ``feature_column`` is missing.
        TypeError: If ``feature_column`` is not numeric.
        ValueError: If ``entry_threshold`` is less than or equal to zero.
    """
    _validate_numeric_column(df, feature_column)
    _validate_entry_threshold(entry_threshold)

    name = _generate_output_column_name(feature_column, output_column)
    result = df.copy()
    result[name] = _apply_mean_reversion_logic(
        values=result[feature_column],
        entry_threshold=entry_threshold,
    )
    return result
