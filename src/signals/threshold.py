"""Threshold-based signal generation for quantitative research.

Converts an existing numeric feature column into discrete trading signals
using optional buy and/or sell thresholds. This module does not compute
features, generate positions or trades, or perform backtesting.
"""

from __future__ import annotations

import pandas as pd

from src.signals.constants import (
    FLAT_SIGNAL,
    LONG_SIGNAL,
    SHORT_SIGNAL,
    SIGNAL_SUFFIX,
)
from src.signals.validation import (
    _validate_numeric_column,
    _validate_thresholds,
)

__all__ = [
    "generate_threshold_signal",
]


def _generate_output_column_name(
    feature_column: str,
    output_column: str | None,
) -> str:
    """Resolve the signal output column name.

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


def _apply_threshold_logic(
    values: pd.Series,
    buy_threshold: float | None,
    sell_threshold: float | None,
) -> pd.Series:
    """Map feature values to discrete long/flat/short signals.

    Rules:
        - ``value <= buy_threshold`` -> ``LONG_SIGNAL`` (when buy is set)
        - ``value >= sell_threshold`` -> ``SHORT_SIGNAL`` (when sell is set)
        - otherwise -> ``FLAT_SIGNAL``

    Args:
        values: Numeric feature series to convert.
        buy_threshold: Optional long-entry threshold.
        sell_threshold: Optional short-entry threshold.

    Returns:
        Integer signal series aligned to ``values.index``.
    """
    signal = pd.Series(FLAT_SIGNAL, index=values.index, dtype="int64")
    if buy_threshold is not None:
        signal = signal.mask(values <= buy_threshold, LONG_SIGNAL)
    if sell_threshold is not None:
        signal = signal.mask(values >= sell_threshold, SHORT_SIGNAL)
    return signal


def generate_threshold_signal(
    df: pd.DataFrame,
    feature_column: str,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Generate discrete trading signals from a numeric feature column.

    Supports buy-only, sell-only, or both thresholds. When both are provided,
    ``buy_threshold`` must be strictly less than ``sell_threshold``.

    Args:
        df: Input DataFrame containing ``feature_column``.
        feature_column: Numeric feature used to generate signals.
        buy_threshold: Values at or below this map to ``LONG_SIGNAL``.
        sell_threshold: Values at or above this map to ``SHORT_SIGNAL``.
        output_column: Optional output column name. Defaults to
            ``{feature_column}_signal``.

    Returns:
        A new DataFrame with the signal column appended. The input is
        unchanged.

    Raises:
        KeyError: If ``feature_column`` is missing.
        TypeError: If ``feature_column`` is not numeric.
        ValueError: If both thresholds are ``None``, or if both are provided
            and are not strictly ordered.
    """
    _validate_numeric_column(df, feature_column)
    _validate_thresholds(buy_threshold, sell_threshold)

    name = _generate_output_column_name(feature_column, output_column)
    result = df.copy()
    result[name] = _apply_threshold_logic(
        values=result[feature_column],
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    return result
