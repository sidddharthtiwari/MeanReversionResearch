"""Breakout event signal generation for quantitative research.

Detects discrete breakout and breakdown events against optional upper and
lower thresholds on an existing numeric feature column. This module does not
compute features, manage positions, generate exits, or perform backtesting.
It signals events only, not above/below state.
"""

from __future__ import annotations

import pandas as pd

from src.signals.constants import (
    BREAKOUT_SIGNAL_SUFFIX,
    FLAT_SIGNAL,
    LONG_SIGNAL,
    SHORT_SIGNAL,
)
from src.signals.validation import _validate_numeric_column

__all__ = [
    "generate_breakout_signal",
]


def _generate_output_column_name(
    feature_column: str,
    output_column: str | None,
) -> str:
    """Resolve the breakout signal output column name.

    Args:
        feature_column: Source feature column used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``{feature_column}_breakout_signal``.
    """
    if output_column is not None:
        return output_column
    return f"{feature_column}{BREAKOUT_SIGNAL_SUFFIX}"


def _validate_breakout_thresholds(
    upper_threshold: float | None,
    lower_threshold: float | None,
) -> None:
    """Validate upper/lower breakout threshold configuration.

    Allows upper-only and lower-only configurations. When both thresholds are
    supplied, ``lower_threshold`` must be strictly less than
    ``upper_threshold``.

    Args:
        upper_threshold: Optional bullish-breakout threshold.
        lower_threshold: Optional bearish-breakdown threshold.

    Raises:
        ValueError: If both thresholds are ``None``, or if both are provided
            and ``lower_threshold`` is not strictly less than
            ``upper_threshold``.
    """
    if upper_threshold is None and lower_threshold is None:
        raise ValueError(
            "At least one of upper_threshold or lower_threshold must be provided."
        )
    if (
        upper_threshold is not None
        and lower_threshold is not None
        and not lower_threshold < upper_threshold
    ):
        raise ValueError(
            f"lower_threshold ({lower_threshold}) must be less than "
            f"upper_threshold ({upper_threshold})."
        )


def _apply_breakout_logic(
    values: pd.Series,
    upper_threshold: float | None,
    lower_threshold: float | None,
) -> pd.Series:
    """Map feature values to discrete breakout event signals.

    Rules:
        - Previous ``value <= upper_threshold`` and current
          ``value > upper_threshold`` -> ``LONG_SIGNAL`` (when upper is set)
        - Previous ``value >= lower_threshold`` and current
          ``value < lower_threshold`` -> ``SHORT_SIGNAL`` (when lower is set)
        - Otherwise -> ``FLAT_SIGNAL``

    The first row is always ``FLAT_SIGNAL`` because no previous observation
    exists.

    Args:
        values: Numeric feature series to convert.
        upper_threshold: Optional bullish-breakout threshold.
        lower_threshold: Optional bearish-breakdown threshold.

    Returns:
        Integer signal series aligned to ``values.index``.
    """
    previous = values.shift(1)

    signal = pd.Series(FLAT_SIGNAL, index=values.index, dtype="int64")
    if upper_threshold is not None:
        bullish_breakout = (previous <= upper_threshold) & (values > upper_threshold)
        signal = signal.mask(bullish_breakout, LONG_SIGNAL)
    if lower_threshold is not None:
        bearish_breakdown = (previous >= lower_threshold) & (values < lower_threshold)
        signal = signal.mask(bearish_breakdown, SHORT_SIGNAL)
    return signal


def generate_breakout_signal(
    df: pd.DataFrame,
    feature_column: str,
    upper_threshold: float | None = None,
    lower_threshold: float | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Generate discrete breakout event signals from a numeric feature.

    Supports upper-only, lower-only, or both thresholds. When both are
    provided, ``lower_threshold`` must be strictly less than
    ``upper_threshold``.

    Args:
        df: Input DataFrame containing ``feature_column``.
        feature_column: Numeric feature used to generate signals.
        upper_threshold: Values crossing above this emit ``LONG_SIGNAL``.
        lower_threshold: Values crossing below this emit ``SHORT_SIGNAL``.
        output_column: Optional output column name. Defaults to
            ``{feature_column}_breakout_signal``.

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
    _validate_breakout_thresholds(upper_threshold, lower_threshold)

    name = _generate_output_column_name(feature_column, output_column)
    result = df.copy()
    result[name] = _apply_breakout_logic(
        values=result[feature_column],
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
    return result
