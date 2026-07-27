"""Drawdown analytics for quantitative research.

Computes maximum drawdown and drawdown duration from a period-return column.
This module does not compute risk-adjusted ratios or return-performance
statistics. Equity and drawdown time-series are produced by the performance
package.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.validation import (
    _validate_columns_exist,
    _validate_numeric_series,
)
from src.performance.constants import (
    DEFAULT_DRAWDOWN_COLUMN,
    DEFAULT_EQUITY_COLUMN,
)
from src.performance.drawdown import compute_drawdown_series
from src.performance.equity import compute_equity_curve

__all__ = [
    "generate_drawdown_summary",
]


def _compute_max_drawdown(drawdown: pd.Series) -> float:
    """Compute maximum drawdown from a drawdown series.

    Maximum drawdown is the minimum (most negative) drawdown value.

    Args:
        drawdown: Drawdown series.

    Returns:
        Maximum drawdown as a float.
    """
    return float(drawdown.min())


def _compute_drawdown_duration(drawdown: pd.Series) -> int:
    """Compute the longest consecutive drawdown duration.

    Duration counts consecutive periods where ``drawdown < 0``. The current
    streak resets whenever drawdown returns to zero.

    Args:
        drawdown: Drawdown series.

    Returns:
        Maximum consecutive drawdown duration in periods.
    """
    current_duration = 0
    maximum_duration = 0
    for value in drawdown:
        if value < 0:
            current_duration += 1
            if current_duration > maximum_duration:
                maximum_duration = current_duration
        else:
            current_duration = 0
    return maximum_duration


def generate_drawdown_summary(
    df: pd.DataFrame,
    return_column: str,
) -> dict[str, float | int]:
    """Compute drawdown summary metrics from a period-return column.

    Metrics:

    - ``max_drawdown``: Most negative value in the drawdown series.
    - ``drawdown_duration``: Maximum consecutive periods where drawdown
      remains below zero.

    Args:
        df: Input DataFrame containing ``return_column``.
        return_column: Numeric period-return column.

    Returns:
        Dictionary containing:

        - ``max_drawdown``: Most negative drawdown observed.
        - ``drawdown_duration``: Longest consecutive period spent below the
          previous equity peak.

    Raises:
        TypeError: If ``df`` is not a DataFrame, or ``return_column`` is not
            numeric.
        KeyError: If ``return_column`` is missing.
        ValueError: If ``return_column`` is empty.
    """
    _validate_columns_exist(df, [return_column])
    _validate_numeric_series(df[return_column], return_column)

    equity_frame = compute_equity_curve(df, return_column)
    drawdown_frame = compute_drawdown_series(
        equity_frame,
        DEFAULT_EQUITY_COLUMN,
    )
    drawdown_series = drawdown_frame[DEFAULT_DRAWDOWN_COLUMN]

    return {
        "max_drawdown": _compute_max_drawdown(drawdown_series),
        "drawdown_duration": _compute_drawdown_duration(drawdown_series),
    }
