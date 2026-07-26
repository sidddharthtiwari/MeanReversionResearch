"""Scalar absolute risk metrics for quantitative analytics.

Computes volatility, annualised volatility, and downside deviation from a
period-return column. This module does not compute risk-adjusted ratios,
drawdowns, or relative-performance statistics.
"""

from __future__ import annotations

import math

import pandas as pd

from src.analytics.constants import PERIODS_PER_YEAR_BY_FREQUENCY
from src.analytics.validation import (
    _validate_columns_exist,
    _validate_frequency,
    _validate_numeric_series,
)

__all__ = [
    "generate_risk_summary",
]


def _compute_volatility(returns: pd.Series) -> float:
    """Compute the sample standard deviation of period returns.

    Volatility is ``returns.std(ddof=1)``.

    Args:
        returns: Period return series.

    Returns:
        Sample standard deviation as a float.
    """
    return float(returns.std(ddof=1))


def _compute_annualised_volatility(
    volatility: float,
    periods_per_year: int,
) -> float:
    """Compute annualised volatility.

    Annualised volatility is ``volatility * sqrt(periods_per_year)``.

    Args:
        volatility: Period sample standard deviation.
        periods_per_year: Annualisation factor for the return frequency.

    Returns:
        Annualised volatility as a float.
    """
    return float(volatility * math.sqrt(periods_per_year))


def _compute_downside_deviation(returns: pd.Series) -> float:
    """Compute Sortino-compatible downside deviation.

    Negative returns are clipped at zero, then downside deviation is
    ``sqrt(mean(negative_returns ** 2))``.

    Args:
        returns: Period return series.

    Returns:
        Downside deviation as a float.
    """
    negative_returns = returns.clip(upper=0)
    return float(math.sqrt((negative_returns ** 2).mean()))


def generate_risk_summary(
    df: pd.DataFrame,
    return_column: str,
    frequency: str = "D",
) -> dict[str, float]:
    """Compute scalar absolute risk metrics from a period-return column.

    Metrics:

    - ``volatility``: sample standard deviation (``ddof=1``)
    - ``annualised_volatility``: ``volatility * sqrt(periods_per_year)``
    - ``downside_deviation``: Sortino-compatible downside deviation

    Args:
        df: Input DataFrame containing ``return_column``.
        return_column: Numeric period-return column.
        frequency: Return frequency used for annualisation. Supported values
            are ``"D"``, ``"W"``, and ``"M"``. Defaults to ``"D"``.

    Returns:
        Dictionary mapping metric names to floating-point values.

    Raises:
        TypeError: If ``df`` is not a DataFrame, ``return_column`` is not
            numeric, or ``frequency`` is not a string.
        KeyError: If ``return_column`` is missing.
        ValueError: If ``return_column`` is empty, or if ``frequency`` is not
            a supported value.
    """
    _validate_columns_exist(df, [return_column])
    _validate_numeric_series(df[return_column], return_column)
    _validate_frequency(frequency)

    returns = df[return_column]
    periods_per_year = PERIODS_PER_YEAR_BY_FREQUENCY[frequency]
    volatility = _compute_volatility(returns)

    return {
        "volatility": volatility,
        "annualised_volatility": _compute_annualised_volatility(
            volatility,
            periods_per_year,
        ),
        "downside_deviation": _compute_downside_deviation(returns),
    }
