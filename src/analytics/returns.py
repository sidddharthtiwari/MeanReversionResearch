"""Scalar return-performance metrics for quantitative analytics.

Computes total return, average period return, annualised return, and CAGR from
a period-return column. This module does not compute risk, drawdown, or
relative-performance statistics.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.constants import PERIODS_PER_YEAR_BY_FREQUENCY
from src.analytics.validation import (
    _validate_columns_exist,
    _validate_frequency,
    _validate_numeric_series,
)

__all__ = [
    "generate_return_summary",
]


def _compute_total_return(returns: pd.Series) -> float:
    """Compute total return from a period-return series.

    Total return is ``(1 + returns).prod() - 1``.

    Args:
        returns: Period return series.

    Returns:
        Total compounded return as a float.
    """
    return float((1.0 + returns).prod() - 1.0)


def _compute_average_period_return(returns: pd.Series) -> float:
    """Compute the average period return.

    Args:
        returns: Period return series.

    Returns:
        Arithmetic mean of period returns as a float.
    """
    return float(returns.mean())


def _compute_annualised_return(
    average_period_return: float,
    periods_per_year: int,
) -> float:
    """Compute arithmetic annualised return.

    Annualised return is ``average_period_return * periods_per_year``.

    Args:
        average_period_return: Mean return per period.
        periods_per_year: Annualisation factor for the return frequency.

    Returns:
        Arithmetic annualised return as a float.
    """
    return float(average_period_return * periods_per_year)


def _compute_cagr(
    total_return: float,
    n_periods: int,
    periods_per_year: int,
) -> float:
    """Compute compound annual growth rate (CAGR).

    CAGR is ``ending_value ** (periods_per_year / n_periods) - 1``, where
    ``ending_value`` is ``1 + total_return``.

    Args:
        total_return: Total compounded return over the sample.
        n_periods: Number of return periods in the sample.
        periods_per_year: Annualisation factor for the return frequency.

    Returns:
        CAGR as a float.

    Raises:
        ValueError: If cumulative wealth is zero or negative.
    """
    ending_value = 1.0 + total_return
    if ending_value <= 0:
        raise ValueError(
            "CAGR is undefined when cumulative wealth "
            "is zero or negative."
        )
    return float(ending_value ** (periods_per_year / n_periods) - 1.0)


def generate_return_summary(
    df: pd.DataFrame,
    return_column: str,
    frequency: str = "D",
) -> dict[str, float]:
    """Compute scalar return-performance metrics from a period-return column.

    Metrics:

    - ``total_return``: ``(1 + returns).prod() - 1``
    - ``average_period_return``: mean period return
    - ``annualised_return``: ``average_period_return * periods_per_year``
    - ``cagr``: ``(1 + total_return) ** (periods_per_year / n) - 1``

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
        ValueError: If ``return_column`` is empty, if ``frequency`` is not a
            supported value, or if CAGR is undefined because cumulative wealth
            is zero or negative.
    """
    _validate_columns_exist(df, [return_column])
    _validate_numeric_series(df[return_column], return_column)
    _validate_frequency(frequency)

    returns = df[return_column]
    periods_per_year = PERIODS_PER_YEAR_BY_FREQUENCY[frequency]
    total_return = _compute_total_return(returns)
    average_period_return = _compute_average_period_return(returns)

    return {
        "total_return": total_return,
        "average_period_return": average_period_return,
        "annualised_return": _compute_annualised_return(
            average_period_return,
            periods_per_year,
        ),
        "cagr": _compute_cagr(
            total_return,
            n_periods=len(returns),
            periods_per_year=periods_per_year,
        ),
    }
