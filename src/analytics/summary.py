"""Top-level analytics summary orchestration.

Aggregates return, risk, drawdown, and ratio summary metrics into a single
flat dictionary. This module does not perform mathematical calculations.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.constants import DEFAULT_RISK_FREE_RATE
from src.analytics.drawdown import generate_drawdown_summary
from src.analytics.ratios import generate_ratio_summary
from src.analytics.returns import generate_return_summary
from src.analytics.risk import generate_risk_summary
from src.analytics.validation import (
    _validate_columns_exist,
    _validate_frequency,
    _validate_numeric_series,
    _validate_risk_free_rate,
)

__all__ = [
    "generate_summary",
]


def generate_summary(
    df: pd.DataFrame,
    return_column: str,
    frequency: str = "D",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float | int]:
    """Generate a complete analytics summary for a period-return column.

    Orchestrates return, risk, drawdown, and ratio summaries into one flat
    dictionary. No mathematical calculations are performed in this module.

    Args:
        df: Input DataFrame containing ``return_column``.
        return_column: Numeric period-return column.
        frequency: Return frequency used for annualisation. Supported values
            are ``"D"``, ``"W"``, and ``"M"``. Defaults to ``"D"``.
        risk_free_rate: Annual risk-free rate. Defaults to
            ``DEFAULT_RISK_FREE_RATE``.

    Returns:
        Flat dictionary containing:

        - ``total_return``
        - ``average_period_return``
        - ``annualised_return``
        - ``cagr``
        - ``volatility``
        - ``annualised_volatility``
        - ``downside_deviation``
        - ``max_drawdown``
        - ``drawdown_duration``
        - ``sharpe_ratio``
        - ``sortino_ratio``
        - ``calmar_ratio``

    Raises:
        TypeError: If ``df`` is not a DataFrame, ``return_column`` is not
            numeric, ``frequency`` is not a string, or ``risk_free_rate`` is
            not an ``int`` or ``float``.
        KeyError: If ``return_column`` is missing.
        ValueError: If ``return_column`` is empty, if ``frequency`` is not a
            supported value, if ``risk_free_rate`` is non-finite, or if CAGR
            is undefined because cumulative wealth is zero or negative.
    """
    _validate_columns_exist(df, [return_column])
    _validate_numeric_series(df[return_column], return_column)
    _validate_frequency(frequency)
    _validate_risk_free_rate(risk_free_rate)

    return_summary = generate_return_summary(
        df,
        return_column=return_column,
        frequency=frequency,
    )
    risk_summary = generate_risk_summary(
        df,
        return_column=return_column,
        frequency=frequency,
    )
    drawdown_summary = generate_drawdown_summary(
        df,
        return_column=return_column,
    )
    ratio_summary = generate_ratio_summary(
        df,
        return_column=return_column,
        frequency=frequency,
        risk_free_rate=risk_free_rate,
    )

    summary: dict[str, float | int] = {}
    summary.update(return_summary)
    summary.update(risk_summary)
    summary.update(drawdown_summary)
    summary.update(ratio_summary)
    return summary
