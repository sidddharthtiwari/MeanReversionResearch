"""Risk-adjusted performance ratios for quantitative analytics.

Orchestrates Sharpe, Sortino, and Calmar ratios from existing return, risk,
and drawdown summary APIs. This module does not recompute underlying return,
risk, or drawdown statistics.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.constants import DEFAULT_RISK_FREE_RATE
from src.analytics.drawdown import generate_drawdown_summary
from src.analytics.returns import generate_return_summary
from src.analytics.risk import generate_risk_summary
from src.analytics.validation import (
    _validate_columns_exist,
    _validate_frequency,
    _validate_numeric_series,
    _validate_risk_free_rate,
)

__all__ = [
    "generate_ratio_summary",
]


def _compute_sharpe_ratio(
    annualised_return: float,
    annualised_volatility: float,
    risk_free_rate: float,
) -> float:
    """Compute the Sharpe ratio.

    Sharpe ratio is
    ``(annualised_return - risk_free_rate) / annualised_volatility``.

    Args:
        annualised_return: Arithmetic annualised return.
        annualised_volatility: Annualised return volatility.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Sharpe ratio as a float, or ``0.0`` when volatility is non-positive.
    """
    if annualised_volatility <= 0:
        return 0.0
    return float(
        (annualised_return - risk_free_rate) / annualised_volatility
    )


def _compute_sortino_ratio(
    annualised_return: float,
    downside_deviation: float,
    risk_free_rate: float,
) -> float:
    """Compute the Sortino ratio.

    Sortino ratio is
    ``(annualised_return - risk_free_rate) / downside_deviation``.

    Args:
        annualised_return: Arithmetic annualised return.
        downside_deviation: Sortino-compatible downside deviation.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Sortino ratio as a float, or ``0.0`` when downside deviation is
        non-positive.
    """
    if downside_deviation <= 0:
        return 0.0
    return float((annualised_return - risk_free_rate) / downside_deviation)


def _compute_calmar_ratio(
    cagr: float,
    max_drawdown: float,
) -> float:
    """Compute the Calmar ratio.

    Calmar ratio is ``cagr / abs(max_drawdown)``.

    Args:
        cagr: Compound annual growth rate.
        max_drawdown: Most negative drawdown observed.

    Returns:
        Calmar ratio as a float, or ``0.0`` when absolute max drawdown is
        non-positive.
    """
    absolute_max_drawdown = abs(max_drawdown)
    if absolute_max_drawdown <= 0:
        return 0.0
    return float(cagr / absolute_max_drawdown)


def generate_ratio_summary(
    df: pd.DataFrame,
    return_column: str,
    frequency: str = "D",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float]:
    """Compute risk-adjusted performance ratios from a period-return column.

    Metrics:

    - ``sharpe_ratio``: Excess annualised return divided by annualised
      volatility.
    - ``sortino_ratio``: Excess annualised return divided by downside
      deviation.
    - ``calmar_ratio``: CAGR divided by the absolute maximum drawdown.

    Args:
        df: Input DataFrame containing ``return_column``.
        return_column: Numeric period-return column.
        frequency: Return frequency used for annualisation. Supported values
            are ``"D"``, ``"W"``, and ``"M"``. Defaults to ``"D"``.
        risk_free_rate: Annual risk-free rate. Defaults to
            ``DEFAULT_RISK_FREE_RATE``.

    Returns:
        Dictionary containing:

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

    annualised_return = return_summary["annualised_return"]
    cagr = return_summary["cagr"]
    annualised_volatility = risk_summary["annualised_volatility"]
    downside_deviation = risk_summary["downside_deviation"]
    max_drawdown = drawdown_summary["max_drawdown"]

    return {
        "sharpe_ratio": _compute_sharpe_ratio(
            annualised_return,
            annualised_volatility,
            risk_free_rate,
        ),
        "sortino_ratio": _compute_sortino_ratio(
            annualised_return,
            downside_deviation,
            risk_free_rate,
        ),
        "calmar_ratio": _compute_calmar_ratio(
            cagr,
            max_drawdown,
        ),
    }
