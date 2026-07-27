"""Equity-curve computation for quantitative performance analysis.

Computes cumulative equity time-series from period returns. This module does
not compute scalar metrics or drawdown series.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.validation import (
    _validate_columns_exist,
    _validate_numeric_series,
    _validate_output_column,
)
from src.performance.constants import DEFAULT_EQUITY_COLUMN

__all__ = [
    "compute_equity_curve",
]


def _calculate_equity_curve(returns: pd.Series) -> pd.Series:
    """Compute cumulative equity from a period-return series.

    Equity is ``(1.0 + returns).cumprod()``.

    Args:
        returns: Period return series.

    Returns:
        Cumulative equity series aligned to ``returns.index``.
    """
    return (1.0 + returns).cumprod()


def compute_equity_curve(
    df: pd.DataFrame,
    return_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute cumulative equity from a period-return column.

    Equity is ``(1.0 + returns).cumprod()``.

    Args:
        df: Input DataFrame containing ``return_column``.
        return_column: Numeric period-return column.
        output_column: Optional equity column name. Defaults to
            ``DEFAULT_EQUITY_COLUMN``.

    Returns:
        A new DataFrame with the equity column appended. The input is
        unchanged.

    Raises:
        TypeError: If ``df`` is not a DataFrame, ``return_column`` is not
            numeric, or ``output_column`` is neither a string nor ``None``.
        KeyError: If ``return_column`` is missing.
        ValueError: If ``return_column`` is empty, or if ``output_column``
            equals ``return_column``.
    """
    _validate_columns_exist(df, [return_column])
    _validate_numeric_series(df[return_column], return_column)
    _validate_output_column(output_column)
    if output_column is not None and output_column == return_column:
        raise ValueError(
            "output_column must be different from return_column."
        )

    output_column_name = (
        output_column
        if output_column is not None
        else DEFAULT_EQUITY_COLUMN
    )
    result = df.copy()
    result[output_column_name] = _calculate_equity_curve(result[return_column])
    return result
