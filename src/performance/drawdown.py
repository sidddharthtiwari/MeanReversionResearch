"""Drawdown-series computation for quantitative performance analysis.

Computes drawdown time-series from cumulative equity. This module does not
compute scalar metrics such as maximum drawdown or drawdown duration.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.validation import (
    _validate_columns_exist,
    _validate_numeric_series,
    _validate_output_column,
)
from src.performance.constants import DEFAULT_DRAWDOWN_COLUMN

__all__ = [
    "compute_drawdown_series",
]


def _calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute the drawdown series from cumulative equity.

    Drawdown is ``(equity / running_peak) - 1``, where ``running_peak`` is the
    cumulative maximum of ``equity``.

    Args:
        equity: Cumulative equity series.

    Returns:
        Drawdown series aligned to ``equity.index``.
    """
    running_peak = equity.cummax()
    return (equity / running_peak) - 1.0


def compute_drawdown_series(
    df: pd.DataFrame,
    equity_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute the drawdown series from a cumulative-equity column.

    Drawdown is ``(equity / running_peak) - 1``, where ``running_peak`` is the
    cumulative maximum of equity.

    Args:
        df: Input DataFrame containing ``equity_column``.
        equity_column: Numeric cumulative-equity column.
        output_column: Optional drawdown column name. Defaults to
            ``DEFAULT_DRAWDOWN_COLUMN``.

    Returns:
        A new DataFrame with the drawdown column appended. The input is
        unchanged.

    Raises:
        TypeError: If ``df`` is not a DataFrame, ``equity_column`` is not
            numeric, or ``output_column`` is neither a string nor ``None``.
        KeyError: If ``equity_column`` is missing.
        ValueError: If ``equity_column`` is empty, or if ``output_column``
            equals ``equity_column``.
    """
    _validate_columns_exist(df, [equity_column])
    _validate_numeric_series(df[equity_column], equity_column)
    _validate_output_column(output_column)
    if output_column is not None and output_column == equity_column:
        raise ValueError(
            "output_column must be different from equity_column."
        )

    output_column_name = (
        output_column
        if output_column is not None
        else DEFAULT_DRAWDOWN_COLUMN
    )
    result = df.copy()
    result[output_column_name] = _calculate_drawdown_series(
        result[equity_column]
    )
    return result
