"""Cumulative-return computation for quantitative analytics.

Computes cumulative returns from a period-return column. This module does not
compute drawdowns, risk metrics, or other performance statistics.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.constants import CUMULATIVE_RETURN_SUFFIX
from src.analytics.validation import (
    _validate_columns_exist,
    _validate_numeric_series,
    _validate_output_column,
)

__all__ = [
    "compute_cumulative_returns",
]


def _generate_output_column_name(
    return_column: str,
    output_column: str | None,
) -> str:
    """Generate the cumulative-return output column name.

    Args:
        return_column: Source return column used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``{return_column}{CUMULATIVE_RETURN_SUFFIX}``.
    """
    if output_column is not None:
        return output_column
    return f"{return_column}{CUMULATIVE_RETURN_SUFFIX}"


def _compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative returns from a period-return series.

    Cumulative return is ``(1 + returns).cumprod() - 1``.

    Args:
        returns: Period return series.

    Returns:
        Cumulative return series aligned to ``returns.index``.
    """
    return (1.0 + returns).cumprod() - 1.0


def compute_cumulative_returns(
    df: pd.DataFrame,
    return_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute cumulative returns from a period-return column.

    Cumulative return is ``(1 + returns).cumprod() - 1``.

    Args:
        df: Input DataFrame containing ``return_column``.
        return_column: Numeric period-return column.
        output_column: Optional cumulative-return column name. Defaults to
            ``{return_column}_cumulative_return``.

    Returns:
        A new DataFrame with the cumulative-return column appended. The input
        is unchanged.

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

    output_column_name = _generate_output_column_name(
        return_column,
        output_column,
    )
    result = df.copy()
    result[output_column_name] = _compute_cumulative_returns(
        result[return_column]
    )
    return result
