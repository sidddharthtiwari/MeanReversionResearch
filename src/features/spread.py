"""Arithmetic spread feature computation for quantitative research.

Spread is defined as ``left_column - right_column``. This module only computes
that difference and does not load data, validate datasets, generate signals,
plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "compute_spread",
]

DEFAULT_SPREAD_COLUMN = "spread"


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Validate that ``column`` exists and has a numeric dtype.

    Args:
        df: Input DataFrame containing the column to validate.
        column: Name of the numeric column to validate.

    Raises:
        KeyError: If ``column`` is not present in ``df``.
        TypeError: If ``column`` is not numeric.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' must be numeric, got dtype '{df[column].dtype}'."
        )


def _generate_output_column_name(output_column: str | None) -> str:
    """Resolve the spread output column name.

    Args:
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``spread``.
    """
    if output_column is not None:
        return output_column
    return DEFAULT_SPREAD_COLUMN


def _compute_spread(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Append the arithmetic spread column to a copied DataFrame.

    Args:
        df: Input DataFrame containing ``left_column`` and ``right_column``.
        left_column: Numeric column used as the left operand.
        right_column: Numeric column used as the right operand.
        output_column: Final resolved name for the spread column.

    Returns:
        A new DataFrame with the spread column appended. The caller's DataFrame
        is never modified.
    """
    result = df.copy()
    result[output_column] = result[left_column] - result[right_column]
    return result


def compute_spread(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute the arithmetic spread between two numeric columns.

    Spread is defined as ``left_column - right_column``.

    Args:
        df: Input DataFrame containing both numeric columns.
        left_column: Numeric column used as the left operand.
        right_column: Numeric column used as the right operand.
        output_column: Optional output column name. Defaults to ``spread``.

    Returns:
        A new DataFrame with the spread column appended. The input is unchanged.
    """
    _validate_numeric_column(df, left_column)
    _validate_numeric_column(df, right_column)
    name = _generate_output_column_name(output_column)
    return _compute_spread(
        df=df,
        left_column=left_column,
        right_column=right_column,
        output_column=name,
    )
