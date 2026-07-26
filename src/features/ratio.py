"""Arithmetic ratio feature computation for quantitative research.

Ratio is defined as ``left_column / right_column``. This module only computes
that quotient and does not load data, validate datasets, generate signals,
plot, or backtest.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "compute_ratio",
]

DEFAULT_RATIO_COLUMN = "ratio"


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
    """Resolve the ratio output column name.

    Args:
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``ratio``.
    """
    if output_column is not None:
        return output_column
    return DEFAULT_RATIO_COLUMN


def _compute_ratio(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Append the arithmetic ratio column to a copied DataFrame.

    Args:
        df: Input DataFrame containing ``left_column`` and ``right_column``.
        left_column: Numeric column used as the numerator.
        right_column: Numeric column used as the denominator.
        output_column: Final resolved name for the ratio column.

    Returns:
        A new DataFrame with the ratio column appended. The caller's DataFrame
        is never modified.
    """
    result = df.copy()
    numerator = result[left_column]
    denominator = result[right_column]
    result[output_column] = numerator / denominator
    return result


def compute_ratio(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute the arithmetic ratio between two numeric columns.

    Ratio is defined as ``left_column / right_column``. Division by zero is
    intentionally delegated to pandas/numpy and is not handled by this module.

    Args:
        df: Input DataFrame containing both numeric columns.
        left_column: Numeric column used as the numerator.
        right_column: Numeric column used as the denominator.
        output_column: Optional output column name. Defaults to ``ratio``.

    Returns:
        A new DataFrame with the ratio column appended. The input is unchanged.
    """
    _validate_numeric_column(df, left_column)
    _validate_numeric_column(df, right_column)
    name = _generate_output_column_name(output_column)
    return _compute_ratio(
        df=df,
        left_column=left_column,
        right_column=right_column,
        output_column=name,
    )
