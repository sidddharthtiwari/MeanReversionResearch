"""Logarithmic spread feature computation for quantitative research.

Logarithmic spread is defined as ``log(left_column) - log(right_column)``.
This module only computes that difference and does not load data, validate
datasets, generate signals, plot, or backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_log_spread",
]

DEFAULT_LOG_SPREAD_COLUMN = "log_spread"


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


def _validate_positive_values(df: pd.DataFrame, column: str) -> None:
    """Validate that every value in ``column`` is strictly greater than zero.

    Args:
        df: Input DataFrame containing the column to validate.
        column: Name of the numeric column to validate.

    Raises:
        ValueError: If any value is less than or equal to zero.
    """
    if (df[column] <= 0).any():
        raise ValueError(
            f"Column '{column}' must contain strictly positive values; "
            "logarithmic spread requires positive inputs."
        )


def _generate_output_column_name(output_column: str | None) -> str:
    """Resolve the log-spread output column name.

    Args:
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``log_spread``.
    """
    if output_column is not None:
        return output_column
    return DEFAULT_LOG_SPREAD_COLUMN


def _compute_log_spread(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Append the logarithmic spread column to a copied DataFrame.

    Args:
        df: Input DataFrame containing ``left_column`` and ``right_column``.
        left_column: Positive numeric column used as the left operand.
        right_column: Positive numeric column used as the right operand.
        output_column: Final resolved name for the log-spread column.

    Returns:
        A new DataFrame with the log-spread column appended. The caller's
        DataFrame is never modified.
    """
    result = df.copy()
    left_log = np.log(result[left_column])
    right_log = np.log(result[right_column])
    result[output_column] = left_log - right_log
    return result


def compute_log_spread(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute the logarithmic spread between two positive numeric columns.

    Logarithmic spread is defined as
    ``log(left_column) - log(right_column)``. Both input columns must contain
    strictly positive values.

    Args:
        df: Input DataFrame containing both positive numeric columns.
        left_column: Positive numeric column used as the left operand.
        right_column: Positive numeric column used as the right operand.
        output_column: Optional output column name. Defaults to ``log_spread``.

    Returns:
        A new DataFrame with the log-spread column appended. The input is
        unchanged.
    """
    _validate_numeric_column(df, left_column)
    _validate_numeric_column(df, right_column)
    _validate_positive_values(df, left_column)
    _validate_positive_values(df, right_column)
    name = _generate_output_column_name(output_column)
    return _compute_log_spread(
        df=df,
        left_column=left_column,
        right_column=right_column,
        output_column=name,
    )
