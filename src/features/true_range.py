"""Per-symbol True Range (TR) feature computation for quantitative research.

True Range is the maximum of ``high - low``, ``abs(high - previous_close)``,
and ``abs(low - previous_close)``, with previous close computed independently
per symbol. This module does not compute ATR, generate signals, plot, or
backtest.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "compute_true_range",
]

DEFAULT_HIGH_COLUMN = "high"
DEFAULT_LOW_COLUMN = "low"
DEFAULT_CLOSE_COLUMN = "close"
DEFAULT_TRUE_RANGE_COLUMN = "true_range"


def _validate_symbol_column(df: pd.DataFrame) -> None:
    """Validate that the required ``symbol`` column exists.

    Args:
        df: Input DataFrame expected to contain a ``symbol`` column.

    Raises:
        KeyError: If ``symbol`` is not present in ``df``.
    """
    if "symbol" not in df.columns:
        raise KeyError("Required column 'symbol' not found in DataFrame.")


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


def _group_by_symbol(df: pd.DataFrame):
    """Group rows by symbol without reordering groups.

    Args:
        df: Input DataFrame containing a ``symbol`` column.

    Returns:
        A DataFrameGroupBy object grouped by ``symbol`` with ``sort=False``.
    """
    return df.groupby("symbol", sort=False)


def _generate_output_column_name(output_column: str | None) -> str:
    """Resolve the True Range output column name.

    Args:
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise ``true_range``.
    """
    if output_column is not None:
        return output_column
    return DEFAULT_TRUE_RANGE_COLUMN


def _compute_true_range(
    df: pd.DataFrame,
    high_column: str,
    low_column: str,
    close_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Append per-symbol True Range to a copied DataFrame.

    Args:
        df: Input DataFrame containing ``symbol`` and OHLC columns.
        high_column: High price column.
        low_column: Low price column.
        close_column: Close price column used for previous-close gaps.
        output_column: Final resolved name for the True Range column.

    Returns:
        A new DataFrame with the True Range column appended. The caller's
        DataFrame is never modified.
    """
    result = df.copy()
    high = result[high_column]
    low = result[low_column]
    previous_close = _group_by_symbol(result)[close_column].shift(1)

    high_low = high - low
    high_previous_close = (high - previous_close).abs()
    low_previous_close = (low - previous_close).abs()

    result[output_column] = pd.concat(
        [high_low, high_previous_close, low_previous_close],
        axis=1,
    ).max(axis=1)
    return result


def compute_true_range(
    df: pd.DataFrame,
    high_column: str = DEFAULT_HIGH_COLUMN,
    low_column: str = DEFAULT_LOW_COLUMN,
    close_column: str = DEFAULT_CLOSE_COLUMN,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Compute per-symbol True Range (TR).

    True Range is the maximum of:

    - ``high - low``
    - ``abs(high - previous_close)``
    - ``abs(low - previous_close)``

    Previous close is shifted independently within each symbol.

    Args:
        df: OHLC DataFrame containing ``symbol`` and price columns.
        high_column: High price column.
        low_column: Low price column.
        close_column: Close price column used for previous-close gaps.
        output_column: Optional output column name. Defaults to ``true_range``.

    Returns:
        A new DataFrame with the True Range column appended. The input is
        unchanged.
    """
    _validate_symbol_column(df)
    _validate_numeric_column(df, high_column)
    _validate_numeric_column(df, low_column)
    _validate_numeric_column(df, close_column)
    name = _generate_output_column_name(output_column)
    return _compute_true_range(
        df=df,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
        output_column=name,
    )
