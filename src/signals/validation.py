"""Reusable validation helpers for the signal layer.

This module contains validation logic only. It does not generate signals,
modify DataFrames, resolve output column names, or encode business rules.
"""

from __future__ import annotations

import pandas as pd


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


def _validate_two_numeric_columns(
    df: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> None:
    """Validate that both columns exist and are numeric.

    Args:
        df: Input DataFrame containing both columns.
        first_column: First numeric column to validate.
        second_column: Second numeric column to validate.

    Raises:
        KeyError: If either column is not present in ``df``.
        TypeError: If either column is not numeric.
    """
    _validate_numeric_column(df, first_column)
    _validate_numeric_column(df, second_column)


def _validate_thresholds(
    buy_threshold: float | None,
    sell_threshold: float | None,
) -> None:
    """Validate buy/sell threshold configuration.

    Allows buy-only and sell-only configurations. When both thresholds are
    supplied, ``buy_threshold`` must be strictly less than ``sell_threshold``.

    Args:
        buy_threshold: Optional long-entry threshold.
        sell_threshold: Optional short-entry threshold.

    Raises:
        ValueError: If both thresholds are ``None``, or if both are provided
            and ``buy_threshold >= sell_threshold``.
    """
    if buy_threshold is None and sell_threshold is None:
        raise ValueError(
            "At least one of buy_threshold or sell_threshold must be provided."
        )
    if (
        buy_threshold is not None
        and sell_threshold is not None
        and buy_threshold >= sell_threshold
    ):
        raise ValueError(
            f"buy_threshold ({buy_threshold}) must be less than "
            f"sell_threshold ({sell_threshold})."
        )
