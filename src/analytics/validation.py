"""Shared validation helpers for the analytics package.

This module contains reusable private validation functions only. It does not
perform analytics calculations or expose a public API.
"""

from __future__ import annotations

import math

import pandas as pd
from pandas.api.types import is_numeric_dtype

_SUPPORTED_FREQUENCIES = frozenset({"D", "W", "M"})


def _validate_numeric_series(series: pd.Series, series_name: str) -> None:
    """Validate that ``series`` is a non-empty numeric pandas Series.

    Args:
        series: Series to validate.
        series_name: Name used in exception messages.

    Raises:
        TypeError: If ``series`` is not a ``Series``, or is not numeric.
        ValueError: If ``series`` is empty.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{series_name} must be a pandas Series, "
            f"got {type(series).__name__}."
        )
    if series.empty:
        raise ValueError(f"{series_name} must not be empty.")
    if not is_numeric_dtype(series):
        raise TypeError(
            f"{series_name} must be numeric, got dtype '{series.dtype}'."
        )


def _validate_frequency(frequency: str) -> None:
    """Validate that ``frequency`` is a supported return frequency code.

    Supported values are ``"D"``, ``"W"``, and ``"M"``.

    Args:
        frequency: Frequency code to validate.

    Raises:
        TypeError: If ``frequency`` is not a string.
        ValueError: If ``frequency`` is not a supported value.
    """
    if not isinstance(frequency, str):
        raise TypeError(
            f"frequency must be a string, got {type(frequency).__name__}."
        )
    if frequency not in _SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"frequency must be one of {sorted(_SUPPORTED_FREQUENCIES)}, "
            f"got '{frequency}'."
        )


def _validate_risk_free_rate(risk_free_rate: int | float) -> None:
    """Validate that ``risk_free_rate`` is a finite int or float.

    Args:
        risk_free_rate: Annual risk-free rate to validate.

    Raises:
        TypeError: If ``risk_free_rate`` is not an ``int`` or ``float``.
        ValueError: If ``risk_free_rate`` is NaN or infinite.
    """
    if isinstance(risk_free_rate, bool) or not isinstance(
        risk_free_rate, (int, float)
    ):
        raise TypeError(
            "risk_free_rate must be an int or float, "
            f"got {type(risk_free_rate).__name__}."
        )
    if not math.isfinite(risk_free_rate):
        raise ValueError(
            f"risk_free_rate must be a finite numeric value, "
            f"got {risk_free_rate}."
        )


def _validate_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that ``df`` is a DataFrame containing every required column.

    Args:
        df: DataFrame expected to contain ``columns``.
        columns: Required column names.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        KeyError: If one or more required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_columns)}."
        )


def _validate_output_column(output_column: str | None) -> None:
    """Validate that ``output_column`` is ``None`` or a string.

    Args:
        output_column: Optional output column name to validate.

    Raises:
        TypeError: If ``output_column`` is neither ``None`` nor a string.
    """
    if output_column is not None and not isinstance(output_column, str):
        raise TypeError(
            "output_column must be a string or None, "
            f"got {type(output_column).__name__}."
        )
