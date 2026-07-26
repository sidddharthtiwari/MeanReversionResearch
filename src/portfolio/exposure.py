"""Portfolio net and gross exposure computation.

Computes net exposure and gross exposure from an existing portfolio position
column. This module does not generate positions, compute returns, or perform
risk analytics.
"""

from __future__ import annotations

import pandas as pd

from src.portfolio.constants import GROSS_EXPOSURE_SUFFIX, NET_EXPOSURE_SUFFIX
from src.portfolio.validation import _validate_position_column

__all__ = [
    "generate_exposure",
]


def _generate_net_exposure_column_name(
    position_column: str,
    net_exposure_column: str | None,
) -> str:
    """Resolve the net-exposure output column name.

    Args:
        position_column: Source position column used in the default name.
        net_exposure_column: Explicit output name, or ``None`` for default.

    Returns:
        ``net_exposure_column`` when provided, otherwise
        ``{position_column}{NET_EXPOSURE_SUFFIX}``.
    """
    if net_exposure_column is not None:
        return net_exposure_column
    return f"{position_column}{NET_EXPOSURE_SUFFIX}"


def _generate_gross_exposure_column_name(
    position_column: str,
    gross_exposure_column: str | None,
) -> str:
    """Resolve the gross-exposure output column name.

    Args:
        position_column: Source position column used in the default name.
        gross_exposure_column: Explicit output name, or ``None`` for default.

    Returns:
        ``gross_exposure_column`` when provided, otherwise
        ``{position_column}{GROSS_EXPOSURE_SUFFIX}``.
    """
    if gross_exposure_column is not None:
        return gross_exposure_column
    return f"{position_column}{GROSS_EXPOSURE_SUFFIX}"


def _validate_exposure_inputs(df: pd.DataFrame, position_column: str) -> None:
    """Validate inputs required to compute portfolio exposure.

    Args:
        df: Input DataFrame containing the position column.
        position_column: Discrete portfolio position column.

    Raises:
        KeyError: If ``position_column`` is not present in ``df``.
        TypeError: If ``position_column`` is not numeric.
        ValueError: If ``position_column`` contains invalid position values.
    """
    _validate_position_column(df, position_column)


def _compute_net_exposure(positions: pd.Series) -> pd.Series:
    """Compute net exposure as the signed position.

    Args:
        positions: Discrete portfolio position series.

    Returns:
        Net exposure series aligned to ``positions.index``.
    """
    return positions


def _compute_gross_exposure(positions: pd.Series) -> pd.Series:
    """Compute gross exposure as the absolute position.

    Args:
        positions: Discrete portfolio position series.

    Returns:
        Gross exposure series aligned to ``positions.index``.
    """
    return positions.abs()


def generate_exposure(
    df: pd.DataFrame,
    position_column: str,
    net_exposure_column: str | None = None,
    gross_exposure_column: str | None = None,
) -> pd.DataFrame:
    """Compute net and gross exposure from a portfolio position column.

    Net exposure equals the signed position. Gross exposure equals the
    absolute value of the position.

    Args:
        df: Input DataFrame containing ``position_column``.
        position_column: Discrete portfolio position column.
        net_exposure_column: Optional net-exposure column name. Defaults to
            ``{position_column}_net_exposure``.
        gross_exposure_column: Optional gross-exposure column name. Defaults
            to ``{position_column}_gross_exposure``.

    Returns:
        A new DataFrame with net-exposure and gross-exposure columns
        appended. The input is unchanged.

    Raises:
        KeyError: If ``position_column`` is missing.
        TypeError: If ``position_column`` is not numeric.
        ValueError: If ``position_column`` contains invalid position values.
    """
    _validate_exposure_inputs(df, position_column)

    net_name = _generate_net_exposure_column_name(
        position_column,
        net_exposure_column,
    )
    gross_name = _generate_gross_exposure_column_name(
        position_column,
        gross_exposure_column,
    )

    result = df.copy()
    positions = result[position_column]
    result[net_name] = _compute_net_exposure(positions)
    result[gross_name] = _compute_gross_exposure(positions)
    return result
