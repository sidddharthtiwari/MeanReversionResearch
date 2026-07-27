"""Equity-curve visualization for quantitative performance analysis.

Renders equity time-series as matplotlib figures. This module does not
compute equity, drawdowns, or analytics metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.analytics.validation import (
    _validate_columns_exist,
    _validate_numeric_series,
)
from src.performance.constants import DEFAULT_EQUITY_COLUMN

__all__ = [
    "plot_equity_curve",
]


def plot_equity_curve(
    df: pd.DataFrame,
    equity_column: str = DEFAULT_EQUITY_COLUMN,
) -> Figure:
    """Plot an equity curve from a cumulative-equity column.

    Args:
        df: Input DataFrame containing ``equity_column``.
        equity_column: Numeric cumulative-equity column. Defaults to
            ``DEFAULT_EQUITY_COLUMN``.

    Returns:
        A new matplotlib ``Figure`` containing the equity-curve plot. The
        input DataFrame is not modified.

    Raises:
        TypeError: If ``df`` is not a DataFrame, or ``equity_column`` is not
            numeric.
        KeyError: If ``equity_column`` is missing.
        ValueError: If ``equity_column`` is empty.
    """
    _validate_columns_exist(df, [equity_column])
    _validate_numeric_series(df[equity_column], equity_column)

    figure, axes = plt.subplots()
    axes.plot(df.index, df[equity_column])
    axes.set_title("Equity Curve")
    axes.set_xlabel("Index")
    axes.set_ylabel("Equity")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure
