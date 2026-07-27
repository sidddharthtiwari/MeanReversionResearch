"""Drawdown-curve visualization for quantitative performance analysis.

Renders drawdown time-series as matplotlib figures. This module does not
compute drawdowns, equity, or analytics metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.analytics.validation import (
    _validate_columns_exist,
    _validate_numeric_series,
)
from src.performance.constants import DEFAULT_DRAWDOWN_COLUMN

__all__ = [
    "plot_drawdown_curve",
]


def plot_drawdown_curve(
    df: pd.DataFrame,
    drawdown_column: str = DEFAULT_DRAWDOWN_COLUMN,
) -> Figure:
    """Plot a drawdown curve from a drawdown-series column.

    Args:
        df: Input DataFrame containing ``drawdown_column``.
        drawdown_column: Numeric drawdown column. Defaults to
            ``DEFAULT_DRAWDOWN_COLUMN``.

    Returns:
        A new matplotlib ``Figure`` containing the drawdown-curve plot. The
        input DataFrame is not modified.

    Raises:
        TypeError: If ``df`` is not a DataFrame, or ``drawdown_column`` is not
            numeric.
        KeyError: If ``drawdown_column`` is missing.
        ValueError: If ``drawdown_column`` is empty.
    """
    _validate_columns_exist(df, [drawdown_column])
    _validate_numeric_series(df[drawdown_column], drawdown_column)

    figure, axes = plt.subplots()
    axes.plot(df.index, df[drawdown_column])
    axes.set_title("Drawdown Curve")
    axes.set_xlabel("Index")
    axes.set_ylabel("Drawdown")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure
