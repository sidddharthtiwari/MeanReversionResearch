"""Persistence layer for basket-level research visualizations.

Saves equity and drawdown figures produced by the existing visualization
package. This module does not compute equity, drawdowns, or analytics, and
does not modify input series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.performance.constants import (
    DEFAULT_DRAWDOWN_COLUMN,
    DEFAULT_EQUITY_COLUMN,
)
from src.visualization.drawdown import plot_drawdown_curve
from src.visualization.equity import plot_equity_curve

__all__ = [
    "save_research_visualizations",
]

_EQUITY_CURVE_FILENAME = "equity_curve.png"
_DRAWDOWN_CURVE_FILENAME = "drawdown_curve.png"


def _save_figure(figure: Figure, path: Path) -> None:
    """Persist a matplotlib figure and release its resources.

    Args:
        figure: Figure to write.
        path: Destination image path.
    """
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_research_visualizations(
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    output_directory: Path | str,
) -> None:
    """Persist research equity and drawdown figures to ``output_directory``.

    Creates the directory if needed, then writes ``equity_curve.png`` and
    ``drawdown_curve.png`` using the existing visualization public APIs.
    Input series are not modified.

    Args:
        equity_curve: Cumulative equity series to plot.
        drawdown_curve: Drawdown series to plot.
        output_directory: Destination directory as a string or ``Path``.

    Raises:
        TypeError: If ``equity_curve`` or ``drawdown_curve`` is not a Series,
            or if ``output_directory`` is neither a string nor a ``Path``.
    """
    if not isinstance(equity_curve, pd.Series):
        raise TypeError(
            "equity_curve must be a pandas Series, "
            f"got {type(equity_curve).__name__}."
        )
    if not isinstance(drawdown_curve, pd.Series):
        raise TypeError(
            "drawdown_curve must be a pandas Series, "
            f"got {type(drawdown_curve).__name__}."
        )
    if not isinstance(output_directory, (str, Path)):
        raise TypeError(
            "output_directory must be a string or pathlib.Path, "
            f"got {type(output_directory).__name__}."
        )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    equity_figure = plot_equity_curve(
        equity_curve.to_frame(name=DEFAULT_EQUITY_COLUMN),
    )
    _save_figure(equity_figure, output_path / _EQUITY_CURVE_FILENAME)

    drawdown_figure = plot_drawdown_curve(
        drawdown_curve.to_frame(name=DEFAULT_DRAWDOWN_COLUMN),
    )
    _save_figure(drawdown_figure, output_path / _DRAWDOWN_CURVE_FILENAME)
