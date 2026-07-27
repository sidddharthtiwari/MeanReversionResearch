"""Persistence layer for research pipeline outputs.

Writes a completed ``PipelineResult`` to disk using deterministic filenames.
This module does not execute the pipeline, calculate metrics, or transform
business data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.pipeline.result import PipelineResult

__all__ = [
    "save_pipeline_outputs",
]

_SIGNALS_FILENAME = "signals.csv"
_PORTFOLIO_FILENAME = "portfolio.csv"
_BACKTEST_FILENAME = "backtest.csv"
_ANALYTICS_FILENAME = "analytics.json"


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to CSV without modifying the input.

    Args:
        frame: DataFrame to write.
        path: Destination CSV path.
    """
    frame.to_csv(path, index=False)


def _save_analytics(
    analytics: Mapping[str, float | int],
    path: Path,
) -> None:
    """Persist analytics metrics to a deterministic JSON file.

    Args:
        analytics: Mapping of metric names to scalar values.
        path: Destination JSON path.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            dict(analytics),
            file,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
        )


def save_pipeline_outputs(
    result: PipelineResult,
    output_directory: Path | str,
) -> None:
    """Persist pipeline outputs to ``output_directory``.

    Creates the directory if needed, then writes ``signals.csv``,
    ``portfolio.csv``, ``backtest.csv``, and ``analytics.json``. Existing
    files with those names are overwritten.

    Args:
        result: Immutable pipeline outputs to persist.
        output_directory: Destination directory as a string or ``Path``.

    Raises:
        TypeError: If ``result`` is not a ``PipelineResult``, or if
            ``output_directory`` is neither a string nor a ``Path``.
    """
    if not isinstance(result, PipelineResult):
        raise TypeError(
            f"result must be a PipelineResult, got {type(result).__name__}."
        )
    if not isinstance(output_directory, (str, Path)):
        raise TypeError(
            "output_directory must be a string or pathlib.Path, "
            f"got {type(output_directory).__name__}."
        )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_dataframe(result.signals, output_path / _SIGNALS_FILENAME)
    _save_dataframe(result.portfolio, output_path / _PORTFOLIO_FILENAME)
    _save_dataframe(result.backtest, output_path / _BACKTEST_FILENAME)
    _save_analytics(result.analytics, output_path / _ANALYTICS_FILENAME)
