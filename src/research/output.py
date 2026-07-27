"""Persistence layer for basket-level research outputs.

Writes a completed ``ResearchResult`` and analytics mapping to disk using
deterministic filenames. This module does not execute research, calculate
metrics, generate plots, or modify ``ResearchResult``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.research.result import ResearchResult

__all__ = [
    "save_research_outputs",
]

_PORTFOLIO_RETURNS_FILENAME = "portfolio_returns.csv"
_ALIGNED_RETURNS_FILENAME = "aligned_returns.csv"
_WEIGHTS_FILENAME = "weights.csv"
_ANALYTICS_FILENAME = "analytics.json"


def _save_dataframe(data: pd.DataFrame | pd.Series, path: Path) -> None:
    """Persist a DataFrame or Series to CSV without modifying the input.

    The index is preserved because research outputs are date-indexed or
    symbol-indexed.

    Args:
        data: DataFrame or Series to write.
        path: Destination CSV path.
    """
    data.to_csv(path, index=True)


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


def save_research_outputs(
    result: ResearchResult,
    analytics: Mapping[str, float | int],
    output_directory: Path | str,
) -> None:
    """Persist basket research outputs to ``output_directory``.

    Creates the directory if needed, then writes ``portfolio_returns.csv``,
    ``aligned_returns.csv``, ``weights.csv``, and ``analytics.json``.
    Existing files with those names are overwritten. CSV indexes are
    preserved.

    Args:
        result: Immutable basket-level research outputs to persist.
        analytics: Mapping of scalar analytics summary metrics.
        output_directory: Destination directory as a string or ``Path``.

    Raises:
        TypeError: If ``result`` is not a ``ResearchResult``, if
            ``analytics`` is not a ``Mapping``, or if ``output_directory``
            is neither a string nor a ``Path``.
    """
    if not isinstance(result, ResearchResult):
        raise TypeError(
            f"result must be a ResearchResult, got {type(result).__name__}."
        )
    if not isinstance(analytics, Mapping):
        raise TypeError(
            f"analytics must be a Mapping, got {type(analytics).__name__}."
        )
    if not isinstance(output_directory, (str, Path)):
        raise TypeError(
            "output_directory must be a string or pathlib.Path, "
            f"got {type(output_directory).__name__}."
        )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_dataframe(
        result.portfolio_returns.rename("portfolio_return"),
        output_path / _PORTFOLIO_RETURNS_FILENAME,
    )
    _save_dataframe(
        result.aligned_returns,
        output_path / _ALIGNED_RETURNS_FILENAME,
    )
    _save_dataframe(
        result.weights.rename("weight"),
        output_path / _WEIGHTS_FILENAME,
    )
    _save_analytics(analytics, output_path / _ANALYTICS_FILENAME)
