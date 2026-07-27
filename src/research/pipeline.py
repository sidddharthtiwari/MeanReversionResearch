"""Public orchestration entry point for basket-level research.

Coordinates the basket runner and aggregation modules into a single research
workflow. This module performs orchestration only and does not compute
analytics, weights, plots, or persistence.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.pipeline.config import PipelineConfig
from src.research.aggregation import aggregate
from src.research.result import ResearchResult
from src.research.runner import run_basket
from src.research.weighting import compute_equal_weights

__all__ = [
    "run_research",
]


def run_research(
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
    weighting_function: Callable[
        [pd.DataFrame],
        pd.Series,
    ] = compute_equal_weights,
) -> ResearchResult:
    """Execute complete basket-level research for a sector.

    Runs the basket pipeline for every metadata symbol, aggregates the
    resulting ``PipelineResult`` objects, and returns an immutable
    ``ResearchResult``.

    Args:
        sector_data: Sector OHLC DataFrame containing a ``symbol`` column.
        metadata: Basket metadata DataFrame containing a ``Symbol`` column.
        config: Immutable pipeline configuration shared across symbols.
        weighting_function: Callable that accepts the aligned return matrix
            and returns a weight series indexed by symbol. Defaults to
            ``compute_equal_weights``.

    Returns:
        Immutable basket-level research result.
    """
    results = run_basket(sector_data, metadata, config)
    return aggregate(
        results=results,
        weighting_function=weighting_function,
    )
