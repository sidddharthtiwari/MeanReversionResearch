"""Immutable result object for basket-level research outputs.

Stores the aggregated portfolio returns and supporting research artefacts
produced by a complete basket run. This module contains result structure only
and does not execute aggregation, analytics, or persistence logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "ResearchResult",
]


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Immutable outputs from a complete basket-level research run.

    Attributes:
        portfolio_returns: Aggregated portfolio return series used by
            performance, analytics, and visualization modules.
        aligned_returns: Date-aligned DataFrame of per-symbol strategy
            returns, with dates as the index and symbols as columns.
        weights: Portfolio weights used during aggregation, indexed by
            processed symbol.
        processed_symbols: Symbols successfully processed by the basket
            runner, in aggregation order.
        skipped_symbols: Symbols skipped due to missing data or pipeline
            failures.
    """

    portfolio_returns: pd.Series
    aligned_returns: pd.DataFrame
    weights: pd.Series
    processed_symbols: tuple[str, ...]
    skipped_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate result field types after initialisation.

        Raises:
            TypeError: If any field has an unexpected type.
        """
        if not isinstance(self.portfolio_returns, pd.Series):
            raise TypeError(
                "portfolio_returns must be a pandas Series, "
                f"got {type(self.portfolio_returns).__name__}."
            )
        if not isinstance(self.aligned_returns, pd.DataFrame):
            raise TypeError(
                "aligned_returns must be a pandas DataFrame, "
                f"got {type(self.aligned_returns).__name__}."
            )
        if not isinstance(self.weights, pd.Series):
            raise TypeError(
                "weights must be a pandas Series, "
                f"got {type(self.weights).__name__}."
            )
        if not isinstance(self.processed_symbols, tuple):
            raise TypeError(
                "processed_symbols must be a tuple, "
                f"got {type(self.processed_symbols).__name__}."
            )
        if not isinstance(self.skipped_symbols, tuple):
            raise TypeError(
                "skipped_symbols must be a tuple, "
                f"got {type(self.skipped_symbols).__name__}."
            )
