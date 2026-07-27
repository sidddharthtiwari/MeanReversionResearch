"""Basket-level return aggregation for the MeanReversionResearch framework.

Converts multiple ``PipelineResult`` objects into a single immutable
``ResearchResult`` by aligning strategy returns and applying an injected
weighting function.

Weighting algorithms are intentionally supplied by the caller so this module
can stay focused on orchestration. New weighting methodologies can be added
elsewhere without changing aggregation itself.

This module does not compute analytics, plot, persist outputs, or implement
weighting methodologies.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.pipeline.result import PipelineResult
from src.portfolio.constants import STRATEGY_RETURN_SUFFIX
from src.research.result import ResearchResult
from src.research.weighting import compute_equal_weights

__all__ = [
    "aggregate",
]

_DATE_COLUMN = "date"


def _validate_results(results: dict[str, PipelineResult]) -> None:
    """Validate basket aggregation inputs before any transformation.

    Responsibilities:
        Confirm ``results`` is a non-empty mapping whose values are
        ``PipelineResult`` instances.

    Immutability:
        Performs no mutation; inputs are inspected only.

    Args:
        results: Mapping of symbol to ``PipelineResult``.

    Raises:
        TypeError: If ``results`` is not a dictionary, or if any value is not
            a ``PipelineResult``.
        ValueError: If ``results`` is empty.
    """
    if not isinstance(results, dict):
        raise TypeError(
            f"results must be a dict, got {type(results).__name__}."
        )
    if not results:
        raise ValueError("results must not be empty.")
    for symbol, result in results.items():
        if not isinstance(result, PipelineResult):
            raise TypeError(
                f"results['{symbol}'] must be a PipelineResult, "
                f"got {type(result).__name__}."
            )


def _extract_returns(
    results: dict[str, PipelineResult],
) -> dict[str, pd.Series]:
    """Extract strategy-return series from each ``PipelineResult``.

    Responsibilities:
        Locate each symbol's strategy-return column and expose it as a
        date-indexed series without cross-symbol alignment.

    Immutability:
        Reads portfolio frames only; source ``PipelineResult`` objects are
        unchanged.

    Args:
        results: Mapping of symbol to ``PipelineResult``.

    Returns:
        New mapping of symbol to strategy-return series indexed by date.
        Series values are not transformed.

    Raises:
        KeyError: If a portfolio frame is missing the date column or does not
            contain exactly one strategy-return column.
    """
    extracted: dict[str, pd.Series] = {}
    for symbol, result in results.items():
        portfolio = result.portfolio
        if _DATE_COLUMN not in portfolio.columns:
            raise KeyError(
                f"Missing required portfolio column '{_DATE_COLUMN}' "
                f"for symbol '{symbol}'."
            )
        strategy_columns = [
            column
            for column in portfolio.columns
            if column.endswith(STRATEGY_RETURN_SUFFIX)
        ]
        if len(strategy_columns) != 1:
            raise KeyError(
                "Expected exactly one strategy-return column for symbol "
                f"'{symbol}', found {len(strategy_columns)}."
            )
        extracted[symbol] = pd.Series(
            portfolio[strategy_columns[0]].to_numpy(),
            index=pd.Index(portfolio[_DATE_COLUMN], name=_DATE_COLUMN),
            name=symbol,
        )
    return extracted


def _align_returns(return_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Align symbol return series onto a shared date index.

    Responsibilities:
        Build a symbol-column return matrix and retain only dates present for
        every symbol so portfolio aggregation never mixes incomplete baskets.

    Immutability:
        Constructs a new DataFrame; the input mapping is unchanged.

    Args:
        return_series: Mapping of symbol to date-indexed return series.

    Returns:
        New date-aligned DataFrame with symbols as columns, missing-date rows
        removed (inner alignment), and a chronologically sorted index.
    """
    # Inner alignment keeps only shared trading dates so every period's
    # portfolio return reflects the same complete symbol set.
    aligned_returns = pd.DataFrame(return_series)
    aligned_returns = aligned_returns.dropna(how="any")
    return aligned_returns.sort_index()


def _aggregate_returns(
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Aggregate aligned symbol returns into a portfolio return series.

    Responsibilities:
        Apply caller-supplied weights to the aligned return matrix and return
        the resulting portfolio return series.

    Immutability:
        Does not mutate ``aligned_returns`` or ``weights``.

    Args:
        aligned_returns: Date-aligned return matrix with symbols as columns.
        weights: Portfolio weights indexed by symbol.

    Returns:
        New weighted portfolio return series aligned to
        ``aligned_returns.index`` with dtype ``float64``.

    Raises:
        ValueError: If ``weights`` is missing any symbol present in
            ``aligned_returns``.
    """
    # Reindex onto matrix columns so multiplication is column-aligned even
    # when the weighting function returns a differently ordered index.
    ordered_weights = weights.reindex(aligned_returns.columns)
    missing = ordered_weights[ordered_weights.isna()].index.tolist()
    if missing:
        # Fail fast: silent NaNs would corrupt portfolio returns without a
        # clear signal that the weighting function was incomplete.
        raise ValueError(
            "Weighting function did not return weights for all symbols. "
            f"Missing weights for: {missing}."
        )

    portfolio_returns = aligned_returns.mul(ordered_weights, axis=1).sum(axis=1)
    return portfolio_returns.astype("float64")


def _build_result(
    portfolio_returns: pd.Series,
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
    processed_symbols: tuple[str, ...],
    skipped_symbols: tuple[str, ...],
) -> ResearchResult:
    """Construct an immutable ``ResearchResult``.

    Responsibilities:
        Assemble validated aggregation outputs into the research-layer result
        contract without additional computation.

    Immutability:
        Does not mutate inputs; returns a frozen ``ResearchResult``.

    Args:
        portfolio_returns: Aggregated portfolio return series.
        aligned_returns: Date-aligned per-symbol return matrix.
        weights: Portfolio weights indexed by symbol.
        processed_symbols: Successfully aggregated symbols in order.
        skipped_symbols: Symbols excluded from aggregation.

    Returns:
        Immutable basket-level research result.
    """
    return ResearchResult(
        portfolio_returns=portfolio_returns,
        aligned_returns=aligned_returns,
        weights=weights,
        processed_symbols=processed_symbols,
        skipped_symbols=skipped_symbols,
    )


def aggregate(
    results: dict[str, PipelineResult],
    weighting_function: Callable[
        [pd.DataFrame],
        pd.Series,
    ] = compute_equal_weights,
) -> ResearchResult:
    """Aggregate multiple ``PipelineResult`` objects into a ``ResearchResult``.

    Extracts strategy returns, aligns them by date, applies
    ``weighting_function``, and builds an immutable research result. No
    analytics or persistence are performed.

    Args:
        results: Mapping of symbol to ``PipelineResult`` from basket execution.
        weighting_function: Callable that accepts the aligned return matrix and
            returns a weight series indexed by symbol. Defaults to
            ``compute_equal_weights``.

    Returns:
        Immutable ``ResearchResult`` containing portfolio returns, aligned
        symbol returns, weights, and processed-symbol metadata.

    Raises:
        TypeError: If ``results`` is not a dictionary, or if any value is not
            a ``PipelineResult``.
        ValueError: If ``results`` is empty, or if ``weighting_function``
            omits weights for one or more symbols.
        KeyError: If a portfolio frame is missing the date column or does not
            contain exactly one strategy-return column.
    """
    _validate_results(results)

    return_series = _extract_returns(results)
    aligned_returns = _align_returns(return_series)
    weights = weighting_function(aligned_returns)
    portfolio_returns = _aggregate_returns(aligned_returns, weights)
    processed_symbols = tuple(aligned_returns.columns)

    return _build_result(
        portfolio_returns=portfolio_returns,
        aligned_returns=aligned_returns,
        weights=weights,
        processed_symbols=processed_symbols,
        skipped_symbols=(),
    )
