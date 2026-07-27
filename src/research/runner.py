"""Basket-level research orchestration for the MeanReversionResearch framework.

Runs the existing single-asset research pipeline once per symbol in a sector
basket. This module does not construct portfolios, compute analytics, plot,
persist outputs, or duplicate pipeline logic.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.pipeline.config import PipelineConfig
from src.pipeline.result import PipelineResult
from src.pipeline.runner import run_pipeline

__all__ = [
    "run_basket",
]

logger = logging.getLogger(__name__)

_METADATA_SYMBOL_COLUMN = "Symbol"
_OHLC_SYMBOL_COLUMN = "symbol"
_DATE_COLUMN = "date"


def _validate_inputs(
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Validate basket-runner inputs.

    Args:
        sector_data: Sector OHLC DataFrame.
        metadata: Basket metadata DataFrame.
        config: Immutable pipeline configuration.

    Raises:
        TypeError: If ``sector_data`` or ``metadata`` is not a DataFrame, or
            if ``config`` is not a ``PipelineConfig``.
        ValueError: If ``sector_data`` is empty.
        KeyError: If ``metadata`` is missing ``Symbol``, or if
            ``sector_data`` is missing ``symbol``.
    """
    if not isinstance(sector_data, pd.DataFrame):
        raise TypeError(
            "sector_data must be a pandas DataFrame, "
            f"got {type(sector_data).__name__}."
        )
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "metadata must be a pandas DataFrame, "
            f"got {type(metadata).__name__}."
        )
    if not isinstance(config, PipelineConfig):
        raise TypeError(
            f"config must be a PipelineConfig, got {type(config).__name__}."
        )
    if sector_data.empty:
        raise ValueError("sector_data must not be empty.")
    if _OHLC_SYMBOL_COLUMN not in sector_data.columns:
        raise KeyError(
            f"Missing required sector data column: {_OHLC_SYMBOL_COLUMN}."
        )
    if _METADATA_SYMBOL_COLUMN not in metadata.columns:
        raise KeyError(
            f"Missing required metadata column: {_METADATA_SYMBOL_COLUMN}."
        )


def _prepare_stock_dataframe(
    sector_data: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Prepare a single-symbol OHLC DataFrame for pipeline execution.

    Filters ``sector_data`` to ``symbol``, sorts chronologically when a date
    column is present, and returns a new DataFrame. The input is unchanged.

    Args:
        sector_data: Sector OHLC DataFrame.
        symbol: Symbol identifier to extract.

    Returns:
        Filtered single-symbol DataFrame, possibly empty when ``symbol`` is
        absent from ``sector_data``.
    """
    stock_data = sector_data.loc[
        sector_data[_OHLC_SYMBOL_COLUMN] == symbol
    ].copy()
    if stock_data.empty:
        return stock_data
    if _DATE_COLUMN in stock_data.columns:
        stock_data = stock_data.sort_values(_DATE_COLUMN)
    return stock_data


def run_basket(
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, PipelineResult]:
    """Run the research pipeline for every symbol in a sector basket.

    Extracts unique symbols from basket metadata, filters the sector OHLC
    frame for each symbol, and delegates execution to ``run_pipeline``.
    Missing symbols, empty datasets, and per-symbol pipeline failures are
    skipped so remaining symbols continue processing.

    Args:
        sector_data: Sector OHLC DataFrame containing a ``symbol`` column.
        metadata: Basket metadata DataFrame containing a ``Symbol`` column.
        config: Immutable pipeline configuration shared across symbols.

    Returns:
        Mapping of successfully processed symbol to ``PipelineResult``.

    Raises:
        TypeError: If ``sector_data`` or ``metadata`` is not a DataFrame, or
            if ``config`` is not a ``PipelineConfig``.
        ValueError: If ``sector_data`` is empty.
        KeyError: If ``metadata`` is missing ``Symbol``, or if
            ``sector_data`` is missing ``symbol``.
    """
    _validate_inputs(sector_data, metadata, config)

    symbols = (
        metadata[_METADATA_SYMBOL_COLUMN]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    results: dict[str, PipelineResult] = {}
    for symbol in symbols:
        stock_data = _prepare_stock_dataframe(sector_data, symbol)
        if stock_data.empty:
            logger.warning(
                "Skipping symbol '%s': no matching OHLC rows.",
                symbol,
            )
            continue

        try:
            results[symbol] = run_pipeline(stock_data, config)
        except Exception as error:
            logger.warning(
                "Skipping symbol '%s' after pipeline failure: %s",
                symbol,
                error,
            )
            continue

    return results
