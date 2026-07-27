"""Basket research usage example for the MeanReversionResearch framework.

Demonstrates end-to-end portfolio research with the public research API.

Datasets are located under the top-level ``data/`` directory:

    data/sectors/bank_sector.csv
    data/metadata/bank_metadata.csv

Replace those files, or update the path constants below, to run the example
on your own sector basket.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from examples.utils import (
    load_csv,
    print_footer,
    print_header,
    print_key_value,
    print_section,
)
from src.analytics.summary import generate_summary
from src.pipeline.config import PipelineConfig
from src.research.pipeline import run_research

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

SECTOR = "Bank"
SECTOR_DATA = Path("data") / "sectors" / "bank_sector.csv"
METADATA = Path("data") / "metadata" / "bank_metadata.csv"
LOOKBACK = 20
ENTRY_ZSCORE = 2.0
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005


def main() -> None:
    """Run a basket research example and print the portfolio summary."""
    print_header(
        "Mean Reversion Research Framework - Basket Research Example"
    )
    print_key_value("Sector", SECTOR)
    print_key_value("Sector Dataset", SECTOR_DATA)
    print_key_value("Metadata Dataset", METADATA)
    print()

    print_section("Pipeline Configuration")
    print_key_value("Lookback", LOOKBACK)
    print_key_value("Entry Z-Score", ENTRY_ZSCORE)
    print_key_value("Transaction Cost", TRANSACTION_COST)
    print_key_value("Slippage", SLIPPAGE)
    print()

    sector_data = load_csv(SECTOR_DATA)
    metadata = load_csv(METADATA)
    config = PipelineConfig(
        lookback=LOOKBACK,
        entry_zscore=ENTRY_ZSCORE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
    )
    result = run_research(sector_data, metadata, config)

    portfolio_returns = result.portfolio_returns.to_frame(
        name="portfolio_return"
    )
    analytics = generate_summary(
        portfolio_returns,
        return_column="portfolio_return",
    )

    print_section("Research Summary")
    print_key_value("Processed Symbols", len(result.processed_symbols))
    print_key_value("Skipped Symbols", len(result.skipped_symbols))
    print()

    print_section("Portfolio Weights")
    for symbol, weight in result.weights.items():
        print_key_value(symbol, round(weight, 4))
    print()

    print_section("Portfolio Analytics")
    for metric_name, metric_value in analytics.items():
        print_key_value(metric_name, metric_value)

    print_footer("Basket research completed successfully.")


if __name__ == "__main__":
    main()
