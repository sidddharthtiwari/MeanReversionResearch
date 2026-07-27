"""Single-asset usage example for the MeanReversionResearch framework.

Demonstrates how to analyse one stock with the public pipeline API.

Dataset location:

    data/single_asset/{SYMBOL}.csv

Replace ``SYMBOL`` or point ``CSV_PATH`` at your own OHLC CSV to run the
example on a different instrument.
"""

from __future__ import annotations

from pathlib import Path

from examples.utils import (
    load_csv,
    print_footer,
    print_header,
    print_key_value,
    print_section,
)
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import run_pipeline

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

SYMBOL = "AUBANK"
CSV_PATH = Path("data") / "single_asset" / f"{SYMBOL}.csv"
LOOKBACK = 20
ENTRY_ZSCORE = 2.0
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005


def main() -> None:
    """Run a single-asset research example and print the analytics summary."""
    print_header(
        "Mean Reversion Research Framework - Single Asset Example"
    )
    print_key_value("Symbol", SYMBOL)
    print_key_value("Dataset", CSV_PATH)
    print()

    print_section("Pipeline Configuration")
    print_key_value("Lookback", LOOKBACK)
    print_key_value("Entry Z-Score", ENTRY_ZSCORE)
    print_key_value("Transaction Cost", TRANSACTION_COST)
    print_key_value("Slippage", SLIPPAGE)
    print()

    data = load_csv(CSV_PATH)
    config = PipelineConfig(
        lookback=LOOKBACK,
        entry_zscore=ENTRY_ZSCORE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
    )
    result = run_pipeline(data, config)

    print_section("Analytics Summary")
    for metric_name, metric_value in result.analytics.items():
        print_key_value(metric_name, metric_value)

    print_footer("Single-asset research completed successfully.")


if __name__ == "__main__":
    main()
