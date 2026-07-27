"""PipelineConfig preset examples for the MeanReversionResearch framework.

Demonstrates common ``PipelineConfig`` presets for different research
objectives. Copy any of these configurations into the other example scripts
to customise lookback, entry thresholds, and trading costs.

This module does not load datasets, run research, or generate analytics.
"""

from __future__ import annotations

from examples.utils import (
    print_footer,
    print_header,
    print_key_value,
    print_section,
)
from src.pipeline.config import PipelineConfig


def print_config(config: PipelineConfig) -> None:
    """Print the primary fields of a ``PipelineConfig``.

    Args:
        config: Pipeline configuration to display.
    """
    print_key_value("Lookback", config.lookback)
    print_key_value("Entry Z-Score", config.entry_zscore)
    print_key_value("Transaction Cost", config.transaction_cost)
    print_key_value("Slippage", config.slippage)
    print()


def main() -> None:
    """Display common PipelineConfig presets."""
    print_header(
        "Mean Reversion Research Framework - Configuration Examples"
    )

    print_section("Default Configuration")
    default_config = PipelineConfig(
        lookback=20,
        entry_zscore=2.0,
        transaction_cost=0.001,
        slippage=0.0005,
    )
    print_config(default_config)

    print_section("Conservative Configuration")
    conservative_config = PipelineConfig(
        lookback=40,
        entry_zscore=2.5,
        transaction_cost=0.001,
        slippage=0.0005,
    )
    print_config(conservative_config)

    print_section("Aggressive Configuration")
    aggressive_config = PipelineConfig(
        lookback=10,
        entry_zscore=1.5,
        transaction_cost=0.001,
        slippage=0.0005,
    )
    print_config(aggressive_config)

    print_section("High Trading Cost Configuration")
    high_cost_config = PipelineConfig(
        lookback=20,
        entry_zscore=2.0,
        transaction_cost=0.003,
        slippage=0.0015,
    )
    print_config(high_cost_config)

    print_footer("Configuration examples completed successfully.")


if __name__ == "__main__":
    main()
