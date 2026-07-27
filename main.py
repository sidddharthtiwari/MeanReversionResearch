"""Official entry point for the MeanReversionResearch framework.

Orchestrates basket-level research across every available sector by
delegating to existing public APIs. This module contains no business logic
or mathematical calculations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analytics.summary import generate_summary
from src.data.loader import (
    available_sectors,
    load_sector,
    load_sector_metadata,
)
from src.performance.constants import (
    DEFAULT_DRAWDOWN_COLUMN,
    DEFAULT_EQUITY_COLUMN,
)
from src.performance.drawdown import compute_drawdown_series
from src.performance.equity import compute_equity_curve
from src.pipeline.config import PipelineConfig
from src.research.output import save_research_outputs
from src.research.pipeline import run_research
from src.research.summary import create_summary_row
from src.research.visualization import save_research_visualizations

logger = logging.getLogger(__name__)

PORTFOLIO_RETURN_COLUMN = "portfolio_return"
BASKET_COMPARISON_FILENAME = "basket_comparison.csv"


def _process_basket(
    basket_name: str,
    config: PipelineConfig,
) -> dict[str, object]:
    """Run the complete research workflow for one basket.

    Args:
        basket_name: Sector identifier discovered from OHLC parquet files.
        config: Shared immutable pipeline configuration.

    Returns:
        One summary-row dictionary for basket comparison.
    """
    sector_data = load_sector(basket_name)
    metadata = load_sector_metadata(basket_name)
    result = run_research(sector_data, metadata, config)

    portfolio_returns = result.portfolio_returns.to_frame(
        name=PORTFOLIO_RETURN_COLUMN,
    )
    analytics = generate_summary(
        portfolio_returns,
        return_column=PORTFOLIO_RETURN_COLUMN,
        frequency=config.rebalance_frequency,
    )

    equity_frame = compute_equity_curve(
        portfolio_returns,
        return_column=PORTFOLIO_RETURN_COLUMN,
    )
    drawdown_frame = compute_drawdown_series(
        equity_frame,
        equity_column=DEFAULT_EQUITY_COLUMN,
    )

    basket_output_directory = Path(config.output_directory) / basket_name
    save_research_outputs(result, analytics, basket_output_directory)
    save_research_visualizations(
        equity_frame[DEFAULT_EQUITY_COLUMN],
        drawdown_frame[DEFAULT_DRAWDOWN_COLUMN],
        basket_output_directory,
    )

    return create_summary_row(basket_name, result, analytics)


def main() -> None:
    """Execute basket research for every available sector.

    Discovers sector parquet files, orchestrates the public research workflow
    for each basket, and writes a cross-basket comparison table.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig()
    output_root = Path(config.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)

    baskets = available_sectors()
    logger.info("Discovered %d baskets.", len(baskets))

    summary_rows: list[dict[str, object]] = []
    for basket_name in baskets:
        logger.info("Starting basket '%s'.", basket_name)
        try:
            summary_row = _process_basket(basket_name, config)
        except Exception:
            logger.exception("Skipped basket '%s'.", basket_name)
            continue

        summary_rows.append(summary_row)
        logger.info("Completed basket '%s'.", basket_name)

    if summary_rows:
        comparison = pd.DataFrame(summary_rows)
        comparison_path = output_root / BASKET_COMPARISON_FILENAME
        comparison.to_csv(comparison_path, index=False)
        logger.info(
            "Final summary: completed %d of %d baskets. "
            "Comparison written to %s.",
            len(summary_rows),
            len(baskets),
            comparison_path,
        )
    else:
        logger.warning("No baskets completed successfully.")


if __name__ == "__main__":
    main()
