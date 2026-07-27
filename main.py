"""Official entry point for the MeanReversionResearch framework.

Demonstrates the complete end-to-end research workflow by orchestrating
existing public APIs. This module contains no business logic, calculations,
or private helpers.
"""

from __future__ import annotations

from pathlib import Path

from src.backtest.runner import resolve_backtest_return_column
from src.data.loader import load_sector
from src.features.zscore import generate_zscore_column_name
from src.performance.constants import DEFAULT_EQUITY_COLUMN
from src.performance.drawdown import compute_drawdown_series
from src.performance.equity import compute_equity_curve
from src.pipeline.config import PipelineConfig
from src.pipeline.output import save_pipeline_outputs
from src.pipeline.runner import run_pipeline
from src.portfolio.positions import generate_position_column_name
from src.portfolio.returns import generate_strategy_return_column_name
from src.signals.mean_reversion import generate_mean_reversion_signal_column_name
from src.visualization.drawdown import plot_drawdown_curve
from src.visualization.equity import plot_equity_curve

# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

SECTOR = "bank"
SYMBOL = "AUBANK"
OUTPUT_DIRECTORY = "outputs"

LOOKBACK = 20
ENTRY_ZSCORE = 2.0
EXIT_ZSCORE = 0.5
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
REBALANCE_FREQUENCY = "D"

EQUITY_FIGURE_FILENAME = "equity_curve.png"
DRAWDOWN_FIGURE_FILENAME = "drawdown_curve.png"


def main() -> None:
    """Run the complete MeanReversionResearch workflow.

    Orchestrates data loading, pipeline execution, analytics reporting,
    visualisation, and output persistence using public framework APIs only.
    """
    try:
        # ------------------------------------------------
        # Stage 1: Load market data
        # ------------------------------------------------
        print("Loading market data...")
        market_data = load_sector(SECTOR)
        market_data = market_data.loc[market_data["symbol"] == SYMBOL].copy()

        # ------------------------------------------------
        # Stage 2: Execute research pipeline
        # ------------------------------------------------
        print("Running research pipeline...")
        config = PipelineConfig(
            lookback=LOOKBACK,
            entry_zscore=ENTRY_ZSCORE,
            exit_zscore=EXIT_ZSCORE,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            rebalance_frequency=REBALANCE_FREQUENCY,
            output_directory=OUTPUT_DIRECTORY,
        )
        result = run_pipeline(market_data, config)

        # ------------------------------------------------
        # Stage 3: Print analytics summary
        # ------------------------------------------------
        print("Generating analytics summary...")
        print()
        print("Analytics Summary")
        print("-" * 40)
        for metric_name, metric_value in result.analytics.items():
            print(f"{metric_name}: {metric_value}")
        print()

        # ------------------------------------------------
        # Stage 4: Generate visualisations
        # ------------------------------------------------
        print("Generating visualisations...")
        zscore_column = generate_zscore_column_name(window=LOOKBACK)
        signal_column = generate_mean_reversion_signal_column_name(
            feature_column=zscore_column,
        )
        position_column = generate_position_column_name(
            signal_column=signal_column,
        )
        strategy_return_column = generate_strategy_return_column_name(
            position_column=position_column,
        )
        analytics_return_column = resolve_backtest_return_column(
            strategy_return_column=strategy_return_column,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
        )

        equity_frame = compute_equity_curve(
            result.backtest,
            return_column=analytics_return_column,
        )
        drawdown_frame = compute_drawdown_series(
            equity_frame,
            equity_column=DEFAULT_EQUITY_COLUMN,
        )
        equity_figure = plot_equity_curve(equity_frame)
        drawdown_figure = plot_drawdown_curve(drawdown_frame)

        # ------------------------------------------------
        # Stage 5: Save outputs
        # ------------------------------------------------
        print("Saving outputs...")
        output_path = Path(OUTPUT_DIRECTORY)
        save_pipeline_outputs(result, output_path)
        equity_figure.savefig(output_path / EQUITY_FIGURE_FILENAME)
        drawdown_figure.savefig(output_path / DRAWDOWN_FIGURE_FILENAME)

        # ------------------------------------------------
        # Stage 6: Completion
        # ------------------------------------------------
        print("Research completed successfully.")
    except Exception as error:
        print(f"Research pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()
