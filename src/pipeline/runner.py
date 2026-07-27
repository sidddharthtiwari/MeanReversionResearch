"""Research pipeline orchestration for quantitative workflows.

This module is the conductor of the research framework. It coordinates
existing public APIs across features, signals, portfolio, backtest, and
analytics packages into one deterministic workflow.

Architectural philosophy:
    The runner intentionally performs no business logic and no calculations.
    Every transformation is delegated to the specialised module that owns it.
    The runner only validates top-level inputs, calls public APIs in order,
    and packages the stage outputs into ``PipelineResult``.

This file is intentionally small so the research workflow remains easy to
read and hard to misuse.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.summary import generate_summary
from src.backtest.runner import (
    resolve_backtest_return_column,
    run_backtest,
)
from src.features.returns import DEFAULT_SIMPLE_RETURN, compute_simple_returns
from src.features.zscore import compute_zscore, generate_zscore_column_name
from src.pipeline.config import PipelineConfig
from src.pipeline.result import PipelineResult
from src.portfolio.positions import (
    generate_position_column_name,
    generate_positions,
)
from src.portfolio.returns import (
    generate_strategy_return_column_name,
    generate_strategy_returns,
)
from src.signals.mean_reversion import (
    generate_mean_reversion_signal,
    generate_mean_reversion_signal_column_name,
)

__all__ = [
    "run_pipeline",
]


def run_pipeline(
    data: pd.DataFrame,
    config: PipelineConfig,
) -> PipelineResult:
    """Execute a complete mean-reversion research pipeline.

    Orchestrates existing public APIs from features, signals, portfolio,
    backtest, and analytics packages. Returns an immutable
    ``PipelineResult`` without persisting outputs.

    Args:
        data: Input OHLC research DataFrame.
        config: Immutable pipeline configuration.

    Returns:
        Immutable pipeline outputs containing signal, portfolio, backtest,
        and analytics summary stages.

    Raises:
        TypeError: If ``data`` is not a DataFrame or ``config`` is not a
            ``PipelineConfig``.
    """
    # ------------------------------------------------
    # Input validation
    # ------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame, got {type(data).__name__}."
        )
    if not isinstance(config, PipelineConfig):
        raise TypeError(
            f"config must be a PipelineConfig, got {type(config).__name__}."
        )

    # ------------------------------------------------
    # Feature engineering
    # ------------------------------------------------
    returns_frame = compute_simple_returns(data)
    feature_frame = compute_zscore(
        returns_frame,
        window=config.lookback,
        column=DEFAULT_SIMPLE_RETURN,
    )
    zscore_column = generate_zscore_column_name(window=config.lookback)

    # ------------------------------------------------
    # Signal generation
    # ------------------------------------------------
    signal_frame = generate_mean_reversion_signal(
        feature_frame,
        feature_column=zscore_column,
        entry_threshold=config.entry_zscore,
    )
    signal_column = generate_mean_reversion_signal_column_name(
        feature_column=zscore_column,
    )

    # ------------------------------------------------
    # Portfolio construction
    # ------------------------------------------------
    position_frame = generate_positions(
        signal_frame,
        signal_column=signal_column,
    )
    position_column = generate_position_column_name(
        signal_column=signal_column,
    )
    portfolio_frame = generate_strategy_returns(
        position_frame,
        position_column=position_column,
        asset_return_column=DEFAULT_SIMPLE_RETURN,
    )
    strategy_return_column = generate_strategy_return_column_name(
        position_column=position_column,
    )

    # ------------------------------------------------
    # Execution modelling
    # ------------------------------------------------
    # Transaction costs and slippage are applied by the backtest runner.

    # ------------------------------------------------
    # Backtesting
    # ------------------------------------------------
    backtest_frame = run_backtest(
        portfolio_frame,
        strategy_return_column=strategy_return_column,
        position_column=position_column,
        transaction_cost=config.transaction_cost,
        slippage=config.slippage,
    )
    analytics_return_column = resolve_backtest_return_column(
        strategy_return_column=strategy_return_column,
        transaction_cost=config.transaction_cost,
        slippage=config.slippage,
    )

    # ------------------------------------------------
    # Analytics
    # ------------------------------------------------
    analytics = generate_summary(
        backtest_frame,
        return_column=analytics_return_column,
        frequency=config.rebalance_frequency,
    )

    return PipelineResult(
        signals=signal_frame,
        portfolio=portfolio_frame,
        backtest=backtest_frame,
        analytics=analytics,
    )
