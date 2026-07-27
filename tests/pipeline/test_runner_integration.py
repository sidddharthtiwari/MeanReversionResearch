import copy

import pandas as pd
import pandas.testing as pdt

from src.backtest.runner import resolve_backtest_return_column
from src.features.returns import DEFAULT_SIMPLE_RETURN
from src.features.zscore import generate_zscore_column_name
from src.pipeline.config import PipelineConfig
from src.pipeline.result import PipelineResult
from src.pipeline.runner import run_pipeline
from src.portfolio.positions import generate_position_column_name
from src.portfolio.returns import generate_strategy_return_column_name
from src.signals.mean_reversion import generate_mean_reversion_signal_column_name

_EXPECTED_ANALYTICS_KEYS = {
    "total_return",
    "average_period_return",
    "annualised_return",
    "cagr",
    "volatility",
    "annualised_volatility",
    "downside_deviation",
    "max_drawdown",
    "drawdown_duration",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
}


# --------------------------------------------------
# INTEGRATION TEST
# Research Pipeline Runner
# --------------------------------------------------


def test_runner_integration() -> None:
    data = pd.DataFrame(
        {
            "symbol": ["A"] * 20,
            "close": [
                100.0,
                101.0,
                100.5,
                102.0,
                101.5,
                103.0,
                90.0,
                88.0,
                87.0,
                89.0,
                120.0,
                125.0,
                124.0,
                126.0,
                80.0,
                78.0,
                79.0,
                110.0,
                112.0,
                111.0,
            ],
        },
        index=pd.Index(range(20), name="row_id"),
    )
    data_before = copy.deepcopy(data)

    config = PipelineConfig(
        lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.5,
        transaction_cost=0.001,
        slippage=0.001,
        rebalance_frequency="D",
    )

    zscore_column = generate_zscore_column_name(window=config.lookback)
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
        transaction_cost=config.transaction_cost,
        slippage=config.slippage,
    )

    result = run_pipeline(data, config)

    # Input → PipelineResult
    assert isinstance(result, PipelineResult)

    # Features
    assert DEFAULT_SIMPLE_RETURN in result.signals.columns
    assert zscore_column in result.signals.columns

    # Signals
    assert signal_column in result.signals.columns

    # Portfolio
    assert position_column in result.portfolio.columns
    assert strategy_return_column in result.portfolio.columns
    assert signal_column in result.portfolio.columns
    assert zscore_column in result.portfolio.columns

    # Backtest
    assert strategy_return_column in result.backtest.columns
    assert analytics_return_column in result.backtest.columns
    assert position_column in result.backtest.columns

    # Analytics
    assert set(result.analytics) == _EXPECTED_ANALYTICS_KEYS
    assert len(result.analytics) == 12

    # Row counts are preserved.
    assert len(result.signals) == len(data)
    assert len(result.portfolio) == len(data)
    assert len(result.backtest) == len(data)

    # Index is preserved.
    pdt.assert_index_equal(result.signals.index, data.index)
    pdt.assert_index_equal(result.portfolio.index, data.index)
    pdt.assert_index_equal(result.backtest.index, data.index)

    # Original input remains unchanged.
    pdt.assert_frame_equal(data, data_before)

    # Stage outputs are chained: later frames contain earlier columns.
    assert set(result.signals.columns).issubset(set(result.portfolio.columns))
    assert set(result.portfolio.columns).issubset(set(result.backtest.columns))


def main() -> None:
    test_runner_integration()

    print("🎉 PIPELINE RUNNER INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
