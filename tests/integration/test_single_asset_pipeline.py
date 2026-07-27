import pandas as pd
import pytest

from src.features.zscore import generate_zscore_column_name
from src.pipeline.config import PipelineConfig
from src.pipeline.result import PipelineResult
from src.pipeline.runner import run_pipeline
from src.portfolio.positions import generate_position_column_name
from src.portfolio.returns import generate_strategy_return_column_name
from src.signals.mean_reversion import generate_mean_reversion_signal_column_name


@pytest.fixture
def ohlc_data() -> pd.DataFrame:
    close = [
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
    ]
    return pd.DataFrame(
        {
            "symbol": ["AUBANK"] * len(close),
            "date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
            "open": close,
            "high": [price * 1.01 for price in close],
            "low": [price * 0.99 for price in close],
            "close": close,
            "volume": [1_000] * len(close),
        }
    )


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(
        lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.5,
        transaction_cost=0.001,
        slippage=0.001,
        rebalance_frequency="D",
    )


# --------------------------------------------------
# INTEGRATION TEST 1
# Single-Asset Pipeline End To End
# --------------------------------------------------


def test_single_asset_pipeline_end_to_end(
    ohlc_data: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    zscore_column = generate_zscore_column_name(window=config.lookback)
    signal_column = generate_mean_reversion_signal_column_name(
        feature_column=zscore_column,
    )
    position_column = generate_position_column_name(signal_column=signal_column)
    strategy_return_column = generate_strategy_return_column_name(
        position_column=position_column,
    )

    result = run_pipeline(ohlc_data, config)

    assert isinstance(result, PipelineResult)
    assert not result.signals.empty
    assert not result.portfolio.empty
    assert not result.backtest.empty
    assert result.analytics
    assert strategy_return_column in result.portfolio.columns
    assert strategy_return_column in result.backtest.columns

    post_warmup = result.signals.iloc[config.lookback :]
    assert not post_warmup.empty
    assert post_warmup[zscore_column].notna().any()
    assert result.portfolio.iloc[config.lookback :][
        strategy_return_column
    ].notna().all()


# --------------------------------------------------
# INTEGRATION TEST 2
# Invalid Input Type
# --------------------------------------------------


def test_invalid_input_type(config: PipelineConfig) -> None:
    with pytest.raises(TypeError):
        run_pipeline([], config)  # type: ignore[arg-type]
