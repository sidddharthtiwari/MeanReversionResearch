import copy
from dataclasses import FrozenInstanceError

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


def _make_price_frame(
    close: list[float],
    symbol: str = "A",
) -> pd.DataFrame:
    """Build a small deterministic single-symbol price frame."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(close),
            "close": close,
        }
    )


def _make_minimal_valid_frame() -> pd.DataFrame:
    """Build a minimal deterministic frame large enough for lookback=3."""
    return _make_price_frame(
        [
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
        ]
    )


def _make_test_config() -> PipelineConfig:
    """Build a small deterministic pipeline configuration."""
    return PipelineConfig(
        lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.5,
        transaction_cost=0.001,
        slippage=0.001,
    )


# --------------------------------------------------
# TEST 1
# Returns PipelineResult
# --------------------------------------------------


def test_run_pipeline_returns_pipeline_result() -> None:
    data = _make_minimal_valid_frame()
    config = _make_test_config()

    result = run_pipeline(data, config)

    assert isinstance(result, PipelineResult)


# --------------------------------------------------
# TEST 2
# Invalid Data Raises TypeError
# --------------------------------------------------


def test_invalid_data_raises_type_error() -> None:
    config = _make_test_config()

    try:
        run_pipeline([1, 2, 3], config)  # type: ignore[arg-type]
        raised = False
    except TypeError as error:
        raised = True
        assert "data" in str(error)

    assert raised


# --------------------------------------------------
# TEST 3
# Invalid Config Raises TypeError
# --------------------------------------------------


def test_invalid_config_raises_type_error() -> None:
    data = _make_minimal_valid_frame()

    try:
        run_pipeline(data, {"lookback": 3})  # type: ignore[arg-type]
        raised = False
    except TypeError as error:
        raised = True
        assert "config" in str(error)

    assert raised


# --------------------------------------------------
# TEST 4
# PipelineResult Contains All Expected Outputs
# --------------------------------------------------


def test_pipeline_result_contains_all_expected_outputs() -> None:
    data = _make_minimal_valid_frame()
    config = _make_test_config()

    result = run_pipeline(data, config)

    assert result.signals is not None
    assert result.portfolio is not None
    assert result.backtest is not None
    assert result.analytics is not None


# --------------------------------------------------
# TEST 5
# Returned DataFrames Have Expected Types
# --------------------------------------------------


def test_returned_dataframes_have_expected_types() -> None:
    data = _make_minimal_valid_frame()
    config = _make_test_config()

    result = run_pipeline(data, config)

    assert isinstance(result.signals, pd.DataFrame)
    assert isinstance(result.portfolio, pd.DataFrame)
    assert isinstance(result.backtest, pd.DataFrame)
    assert isinstance(result.analytics, dict)


# --------------------------------------------------
# TEST 6
# Analytics Output Exists
# --------------------------------------------------


def test_analytics_output_exists() -> None:
    data = _make_minimal_valid_frame()
    config = _make_test_config()

    result = run_pipeline(data, config)

    assert set(result.analytics) == _EXPECTED_ANALYTICS_KEYS
    assert len(result.analytics) == 12


# --------------------------------------------------
# TEST 7
# Original Input DataFrame Is Unchanged
# --------------------------------------------------


def test_original_input_dataframe_is_unchanged() -> None:
    data = _make_minimal_valid_frame()
    data_before = copy.deepcopy(data)
    config = _make_test_config()

    run_pipeline(data, config)

    pdt.assert_frame_equal(data, data_before)


# --------------------------------------------------
# TEST 8
# PipelineResult Is Frozen
# --------------------------------------------------


def test_pipeline_result_is_frozen() -> None:
    data = _make_minimal_valid_frame()
    config = _make_test_config()
    result = run_pipeline(data, config)

    try:
        result.analytics = {}  # type: ignore[misc]
        raised = False
    except FrozenInstanceError:
        raised = True

    assert raised


# --------------------------------------------------
# TEST 9
# Minimal Valid Dataset Executes Successfully
# --------------------------------------------------


def test_minimal_valid_dataset_executes_successfully() -> None:
    data = _make_minimal_valid_frame()
    config = _make_test_config()

    result = run_pipeline(data, config)

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

    assert DEFAULT_SIMPLE_RETURN in result.signals.columns
    assert zscore_column in result.signals.columns
    assert signal_column in result.signals.columns
    assert position_column in result.portfolio.columns
    assert strategy_return_column in result.portfolio.columns
    assert analytics_return_column in result.backtest.columns
    assert len(result.backtest) == len(data)


# --------------------------------------------------
# TEST 10
# Single-Row Dataset Behaviour
# --------------------------------------------------


def test_single_row_dataset_behaviour() -> None:
    data = _make_price_frame([100.0])
    config = PipelineConfig(
        lookback=1,
        entry_zscore=1.0,
        exit_zscore=0.5,
        transaction_cost=0.0,
        slippage=0.0,
    )

    result = run_pipeline(data, config)

    assert isinstance(result, PipelineResult)
    assert len(result.signals) == 1
    assert len(result.portfolio) == 1
    assert len(result.backtest) == 1
    assert set(result.analytics) == _EXPECTED_ANALYTICS_KEYS


def main() -> None:
    test_run_pipeline_returns_pipeline_result()
    test_invalid_data_raises_type_error()
    test_invalid_config_raises_type_error()
    test_pipeline_result_contains_all_expected_outputs()
    test_returned_dataframes_have_expected_types()
    test_analytics_output_exists()
    test_original_input_dataframe_is_unchanged()
    test_pipeline_result_is_frozen()
    test_minimal_valid_dataset_executes_successfully()
    test_single_row_dataset_behaviour()

    print("🎉 ALL PIPELINE RUNNER UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
