import pandas as pd
import pytest

from src.pipeline.config import PipelineConfig
from src.pipeline.result import PipelineResult
from src.research import runner as runner_module
from src.research.runner import run_basket


@pytest.fixture
def pipeline_result() -> PipelineResult:
    return PipelineResult(
        signals=pd.DataFrame({"signal": [0, 1]}),
        portfolio=pd.DataFrame({"position": [0, 1]}),
        backtest=pd.DataFrame({"net_return": [0.0, 0.01]}),
        analytics={"total_return": 0.01},
    )


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(lookback=20)


@pytest.fixture
def sector_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "AUBANK",
                "AUBANK",
                "AXISBANK",
                "AXISBANK",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "close": [101.0, 100.0, 200.0, 202.0],
        }
    )


@pytest.fixture
def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Company Name": ["AU Small Finance Bank", "Axis Bank"],
            "Symbol": ["AUBANK", "AXISBANK"],
        }
    )


# --------------------------------------------------
# TEST 1
# Run Basket Success
# --------------------------------------------------


def test_run_basket_success(
    monkeypatch: pytest.MonkeyPatch,
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
    pipeline_result: PipelineResult,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "run_pipeline",
        lambda stock_data, pipeline_config: pipeline_result,
    )

    results = run_basket(sector_data, metadata, config)

    assert isinstance(results, dict)
    assert set(results) == {"AUBANK", "AXISBANK"}
    assert all(isinstance(value, PipelineResult) for value in results.values())


# --------------------------------------------------
# TEST 2
# Invalid sector_data Type
# --------------------------------------------------


def test_invalid_sector_data_type(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    with pytest.raises(TypeError):
        run_basket([], metadata, config)  # type: ignore[arg-type]


# --------------------------------------------------
# TEST 3
# Invalid metadata Type
# --------------------------------------------------


def test_invalid_metadata_type(
    sector_data: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    with pytest.raises(TypeError):
        run_basket(sector_data, [], config)  # type: ignore[arg-type]


# --------------------------------------------------
# TEST 4
# Invalid config Type
# --------------------------------------------------


def test_invalid_config_type(
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    with pytest.raises(TypeError):
        run_basket(sector_data, metadata, object())  # type: ignore[arg-type]


# --------------------------------------------------
# TEST 5
# Empty sector_data
# --------------------------------------------------


def test_empty_sector_data(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    with pytest.raises(ValueError):
        run_basket(pd.DataFrame(), metadata, config)


# --------------------------------------------------
# TEST 6
# Missing symbol Column
# --------------------------------------------------


def test_missing_symbol_column(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    sector_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(KeyError):
        run_basket(sector_data, metadata, config)


# --------------------------------------------------
# TEST 7
# Missing Metadata Symbol Column
# --------------------------------------------------


def test_missing_metadata_symbol_column(
    sector_data: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    metadata = pd.DataFrame({"Company Name": ["AU Small Finance Bank"]})

    with pytest.raises(KeyError):
        run_basket(sector_data, metadata, config)


# --------------------------------------------------
# TEST 8
# Missing Symbol Rows Are Skipped
# --------------------------------------------------


def test_missing_symbol_rows_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    sector_data: pd.DataFrame,
    config: PipelineConfig,
    pipeline_result: PipelineResult,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "run_pipeline",
        lambda stock_data, pipeline_config: pipeline_result,
    )
    metadata = pd.DataFrame({"Symbol": ["MISSINGXYZ"]})

    results = run_basket(sector_data, metadata, config)

    assert results == {}


# --------------------------------------------------
# TEST 9
# Pipeline Failure Skips Symbol
# --------------------------------------------------


def test_pipeline_failure_skips_symbol(
    monkeypatch: pytest.MonkeyPatch,
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
    pipeline_result: PipelineResult,
) -> None:
    def fake_run_pipeline(
        stock_data: pd.DataFrame,
        pipeline_config: PipelineConfig,
    ) -> PipelineResult:
        symbol = stock_data["symbol"].iloc[0]
        if symbol == "AUBANK":
            raise RuntimeError("pipeline failed")
        return pipeline_result

    monkeypatch.setattr(runner_module, "run_pipeline", fake_run_pipeline)

    results = run_basket(sector_data, metadata, config)

    assert set(results) == {"AXISBANK"}
    assert isinstance(results["AXISBANK"], PipelineResult)
    assert "AUBANK" not in results
