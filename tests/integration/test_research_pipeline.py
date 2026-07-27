import pandas as pd
import pytest

from src.pipeline.config import PipelineConfig
from src.research import ResearchResult, run_research


@pytest.fixture
def trading_dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=20, freq="B")


@pytest.fixture
def sector_data(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    aubank_close = [
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
    axisbank_close = [
        50.0,
        50.5,
        51.0,
        50.2,
        49.8,
        52.0,
        48.0,
        47.0,
        46.5,
        49.0,
        55.0,
        56.0,
        55.5,
        57.0,
        45.0,
        44.0,
        44.5,
        52.0,
        53.0,
        52.5,
    ]
    closes = aubank_close + axisbank_close
    return pd.DataFrame(
        {
            "symbol": ["AUBANK"] * 20 + ["AXISBANK"] * 20,
            "date": list(trading_dates) * 2,
            "open": closes,
            "high": [close * 1.01 for close in closes],
            "low": [close * 0.99 for close in closes],
            "close": closes,
            "volume": [1_000] * 40,
        }
    )


@pytest.fixture
def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Company Name": ["AU Small Finance Bank Ltd.", "Axis Bank Ltd."],
            "Industry": ["Financial Services", "Financial Services"],
            "Symbol": ["AUBANK", "AXISBANK"],
            "Series": ["EQ", "EQ"],
            "ISIN Code": ["INE949L01017", "INE238A01034"],
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
# Research Pipeline End To End
# --------------------------------------------------


def test_research_pipeline_end_to_end(
    sector_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    result = run_research(sector_data, metadata, config)

    assert isinstance(result, ResearchResult)
    assert set(result.processed_symbols) == {"AUBANK", "AXISBANK"}
    assert result.skipped_symbols == ()
    assert not result.aligned_returns.empty
    assert not result.portfolio_returns.empty
    assert float(result.weights.sum()) == pytest.approx(1.0)
    assert list(result.aligned_returns.columns) == list(result.processed_symbols)


# --------------------------------------------------
# INTEGRATION TEST 2
# Missing Symbol Is Skipped
# --------------------------------------------------


def test_missing_symbol_is_skipped(
    sector_data: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    metadata = pd.DataFrame(
        {
            "Company Name": [
                "AU Small Finance Bank Ltd.",
                "Missing Bank Ltd.",
                "Axis Bank Ltd.",
            ],
            "Symbol": ["AUBANK", "MISSINGXYZ", "AXISBANK"],
        }
    )

    result = run_research(sector_data, metadata, config)

    assert isinstance(result, ResearchResult)
    assert set(result.processed_symbols) == {"AUBANK", "AXISBANK"}
    assert "MISSINGXYZ" not in result.processed_symbols
    assert not result.portfolio_returns.empty
