from dataclasses import FrozenInstanceError

import pandas as pd
import pandas.testing as pdt
import pytest

from src.research.result import ResearchResult


@pytest.fixture
def portfolio_returns() -> pd.Series:
    return pd.Series(
        [0.01, -0.02, 0.015],
        index=pd.Index(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            name="date",
        ),
        name="portfolio_returns",
    )


@pytest.fixture
def aligned_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AUBANK": [0.01, -0.02, 0.03],
            "AXISBANK": [0.0, 0.01, -0.01],
        },
        index=pd.Index(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            name="date",
        ),
    )


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series({"AUBANK": 0.5, "AXISBANK": 0.5})


@pytest.fixture
def processed_symbols() -> tuple[str, ...]:
    return ("AUBANK", "AXISBANK")


@pytest.fixture
def skipped_symbols() -> tuple[str, ...]:
    return ("MISSING",)


@pytest.fixture
def research_result(
    portfolio_returns: pd.Series,
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
    processed_symbols: tuple[str, ...],
    skipped_symbols: tuple[str, ...],
) -> ResearchResult:
    return ResearchResult(
        portfolio_returns=portfolio_returns,
        aligned_returns=aligned_returns,
        weights=weights,
        processed_symbols=processed_symbols,
        skipped_symbols=skipped_symbols,
    )


# --------------------------------------------------
# TEST 1
# Constructs With Valid Inputs
# --------------------------------------------------


def test_constructs_with_valid_inputs(
    portfolio_returns: pd.Series,
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
    processed_symbols: tuple[str, ...],
    skipped_symbols: tuple[str, ...],
) -> None:
    result = ResearchResult(
        portfolio_returns=portfolio_returns,
        aligned_returns=aligned_returns,
        weights=weights,
        processed_symbols=processed_symbols,
        skipped_symbols=skipped_symbols,
    )

    assert isinstance(result, ResearchResult)
    pdt.assert_series_equal(result.portfolio_returns, portfolio_returns)
    pdt.assert_frame_equal(result.aligned_returns, aligned_returns)
    pdt.assert_series_equal(result.weights, weights)
    assert result.processed_symbols == processed_symbols
    assert result.skipped_symbols == skipped_symbols


# --------------------------------------------------
# TEST 2
# Result Is Immutable
# --------------------------------------------------


def test_result_is_immutable(research_result: ResearchResult) -> None:
    with pytest.raises(FrozenInstanceError):
        research_result.weights = pd.Series({"AUBANK": 1.0})


# --------------------------------------------------
# TEST 3
# Invalid portfolio_returns Type
# --------------------------------------------------


def test_invalid_portfolio_returns_type(
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
    processed_symbols: tuple[str, ...],
    skipped_symbols: tuple[str, ...],
) -> None:
    with pytest.raises(TypeError, match="portfolio_returns must be"):
        ResearchResult(
            portfolio_returns=[0.01, -0.02],  # type: ignore[arg-type]
            aligned_returns=aligned_returns,
            weights=weights,
            processed_symbols=processed_symbols,
            skipped_symbols=skipped_symbols,
        )


# --------------------------------------------------
# TEST 4
# Invalid aligned_returns Type
# --------------------------------------------------


def test_invalid_aligned_returns_type(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    processed_symbols: tuple[str, ...],
    skipped_symbols: tuple[str, ...],
) -> None:
    with pytest.raises(TypeError, match="aligned_returns must be"):
        ResearchResult(
            portfolio_returns=portfolio_returns,
            aligned_returns=[{"AUBANK": 0.01}],  # type: ignore[arg-type]
            weights=weights,
            processed_symbols=processed_symbols,
            skipped_symbols=skipped_symbols,
        )


# --------------------------------------------------
# TEST 5
# Invalid weights Type
# --------------------------------------------------


def test_invalid_weights_type(
    portfolio_returns: pd.Series,
    aligned_returns: pd.DataFrame,
    processed_symbols: tuple[str, ...],
    skipped_symbols: tuple[str, ...],
) -> None:
    with pytest.raises(TypeError, match="weights must be"):
        ResearchResult(
            portfolio_returns=portfolio_returns,
            aligned_returns=aligned_returns,
            weights={"AUBANK": 0.5, "AXISBANK": 0.5},  # type: ignore[arg-type]
            processed_symbols=processed_symbols,
            skipped_symbols=skipped_symbols,
        )


# --------------------------------------------------
# TEST 6
# Invalid processed_symbols Type
# --------------------------------------------------


def test_invalid_processed_symbols_type(
    portfolio_returns: pd.Series,
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
    skipped_symbols: tuple[str, ...],
) -> None:
    with pytest.raises(TypeError, match="processed_symbols must be"):
        ResearchResult(
            portfolio_returns=portfolio_returns,
            aligned_returns=aligned_returns,
            weights=weights,
            processed_symbols=["AUBANK", "AXISBANK"],  # type: ignore[arg-type]
            skipped_symbols=skipped_symbols,
        )


# --------------------------------------------------
# TEST 7
# Invalid skipped_symbols Type
# --------------------------------------------------


def test_invalid_skipped_symbols_type(
    portfolio_returns: pd.Series,
    aligned_returns: pd.DataFrame,
    weights: pd.Series,
    processed_symbols: tuple[str, ...],
) -> None:
    with pytest.raises(TypeError, match="skipped_symbols must be"):
        ResearchResult(
            portfolio_returns=portfolio_returns,
            aligned_returns=aligned_returns,
            weights=weights,
            processed_symbols=processed_symbols,
            skipped_symbols=["MISSING"],  # type: ignore[arg-type]
        )
