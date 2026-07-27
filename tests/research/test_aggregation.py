import pandas as pd
import pandas.testing as pdt
import pytest

from src.pipeline.result import PipelineResult
from src.research.aggregation import aggregate
from src.research.result import ResearchResult

_STRATEGY_RETURN_COLUMN = "signal_position_strategy_return"


@pytest.fixture
def make_pipeline_result():
    def _make(
        dates: list[str],
        strategy_returns: list[float],
        strategy_column: str = _STRATEGY_RETURN_COLUMN,
        include_date: bool = True,
        extra_strategy_columns: dict[str, list[float]] | None = None,
    ) -> PipelineResult:
        portfolio_data: dict[str, object] = {
            strategy_column: strategy_returns,
        }
        if include_date:
            portfolio_data["date"] = pd.to_datetime(dates)
        if extra_strategy_columns is not None:
            portfolio_data.update(extra_strategy_columns)

        return PipelineResult(
            signals=pd.DataFrame({"signal": [0] * len(strategy_returns)}),
            portfolio=pd.DataFrame(portfolio_data),
            backtest=pd.DataFrame({"value": [0.0] * len(strategy_returns)}),
            analytics={"total_return": 0.0},
        )

    return _make


@pytest.fixture
def aubank_result(make_pipeline_result) -> PipelineResult:
    return make_pipeline_result(
        dates=["2024-01-01", "2024-01-02", "2024-01-03"],
        strategy_returns=[0.01, -0.02, 0.03],
    )


@pytest.fixture
def axisbank_result(make_pipeline_result) -> PipelineResult:
    return make_pipeline_result(
        dates=["2024-01-01", "2024-01-02", "2024-01-03"],
        strategy_returns=[0.0, 0.04, -0.01],
    )


@pytest.fixture
def basket_results(
    aubank_result: PipelineResult,
    axisbank_result: PipelineResult,
) -> dict[str, PipelineResult]:
    return {
        "AUBANK": aubank_result,
        "AXISBANK": axisbank_result,
    }


# --------------------------------------------------
# TEST 1
# Aggregate Returns ResearchResult
# --------------------------------------------------


def test_aggregate_returns_research_result(
    basket_results: dict[str, PipelineResult],
) -> None:
    result = aggregate(basket_results)

    assert isinstance(result, ResearchResult)
    assert result.processed_symbols == ("AUBANK", "AXISBANK")
    assert result.skipped_symbols == ()
    assert list(result.aligned_returns.columns) == ["AUBANK", "AXISBANK"]

    expected_portfolio_returns = result.aligned_returns.mean(axis=1)
    pdt.assert_series_equal(
        result.portfolio_returns,
        expected_portfolio_returns.astype("float64"),
        check_names=False,
    )
    assert float(result.weights.sum()) == pytest.approx(1.0)


# --------------------------------------------------
# TEST 2
# Invalid Results Type
# --------------------------------------------------


def test_invalid_results_type() -> None:
    with pytest.raises(TypeError, match="results must be"):
        aggregate([])  # type: ignore[arg-type]


# --------------------------------------------------
# TEST 3
# Empty Results
# --------------------------------------------------


def test_empty_results() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate({})


# --------------------------------------------------
# TEST 4
# Invalid PipelineResult
# --------------------------------------------------


def test_invalid_pipeline_result() -> None:
    with pytest.raises(TypeError, match="PipelineResult"):
        aggregate({"AUBANK": object()})  # type: ignore[dict-item]


# --------------------------------------------------
# TEST 5
# Missing Date Column
# --------------------------------------------------


def test_missing_date_column(make_pipeline_result) -> None:
    result = make_pipeline_result(
        dates=["2024-01-01", "2024-01-02"],
        strategy_returns=[0.01, -0.02],
        include_date=False,
    )

    with pytest.raises(KeyError):
        aggregate({"AUBANK": result})


# --------------------------------------------------
# TEST 6
# Missing Strategy Return Column
# --------------------------------------------------


def test_missing_strategy_return_column() -> None:
    result = PipelineResult(
        signals=pd.DataFrame({"signal": [0, 0]}),
        portfolio=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "close": [100.0, 101.0],
            }
        ),
        backtest=pd.DataFrame({"value": [0.0, 0.0]}),
        analytics={},
    )

    with pytest.raises(KeyError):
        aggregate({"AUBANK": result})


# --------------------------------------------------
# TEST 7
# Multiple Strategy Return Columns
# --------------------------------------------------


def test_multiple_strategy_return_columns(make_pipeline_result) -> None:
    result = make_pipeline_result(
        dates=["2024-01-01", "2024-01-02"],
        strategy_returns=[0.01, -0.02],
        extra_strategy_columns={
            "alt_signal_position_strategy_return": [0.02, 0.0],
        },
    )

    with pytest.raises(KeyError):
        aggregate({"AUBANK": result})


# --------------------------------------------------
# TEST 8
# Inner Alignment Keeps Common Dates
# --------------------------------------------------


def test_inner_alignment_keeps_common_dates(make_pipeline_result) -> None:
    first = make_pipeline_result(
        dates=["2024-01-01", "2024-01-02", "2024-01-03"],
        strategy_returns=[0.01, -0.02, 0.03],
    )
    second = make_pipeline_result(
        dates=["2024-01-02", "2024-01-03", "2024-01-04"],
        strategy_returns=[0.0, 0.04, -0.01],
    )

    result = aggregate({"AUBANK": first, "AXISBANK": second})

    expected_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pdt.assert_index_equal(
        result.aligned_returns.index,
        pd.Index(expected_dates, name="date"),
    )


# --------------------------------------------------
# TEST 9
# Custom Weighting Function
# --------------------------------------------------


def test_custom_weighting_function(
    basket_results: dict[str, PipelineResult],
) -> None:
    def custom_weights(aligned_returns: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {"AUBANK": 0.7, "AXISBANK": 0.3},
            dtype="float64",
        )

    result = aggregate(
        basket_results,
        weighting_function=custom_weights,
    )

    expected = (
        result.aligned_returns["AUBANK"] * 0.7
        + result.aligned_returns["AXISBANK"] * 0.3
    ).astype("float64")
    pdt.assert_series_equal(
        result.portfolio_returns,
        expected,
        check_names=False,
    )
    assert result.weights["AUBANK"] == pytest.approx(0.7)
    assert result.weights["AXISBANK"] == pytest.approx(0.3)


# --------------------------------------------------
# TEST 10
# Missing Weight From Weighting Function
# --------------------------------------------------


def test_missing_weight_from_weighting_function(
    basket_results: dict[str, PipelineResult],
) -> None:
    def incomplete_weights(aligned_returns: pd.DataFrame) -> pd.Series:
        return pd.Series({"AUBANK": 1.0}, dtype="float64")

    with pytest.raises(ValueError, match="Missing weights"):
        aggregate(
            basket_results,
            weighting_function=incomplete_weights,
        )
