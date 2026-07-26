from dataclasses import FrozenInstanceError

import pandas as pd
import pandas.testing as pdt

from src.pipeline.result import PipelineResult


def _make_frame(column: str = "value") -> pd.DataFrame:
    """Build a small deterministic DataFrame for PipelineResult fields."""
    return pd.DataFrame({column: [1.0, 2.0]})


def _make_valid_result() -> PipelineResult:
    """Build a valid PipelineResult for reuse across tests."""
    return PipelineResult(
        signals=_make_frame("signal"),
        portfolio=_make_frame("position"),
        backtest=_make_frame("net_return"),
        analytics={"total_return": 0.1, "drawdown_duration": 2},
    )


# --------------------------------------------------
# TEST 1
# Valid Pipeline Result
# --------------------------------------------------


def test_valid_pipeline_result() -> None:
    signals = _make_frame("signal")
    portfolio = _make_frame("position")
    backtest = _make_frame("net_return")
    analytics = {"total_return": 0.1, "drawdown_duration": 2}

    result = PipelineResult(
        signals=signals,
        portfolio=portfolio,
        backtest=backtest,
        analytics=analytics,
    )

    pdt.assert_frame_equal(result.signals, signals)
    pdt.assert_frame_equal(result.portfolio, portfolio)
    pdt.assert_frame_equal(result.backtest, backtest)
    assert result.analytics == analytics


# --------------------------------------------------
# TEST 2
# Invalid Signals
# --------------------------------------------------


def test_invalid_signals() -> None:
    try:
        PipelineResult(
            signals=[1, 2, 3],  # type: ignore[arg-type]
            portfolio=_make_frame("position"),
            backtest=_make_frame("net_return"),
            analytics={},
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "signals" in str(error)

    assert raised


# --------------------------------------------------
# TEST 3
# Invalid Portfolio
# --------------------------------------------------


def test_invalid_portfolio() -> None:
    try:
        PipelineResult(
            signals=_make_frame("signal"),
            portfolio=[1, 2, 3],  # type: ignore[arg-type]
            backtest=_make_frame("net_return"),
            analytics={},
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "portfolio" in str(error)

    assert raised


# --------------------------------------------------
# TEST 4
# Invalid Backtest
# --------------------------------------------------


def test_invalid_backtest() -> None:
    try:
        PipelineResult(
            signals=_make_frame("signal"),
            portfolio=_make_frame("position"),
            backtest=[1, 2, 3],  # type: ignore[arg-type]
            analytics={},
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "backtest" in str(error)

    assert raised


# --------------------------------------------------
# TEST 5
# Invalid Analytics
# --------------------------------------------------


def test_invalid_analytics() -> None:
    try:
        PipelineResult(
            signals=_make_frame("signal"),
            portfolio=_make_frame("position"),
            backtest=_make_frame("net_return"),
            analytics=["not", "a", "mapping"],  # type: ignore[arg-type]
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "analytics" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Pipeline Result Is Frozen
# --------------------------------------------------


def test_pipeline_result_is_frozen() -> None:
    result = _make_valid_result()

    try:
        result.analytics = {}  # type: ignore[misc]
        raised = False
    except FrozenInstanceError:
        raised = True

    assert raised


def main() -> None:
    test_valid_pipeline_result()
    test_invalid_signals()
    test_invalid_portfolio()
    test_invalid_backtest()
    test_invalid_analytics()
    test_pipeline_result_is_frozen()

    print("🎉 ALL PIPELINE RESULT TESTS PASSED")


if __name__ == "__main__":
    main()
