import json
import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.pipeline.output import save_pipeline_outputs
from src.pipeline.result import PipelineResult


def _make_pipeline_result(
    signals: pd.DataFrame | None = None,
    portfolio: pd.DataFrame | None = None,
    backtest: pd.DataFrame | None = None,
    analytics: dict[str, float | int] | None = None,
) -> PipelineResult:
    """Build a deterministic PipelineResult for persistence tests."""
    if signals is None:
        signals = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "signal": [1, -1],
            }
        )
    if portfolio is None:
        portfolio = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "position": [1, -1],
                "strategy_return": [0.01, -0.02],
            }
        )
    if backtest is None:
        backtest = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "equity": [1.01, 0.99],
            }
        )
    if analytics is None:
        analytics = {
            "sharpe_ratio": 1.5,
            "annual_return": 0.20,
            "max_drawdown": -0.10,
        }
    return PipelineResult(
        signals=signals,
        portfolio=portfolio,
        backtest=backtest,
        analytics=analytics,
    )


# --------------------------------------------------
# TEST 1
# Invalid Result Type
# --------------------------------------------------


def test_invalid_result_type() -> None:
    try:
        save_pipeline_outputs(
            {"signals": 1},  # type: ignore[arg-type]
            tempfile.gettempdir(),
        )
    except TypeError as error:
        assert "result" in str(error)
    else:
        raise AssertionError("Expected TypeError.")


# --------------------------------------------------
# TEST 2
# Invalid Output Directory Type
# --------------------------------------------------


def test_invalid_output_directory_type() -> None:
    result = _make_pipeline_result()

    try:
        save_pipeline_outputs(result, 123)  # type: ignore[arg-type]
    except TypeError as error:
        assert "output_directory" in str(error)
    else:
        raise AssertionError("Expected TypeError.")


# --------------------------------------------------
# TEST 3
# Output Directory Created
# --------------------------------------------------


def test_output_directory_created() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root) / "nested" / "outputs"
        assert not output_directory.exists()

        save_pipeline_outputs(result, output_directory)

        assert output_directory.exists()
        assert output_directory.is_dir()


# --------------------------------------------------
# TEST 4
# All Expected Files Created
# --------------------------------------------------


def test_all_expected_files_created() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root)
        save_pipeline_outputs(result, output_directory)

        assert (output_directory / "signals.csv").is_file()
        assert (output_directory / "portfolio.csv").is_file()
        assert (output_directory / "backtest.csv").is_file()
        assert (output_directory / "analytics.json").is_file()


# --------------------------------------------------
# TEST 5
# Returns None
# --------------------------------------------------


def test_returns_none() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        returned = save_pipeline_outputs(result, temporary_root)

    assert returned is None


# --------------------------------------------------
# TEST 6
# Existing Directory Supported
# --------------------------------------------------


def test_existing_directory_supported() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root)
        assert output_directory.is_dir()

        save_pipeline_outputs(result, output_directory)

        assert (output_directory / "signals.csv").is_file()
        assert (output_directory / "analytics.json").is_file()


# --------------------------------------------------
# TEST 7
# Existing Files Are Overwritten
# --------------------------------------------------


def test_existing_files_are_overwritten() -> None:
    first_result = _make_pipeline_result(
        signals=pd.DataFrame({"date": ["2024-01-01"], "signal": [1]}),
        analytics={"sharpe_ratio": 1.0},
    )
    second_result = _make_pipeline_result(
        signals=pd.DataFrame({"date": ["2024-01-02"], "signal": [-1]}),
        analytics={"sharpe_ratio": 2.5},
    )

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root)
        save_pipeline_outputs(first_result, output_directory)
        save_pipeline_outputs(second_result, output_directory)

        loaded_signals = pd.read_csv(output_directory / "signals.csv")
        with (output_directory / "analytics.json").open(
            encoding="utf-8"
        ) as file:
            loaded_analytics = json.load(file)

        pdt.assert_frame_equal(loaded_signals, second_result.signals)
        assert loaded_analytics == dict(second_result.analytics)


# --------------------------------------------------
# TEST 8
# Saved CSV Matches PipelineResult
# --------------------------------------------------


def test_saved_csv_matches_pipeline_result() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root)
        save_pipeline_outputs(result, output_directory)

        loaded_signals = pd.read_csv(output_directory / "signals.csv")
        loaded_portfolio = pd.read_csv(output_directory / "portfolio.csv")
        loaded_backtest = pd.read_csv(output_directory / "backtest.csv")

        pdt.assert_frame_equal(loaded_signals, result.signals)
        pdt.assert_frame_equal(loaded_portfolio, result.portfolio)
        pdt.assert_frame_equal(loaded_backtest, result.backtest)


# --------------------------------------------------
# TEST 9
# Saved Analytics Matches PipelineResult
# --------------------------------------------------


def test_saved_analytics_matches_pipeline_result() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root)
        save_pipeline_outputs(result, output_directory)

        with (output_directory / "analytics.json").open(
            encoding="utf-8"
        ) as file:
            loaded_analytics = json.load(file)

        assert loaded_analytics == dict(result.analytics)


# --------------------------------------------------
# TEST 10
# String Output Directory Supported
# --------------------------------------------------


def test_string_output_directory_supported() -> None:
    result = _make_pipeline_result()

    with tempfile.TemporaryDirectory() as temporary_root:
        save_pipeline_outputs(result, temporary_root)

        output_directory = Path(temporary_root)
        assert (output_directory / "signals.csv").is_file()
        assert (output_directory / "portfolio.csv").is_file()
        assert (output_directory / "backtest.csv").is_file()
        assert (output_directory / "analytics.json").is_file()


def main() -> None:
    test_invalid_result_type()
    test_invalid_output_directory_type()
    test_output_directory_created()
    test_all_expected_files_created()
    test_returns_none()
    test_existing_directory_supported()
    test_existing_files_are_overwritten()
    test_saved_csv_matches_pipeline_result()
    test_saved_analytics_matches_pipeline_result()
    test_string_output_directory_supported()

    print("ALL PIPELINE OUTPUT UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
