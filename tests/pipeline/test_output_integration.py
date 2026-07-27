import json
import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.pipeline.config import PipelineConfig
from src.pipeline.output import save_pipeline_outputs
from src.pipeline.runner import run_pipeline


# --------------------------------------------------
# INTEGRATION TEST
# Pipeline Output Persistence
# --------------------------------------------------


def test_output_integration() -> None:
    data = pd.DataFrame(
        {
            "symbol": ["A"] * 12,
            "close": [
                100.0,
                101.0,
                99.0,
                102.0,
                98.0,
                103.0,
                97.0,
                104.0,
                96.0,
                105.0,
                95.0,
                106.0,
            ],
        }
    )
    config = PipelineConfig(
        lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.5,
        transaction_cost=0.001,
        slippage=0.001,
        rebalance_frequency="D",
    )

    result = run_pipeline(data, config)

    with tempfile.TemporaryDirectory() as temporary_root:
        output_directory = Path(temporary_root)
        save_pipeline_outputs(result, output_directory)

        signals_path = output_directory / "signals.csv"
        portfolio_path = output_directory / "portfolio.csv"
        backtest_path = output_directory / "backtest.csv"
        analytics_path = output_directory / "analytics.json"

        assert signals_path.is_file()
        assert portfolio_path.is_file()
        assert backtest_path.is_file()
        assert analytics_path.is_file()

        loaded_signals = pd.read_csv(signals_path)
        loaded_portfolio = pd.read_csv(portfolio_path)
        loaded_backtest = pd.read_csv(backtest_path)

        pdt.assert_frame_equal(loaded_signals, result.signals)
        pdt.assert_frame_equal(loaded_portfolio, result.portfolio)
        pdt.assert_frame_equal(loaded_backtest, result.backtest)

        with analytics_path.open(encoding="utf-8") as file:
            loaded_analytics = json.load(file)

        assert loaded_analytics == dict(result.analytics)


def main() -> None:
    test_output_integration()

    print("PIPELINE OUTPUT INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
