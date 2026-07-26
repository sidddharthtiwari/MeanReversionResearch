import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.features.rolling import compute_rolling_mean, compute_rolling_std
from src.features.zscore import compute_zscore
from src.portfolio.positions import generate_positions
from src.portfolio.returns import generate_strategy_returns
from src.signals.mean_reversion import generate_mean_reversion_signal


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


def _run_pipeline(
    df: pd.DataFrame,
    window: int,
    entry_threshold: float,
) -> pd.DataFrame:
    """Run the full feature → signal → position → return research pipeline."""
    with_returns = compute_simple_returns(df)
    with_mean = compute_rolling_mean(
        with_returns,
        window=window,
        column="simple_return",
    )
    with_std = compute_rolling_std(
        with_mean,
        window=window,
        column="simple_return",
    )
    with_zscore = compute_zscore(
        with_std,
        window=window,
        column="simple_return",
    )
    zscore_column = f"zscore_{window}"
    with_signal = generate_mean_reversion_signal(
        with_zscore,
        feature_column=zscore_column,
        entry_threshold=entry_threshold,
    )
    with_positions = generate_positions(
        with_signal,
        signal_column=f"{zscore_column}_signal",
    )
    # Leading simple-return NaN would poison cumulative products; treat as 0.
    prepared = with_positions.copy()
    prepared["simple_return"] = prepared["simple_return"].fillna(0.0)
    return generate_strategy_returns(
        prepared,
        position_column=f"{zscore_column}_signal_position",
        asset_return_column="simple_return",
    )


# --------------------------------------------------
# INTEGRATION TEST
# Returns Pipeline
# --------------------------------------------------


def test_returns_pipeline() -> None:
    window = 3
    entry_threshold = 1.0
    position_column = f"zscore_{window}_signal_position"
    strategy_return_column = f"{position_column}_strategy_return"
    cumulative_return_column = f"{position_column}_cumulative_return"
    equity_curve_column = f"{position_column}_equity_curve"

    df = _make_price_frame(
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
    )
    df_before = copy.deepcopy(df)

    result = _run_pipeline(
        df,
        window=window,
        entry_threshold=entry_threshold,
    )

    # 1. Pipeline executes successfully.
    assert result is not None

    # 2. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 3. Strategy-return column exists.
    assert strategy_return_column in result.columns

    # 4. Cumulative-return column exists.
    assert cumulative_return_column in result.columns

    # 5. Equity-curve column exists.
    assert equity_curve_column in result.columns

    # 6. Strategy returns equal position × asset return.
    expected_strategy_returns = (
        result[position_column] * result["simple_return"]
    )
    pdt.assert_series_equal(
        result[strategy_return_column],
        expected_strategy_returns,
        check_names=False,
    )

    # 7. Equity curve equals 1 + cumulative return.
    expected_equity_curve = 1.0 + result[cumulative_return_column]
    pdt.assert_series_equal(
        result[equity_curve_column],
        expected_equity_curve,
        check_names=False,
    )

    # 8. Generated columns contain no missing values.
    assert result[strategy_return_column].notna().all()
    assert result[cumulative_return_column].notna().all()
    assert result[equity_curve_column].notna().all()

    # 9. Output index matches the original index.
    pdt.assert_index_equal(result.index, df.index)


def main() -> None:
    test_returns_pipeline()

    print("🎉 RETURNS PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
