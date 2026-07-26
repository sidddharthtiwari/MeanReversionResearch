import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.features.rolling import compute_rolling_mean, compute_rolling_std
from src.features.zscore import compute_zscore
from src.portfolio.exposure import generate_exposure
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
    """Run the full feature → signal → position → return → exposure pipeline."""
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
    position_column = f"{zscore_column}_signal_position"

    # Leading simple-return NaN would poison cumulative products; treat as 0.
    prepared = with_positions.copy()
    prepared["simple_return"] = prepared["simple_return"].fillna(0.0)
    with_strategy_returns = generate_strategy_returns(
        prepared,
        position_column=position_column,
        asset_return_column="simple_return",
    )
    return generate_exposure(
        with_strategy_returns,
        position_column=position_column,
    )


# --------------------------------------------------
# INTEGRATION TEST
# Exposure Pipeline
# --------------------------------------------------


def test_exposure_pipeline() -> None:
    window = 3
    entry_threshold = 1.0
    position_column = f"zscore_{window}_signal_position"
    net_exposure_column = f"{position_column}_net_exposure"
    gross_exposure_column = f"{position_column}_gross_exposure"

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

    # 3. Net-exposure column exists.
    assert net_exposure_column in result.columns

    # 4. Gross-exposure column exists.
    assert gross_exposure_column in result.columns

    # 5. Net exposure equals the position column.
    pdt.assert_series_equal(
        result[net_exposure_column],
        result[position_column],
        check_names=False,
    )

    # 6. Gross exposure equals abs(position).
    pdt.assert_series_equal(
        result[gross_exposure_column],
        result[position_column].abs(),
        check_names=False,
    )

    # 7. Generated columns contain no missing values.
    assert result[net_exposure_column].notna().all()
    assert result[gross_exposure_column].notna().all()

    # 8. Output index matches the original index.
    pdt.assert_index_equal(result.index, df.index)


def main() -> None:
    test_exposure_pipeline()

    print("🎉 EXPOSURE PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
