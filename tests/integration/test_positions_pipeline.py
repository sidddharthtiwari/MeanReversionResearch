import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.features.rolling import compute_rolling_mean, compute_rolling_std
from src.features.zscore import compute_zscore
from src.portfolio.positions import generate_positions
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
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
    """Run the full feature → signal → position research pipeline."""
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
    return generate_positions(
        with_signal,
        signal_column=f"{zscore_column}_signal",
    )


# --------------------------------------------------
# INTEGRATION TEST
# Positions Pipeline
# --------------------------------------------------


def test_positions_pipeline() -> None:
    window = 3
    entry_threshold = 1.0
    signal_column = f"zscore_{window}_signal"
    position_column = f"{signal_column}_position"

    # Realistic volatile path that produces non-flat signals and flats after.
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

    # 1. Pipeline executed successfully (reached here with a result).
    assert result is not None

    # 2. Original DataFrame unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 3. Default position column exists.
    assert position_column in result.columns

    # 4. Position column contains only LONG / FLAT / SHORT.
    allowed_signals = {LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL}
    assert set(result[position_column].unique()).issubset(allowed_signals)

    # 5. Carry-forward occurred: flat signal after non-flat keeps prior position.
    previous_signal = result[signal_column].shift(1)
    flat_after_non_flat = (
        result[signal_column].eq(FLAT_SIGNAL)
        & previous_signal.ne(FLAT_SIGNAL)
        & previous_signal.notna()
    )
    carried_forward = flat_after_non_flat & result[position_column].eq(
        result[position_column].shift(1)
    )
    assert carried_forward.any()

    # 6. Output DataFrame preserves the original index.
    pdt.assert_index_equal(result.index, df.index)

    # 7. Generated position column contains no missing values.
    assert result[position_column].notna().all()


def main() -> None:
    test_positions_pipeline()

    print("🎉 POSITIONS PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
