import copy

import pandas as pd
import pandas.testing as pdt

from src.features.rolling import compute_rolling_mean
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.crossover import generate_crossover_signal


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
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    """Run fast MA → slow MA → crossover-signal pipeline."""
    fast_column = f"rolling_mean_{fast_window}"
    slow_column = f"rolling_mean_{slow_window}"

    with_fast = compute_rolling_mean(df, window=fast_window)
    with_slow = compute_rolling_mean(with_fast, window=slow_window)
    return generate_crossover_signal(
        with_slow,
        fast_column=fast_column,
        slow_column=slow_column,
    )


# --------------------------------------------------
# INTEGRATION TEST
# Feature → Crossover Pipeline
# --------------------------------------------------


def test_feature_crossover_pipeline() -> None:
    fast_window = 2
    slow_window = 5
    fast_column = f"rolling_mean_{fast_window}"
    slow_column = f"rolling_mean_{slow_window}"
    signal_column = f"{fast_column}_cross_signal"

    # Downtrend then sharp uptrend so the fast MA crosses the slow MA.
    df = _make_price_frame(
        [
            100.0,
            98.0,
            96.0,
            94.0,
            92.0,
            90.0,
            88.0,
            95.0,
            105.0,
            115.0,
            120.0,
            125.0,
        ]
    )
    df_before = copy.deepcopy(df)

    result = _run_pipeline(
        df,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    # 1. Fast MA column exists.
    assert fast_column in result.columns

    # 2. Slow MA column exists.
    assert slow_column in result.columns

    # 3. Signal column exists.
    assert signal_column in result.columns

    # 4. Row count unchanged.
    assert len(result) == len(df)

    # 5. Original DataFrame unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 6. Pipeline schema correct.
    expected_columns = list(df.columns) + [
        fast_column,
        slow_column,
        signal_column,
    ]
    assert list(result.columns) == expected_columns

    # 7. Signal values belong only to LONG / FLAT / SHORT.
    allowed_signals = {LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL}
    assert set(result[signal_column].unique()).issubset(allowed_signals)

    # 8. First row is FLAT_SIGNAL.
    assert result[signal_column].iloc[0] == FLAT_SIGNAL

    # 9. At least one crossover event exists.
    assert (
        (result[signal_column] == LONG_SIGNAL).any()
        or (result[signal_column] == SHORT_SIGNAL).any()
    )


def main() -> None:
    test_feature_crossover_pipeline()

    print("🎉 FEATURE → CROSSOVER PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
