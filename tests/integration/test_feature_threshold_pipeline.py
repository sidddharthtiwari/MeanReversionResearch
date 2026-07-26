import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.features.zscore import compute_zscore
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.threshold import generate_threshold_signal


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
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Run returns → z-score → threshold-signal pipeline."""
    with_returns = compute_simple_returns(df)
    with_zscore = compute_zscore(
        with_returns,
        window=window,
        column="simple_return",
    )
    return generate_threshold_signal(
        with_zscore,
        feature_column=f"zscore_{window}",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


# --------------------------------------------------
# INTEGRATION TEST
# Feature → Threshold Pipeline
# --------------------------------------------------


def test_feature_threshold_pipeline() -> None:
    window = 3
    buy_threshold = -1.0
    sell_threshold = 1.0
    zscore_column = f"zscore_{window}"
    signal_column = f"{zscore_column}_signal"

    df = _make_price_frame(
        [
            100.0,
            102.0,
            101.0,
            103.0,
            98.0,
            97.0,
            99.0,
            105.0,
            104.0,
            106.0,
        ]
    )
    df_before = copy.deepcopy(df)

    result = _run_pipeline(
        df,
        window=window,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 1. Returns column exists.
    assert "simple_return" in result.columns

    # 2. Z-score column exists.
    assert zscore_column in result.columns

    # 3. Signal column exists.
    assert signal_column in result.columns

    # 4. Row count unchanged.
    assert len(result) == len(df)

    # 5. Original DataFrame unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 6. Pipeline schema correct.
    expected_columns = list(df.columns) + [
        "simple_return",
        zscore_column,
        signal_column,
    ]
    assert list(result.columns) == expected_columns

    # 7. Signal values belong only to LONG / FLAT / SHORT.
    allowed_signals = {LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL}
    assert set(result[signal_column].unique()).issubset(allowed_signals)


def main() -> None:
    test_feature_threshold_pipeline()

    print("🎉 FEATURE → THRESHOLD PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
