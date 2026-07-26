import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.features.zscore import compute_zscore
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
    """Run returns → z-score → mean-reversion-signal pipeline."""
    with_returns = compute_simple_returns(
        df,
        output_column="close_return",
    )
    with_zscore = compute_zscore(
        with_returns,
        window=window,
        column="close_return",
        output_column="zscore",
    )
    return generate_mean_reversion_signal(
        with_zscore,
        feature_column="zscore",
        entry_threshold=entry_threshold,
    )


# --------------------------------------------------
# INTEGRATION TEST
# Feature → Mean Reversion Pipeline
# --------------------------------------------------


def test_feature_mean_reversion_pipeline() -> None:
    window = 3
    entry_threshold = 1.0

    # Volatile path so rolling z-scores exceed the entry threshold.
    df = _make_price_frame(
        [
            100.0,
            101.0,
            100.5,
            102.0,
            90.0,
            88.0,
            120.0,
            125.0,
            80.0,
            78.0,
            110.0,
            115.0,
        ]
    )
    df_before = copy.deepcopy(df)

    result = _run_pipeline(
        df,
        window=window,
        entry_threshold=entry_threshold,
    )

    # 1. Returns column exists.
    assert "close_return" in result.columns

    # 2. Z-score column exists.
    assert "zscore" in result.columns

    # 3. Signal column exists.
    assert "zscore_signal" in result.columns

    # 4. Row count unchanged.
    assert len(result) == len(df)

    # 5. Original DataFrame unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 6. Pipeline schema contains required columns.
    for column in ("close", "close_return", "zscore", "zscore_signal"):
        assert column in result.columns

    # 7. Signal values belong only to LONG / FLAT / SHORT.
    allowed_signals = {LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL}
    assert set(result["zscore_signal"].unique()).issubset(allowed_signals)

    # 8. At least one LONG_SIGNAL or SHORT_SIGNAL exists.
    assert (
        (result["zscore_signal"] == LONG_SIGNAL).any()
        or (result["zscore_signal"] == SHORT_SIGNAL).any()
    )

    # 9. Non-flat signals require |zscore| >= entry_threshold.
    non_flat = result.loc[
        result["zscore_signal"] != FLAT_SIGNAL,
        ["zscore", "zscore_signal"],
    ]
    assert (non_flat["zscore"].abs() >= entry_threshold).all()


def main() -> None:
    test_feature_mean_reversion_pipeline()

    print("🎉 FEATURE → MEAN REVERSION PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
