import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.signals.breakout import generate_breakout_signal
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL


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
    upper_threshold: float,
    lower_threshold: float,
) -> pd.DataFrame:
    """Run returns → breakout-signal pipeline."""
    with_returns = compute_simple_returns(df)
    return generate_breakout_signal(
        with_returns,
        feature_column="simple_return",
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )


# --------------------------------------------------
# INTEGRATION TEST
# Feature → Breakout Pipeline
# --------------------------------------------------


def test_feature_breakout_pipeline() -> None:
    upper_threshold = 0.03
    lower_threshold = -0.02
    signal_column = "simple_return_breakout_signal"

    # Deterministic closes that produce returns crossing both thresholds.
    df = _make_price_frame(
        [
            100.0,
            101.0,
            105.0,
            104.0,
            100.0,
            99.0,
            103.0,
            102.0,
            98.0,
            102.0,
        ]
    )
    df_before = copy.deepcopy(df)

    result = _run_pipeline(
        df,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )

    # Original DataFrame unchanged.
    pdt.assert_frame_equal(df, df_before)

    # Row count preserved.
    assert len(result) == len(df)

    # Required columns exist.
    for column in ("close", "simple_return", signal_column):
        assert column in result.columns

    # First row flat.
    assert result[signal_column].iloc[0] == FLAT_SIGNAL

    # Signals only contain LONG / FLAT / SHORT.
    allowed_signals = {LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL}
    assert set(result[signal_column].unique()).issubset(allowed_signals)

    # At least one breakout event exists.
    assert (
        (result[signal_column] == LONG_SIGNAL).any()
        or (result[signal_column] == SHORT_SIGNAL).any()
    )

    returns = result["simple_return"]
    previous_returns = returns.shift(1)

    # Every LONG signal satisfies previous <= upper and current > upper.
    long_mask = result[signal_column] == LONG_SIGNAL
    if long_mask.any():
        assert (previous_returns.loc[long_mask] <= upper_threshold).all()
        assert (returns.loc[long_mask] > upper_threshold).all()

    # Every SHORT signal satisfies previous >= lower and current < lower.
    short_mask = result[signal_column] == SHORT_SIGNAL
    if short_mask.any():
        assert (previous_returns.loc[short_mask] >= lower_threshold).all()
        assert (returns.loc[short_mask] < lower_threshold).all()


def main() -> None:
    test_feature_breakout_pipeline()

    print("🎉 FEATURE → BREAKOUT PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
