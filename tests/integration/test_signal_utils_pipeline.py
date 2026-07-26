import copy

import pandas as pd
import pandas.testing as pdt

from src.features.returns import compute_simple_returns
from src.features.zscore import compute_zscore
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.mean_reversion import generate_mean_reversion_signal
from src.signals.signal_utils import (
    count_signal_changes,
    invert_signal,
    summarize_signal,
    validate_signal_column,
)


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


def _run_feature_pipeline(
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
# Signal Utils Pipeline
# --------------------------------------------------


def test_signal_utils_pipeline() -> None:
    window = 3
    entry_threshold = 1.0
    signal_column = "zscore_signal"
    inverted_column = f"{signal_column}_inverted"

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

    with_signals = _run_feature_pipeline(
        df,
        window=window,
        entry_threshold=entry_threshold,
    )
    result = invert_signal(with_signals, signal_column)

    # Original DataFrame unchanged.
    pdt.assert_frame_equal(df, df_before)

    # Required columns exist.
    for column in (
        "close",
        "close_return",
        "zscore",
        signal_column,
        inverted_column,
    ):
        assert column in result.columns

    # validate_signal_column succeeds.
    validate_signal_column(result, signal_column)
    validate_signal_column(result, inverted_column)

    # invert_signal creates zscore_signal_inverted.
    assert inverted_column in result.columns

    # Inverted signal is exactly the negative of the original signal.
    pdt.assert_series_equal(
        result[inverted_column],
        result[signal_column] * -1,
        check_names=False,
    )

    # count_signal_changes returns a non-negative integer.
    changes = count_signal_changes(result, signal_column)
    assert isinstance(changes, int)
    assert changes >= 0

    # summarize_signal returns long / flat / short.
    summary = summarize_signal(result, signal_column)
    assert list(summary.index) == ["long", "flat", "short"]

    # Summary counts sum to DataFrame length.
    assert int(summary.sum()) == len(result)

    # Signal values limited to LONG / FLAT / SHORT.
    allowed_signals = {LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL}
    assert set(result[signal_column].unique()).issubset(allowed_signals)
    assert set(result[inverted_column].unique()).issubset(allowed_signals)


def main() -> None:
    test_signal_utils_pipeline()

    print("🎉 SIGNAL UTILS PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
