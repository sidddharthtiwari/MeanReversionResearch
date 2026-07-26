import copy

import pandas as pd
import pandas.testing as pdt

from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.threshold import generate_threshold_signal


def _make_feature_frame(
    values: list[float],
    feature_column: str = "feature",
) -> pd.DataFrame:
    """Build a small deterministic single-column feature frame."""
    return pd.DataFrame({feature_column: values})


# --------------------------------------------------
# TEST 1
# Basic Threshold Signal
# --------------------------------------------------


def test_basic_threshold_signal() -> None:
    df = _make_feature_frame([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = generate_threshold_signal(
        df,
        feature_column="feature",
        buy_threshold=-1.0,
        sell_threshold=1.0,
    )

    expected = pd.Series(
        [LONG_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL, SHORT_SIGNAL],
        dtype="int64",
    )
    assert "feature_signal" in result.columns
    assert len(result) == len(df)
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Buy Only Threshold
# --------------------------------------------------


def test_buy_only_threshold() -> None:
    df = _make_feature_frame([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = generate_threshold_signal(
        df,
        feature_column="feature",
        buy_threshold=0.0,
    )

    expected = pd.Series(
        [LONG_SIGNAL, LONG_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# Sell Only Threshold
# --------------------------------------------------


def test_sell_only_threshold() -> None:
    df = _make_feature_frame([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = generate_threshold_signal(
        df,
        feature_column="feature",
        sell_threshold=0.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL, SHORT_SIGNAL, SHORT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0])
    result = generate_threshold_signal(
        df,
        feature_column="feature",
        buy_threshold=-1.0,
        sell_threshold=1.0,
        output_column="custom_signal",
    )

    assert "custom_signal" in result.columns
    assert "feature_signal" not in result.columns


# --------------------------------------------------
# TEST 5
# Default Output Column
# --------------------------------------------------


def test_default_output_column() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0], feature_column="zscore")
    result = generate_threshold_signal(
        df,
        feature_column="zscore",
        buy_threshold=-1.0,
        sell_threshold=1.0,
        output_column=None,
    )

    assert "zscore_signal" in result.columns


# --------------------------------------------------
# TEST 6
# Invalid Threshold Order
# --------------------------------------------------


def test_invalid_threshold_order() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0])

    try:
        generate_threshold_signal(
            df,
            feature_column="feature",
            buy_threshold=1.0,
            sell_threshold=0.0,
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "buy_threshold" in str(error)
        assert "sell_threshold" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Missing Feature Column
# --------------------------------------------------


def test_missing_feature_column() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0])

    try:
        generate_threshold_signal(
            df,
            feature_column="missing",
            buy_threshold=-1.0,
            sell_threshold=1.0,
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Non-Numeric Feature
# --------------------------------------------------


def test_non_numeric_feature() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0])
    df["feature"] = df["feature"].astype(str)

    try:
        generate_threshold_signal(
            df,
            feature_column="feature",
            buy_threshold=-1.0,
            sell_threshold=1.0,
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# No Thresholds
# --------------------------------------------------


def test_no_thresholds() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0])

    try:
        generate_threshold_signal(df, feature_column="feature")
        raised = False
    except ValueError as error:
        raised = True
        assert "buy_threshold" in str(error) or "sell_threshold" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_feature_frame([-2.0, -1.0, 0.0, 1.0, 2.0])
    df_before = copy.deepcopy(df)

    generate_threshold_signal(
        df,
        feature_column="feature",
        buy_threshold=-1.0,
        sell_threshold=1.0,
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 11
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_feature_frame([-1.0, 0.0, 1.0])
    result = generate_threshold_signal(
        df,
        feature_column="feature",
        buy_threshold=-1.0,
        sell_threshold=1.0,
    )

    expected_columns = list(df.columns) + ["feature_signal"]
    assert list(result.columns) == expected_columns


# --------------------------------------------------
# TEST 12
# Boundary Conditions
# --------------------------------------------------


def test_boundary_conditions() -> None:
    buy_threshold = -1.0
    sell_threshold = 1.0
    df = _make_feature_frame(
        [
            buy_threshold,
            buy_threshold - 0.1,
            0.0,
            sell_threshold,
            sell_threshold + 0.1,
        ]
    )
    result = generate_threshold_signal(
        df,
        feature_column="feature",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # value <= buy_threshold -> LONG
    # value >= sell_threshold -> SHORT
    expected = pd.Series(
        [LONG_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL, SHORT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


def main() -> None:
    test_basic_threshold_signal()
    test_buy_only_threshold()
    test_sell_only_threshold()
    test_custom_output_column()
    test_default_output_column()
    test_invalid_threshold_order()
    test_missing_feature_column()
    test_non_numeric_feature()
    test_no_thresholds()
    test_input_dataframe_unchanged()
    test_output_schema()
    test_boundary_conditions()

    print("🎉 ALL THRESHOLD SIGNAL TESTS PASSED")


if __name__ == "__main__":
    main()
