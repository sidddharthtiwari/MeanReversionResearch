import copy

import pandas as pd
import pandas.testing as pdt

from src.signals.breakout import generate_breakout_signal
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL


def _make_feature_frame(
    values: list[float],
    feature_column: str = "feature",
) -> pd.DataFrame:
    """Build a small deterministic single-column feature frame."""
    return pd.DataFrame({feature_column: values})


# --------------------------------------------------
# TEST 1
# Bullish Breakout
# --------------------------------------------------


def test_bullish_breakout() -> None:
    df = _make_feature_frame([0.5, 0.8, 1.5])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, LONG_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Bearish Breakdown
# --------------------------------------------------


def test_bearish_breakdown() -> None:
    df = _make_feature_frame([-0.5, -0.8, -1.5])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        lower_threshold=-1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# No Breakout
# --------------------------------------------------


def test_no_breakout() -> None:
    df = _make_feature_frame([0.0, 0.2, 0.5, 0.8])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
        lower_threshold=-1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# First Row Flat
# --------------------------------------------------


def test_first_row_flat() -> None:
    df = _make_feature_frame([2.0, 3.0])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
    )

    assert result["feature_breakout_signal"].iloc[0] == FLAT_SIGNAL


# --------------------------------------------------
# TEST 5
# Upper Threshold Only
# --------------------------------------------------


def test_upper_threshold_only() -> None:
    df = _make_feature_frame([0.5, 1.5, -2.0])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Lower Threshold Only
# --------------------------------------------------


def test_lower_threshold_only() -> None:
    df = _make_feature_frame([-0.5, -1.5, 2.0])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        lower_threshold=-1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 7
# Both Thresholds
# --------------------------------------------------


def test_both_thresholds() -> None:
    df = _make_feature_frame([0.0, 1.5, 0.0, -1.5])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
        lower_threshold=-1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 8
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_feature_frame([0.5, 1.5])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
        output_column="custom_breakout",
    )

    assert "custom_breakout" in result.columns
    assert "feature_breakout_signal" not in result.columns


# --------------------------------------------------
# TEST 9
# Default Output Column
# --------------------------------------------------


def test_default_output_column() -> None:
    df = _make_feature_frame([0.5, 1.5], feature_column="close")
    result = generate_breakout_signal(
        df,
        feature_column="close",
        upper_threshold=1.0,
        output_column=None,
    )

    assert "close_breakout_signal" in result.columns


# --------------------------------------------------
# TEST 10
# Missing Feature Column
# --------------------------------------------------


def test_missing_feature_column() -> None:
    df = _make_feature_frame([0.5, 1.5])

    try:
        generate_breakout_signal(
            df,
            feature_column="missing",
            upper_threshold=1.0,
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Non-Numeric Feature Column
# --------------------------------------------------


def test_non_numeric_feature_column() -> None:
    df = _make_feature_frame([0.5, 1.5])
    df["feature"] = df["feature"].astype(str)

    try:
        generate_breakout_signal(
            df,
            feature_column="feature",
            upper_threshold=1.0,
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Missing Thresholds
# --------------------------------------------------


def test_missing_thresholds() -> None:
    df = _make_feature_frame([0.5, 1.5])

    try:
        generate_breakout_signal(df, feature_column="feature")
        raised = False
    except ValueError as error:
        raised = True
        assert "upper_threshold" in str(error) or "lower_threshold" in str(error)

    assert raised


# --------------------------------------------------
# TEST 13
# Invalid Threshold Order
# --------------------------------------------------


def test_invalid_threshold_order() -> None:
    df = _make_feature_frame([0.5, 1.5])

    try:
        generate_breakout_signal(
            df,
            feature_column="feature",
            upper_threshold=0.0,
            lower_threshold=1.0,
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "lower_threshold" in str(error)
        assert "upper_threshold" in str(error)

    assert raised


# --------------------------------------------------
# TEST 14
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_feature_frame([0.5, 1.5, -1.5])
    df_before = copy.deepcopy(df)

    generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
        lower_threshold=-1.0,
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 15
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_feature_frame([0.5, 1.5])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
    )

    expected_columns = list(df.columns) + ["feature_breakout_signal"]
    assert list(result.columns) == expected_columns


# --------------------------------------------------
# TEST 16
# Touch Without Breakout
# --------------------------------------------------


def test_touch_without_breakout() -> None:
    df = _make_feature_frame([0.5, 1.0, 1.0])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
    )

    # Equality with the threshold must not generate a breakout event.
    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 17
# Multiple Breakout Events
# --------------------------------------------------


def test_multiple_breakout_events() -> None:
    df = _make_feature_frame([0.5, 1.5, 0.5, 1.2, 0.8, 2.0])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        upper_threshold=1.0,
    )

    expected = pd.Series(
        [
            FLAT_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
            LONG_SIGNAL,
        ],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 18
# Multiple Breakdown Events
# --------------------------------------------------


def test_multiple_breakdown_events() -> None:
    df = _make_feature_frame([-0.5, -1.5, -0.5, -1.2, -0.8, -2.0])
    result = generate_breakout_signal(
        df,
        feature_column="feature",
        lower_threshold=-1.0,
    )

    expected = pd.Series(
        [
            FLAT_SIGNAL,
            SHORT_SIGNAL,
            FLAT_SIGNAL,
            SHORT_SIGNAL,
            FLAT_SIGNAL,
            SHORT_SIGNAL,
        ],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_breakout_signal"],
        expected,
        check_names=False,
    )


def main() -> None:
    test_bullish_breakout()
    test_bearish_breakdown()
    test_no_breakout()
    test_first_row_flat()
    test_upper_threshold_only()
    test_lower_threshold_only()
    test_both_thresholds()
    test_custom_output_column()
    test_default_output_column()
    test_missing_feature_column()
    test_non_numeric_feature_column()
    test_missing_thresholds()
    test_invalid_threshold_order()
    test_input_dataframe_unchanged()
    test_output_schema()
    test_touch_without_breakout()
    test_multiple_breakout_events()
    test_multiple_breakdown_events()

    print("🎉 ALL BREAKOUT SIGNAL TESTS PASSED")


if __name__ == "__main__":
    main()
