import copy

import pandas as pd
import pandas.testing as pdt

from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.mean_reversion import generate_mean_reversion_signal


def _make_feature_frame(
    values: list[float],
    feature_column: str = "feature",
) -> pd.DataFrame:
    """Build a small deterministic single-column feature frame."""
    return pd.DataFrame({feature_column: values})


# --------------------------------------------------
# TEST 1
# Long Signal
# --------------------------------------------------


def test_long_signal() -> None:
    df = _make_feature_frame([-2.5, -2.0])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    expected = pd.Series([LONG_SIGNAL, LONG_SIGNAL], dtype="int64")
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Short Signal
# --------------------------------------------------


def test_short_signal() -> None:
    df = _make_feature_frame([2.0, 2.5])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    expected = pd.Series([SHORT_SIGNAL, SHORT_SIGNAL], dtype="int64")
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# Flat Signal
# --------------------------------------------------


def test_flat_signal() -> None:
    df = _make_feature_frame([-0.5, 0.0, 0.5])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Negative Threshold Boundary
# --------------------------------------------------


def test_negative_threshold_boundary() -> None:
    entry_threshold = 1.5
    df = _make_feature_frame([-entry_threshold])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=entry_threshold,
    )

    assert result["feature_signal"].iloc[0] == LONG_SIGNAL


# --------------------------------------------------
# TEST 5
# Positive Threshold Boundary
# --------------------------------------------------


def test_positive_threshold_boundary() -> None:
    entry_threshold = 1.5
    df = _make_feature_frame([entry_threshold])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=entry_threshold,
    )

    assert result["feature_signal"].iloc[0] == SHORT_SIGNAL


# --------------------------------------------------
# TEST 6
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
        output_column="custom_mr_signal",
    )

    assert "custom_mr_signal" in result.columns
    assert "feature_signal" not in result.columns


# --------------------------------------------------
# TEST 7
# Default Output Column
# --------------------------------------------------


def test_default_output_column() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0], feature_column="zscore")
    result = generate_mean_reversion_signal(
        df,
        feature_column="zscore",
        entry_threshold=1.0,
        output_column=None,
    )

    assert "zscore_signal" in result.columns


# --------------------------------------------------
# TEST 8
# Missing Feature Column
# --------------------------------------------------


def test_missing_feature_column() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0])

    try:
        generate_mean_reversion_signal(
            df,
            feature_column="missing",
            entry_threshold=1.0,
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Non-Numeric Feature Column
# --------------------------------------------------


def test_non_numeric_feature_column() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0])
    df["feature"] = df["feature"].astype(str)

    try:
        generate_mean_reversion_signal(
            df,
            feature_column="feature",
            entry_threshold=1.0,
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# Invalid Entry Threshold
# --------------------------------------------------


def test_invalid_entry_threshold() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0])

    for invalid_threshold in (0, -1.0):
        try:
            generate_mean_reversion_signal(
                df,
                feature_column="feature",
                entry_threshold=invalid_threshold,
            )
            raised = False
        except ValueError as error:
            raised = True
            assert "entry_threshold" in str(error)

        assert raised


# --------------------------------------------------
# TEST 11
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0])
    df_before = copy.deepcopy(df)

    generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 12
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_feature_frame([-2.0, 0.0, 2.0])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    expected_columns = list(df.columns) + ["feature_signal"]
    assert list(result.columns) == expected_columns


# --------------------------------------------------
# TEST 13
# All Flat
# --------------------------------------------------


def test_all_flat() -> None:
    df = _make_feature_frame([-0.9, -0.5, 0.0, 0.5, 0.9])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    expected = pd.Series(
        [FLAT_SIGNAL] * len(df),
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 14
# All Long
# --------------------------------------------------


def test_all_long() -> None:
    df = _make_feature_frame([-3.0, -2.5, -2.0, -1.5])
    result = generate_mean_reversion_signal(
        df,
        feature_column="feature",
        entry_threshold=1.0,
    )

    expected = pd.Series(
        [LONG_SIGNAL] * len(df),
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["feature_signal"],
        expected,
        check_names=False,
    )


def main() -> None:
    test_long_signal()
    test_short_signal()
    test_flat_signal()
    test_negative_threshold_boundary()
    test_positive_threshold_boundary()
    test_custom_output_column()
    test_default_output_column()
    test_missing_feature_column()
    test_non_numeric_feature_column()
    test_invalid_entry_threshold()
    test_input_dataframe_unchanged()
    test_output_schema()
    test_all_flat()
    test_all_long()

    print("🎉 ALL MEAN REVERSION SIGNAL TESTS PASSED")


if __name__ == "__main__":
    main()
