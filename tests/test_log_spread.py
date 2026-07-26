import copy

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features.log_spread import compute_log_spread


def _make_pair_frame(
    left: list[float],
    right: list[float],
) -> pd.DataFrame:
    """Build a small deterministic two-column frame for log-spread tests."""
    return pd.DataFrame(
        {
            "left": left,
            "right": right,
        }
    )


# --------------------------------------------------
# TEST 1
# Basic Log Spread
# --------------------------------------------------


def test_compute_log_spread_basic() -> None:
    df = _make_pair_frame([100.0, 110.0, 121.0], [50.0, 55.0, 60.0])
    result = compute_log_spread(df, left_column="left", right_column="right")

    assert "log_spread" in result.columns
    assert len(result) == len(df)
    pdt.assert_series_equal(
        result["log_spread"],
        np.log(df["left"]) - np.log(df["right"]),
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_pair_frame([100.0, 110.0], [50.0, 55.0])
    result = compute_log_spread(
        df,
        left_column="left",
        right_column="right",
        output_column="custom_log_spread",
    )

    assert "custom_log_spread" in result.columns
    assert "log_spread" not in result.columns


# --------------------------------------------------
# TEST 3
# Zero Values
# --------------------------------------------------


def test_zero_values() -> None:
    df = _make_pair_frame([100.0, 0.0], [50.0, 55.0])

    try:
        compute_log_spread(df, left_column="left", right_column="right")
        raised = False
    except ValueError as error:
        raised = True
        assert "strictly positive" in str(error)

    assert raised


# --------------------------------------------------
# TEST 4
# Negative Values
# --------------------------------------------------


def test_negative_values() -> None:
    df = _make_pair_frame([100.0, -10.0], [50.0, 55.0])

    try:
        compute_log_spread(df, left_column="left", right_column="right")
        raised = False
    except ValueError as error:
        raised = True
        assert "strictly positive" in str(error)

    assert raised


# --------------------------------------------------
# TEST 5
# Missing Left Column
# --------------------------------------------------


def test_missing_left_column() -> None:
    df = _make_pair_frame([100.0, 110.0], [50.0, 55.0])

    try:
        compute_log_spread(df, left_column="missing", right_column="right")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Missing Right Column
# --------------------------------------------------


def test_missing_right_column() -> None:
    df = _make_pair_frame([100.0, 110.0], [50.0, 55.0])

    try:
        compute_log_spread(df, left_column="left", right_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Non-Numeric Left Column
# --------------------------------------------------


def test_non_numeric_left_column() -> None:
    df = _make_pair_frame([100.0, 110.0], [50.0, 55.0])
    df["left"] = df["left"].astype(str)

    try:
        compute_log_spread(df, left_column="left", right_column="right")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Non-Numeric Right Column
# --------------------------------------------------


def test_non_numeric_right_column() -> None:
    df = _make_pair_frame([100.0, 110.0], [50.0, 55.0])
    df["right"] = df["right"].astype(str)

    try:
        compute_log_spread(df, left_column="left", right_column="right")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_pair_frame([100.0, 110.0, 121.0], [50.0, 55.0, 60.0])
    df_before = copy.deepcopy(df)

    compute_log_spread(df, left_column="left", right_column="right")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 10
# Matches Manual Formula and Schema
# --------------------------------------------------


def test_matches_manual_formula_and_schema() -> None:
    df = _make_pair_frame([100.0, 110.0, 121.0], [50.0, 55.0, 60.0])
    result = compute_log_spread(df, left_column="left", right_column="right")
    expected = np.log(df["left"]) - np.log(df["right"])

    pdt.assert_series_equal(
        result["log_spread"],
        expected,
        check_names=False,
    )
    assert list(result.columns) == list(df.columns) + ["log_spread"]


def main() -> None:
    test_compute_log_spread_basic()
    test_custom_output_column()
    test_zero_values()
    test_negative_values()
    test_missing_left_column()
    test_missing_right_column()
    test_non_numeric_left_column()
    test_non_numeric_right_column()
    test_input_dataframe_unchanged()
    test_matches_manual_formula_and_schema()

    print("🎉 ALL LOG SPREAD TESTS PASSED")


if __name__ == "__main__":
    main()
