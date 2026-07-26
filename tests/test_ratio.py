import copy

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features.ratio import compute_ratio


def _make_pair_frame(
    left: list[float],
    right: list[float],
) -> pd.DataFrame:
    """Build a small deterministic two-column frame for ratio tests."""
    return pd.DataFrame(
        {
            "left": left,
            "right": right,
        }
    )


# --------------------------------------------------
# TEST 1
# Basic Ratio
# --------------------------------------------------


def test_compute_ratio_basic() -> None:
    df = _make_pair_frame([10.0, 20.0, 30.0], [2.0, 5.0, 10.0])
    result = compute_ratio(df, left_column="left", right_column="right")

    assert "ratio" in result.columns
    assert len(result) == len(df)
    pdt.assert_series_equal(
        result["ratio"],
        df["left"] / df["right"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [2.0, 4.0])
    result = compute_ratio(
        df,
        left_column="left",
        right_column="right",
        output_column="price_ratio",
    )

    assert "price_ratio" in result.columns
    assert "ratio" not in result.columns


# --------------------------------------------------
# TEST 3
# Division by Zero
# --------------------------------------------------


def test_division_by_zero() -> None:
    df = _make_pair_frame([10.0, 20.0, 0.0], [2.0, 0.0, 0.0])
    result = compute_ratio(df, left_column="left", right_column="right")

    assert "ratio" in result.columns
    assert np.isinf(result["ratio"].iloc[1]) or np.isnan(result["ratio"].iloc[1])
    assert np.isnan(result["ratio"].iloc[2]) or np.isinf(result["ratio"].iloc[2])


# --------------------------------------------------
# TEST 4
# Missing Left Column
# --------------------------------------------------


def test_missing_left_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [2.0, 4.0])

    try:
        compute_ratio(df, left_column="missing", right_column="right")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 5
# Missing Right Column
# --------------------------------------------------


def test_missing_right_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [2.0, 4.0])

    try:
        compute_ratio(df, left_column="left", right_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Non-Numeric Left Column
# --------------------------------------------------


def test_non_numeric_left_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [2.0, 4.0])
    df["left"] = df["left"].astype(str)

    try:
        compute_ratio(df, left_column="left", right_column="right")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Non-Numeric Right Column
# --------------------------------------------------


def test_non_numeric_right_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [2.0, 4.0])
    df["right"] = df["right"].astype(str)

    try:
        compute_ratio(df, left_column="left", right_column="right")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_pair_frame([10.0, 20.0, 30.0], [2.0, 5.0, 10.0])
    df_before = copy.deepcopy(df)

    compute_ratio(df, left_column="left", right_column="right")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 9
# Matches Manual Formula
# --------------------------------------------------


def test_matches_manual_formula() -> None:
    df = _make_pair_frame([100.0, 110.0, 121.0], [10.0, 20.0, 11.0])
    result = compute_ratio(df, left_column="left", right_column="right")
    manual = df["left"] / df["right"]

    pdt.assert_series_equal(
        result["ratio"],
        manual,
        check_names=False,
    )


# --------------------------------------------------
# TEST 10
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_pair_frame([10.0, 20.0], [2.0, 4.0])
    result = compute_ratio(df, left_column="left", right_column="right")

    expected_columns = list(df.columns) + ["ratio"]
    assert list(result.columns) == expected_columns


def main() -> None:
    test_compute_ratio_basic()
    test_custom_output_column()
    test_division_by_zero()
    test_missing_left_column()
    test_missing_right_column()
    test_non_numeric_left_column()
    test_non_numeric_right_column()
    test_input_dataframe_unchanged()
    test_matches_manual_formula()
    test_output_schema()

    print("🎉 ALL RATIO TESTS PASSED")


if __name__ == "__main__":
    main()
