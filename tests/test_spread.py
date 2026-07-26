import copy

import pandas as pd
import pandas.testing as pdt

from src.features.spread import compute_spread


def _make_pair_frame(
    left: list[float],
    right: list[float],
) -> pd.DataFrame:
    """Build a small deterministic two-column frame for spread tests."""
    return pd.DataFrame(
        {
            "left": left,
            "right": right,
        }
    )


# --------------------------------------------------
# TEST 1
# Basic Spread
# --------------------------------------------------


def test_compute_spread_basic() -> None:
    df = _make_pair_frame([10.0, 20.0, 30.0], [3.0, 5.0, 8.0])
    result = compute_spread(df, left_column="left", right_column="right")

    assert "spread" in result.columns
    assert len(result) == len(df)
    pdt.assert_series_equal(
        result["spread"],
        df["left"] - df["right"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [1.0, 2.0])
    result = compute_spread(
        df,
        left_column="left",
        right_column="right",
        output_column="bank_spread",
    )

    assert "bank_spread" in result.columns
    assert "spread" not in result.columns


# --------------------------------------------------
# TEST 3
# Negative Spread
# --------------------------------------------------


def test_negative_spread() -> None:
    df = _make_pair_frame([5.0, 10.0, 15.0], [10.0, 20.0, 25.0])
    result = compute_spread(df, left_column="left", right_column="right")

    expected = pd.Series([-5.0, -10.0, -10.0], dtype="float64")
    pdt.assert_series_equal(
        result["spread"].reset_index(drop=True),
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Missing Left Column
# --------------------------------------------------


def test_missing_left_column() -> None:
    df = _make_pair_frame([10.0, 20.0], [1.0, 2.0])

    try:
        compute_spread(df, left_column="missing", right_column="right")
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
    df = _make_pair_frame([10.0, 20.0], [1.0, 2.0])

    try:
        compute_spread(df, left_column="left", right_column="missing")
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
    df = _make_pair_frame([10.0, 20.0], [1.0, 2.0])
    df["left"] = df["left"].astype(str)

    try:
        compute_spread(df, left_column="left", right_column="right")
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
    df = _make_pair_frame([10.0, 20.0], [1.0, 2.0])
    df["right"] = df["right"].astype(str)

    try:
        compute_spread(df, left_column="left", right_column="right")
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
    df = _make_pair_frame([10.0, 20.0, 30.0], [3.0, 5.0, 8.0])
    df_before = copy.deepcopy(df)

    compute_spread(df, left_column="left", right_column="right")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 9
# Matches Manual Formula
# --------------------------------------------------


def test_matches_manual_formula() -> None:
    df = _make_pair_frame([100.0, 110.0, 121.0], [90.0, 95.0, 100.0])
    result = compute_spread(df, left_column="left", right_column="right")
    manual = df["left"] - df["right"]

    pdt.assert_series_equal(
        result["spread"],
        manual,
        check_names=False,
    )


# --------------------------------------------------
# TEST 10
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_pair_frame([10.0, 20.0], [1.0, 2.0])
    result = compute_spread(df, left_column="left", right_column="right")

    expected_columns = list(df.columns) + ["spread"]
    assert list(result.columns) == expected_columns


def main() -> None:
    test_compute_spread_basic()
    test_custom_output_column()
    test_negative_spread()
    test_missing_left_column()
    test_missing_right_column()
    test_non_numeric_left_column()
    test_non_numeric_right_column()
    test_input_dataframe_unchanged()
    test_matches_manual_formula()
    test_output_schema()

    print("🎉 ALL SPREAD TESTS PASSED")


if __name__ == "__main__":
    main()
