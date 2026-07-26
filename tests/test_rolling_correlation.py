import copy

import pandas as pd
import pandas.testing as pdt

from src.features.rolling_correlation import compute_rolling_correlation


def _make_pair_frame(
    left: list[float],
    right: list[float],
    symbol: str = "A",
) -> pd.DataFrame:
    """Build a small deterministic single-symbol frame."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(left),
            "left": left,
            "right": right,
        }
    )


def _make_multi_symbol_frame() -> pd.DataFrame:
    """Build a deterministic two-symbol frame for isolation checks."""
    symbol_a = _make_pair_frame(
        left=[1.0, 2.0, 3.0, 4.0],
        right=[1.0, 2.0, 3.0, 4.0],
        symbol="A",
    )
    symbol_b = _make_pair_frame(
        left=[10.0, 20.0, 30.0, 40.0],
        right=[40.0, 30.0, 20.0, 10.0],
        symbol="B",
    )
    return pd.concat([symbol_a, symbol_b], ignore_index=True)


# --------------------------------------------------
# TEST 1
# Basic Rolling Correlation
# --------------------------------------------------


def test_basic_rolling_correlation() -> None:
    df = _make_pair_frame(
        left=[1.0, 2.0, 3.0, 4.0, 5.0],
        right=[2.0, 4.0, 5.0, 7.0, 8.0],
    )
    result = compute_rolling_correlation(
        df,
        window=3,
        left_column="left",
        right_column="right",
    )

    assert "rolling_correlation_3" in result.columns
    assert len(result) == len(df)


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_pair_frame(
        left=[1.0, 2.0, 3.0],
        right=[2.0, 4.0, 6.0],
    )
    result = compute_rolling_correlation(
        df,
        window=2,
        left_column="left",
        right_column="right",
        output_column="custom_corr",
    )

    assert "custom_corr" in result.columns
    assert "rolling_correlation_2" not in result.columns


# --------------------------------------------------
# TEST 3
# Group Isolation
# --------------------------------------------------


def test_group_isolation() -> None:
    df = _make_multi_symbol_frame()
    window = 3
    result = compute_rolling_correlation(
        df,
        window=window,
        left_column="left",
        right_column="right",
    )
    column = f"rolling_correlation_{window}"

    only_a = compute_rolling_correlation(
        df[df["symbol"] == "A"].reset_index(drop=True),
        window=window,
        left_column="left",
        right_column="right",
    )
    only_b = compute_rolling_correlation(
        df[df["symbol"] == "B"].reset_index(drop=True),
        window=window,
        left_column="left",
        right_column="right",
    )

    pdt.assert_series_equal(
        result.loc[result["symbol"] == "A", column].reset_index(drop=True),
        only_a[column].reset_index(drop=True),
        check_names=False,
    )
    pdt.assert_series_equal(
        result.loc[result["symbol"] == "B", column].reset_index(drop=True),
        only_b[column].reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Matches Manual Formula
# --------------------------------------------------


def test_matches_manual_formula() -> None:
    df = _make_pair_frame(
        left=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        right=[2.0, 4.0, 5.0, 7.0, 8.0, 11.0],
    )
    window = 3
    result = compute_rolling_correlation(
        df,
        window=window,
        left_column="left",
        right_column="right",
    )
    expected = df["left"].rolling(window=window).corr(df["right"])

    pdt.assert_series_equal(
        result[f"rolling_correlation_{window}"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Invalid Window
# --------------------------------------------------


def test_invalid_window() -> None:
    df = _make_pair_frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

    try:
        compute_rolling_correlation(
            df,
            window=0,
            left_column="left",
            right_column="right",
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "window must be > 0" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Invalid min_periods
# --------------------------------------------------


def test_invalid_min_periods() -> None:
    df = _make_pair_frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

    try:
        compute_rolling_correlation(
            df,
            window=3,
            left_column="left",
            right_column="right",
            min_periods=0,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised


# --------------------------------------------------
# TEST 7
# Missing Left Column
# --------------------------------------------------


def test_missing_left_column() -> None:
    df = _make_pair_frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

    try:
        compute_rolling_correlation(
            df,
            window=2,
            left_column="missing",
            right_column="right",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Missing Right Column
# --------------------------------------------------


def test_missing_right_column() -> None:
    df = _make_pair_frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

    try:
        compute_rolling_correlation(
            df,
            window=2,
            left_column="left",
            right_column="missing",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Non-Numeric Left Column
# --------------------------------------------------


def test_non_numeric_left_column() -> None:
    df = _make_pair_frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    df["left"] = df["left"].astype(str)

    try:
        compute_rolling_correlation(
            df,
            window=2,
            left_column="left",
            right_column="right",
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# Non-Numeric Right Column
# --------------------------------------------------


def test_non_numeric_right_column() -> None:
    df = _make_pair_frame([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    df["right"] = df["right"].astype(str)

    try:
        compute_rolling_correlation(
            df,
            window=2,
            left_column="left",
            right_column="right",
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_pair_frame(
        left=[1.0, 2.0, 3.0, 4.0],
        right=[2.0, 4.0, 5.0, 7.0],
    )
    df_before = copy.deepcopy(df)

    compute_rolling_correlation(
        df,
        window=3,
        left_column="left",
        right_column="right",
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 12
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_pair_frame(
        left=[1.0, 2.0, 3.0],
        right=[2.0, 4.0, 6.0],
    )
    result = compute_rolling_correlation(
        df,
        window=2,
        left_column="left",
        right_column="right",
    )

    expected_columns = list(df.columns) + ["rolling_correlation_2"]
    assert list(result.columns) == expected_columns


def main() -> None:
    test_basic_rolling_correlation()
    test_custom_output_column()
    test_group_isolation()
    test_matches_manual_formula()
    test_invalid_window()
    test_invalid_min_periods()
    test_missing_left_column()
    test_missing_right_column()
    test_non_numeric_left_column()
    test_non_numeric_right_column()
    test_input_dataframe_unchanged()
    test_output_schema()

    print("🎉 ALL ROLLING CORRELATION TESTS PASSED")


if __name__ == "__main__":
    main()
