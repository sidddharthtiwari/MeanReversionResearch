import copy

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features.zscore import compute_zscore


def _make_ohlc(
    closes: list[float],
    symbol: str = "A",
) -> pd.DataFrame:
    """Build a small deterministic single-symbol OHLC frame."""
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "date": dates,
            "close": closes,
        }
    )


def _make_multi_symbol() -> pd.DataFrame:
    """Build a deterministic two-symbol frame for leakage checks."""
    symbol_a = _make_ohlc([100.0, 110.0, 120.0, 130.0], symbol="A")
    symbol_b = _make_ohlc([10.0, 20.0, 30.0, 40.0], symbol="B")
    return pd.concat([symbol_a, symbol_b], ignore_index=True)


# --------------------------------------------------
# TEST 1
# Basic Z-Score
# --------------------------------------------------


def test_compute_zscore_basic() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0, 130.0, 140.0])
    window = 3
    result = compute_zscore(df, window=window, column="close")

    assert f"zscore_{window}" in result.columns
    assert len(result) == len(df)

    expected_mean = df["close"].rolling(window).mean()
    expected_std = df["close"].rolling(window).std()
    expected = (df["close"] - expected_mean) / expected_std

    pdt.assert_series_equal(
        result[f"zscore_{window}"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Multiple Symbols Are Independent
# --------------------------------------------------


def test_multiple_symbols_are_independent() -> None:
    df = _make_multi_symbol()
    window = 2
    result = compute_zscore(df, window=window, column="close")

    only_a = compute_zscore(
        df[df["symbol"] == "A"].reset_index(drop=True),
        window=window,
        column="close",
    )
    only_b = compute_zscore(
        df[df["symbol"] == "B"].reset_index(drop=True),
        window=window,
        column="close",
    )

    pdt.assert_series_equal(
        result.loc[result["symbol"] == "A", f"zscore_{window}"].reset_index(
            drop=True
        ),
        only_a[f"zscore_{window}"].reset_index(drop=True),
        check_names=False,
    )
    pdt.assert_series_equal(
        result.loc[result["symbol"] == "B", f"zscore_{window}"].reset_index(
            drop=True
        ),
        only_b[f"zscore_{window}"].reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# Window = 1
# --------------------------------------------------


def test_window_one() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0])
    result = compute_zscore(df, window=1, column="close")

    assert "zscore_1" in result.columns
    assert result["zscore_1"].isna().all()


# --------------------------------------------------
# TEST 4
# Window Larger Than Dataset
# --------------------------------------------------


def test_window_larger_than_dataset() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0])
    result = compute_zscore(df, window=50, column="close")

    assert result["zscore_50"].isna().all()


# --------------------------------------------------
# TEST 5
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0, 130.0])
    result = compute_zscore(
        df,
        window=2,
        column="close",
        output_column="custom_z",
    )

    assert "custom_z" in result.columns
    assert "zscore_2" not in result.columns


# --------------------------------------------------
# TEST 6
# Custom Input Column
# --------------------------------------------------


def test_custom_input_column() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0, 130.0, 140.0])
    df = df.rename(columns={"close": "spread"})
    window = 3

    result = compute_zscore(df, window=window, column="spread")

    expected_mean = df["spread"].rolling(window).mean()
    expected_std = df["spread"].rolling(window).std()
    expected = (df["spread"] - expected_mean) / expected_std

    pdt.assert_series_equal(
        result[f"zscore_{window}"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 7
# Temporary Columns Not Exposed
# --------------------------------------------------


def test_temporary_columns_not_exposed() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0, 130.0])
    result = compute_zscore(df, window=2, column="close")

    assert "__rolling_mean_temp__" not in result.columns
    assert "__rolling_std_temp__" not in result.columns


# --------------------------------------------------
# TEST 8
# Original Columns Preserved
# --------------------------------------------------


def test_original_columns_preserved() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0, 130.0])
    result = compute_zscore(df, window=2, column="close")

    expected_columns = list(df.columns) + ["zscore_2"]
    assert list(result.columns) == expected_columns


# --------------------------------------------------
# TEST 9
# Missing Symbol Column
# --------------------------------------------------


def test_missing_symbol_column() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0]).drop(columns=["symbol"])

    try:
        compute_zscore(df, window=2, column="close")
        raised = False
    except KeyError as error:
        raised = True
        assert error.args[0] == "Required column 'symbol' not found in DataFrame."

    assert raised


# --------------------------------------------------
# TEST 10
# Missing Numeric Column
# --------------------------------------------------


def test_missing_numeric_column() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0]).drop(columns=["close"])

    try:
        compute_zscore(df, window=2, column="close")
        raised = False
    except KeyError as error:
        raised = True
        assert "close" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Non-Numeric Column
# --------------------------------------------------


def test_non_numeric_column() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0])
    df["close"] = df["close"].astype(str)

    try:
        compute_zscore(df, window=2, column="close")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Invalid Window
# --------------------------------------------------


def test_invalid_window() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0])

    try:
        compute_zscore(df, window=0, column="close")
        raised = False
    except ValueError as error:
        raised = True
        assert "window must be > 0" in str(error)

    assert raised


# --------------------------------------------------
# TEST 13
# Invalid min_periods
# --------------------------------------------------


def test_invalid_min_periods() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0])

    try:
        compute_zscore(df, window=3, column="close", min_periods=0)
        raised = False
    except ValueError:
        raised = True

    assert raised


# --------------------------------------------------
# TEST 14
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_ohlc([100.0, 110.0, 120.0, 130.0])
    df_before = copy.deepcopy(df)

    compute_zscore(df, window=2, column="close")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 15
# Zero Standard Deviation
# --------------------------------------------------


def test_zero_standard_deviation() -> None:
    df = _make_ohlc([100.0, 100.0, 100.0, 100.0])
    result = compute_zscore(df, window=3, column="close")

    assert "zscore_3" in result.columns
    # Constant series => rolling std is 0; pandas yields NaN from 0/0.
    assert result["zscore_3"].isna().all()


# --------------------------------------------------
# TEST 16
# Matches Manual Formula
# --------------------------------------------------


def test_matches_manual_formula() -> None:
    df = _make_ohlc([100.0, 110.0, 121.0, 133.1, 146.41])
    window = 3

    result = compute_zscore(df, window=window, column="close")

    rolling_mean = df["close"].rolling(window=window).mean()
    rolling_std = df["close"].rolling(window=window).std()
    manual = (df["close"] - rolling_mean) / rolling_std

    pdt.assert_series_equal(
        result[f"zscore_{window}"],
        manual,
        check_names=False,
    )


if __name__ == "__main__":
    test_compute_zscore_basic()
    test_multiple_symbols_are_independent()
    test_window_one()
    test_window_larger_than_dataset()
    test_custom_output_column()
    test_custom_input_column()
    test_temporary_columns_not_exposed()
    test_original_columns_preserved()
    test_missing_symbol_column()
    test_missing_numeric_column()
    test_non_numeric_column()
    test_invalid_window()
    test_invalid_min_periods()
    test_input_dataframe_unchanged()
    test_zero_standard_deviation()
    test_matches_manual_formula()

    print("🎉 ALL Z-SCORE TESTS PASSED")
