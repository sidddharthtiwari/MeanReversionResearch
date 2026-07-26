import copy

import pandas as pd
import pandas.testing as pdt

from src.features.atr import compute_atr


def _make_ohlc_frame(
    high: list[float],
    low: list[float],
    close: list[float],
    symbol: str = "A",
) -> pd.DataFrame:
    """Build a small deterministic single-symbol OHLC frame."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(high),
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _make_multi_symbol_frame() -> pd.DataFrame:
    """Build a deterministic two-symbol frame for isolation checks."""
    symbol_a = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0, 16.0],
        low=[10.0, 11.0, 12.0, 13.0],
        close=[11.0, 14.0, 13.0, 15.0],
        symbol="A",
    )
    symbol_b = _make_ohlc_frame(
        high=[100.0, 110.0, 105.0, 120.0],
        low=[90.0, 95.0, 100.0, 110.0],
        close=[95.0, 105.0, 102.0, 115.0],
        symbol="B",
    )
    return pd.concat([symbol_a, symbol_b], ignore_index=True)


# --------------------------------------------------
# TEST 1
# Basic ATR
# --------------------------------------------------


def test_basic_atr() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0, 16.0],
        low=[10.0, 11.0, 12.0, 13.0],
        close=[11.0, 14.0, 13.0, 15.0],
    )
    result = compute_atr(df, window=3)

    assert "atr_3" in result.columns
    assert len(result) == len(df)


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )
    result = compute_atr(df, window=2, output_column="custom_atr")

    assert "custom_atr" in result.columns
    assert "atr_2" not in result.columns


# --------------------------------------------------
# TEST 3
# Group Isolation
# --------------------------------------------------


def test_group_isolation() -> None:
    df = _make_multi_symbol_frame()
    window = 3
    result = compute_atr(df, window=window)
    column = f"atr_{window}"

    only_a = compute_atr(
        df[df["symbol"] == "A"].reset_index(drop=True),
        window=window,
    )
    only_b = compute_atr(
        df[df["symbol"] == "B"].reset_index(drop=True),
        window=window,
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
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0, 16.0, 17.0],
        low=[10.0, 11.0, 12.0, 13.0, 14.0],
        close=[11.0, 14.0, 13.0, 15.0, 16.0],
    )
    window = 3
    result = compute_atr(df, window=window)

    previous_close = df.groupby("symbol", sort=False)["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected = true_range.rolling(window=window).mean()

    pdt.assert_series_equal(
        result[f"atr_{window}"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Invalid Window
# --------------------------------------------------


def test_invalid_window() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )

    try:
        compute_atr(df, window=0)
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
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )

    try:
        compute_atr(df, window=3, min_periods=0)
        raised = False
    except ValueError:
        raised = True

    assert raised


# --------------------------------------------------
# TEST 7
# Missing Symbol Column
# --------------------------------------------------


def test_missing_symbol_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["symbol"])

    try:
        compute_atr(df, window=2)
        raised = False
    except KeyError as error:
        raised = True
        assert error.args[0] == "Required column 'symbol' not found in DataFrame."

    assert raised


# --------------------------------------------------
# TEST 8
# Missing High Column
# --------------------------------------------------


def test_missing_high_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["high"])

    try:
        compute_atr(df, window=2)
        raised = False
    except KeyError as error:
        raised = True
        assert "high" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Missing Low Column
# --------------------------------------------------


def test_missing_low_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["low"])

    try:
        compute_atr(df, window=2)
        raised = False
    except KeyError as error:
        raised = True
        assert "low" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# Missing Close Column
# --------------------------------------------------


def test_missing_close_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["close"])

    try:
        compute_atr(df, window=2)
        raised = False
    except KeyError as error:
        raised = True
        assert "close" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Input DataFrame Unchanged
# --------------------------------------------------


def test_input_dataframe_unchanged() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )
    df_before = copy.deepcopy(df)

    compute_atr(df, window=2)

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 12
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )
    result = compute_atr(df, window=2)

    expected_columns = list(df.columns) + ["atr_2"]
    assert list(result.columns) == expected_columns


def main() -> None:
    test_basic_atr()
    test_custom_output_column()
    test_group_isolation()
    test_matches_manual_formula()
    test_invalid_window()
    test_invalid_min_periods()
    test_missing_symbol_column()
    test_missing_high_column()
    test_missing_low_column()
    test_missing_close_column()
    test_input_dataframe_unchanged()
    test_output_schema()

    print("🎉 ALL ATR TESTS PASSED")


if __name__ == "__main__":
    main()
