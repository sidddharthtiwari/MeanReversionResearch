import copy

import pandas as pd
import pandas.testing as pdt

from src.features.true_range import compute_true_range


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
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
        symbol="A",
    )
    symbol_b = _make_ohlc_frame(
        high=[100.0, 110.0, 105.0],
        low=[90.0, 95.0, 100.0],
        close=[95.0, 105.0, 102.0],
        symbol="B",
    )
    return pd.concat([symbol_a, symbol_b], ignore_index=True)


# --------------------------------------------------
# TEST 1
# Basic True Range
# --------------------------------------------------


def test_basic_true_range() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )
    result = compute_true_range(df)

    assert "true_range" in result.columns
    assert len(result) == len(df)


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    )
    result = compute_true_range(df, output_column="custom_tr")

    assert "custom_tr" in result.columns
    assert "true_range" not in result.columns


# --------------------------------------------------
# TEST 3
# Group Isolation
# --------------------------------------------------


def test_group_isolation() -> None:
    df = _make_multi_symbol_frame()
    result = compute_true_range(df)

    only_a = compute_true_range(
        df[df["symbol"] == "A"].reset_index(drop=True),
    )
    only_b = compute_true_range(
        df[df["symbol"] == "B"].reset_index(drop=True),
    )

    pdt.assert_series_equal(
        result.loc[result["symbol"] == "A", "true_range"].reset_index(drop=True),
        only_a["true_range"].reset_index(drop=True),
        check_names=False,
    )
    pdt.assert_series_equal(
        result.loc[result["symbol"] == "B", "true_range"].reset_index(drop=True),
        only_b["true_range"].reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Matches Manual Formula
# --------------------------------------------------


def test_matches_manual_formula() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0, 16.0],
        low=[10.0, 11.0, 12.0, 13.0],
        close=[11.0, 14.0, 13.0, 15.0],
    )
    result = compute_true_range(df)

    previous_close = df.groupby("symbol", sort=False)["close"].shift(1)
    expected = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    pdt.assert_series_equal(
        result["true_range"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Missing Symbol Column
# --------------------------------------------------


def test_missing_symbol_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["symbol"])

    try:
        compute_true_range(df)
        raised = False
    except KeyError as error:
        raised = True
        assert error.args[0] == "Required column 'symbol' not found in DataFrame."

    assert raised


# --------------------------------------------------
# TEST 6
# Missing High Column
# --------------------------------------------------


def test_missing_high_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["high"])

    try:
        compute_true_range(df)
        raised = False
    except KeyError as error:
        raised = True
        assert "high" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Missing Low Column
# --------------------------------------------------


def test_missing_low_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["low"])

    try:
        compute_true_range(df)
        raised = False
    except KeyError as error:
        raised = True
        assert "low" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Missing Close Column
# --------------------------------------------------


def test_missing_close_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    ).drop(columns=["close"])

    try:
        compute_true_range(df)
        raised = False
    except KeyError as error:
        raised = True
        assert "close" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Non-Numeric High Column
# --------------------------------------------------


def test_non_numeric_high_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    )
    df["high"] = df["high"].astype(str)

    try:
        compute_true_range(df)
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# Non-Numeric Low Column
# --------------------------------------------------


def test_non_numeric_low_column() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    )
    df["low"] = df["low"].astype(str)

    try:
        compute_true_range(df)
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
    df = _make_ohlc_frame(
        high=[12.0, 15.0, 14.0],
        low=[10.0, 11.0, 12.0],
        close=[11.0, 14.0, 13.0],
    )
    df_before = copy.deepcopy(df)

    compute_true_range(df)

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 12
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_ohlc_frame(
        high=[12.0, 15.0],
        low=[10.0, 11.0],
        close=[11.0, 14.0],
    )
    result = compute_true_range(df)

    expected_columns = list(df.columns) + ["true_range"]
    assert list(result.columns) == expected_columns


def main() -> None:
    test_basic_true_range()
    test_custom_output_column()
    test_group_isolation()
    test_matches_manual_formula()
    test_missing_symbol_column()
    test_missing_high_column()
    test_missing_low_column()
    test_missing_close_column()
    test_non_numeric_high_column()
    test_non_numeric_low_column()
    test_input_dataframe_unchanged()
    test_output_schema()

    print("🎉 ALL TRUE RANGE TESTS PASSED")


if __name__ == "__main__":
    main()
