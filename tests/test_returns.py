import copy

import numpy as np
import pandas as pd

from src.data.loader import load_sector, load_sector_metadata
from src.features.returns import (
    compute_forward_returns,
    compute_log_returns,
    compute_simple_returns,
)


def load_bank_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared bank OHLC and metadata fixtures once."""
    return load_sector("bank"), load_sector_metadata("bank")


def _first_row_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Return the first row for each symbol, preserving group order."""
    return df.groupby("symbol", sort=False).head(1)


def _last_row_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last row for each symbol, preserving group order."""
    return df.groupby("symbol", sort=False).tail(1)


# --------------------------------------------------
# TEST 1
# Valid Simple Returns
# --------------------------------------------------


def test_valid_simple_returns(df: pd.DataFrame) -> None:
    result = compute_simple_returns(df)

    assert "simple_return" in result.columns
    assert len(result) == len(df)
    assert result.shape == (df.shape[0], df.shape[1] + 1)
    assert _first_row_per_symbol(result)["simple_return"].isna().all()


# --------------------------------------------------
# TEST 2
# Valid Log Returns
# --------------------------------------------------


def test_valid_log_returns(df: pd.DataFrame) -> None:
    result = compute_log_returns(df)

    assert "log_return" in result.columns
    assert _first_row_per_symbol(result)["log_return"].isna().all()

    previous_close = df.groupby("symbol", sort=False)["close"].shift(1)
    expected = np.log(df["close"] / previous_close)
    pd.testing.assert_series_equal(
        result["log_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# Forward Returns
# --------------------------------------------------


def test_forward_returns(df: pd.DataFrame) -> None:
    result = compute_forward_returns(df, periods=1)

    assert "forward_return_1d" in result.columns
    assert _last_row_per_symbol(result)["forward_return_1d"].isna().all()

    future_close = df.groupby("symbol", sort=False)["close"].shift(-1)
    expected = future_close / df["close"] - 1.0
    pd.testing.assert_series_equal(
        result["forward_return_1d"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column(df: pd.DataFrame) -> None:
    result = compute_simple_returns(df, output_column="custom_simple")

    assert "custom_simple" in result.columns
    assert "simple_return" not in result.columns


# --------------------------------------------------
# TEST 5
# Custom Price Column
# --------------------------------------------------


def test_custom_price_column(df: pd.DataFrame) -> None:
    custom_df = df.copy()
    custom_df["alt_close"] = custom_df["close"]

    default_result = compute_simple_returns(df)
    custom_result = compute_simple_returns(
        custom_df,
        price_column="alt_close",
        output_column="simple_return",
    )

    pd.testing.assert_series_equal(
        default_result["simple_return"],
        custom_result["simple_return"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Missing Symbol Column
# --------------------------------------------------


def test_missing_symbol_column(df: pd.DataFrame) -> None:
    missing_symbol_df = df.drop(columns=["symbol"])

    try:
        compute_simple_returns(missing_symbol_df)
        raised = False
    except KeyError as error:
        raised = True
        assert error.args[0] == "Required column 'symbol' not found in DataFrame."

    assert raised


# --------------------------------------------------
# TEST 7
# Missing Price Column
# --------------------------------------------------


def test_missing_price_column(df: pd.DataFrame) -> None:
    missing_price_df = df.drop(columns=["close"])

    try:
        compute_simple_returns(missing_price_df)
        raised = False
    except KeyError as error:
        raised = True
        assert "close" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Non-Numeric Price Column
# --------------------------------------------------


def test_non_numeric_price_column(df: pd.DataFrame) -> None:
    non_numeric_df = df.copy()
    non_numeric_df["close"] = non_numeric_df["close"].astype(str)

    try:
        compute_simple_returns(non_numeric_df)
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Invalid Periods
# --------------------------------------------------


def test_invalid_periods(df: pd.DataFrame) -> None:
    for periods in (0, -1, -5):
        try:
            compute_forward_returns(df, periods=periods)
            raised = False
        except ValueError as error:
            raised = True
            assert "periods must be > 0" in str(error)

        assert raised


# --------------------------------------------------
# TEST 10
# Single-Row Symbol
# --------------------------------------------------


def test_single_row_symbol() -> None:
    single_row_df = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": [pd.Timestamp("2024-01-01")],
            "close": [100.0],
        }
    )

    simple = compute_simple_returns(single_row_df)
    log = compute_log_returns(single_row_df)
    forward = compute_forward_returns(single_row_df, periods=1)

    assert simple["simple_return"].isna().all()
    assert log["log_return"].isna().all()
    assert forward["forward_return_1d"].isna().all()


# --------------------------------------------------
# TEST 11
# Read-Only Guarantee
# --------------------------------------------------


def test_read_only_guarantee(df: pd.DataFrame) -> None:
    df_before = copy.deepcopy(df)

    compute_simple_returns(df)
    compute_log_returns(df)
    compute_forward_returns(df, periods=1)
    compute_forward_returns(df, periods=5)

    pd.testing.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 12
# Exact Numerical Correctness
# --------------------------------------------------


def test_exact_numerical_correctness() -> None:
    tiny_df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "close": [100.0, 110.0, 121.0],
        }
    )

    simple = compute_simple_returns(tiny_df)
    log = compute_log_returns(tiny_df)
    forward = compute_forward_returns(tiny_df, periods=1)

    expected_simple = pd.Series([np.nan, 0.10, 0.10], dtype="float64")
    expected_log = pd.Series([np.nan, np.log(1.1), np.log(1.1)], dtype="float64")
    expected_forward = pd.Series([0.10, 0.10, np.nan], dtype="float64")

    pd.testing.assert_series_equal(
        simple["simple_return"].reset_index(drop=True),
        expected_simple,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        log["log_return"].reset_index(drop=True),
        expected_log,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        forward["forward_return_1d"].reset_index(drop=True),
        expected_forward,
        check_names=False,
    )


def main() -> None:
    df, _metadata = load_bank_data()

    test_valid_simple_returns(df)
    test_valid_log_returns(df)
    test_forward_returns(df)
    test_custom_output_column(df)
    test_custom_price_column(df)
    test_missing_symbol_column(df)
    test_missing_price_column(df)
    test_non_numeric_price_column(df)
    test_invalid_periods(df)
    test_single_row_symbol()
    test_read_only_guarantee(df)
    test_exact_numerical_correctness()

    print("\n🎉 ALL RETURN TESTS PASSED")


if __name__ == "__main__":
    main()
