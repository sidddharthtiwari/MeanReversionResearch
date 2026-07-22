import copy

import numpy as np
import pandas as pd

from src.data.loader import load_sector, load_sector_metadata
from src.features.rolling import (
    compute_rolling_max,
    compute_rolling_mean,
    compute_rolling_median,
    compute_rolling_min,
    compute_rolling_std,
)


def load_bank_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared bank OHLC and metadata fixtures once."""
    return load_sector("bank"), load_sector_metadata("bank")


def _rows_before_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the first ``window - 1`` rows for each symbol."""
    return df.groupby("symbol", sort=False).head(window - 1)


def _manual_rolling(series: pd.Series, window: int, how: str) -> pd.Series:
    """Compute a single-series rolling statistic for comparison."""
    rolling = series.rolling(window=window, min_periods=window)
    return getattr(rolling, how)()


# --------------------------------------------------
# TEST 1
# Rolling Mean
# --------------------------------------------------


def test_rolling_mean(df: pd.DataFrame) -> None:
    window = 20
    result = compute_rolling_mean(df, window=window)
    column = f"rolling_mean_{window}"

    assert column in result.columns
    assert len(result) == len(df)
    assert result.shape == (df.shape[0], df.shape[1] + 1)
    assert _rows_before_window(result, window)[column].isna().all()


# --------------------------------------------------
# TEST 2
# Rolling Standard Deviation
# --------------------------------------------------


def test_rolling_std(df: pd.DataFrame) -> None:
    window = 20
    result = compute_rolling_std(df, window=window)
    column = f"rolling_std_{window}"

    assert column in result.columns

    symbol = df["symbol"].iloc[0]
    subset = df[df["symbol"] == symbol].copy().reset_index(drop=True)
    actual = compute_rolling_std(subset, window=window)[column]
    expected = _manual_rolling(subset["close"], window, "std")

    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# Rolling Minimum
# --------------------------------------------------


def test_rolling_min(df: pd.DataFrame) -> None:
    window = 10
    symbol = df["symbol"].iloc[0]
    subset = df[df["symbol"] == symbol].copy().reset_index(drop=True)

    result = compute_rolling_min(subset, window=window)
    column = f"rolling_min_{window}"
    expected = _manual_rolling(subset["close"], window, "min")

    pd.testing.assert_series_equal(
        result[column].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Rolling Maximum
# --------------------------------------------------


def test_rolling_max(df: pd.DataFrame) -> None:
    window = 10
    symbol = df["symbol"].iloc[0]
    subset = df[df["symbol"] == symbol].copy().reset_index(drop=True)

    result = compute_rolling_max(subset, window=window)
    column = f"rolling_max_{window}"
    expected = _manual_rolling(subset["close"], window, "max")

    pd.testing.assert_series_equal(
        result[column].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Rolling Median
# --------------------------------------------------


def test_rolling_median(df: pd.DataFrame) -> None:
    window = 10
    symbol = df["symbol"].iloc[0]
    subset = df[df["symbol"] == symbol].copy().reset_index(drop=True)

    result = compute_rolling_median(subset, window=window)
    column = f"rolling_median_{window}"
    expected = _manual_rolling(subset["close"], window, "median")

    pd.testing.assert_series_equal(
        result[column].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column(df: pd.DataFrame) -> None:
    result = compute_rolling_mean(df, window=5, output_column="custom_mean")

    assert "custom_mean" in result.columns
    assert "rolling_mean_5" not in result.columns


# --------------------------------------------------
# TEST 7
# Custom Numeric Column
# --------------------------------------------------


def test_custom_numeric_column(df: pd.DataFrame) -> None:
    custom_df = df.copy()
    custom_df["alt_close"] = custom_df["close"]

    default_result = compute_rolling_mean(df, window=5)
    custom_result = compute_rolling_mean(
        custom_df,
        window=5,
        column="alt_close",
        output_column="rolling_mean_5",
    )

    pd.testing.assert_series_equal(
        default_result["rolling_mean_5"],
        custom_result["rolling_mean_5"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 8
# Window = 1
# --------------------------------------------------


def test_window_equals_one(df: pd.DataFrame) -> None:
    mean = compute_rolling_mean(df, window=1)
    minimum = compute_rolling_min(df, window=1)
    maximum = compute_rolling_max(df, window=1)

    pd.testing.assert_series_equal(
        mean["rolling_mean_1"],
        df["close"].astype(mean["rolling_mean_1"].dtype),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        minimum["rolling_min_1"],
        df["close"].astype(minimum["rolling_min_1"].dtype),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        maximum["rolling_max_1"],
        df["close"].astype(maximum["rolling_max_1"].dtype),
        check_names=False,
    )


# --------------------------------------------------
# TEST 9
# Window Larger Than Dataset
# --------------------------------------------------


def test_window_larger_than_dataset() -> None:
    tiny_df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "close": [100.0, 110.0, 120.0],
        }
    )

    result = compute_rolling_mean(tiny_df, window=10)

    assert result["rolling_mean_10"].isna().all()


# --------------------------------------------------
# TEST 10
# min_periods = 1
# --------------------------------------------------


def test_min_periods_one(df: pd.DataFrame) -> None:
    result = compute_rolling_mean(df, window=20, min_periods=1)

    first_rows = result.groupby("symbol", sort=False).head(1)
    assert first_rows["rolling_mean_20"].notna().all()


# --------------------------------------------------
# TEST 11
# Missing Symbol Column
# --------------------------------------------------


def test_missing_symbol_column(df: pd.DataFrame) -> None:
    missing_symbol_df = df.drop(columns=["symbol"])

    try:
        compute_rolling_mean(missing_symbol_df, window=5)
        raised = False
    except KeyError as error:
        raised = True
        assert error.args[0] == "Required column 'symbol' not found in DataFrame."

    assert raised


# --------------------------------------------------
# TEST 12
# Missing Numeric Column
# --------------------------------------------------


def test_missing_numeric_column(df: pd.DataFrame) -> None:
    missing_close_df = df.drop(columns=["close"])

    try:
        compute_rolling_mean(missing_close_df, window=5)
        raised = False
    except KeyError as error:
        raised = True
        assert "close" in str(error)

    assert raised


# --------------------------------------------------
# TEST 13
# Non-Numeric Column
# --------------------------------------------------


def test_non_numeric_column(df: pd.DataFrame) -> None:
    non_numeric_df = df.copy()
    non_numeric_df["close"] = non_numeric_df["close"].astype(str)

    try:
        compute_rolling_mean(non_numeric_df, window=5)
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 14
# Invalid Window
# --------------------------------------------------


def test_invalid_window(df: pd.DataFrame) -> None:
    try:
        compute_rolling_mean(df, window=2.5)  # type: ignore[arg-type]
        raised_type = False
    except TypeError:
        raised_type = True
    assert raised_type

    for window in (0, -1, -20):
        try:
            compute_rolling_mean(df, window=window)
            raised_value = False
        except ValueError as error:
            raised_value = True
            assert "window must be > 0" in str(error)
        assert raised_value


# --------------------------------------------------
# TEST 15
# Invalid min_periods
# --------------------------------------------------


def test_invalid_min_periods(df: pd.DataFrame) -> None:
    try:
        compute_rolling_mean(df, window=5, min_periods=1.5)  # type: ignore[arg-type]
        raised_type = False
    except TypeError:
        raised_type = True
    assert raised_type

    for min_periods in (0, -1, 6):
        try:
            compute_rolling_mean(df, window=5, min_periods=min_periods)
            raised_value = False
        except ValueError:
            raised_value = True
        assert raised_value


# --------------------------------------------------
# TEST 16
# Read-Only Guarantee
# --------------------------------------------------


def test_read_only_guarantee(df: pd.DataFrame) -> None:
    df_before = copy.deepcopy(df)

    compute_rolling_mean(df, window=5)
    compute_rolling_std(df, window=5)
    compute_rolling_min(df, window=5)
    compute_rolling_max(df, window=5)
    compute_rolling_median(df, window=5)

    pd.testing.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 17
# Multiple Symbols — No Leakage
# --------------------------------------------------


def test_multiple_symbols_no_leakage() -> None:
    multi_df = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "close": [100.0, 200.0, 10.0, 20.0],
        }
    )

    result = compute_rolling_mean(multi_df, window=2)
    values = result["rolling_mean_2"].tolist()

    assert np.isnan(values[0])
    assert values[1] == 150.0
    assert np.isnan(values[2])
    assert values[3] == 15.0


# --------------------------------------------------
# TEST 18
# Exact Mathematical Correctness
# --------------------------------------------------


def test_exact_mathematical_correctness() -> None:
    tiny_df = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "close": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    )
    window = 3
    close = tiny_df["close"]

    mean = compute_rolling_mean(tiny_df, window=window)
    std = compute_rolling_std(tiny_df, window=window)
    minimum = compute_rolling_min(tiny_df, window=window)
    maximum = compute_rolling_max(tiny_df, window=window)
    median = compute_rolling_median(tiny_df, window=window)

    pd.testing.assert_series_equal(
        mean[f"rolling_mean_{window}"],
        close.rolling(window).mean(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        std[f"rolling_std_{window}"],
        close.rolling(window).std(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        minimum[f"rolling_min_{window}"],
        close.rolling(window).min(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        maximum[f"rolling_max_{window}"],
        close.rolling(window).max(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        median[f"rolling_median_{window}"],
        close.rolling(window).median(),
        check_names=False,
    )

    assert mean["rolling_mean_3"].iloc[2] == 110.0
    assert mean["rolling_mean_3"].iloc[3] == 120.0
    assert mean["rolling_mean_3"].iloc[4] == 130.0
    assert minimum["rolling_min_3"].iloc[4] == 120.0
    assert maximum["rolling_max_3"].iloc[4] == 140.0
    assert median["rolling_median_3"].iloc[4] == 130.0


def main() -> None:
    df, _metadata = load_bank_data()

    test_rolling_mean(df)
    test_rolling_std(df)
    test_rolling_min(df)
    test_rolling_max(df)
    test_rolling_median(df)
    test_custom_output_column(df)
    test_custom_numeric_column(df)
    test_window_equals_one(df)
    test_window_larger_than_dataset()
    test_min_periods_one(df)
    test_missing_symbol_column(df)
    test_missing_numeric_column(df)
    test_non_numeric_column(df)
    test_invalid_window(df)
    test_invalid_min_periods(df)
    test_read_only_guarantee(df)
    test_multiple_symbols_no_leakage()
    test_exact_mathematical_correctness()

    print("\n🎉 ALL ROLLING TESTS PASSED")


if __name__ == "__main__":
    main()
