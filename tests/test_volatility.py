import copy

import numpy as np
import pandas as pd

import src.features.volatility as volatility_module
from src.data.loader import load_sector, load_sector_metadata
from src.features.returns import compute_simple_returns
from src.features.volatility import compute_volatility


def load_bank_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared bank OHLC and metadata fixtures once."""
    return load_sector("bank"), load_sector_metadata("bank")


# --------------------------------------------------
# TEST 1
# Basic Volatility Computation
# --------------------------------------------------


def test_basic_volatility_computation(df: pd.DataFrame) -> None:
    result = compute_volatility(df, window=20)

    assert isinstance(result, pd.DataFrame)
    assert "volatility_20" in result.columns
    assert len(result) == len(df)
    assert set(df.columns).issubset(set(result.columns))


# --------------------------------------------------
# TEST 2
# Default Output Name
# --------------------------------------------------


def test_default_output_name(df: pd.DataFrame) -> None:
    window = 15
    result = compute_volatility(df, window=window)

    assert f"volatility_{window}" in result.columns


# --------------------------------------------------
# TEST 3
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column(df: pd.DataFrame) -> None:
    result = compute_volatility(df, window=10, output_column="custom_vol")

    assert "custom_vol" in result.columns
    assert "volatility_10" not in result.columns


# --------------------------------------------------
# TEST 4
# Reuse Existing Return Column
# --------------------------------------------------


def test_reuse_existing_return_column(df: pd.DataFrame) -> None:
    with_returns = compute_simple_returns(df)
    result = compute_volatility(with_returns, window=20)

    assert "simple_return" in result.columns
    assert "volatility_20" in result.columns
    assert list(result.columns).count("simple_return") == 1


# --------------------------------------------------
# TEST 5
# Compute Returns Automatically
# --------------------------------------------------


def test_compute_returns_automatically(df: pd.DataFrame) -> None:
    assert "simple_return" not in df.columns

    result = compute_volatility(df, window=20)

    assert "simple_return" in result.columns
    assert "volatility_20" in result.columns


# --------------------------------------------------
# TEST 6
# Window = 1
# --------------------------------------------------


def test_window_equals_one(df: pd.DataFrame) -> None:
    result = compute_volatility(df, window=1)

    assert result["volatility_1"].isna().all()


# --------------------------------------------------
# TEST 7
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

    result = compute_volatility(tiny_df, window=50)

    assert result["volatility_50"].isna().all()


# --------------------------------------------------
# TEST 8
# Custom Price Column
# --------------------------------------------------


def test_custom_price_column(df: pd.DataFrame) -> None:
    custom_df = df.copy()
    custom_df["custom_close"] = custom_df["close"]

    default_result = compute_volatility(df, window=20)
    custom_result = compute_volatility(
        custom_df,
        window=20,
        price_column="custom_close",
    )

    pd.testing.assert_series_equal(
        default_result["volatility_20"],
        custom_result["volatility_20"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 9
# Custom Return Column
# --------------------------------------------------


def test_custom_return_column(df: pd.DataFrame) -> None:
    with_custom_returns = compute_simple_returns(
        df,
        output_column="my_returns",
    )
    custom_result = compute_volatility(
        with_custom_returns,
        window=20,
        return_column="my_returns",
    )

    default_result = compute_volatility(df, window=20)

    pd.testing.assert_series_equal(
        default_result["volatility_20"],
        custom_result["volatility_20"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 10
# Custom min_periods
# --------------------------------------------------


def test_custom_min_periods(df: pd.DataFrame) -> None:
    early = compute_volatility(df, window=20, min_periods=1)
    default = compute_volatility(df, window=20)

    assert early["volatility_20"].notna().sum() > default["volatility_20"].notna().sum()

    # First return is NaN; with ddof=1 the earliest defined std is at the third row.
    third_rows = early.groupby("symbol", sort=False).nth(2)
    assert third_rows["volatility_20"].notna().all()


# --------------------------------------------------
# TEST 11
# Missing Price Column
# --------------------------------------------------


def test_missing_price_column(df: pd.DataFrame) -> None:
    missing_price_df = df.drop(columns=["close"])

    try:
        compute_volatility(missing_price_df, window=5)
        raised = False
    except KeyError as error:
        raised = True
        assert "close" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Missing Return Column — Auto Computation
# --------------------------------------------------


def test_missing_return_column_auto_computation(df: pd.DataFrame) -> None:
    assert "simple_return" not in df.columns

    result = compute_volatility(df, window=10)

    assert "simple_return" in result.columns
    assert "volatility_10" in result.columns
    assert result["volatility_10"].notna().any()


# --------------------------------------------------
# TEST 13
# Non-Numeric Price Column
# --------------------------------------------------


def test_non_numeric_price_column(df: pd.DataFrame) -> None:
    non_numeric_df = df.copy()
    non_numeric_df["close"] = non_numeric_df["close"].astype(str)

    try:
        compute_volatility(non_numeric_df, window=5)
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
        compute_volatility(df, window=2.5)  # type: ignore[arg-type]
        raised_type = False
    except TypeError:
        raised_type = True
    assert raised_type

    for window in (0, -1):
        try:
            compute_volatility(df, window=window)
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
        compute_volatility(df, window=5, min_periods=1.5)  # type: ignore[arg-type]
        raised_type = False
    except TypeError:
        raised_type = True
    assert raised_type

    for min_periods in (0, -1, 6):
        try:
            compute_volatility(df, window=5, min_periods=min_periods)
            raised_value = False
        except ValueError:
            raised_value = True
        assert raised_value


# --------------------------------------------------
# TEST 16
# Exact Mathematical Correctness
# --------------------------------------------------


def test_exact_mathematical_correctness() -> None:
    tiny_df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "close": [100.0, 110.0, 121.0, 133.1],
        }
    )

    result = compute_volatility(tiny_df, window=2)

    expected_returns = tiny_df["close"] / tiny_df["close"].shift(1) - 1.0
    expected_volatility = expected_returns.rolling(window=2, min_periods=2).std()

    pd.testing.assert_series_equal(
        result["simple_return"].reset_index(drop=True),
        expected_returns.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["volatility_2"].reset_index(drop=True),
        expected_volatility.reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 17
# Read-Only Guarantee
# --------------------------------------------------


def test_read_only_guarantee(df: pd.DataFrame) -> None:
    df_before = copy.deepcopy(df)

    compute_volatility(df, window=20)
    compute_volatility(df, window=5, min_periods=1)

    pd.testing.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 18
# Multiple Symbols — No Leakage
# --------------------------------------------------


def test_multiple_symbols_no_leakage() -> None:
    multi_df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "close": [100.0, 110.0, 121.0, 10.0, 20.0, 40.0],
        }
    )

    result = compute_volatility(multi_df, window=2)

    symbol_a = result.loc[result["symbol"] == "A", "volatility_2"].reset_index(
        drop=True
    )
    symbol_b = result.loc[result["symbol"] == "B", "volatility_2"].reset_index(
        drop=True
    )

    # Returns for A: nan, 0.1, 0.1 -> final rolling std is 0.0
    assert np.isnan(symbol_a.iloc[0])
    assert np.isclose(symbol_a.iloc[2], 0.0)

    # Symbol B must start independently (first value NaN), not inherit A's history.
    assert np.isnan(symbol_b.iloc[0])
    assert np.isclose(symbol_b.iloc[2], 0.0)

    # Cross-check against independent single-symbol computations.
    only_a = compute_volatility(
        multi_df[multi_df["symbol"] == "A"].reset_index(drop=True),
        window=2,
    )
    only_b = compute_volatility(
        multi_df[multi_df["symbol"] == "B"].reset_index(drop=True),
        window=2,
    )
    pd.testing.assert_series_equal(
        symbol_a,
        only_a["volatility_2"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        symbol_b,
        only_b["volatility_2"].reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------
# TEST 19
# Composition Correctness
# --------------------------------------------------


def test_composition_correctness(df: pd.DataFrame) -> None:
    original_returns = volatility_module.compute_simple_returns
    original_std = volatility_module.compute_rolling_std
    calls = {"returns": 0, "std": 0}

    def wrapped_returns(*args, **kwargs):
        calls["returns"] += 1
        return original_returns(*args, **kwargs)

    def wrapped_std(*args, **kwargs):
        calls["std"] += 1
        return original_std(*args, **kwargs)

    volatility_module.compute_simple_returns = wrapped_returns
    volatility_module.compute_rolling_std = wrapped_std

    try:
        calls["returns"] = 0
        calls["std"] = 0
        compute_volatility(df, window=5)
        assert calls["returns"] == 1
        assert calls["std"] == 1

        with_returns = original_returns(df)
        calls["returns"] = 0
        calls["std"] = 0
        compute_volatility(with_returns, window=5)
        assert calls["returns"] == 0
        assert calls["std"] == 1
    finally:
        volatility_module.compute_simple_returns = original_returns
        volatility_module.compute_rolling_std = original_std


def main() -> None:
    df, _metadata = load_bank_data()

    test_basic_volatility_computation(df)
    test_default_output_name(df)
    test_custom_output_column(df)
    test_reuse_existing_return_column(df)
    test_compute_returns_automatically(df)
    test_window_equals_one(df)
    test_window_larger_than_dataset()
    test_custom_price_column(df)
    test_custom_return_column(df)
    test_custom_min_periods(df)
    test_missing_price_column(df)
    test_missing_return_column_auto_computation(df)
    test_non_numeric_price_column(df)
    test_invalid_window(df)
    test_invalid_min_periods(df)
    test_exact_mathematical_correctness()
    test_read_only_guarantee(df)
    test_multiple_symbols_no_leakage()
    test_composition_correctness(df)

    print("\n🎉 ALL VOLATILITY TESTS PASSED")


if __name__ == "__main__":
    main()
