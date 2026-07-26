import copy

import pandas as pd
import pandas.testing as pdt

from src.features.rolling_volume import compute_rolling_volume


def _make_volume_frame(
    volume: list[float],
    symbol: str = "A",
) -> pd.DataFrame:
    """Build a small deterministic single-symbol volume frame."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(volume),
            "volume": volume,
        }
    )


def _make_multi_symbol_frame() -> pd.DataFrame:
    """Build a deterministic two-symbol frame for isolation checks."""
    symbol_a = _make_volume_frame(
        volume=[100.0, 200.0, 300.0, 400.0],
        symbol="A",
    )
    symbol_b = _make_volume_frame(
        volume=[10.0, 20.0, 30.0, 40.0],
        symbol="B",
    )
    return pd.concat([symbol_a, symbol_b], ignore_index=True)


# --------------------------------------------------
# TEST 1
# Basic Rolling Volume
# --------------------------------------------------


def test_basic_rolling_volume() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0, 400.0, 500.0])
    result = compute_rolling_volume(df, window=3)

    assert "rolling_volume_3" in result.columns
    assert len(result) == len(df)


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0])
    result = compute_rolling_volume(
        df,
        window=2,
        output_column="custom_volume",
    )

    assert "custom_volume" in result.columns
    assert "rolling_volume_2" not in result.columns


# --------------------------------------------------
# TEST 3
# Group Isolation
# --------------------------------------------------


def test_group_isolation() -> None:
    df = _make_multi_symbol_frame()
    window = 3
    result = compute_rolling_volume(df, window=window)
    column = f"rolling_volume_{window}"

    only_a = compute_rolling_volume(
        df[df["symbol"] == "A"].reset_index(drop=True),
        window=window,
    )
    only_b = compute_rolling_volume(
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
    df = _make_volume_frame([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    window = 3
    result = compute_rolling_volume(df, window=window)
    expected = df["volume"].rolling(window=window).mean()

    pdt.assert_series_equal(
        result[f"rolling_volume_{window}"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Invalid Window
# --------------------------------------------------


def test_invalid_window() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0])

    try:
        compute_rolling_volume(df, window=0)
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
    df = _make_volume_frame([100.0, 200.0, 300.0])

    try:
        compute_rolling_volume(df, window=3, min_periods=0)
        raised = False
    except ValueError:
        raised = True

    assert raised


# --------------------------------------------------
# TEST 7
# Missing Volume Column
# --------------------------------------------------


def test_missing_volume_column() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0]).drop(columns=["volume"])

    try:
        compute_rolling_volume(df, window=2)
        raised = False
    except KeyError as error:
        raised = True
        assert "volume" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Non-Numeric Volume Column
# --------------------------------------------------


def test_non_numeric_volume_column() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0])
    df["volume"] = df["volume"].astype(str)

    try:
        compute_rolling_volume(df, window=2)
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
    df = _make_volume_frame([100.0, 200.0, 300.0, 400.0])
    df_before = copy.deepcopy(df)

    compute_rolling_volume(df, window=3)

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 10
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0])
    result = compute_rolling_volume(df, window=2)

    expected_columns = list(df.columns) + ["rolling_volume_2"]
    assert list(result.columns) == expected_columns


# --------------------------------------------------
# TEST 11
# Custom Volume Column
# --------------------------------------------------


def test_custom_volume_column() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0, 400.0])
    df = df.rename(columns={"volume": "trade_volume"})
    window = 3

    result = compute_rolling_volume(
        df,
        window=window,
        volume_column="trade_volume",
    )
    expected = df["trade_volume"].rolling(window=window).mean()

    assert f"rolling_volume_{window}" in result.columns
    pdt.assert_series_equal(
        result[f"rolling_volume_{window}"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 12
# Default Output Name
# --------------------------------------------------


def test_default_output_name() -> None:
    df = _make_volume_frame([100.0, 200.0, 300.0, 400.0, 500.0])
    result = compute_rolling_volume(df, window=5, output_column=None)

    assert "rolling_volume_5" in result.columns


def main() -> None:
    test_basic_rolling_volume()
    test_custom_output_column()
    test_group_isolation()
    test_matches_manual_formula()
    test_invalid_window()
    test_invalid_min_periods()
    test_missing_volume_column()
    test_non_numeric_volume_column()
    test_input_dataframe_unchanged()
    test_output_schema()
    test_custom_volume_column()
    test_default_output_name()

    print("🎉 ALL ROLLING VOLUME TESTS PASSED")


if __name__ == "__main__":
    main()
