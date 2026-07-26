import copy

import pandas as pd
import pandas.testing as pdt

from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.crossover import generate_crossover_signal


def _make_pair_frame(
    fast: list[float],
    slow: list[float],
    fast_column: str = "fast",
    slow_column: str = "slow",
) -> pd.DataFrame:
    """Build a small deterministic two-column frame for crossover tests."""
    return pd.DataFrame(
        {
            fast_column: fast,
            slow_column: slow,
        }
    )


# --------------------------------------------------
# TEST 1
# Bullish Crossover
# --------------------------------------------------


def test_bullish_crossover() -> None:
    df = _make_pair_frame(
        fast=[-1.0, -0.5, 0.5],
        slow=[0.0, 0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, LONG_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["fast_cross_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Bearish Crossover
# --------------------------------------------------


def test_bearish_crossover() -> None:
    df = _make_pair_frame(
        fast=[1.0, 0.5, -0.5],
        slow=[0.0, 0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["fast_cross_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 3
# No Crossover
# --------------------------------------------------


def test_no_crossover() -> None:
    df = _make_pair_frame(
        fast=[-2.0, -1.0, -0.5],
        slow=[0.0, 0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["fast_cross_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# First Row Flat
# --------------------------------------------------


def test_first_row_flat() -> None:
    df = _make_pair_frame(
        fast=[1.0, 2.0],
        slow=[0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    assert result["fast_cross_signal"].iloc[0] == FLAT_SIGNAL


# --------------------------------------------------
# TEST 5
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_pair_frame(
        fast=[-1.0, 0.5],
        slow=[0.0, 0.0],
    )
    result = generate_crossover_signal(
        df,
        fast_column="fast",
        slow_column="slow",
        output_column="custom_cross",
    )

    assert "custom_cross" in result.columns
    assert "fast_cross_signal" not in result.columns


# --------------------------------------------------
# TEST 6
# Default Output Column
# --------------------------------------------------


def test_default_output_column() -> None:
    df = _make_pair_frame(
        fast=[-1.0, 0.5],
        slow=[0.0, 0.0],
        fast_column="ema_fast",
        slow_column="ema_slow",
    )
    result = generate_crossover_signal(
        df,
        fast_column="ema_fast",
        slow_column="ema_slow",
        output_column=None,
    )

    assert "ema_fast_cross_signal" in result.columns


# --------------------------------------------------
# TEST 7
# Missing Fast Column
# --------------------------------------------------


def test_missing_fast_column() -> None:
    df = _make_pair_frame([-1.0, 0.5], [0.0, 0.0])

    try:
        generate_crossover_signal(
            df,
            fast_column="missing",
            slow_column="slow",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Missing Slow Column
# --------------------------------------------------


def test_missing_slow_column() -> None:
    df = _make_pair_frame([-1.0, 0.5], [0.0, 0.0])

    try:
        generate_crossover_signal(
            df,
            fast_column="fast",
            slow_column="missing",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Non-Numeric Fast Column
# --------------------------------------------------


def test_non_numeric_fast_column() -> None:
    df = _make_pair_frame([-1.0, 0.5], [0.0, 0.0])
    df["fast"] = df["fast"].astype(str)

    try:
        generate_crossover_signal(df, fast_column="fast", slow_column="slow")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# Non-Numeric Slow Column
# --------------------------------------------------


def test_non_numeric_slow_column() -> None:
    df = _make_pair_frame([-1.0, 0.5], [0.0, 0.0])
    df["slow"] = df["slow"].astype(str)

    try:
        generate_crossover_signal(df, fast_column="fast", slow_column="slow")
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
        fast=[-1.0, -0.5, 0.5],
        slow=[0.0, 0.0, 0.0],
    )
    df_before = copy.deepcopy(df)

    generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 12
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_pair_frame(
        fast=[-1.0, 0.5],
        slow=[0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    expected_columns = list(df.columns) + ["fast_cross_signal"]
    assert list(result.columns) == expected_columns


# --------------------------------------------------
# TEST 13
# Touch Without Cross
# --------------------------------------------------


def test_touch_without_cross() -> None:
    df = _make_pair_frame(
        fast=[-1.0, 0.0, 0.0],
        slow=[0.0, 0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    # Equality alone must not produce a crossover signal.
    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["fast_cross_signal"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 14
# Multiple Crossovers
# --------------------------------------------------


def test_multiple_crossovers() -> None:
    df = _make_pair_frame(
        fast=[-1.0, -0.5, 0.5, 1.0, -0.5, -1.0, 0.5],
        slow=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    result = generate_crossover_signal(df, fast_column="fast", slow_column="slow")

    expected = pd.Series(
        [
            FLAT_SIGNAL,
            FLAT_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
            SHORT_SIGNAL,
            FLAT_SIGNAL,
            LONG_SIGNAL,
        ],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["fast_cross_signal"],
        expected,
        check_names=False,
    )


def main() -> None:
    test_bullish_crossover()
    test_bearish_crossover()
    test_no_crossover()
    test_first_row_flat()
    test_custom_output_column()
    test_default_output_column()
    test_missing_fast_column()
    test_missing_slow_column()
    test_non_numeric_fast_column()
    test_non_numeric_slow_column()
    test_input_dataframe_unchanged()
    test_output_schema()
    test_touch_without_cross()
    test_multiple_crossovers()

    print("🎉 ALL CROSSOVER SIGNAL TESTS PASSED")


if __name__ == "__main__":
    main()
