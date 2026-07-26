import copy

import pandas as pd
import pandas.testing as pdt

from src.portfolio.positions import generate_positions
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL


def _make_signal_frame(
    values: list[int],
    signal_column: str = "signal",
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a small deterministic single-column signal frame."""
    return pd.DataFrame(
        {signal_column: values},
        dtype="int64",
        index=index,
    )


# --------------------------------------------------
# TEST 1
# Default Output Column
# --------------------------------------------------


def test_default_output_column() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL])
    result = generate_positions(df, signal_column="signal", output_column=None)

    assert "signal_position" in result.columns


# --------------------------------------------------
# TEST 2
# Custom Output Column
# --------------------------------------------------


def test_custom_output_column() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL])
    result = generate_positions(
        df,
        signal_column="signal",
        output_column="custom_position",
    )

    assert "custom_position" in result.columns
    assert "signal_position" not in result.columns


# --------------------------------------------------
# TEST 3
# Carry Forward Position Logic
# --------------------------------------------------


def test_carry_forward_position_logic() -> None:
    df = _make_signal_frame(
        [
            FLAT_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
            FLAT_SIGNAL,
            SHORT_SIGNAL,
            FLAT_SIGNAL,
            LONG_SIGNAL,
        ]
    )
    result = generate_positions(df, signal_column="signal")

    expected = pd.Series(
        [
            FLAT_SIGNAL,
            LONG_SIGNAL,
            LONG_SIGNAL,
            LONG_SIGNAL,
            SHORT_SIGNAL,
            SHORT_SIGNAL,
            LONG_SIGNAL,
        ],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["signal_position"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Leading Flat Signals
# --------------------------------------------------


def test_leading_flat_signals() -> None:
    df = _make_signal_frame(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, LONG_SIGNAL]
    )
    result = generate_positions(df, signal_column="signal")

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, LONG_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["signal_position"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Immediate Reversal
# --------------------------------------------------


def test_immediate_reversal() -> None:
    df = _make_signal_frame([LONG_SIGNAL, SHORT_SIGNAL])
    result = generate_positions(df, signal_column="signal")

    expected = pd.Series([LONG_SIGNAL, SHORT_SIGNAL], dtype="int64")
    pdt.assert_series_equal(
        result["signal_position"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# All Flat Signals
# --------------------------------------------------


def test_all_flat_signals() -> None:
    df = _make_signal_frame(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL]
    )
    result = generate_positions(df, signal_column="signal")

    expected = pd.Series(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["signal_position"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 7
# Single Long Signal
# --------------------------------------------------


def test_single_long_signal() -> None:
    df = _make_signal_frame([LONG_SIGNAL])
    result = generate_positions(df, signal_column="signal")

    assert result["signal_position"].iloc[0] == LONG_SIGNAL


# --------------------------------------------------
# TEST 8
# Single Short Signal
# --------------------------------------------------


def test_single_short_signal() -> None:
    df = _make_signal_frame([SHORT_SIGNAL])
    result = generate_positions(df, signal_column="signal")

    assert result["signal_position"].iloc[0] == SHORT_SIGNAL


# --------------------------------------------------
# TEST 9
# Single Flat Signal
# --------------------------------------------------


def test_single_flat_signal() -> None:
    df = _make_signal_frame([FLAT_SIGNAL])
    result = generate_positions(df, signal_column="signal")

    assert result["signal_position"].iloc[0] == FLAT_SIGNAL


# --------------------------------------------------
# TEST 10
# Input DataFrame Is Immutable
# --------------------------------------------------


def test_input_dataframe_is_immutable() -> None:
    df = _make_signal_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL]
    )
    df_before = copy.deepcopy(df)

    generate_positions(df, signal_column="signal")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 11
# Missing Signal Column
# --------------------------------------------------


def test_missing_signal_column() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL])

    try:
        generate_positions(df, signal_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Non-Numeric Signal Column
# --------------------------------------------------


def test_non_numeric_signal_column() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL])
    df["signal"] = df["signal"].astype(str)

    try:
        generate_positions(df, signal_column="signal")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 13
# Invalid Signal Values
# --------------------------------------------------


def test_invalid_signal_values() -> None:
    df = _make_signal_frame([LONG_SIGNAL, 2, FLAT_SIGNAL])

    try:
        generate_positions(df, signal_column="signal")
        raised = False
    except ValueError as error:
        raised = True
        assert "invalid signal values" in str(error)

    assert raised


# --------------------------------------------------
# TEST 14
# Position Dtype Is Int64
# --------------------------------------------------


def test_position_dtype_is_int64() -> None:
    df = _make_signal_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL]
    )
    result = generate_positions(df, signal_column="signal")

    assert result["signal_position"].dtype == "int64"


# --------------------------------------------------
# TEST 15
# Index Is Preserved
# --------------------------------------------------


def test_index_is_preserved() -> None:
    index = pd.Index([10, 20, 30, 40], name="row_id")
    df = _make_signal_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL],
        index=index,
    )
    result = generate_positions(df, signal_column="signal")

    pdt.assert_index_equal(result.index, df.index)


def main() -> None:
    test_default_output_column()
    test_custom_output_column()
    test_carry_forward_position_logic()
    test_leading_flat_signals()
    test_immediate_reversal()
    test_all_flat_signals()
    test_single_long_signal()
    test_single_short_signal()
    test_single_flat_signal()
    test_input_dataframe_is_immutable()
    test_missing_signal_column()
    test_non_numeric_signal_column()
    test_invalid_signal_values()
    test_position_dtype_is_int64()
    test_index_is_preserved()

    print("🎉 ALL POSITION TESTS PASSED")


if __name__ == "__main__":
    main()
