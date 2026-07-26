import copy

import pandas as pd
import pandas.testing as pdt

from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from src.signals.signal_utils import (
    count_signal_changes,
    invert_signal,
    summarize_signal,
    validate_signal_column,
)


def _make_signal_frame(
    values: list[int],
    signal_column: str = "signal",
) -> pd.DataFrame:
    """Build a small deterministic single-column signal frame."""
    return pd.DataFrame({signal_column: values}, dtype="int64")


# --------------------------------------------------
# TEST 1
# Validate Signal Column Valid
# --------------------------------------------------


def test_validate_signal_column_valid() -> None:
    df = _make_signal_frame(
        [LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL]
    )

    validate_signal_column(df, "signal")


# --------------------------------------------------
# TEST 2
# Validate Signal Column Invalid Values
# --------------------------------------------------


def test_validate_signal_column_invalid_values() -> None:
    df = _make_signal_frame([LONG_SIGNAL, 2, FLAT_SIGNAL])

    try:
        validate_signal_column(df, "signal")
        raised = False
    except ValueError as error:
        raised = True
        assert "invalid signal values" in str(error)
        assert "[2]" in str(error)

    assert raised


# --------------------------------------------------
# TEST 3
# Validate Signal Column Missing Column
# --------------------------------------------------


def test_validate_signal_column_missing_column() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL])

    try:
        validate_signal_column(df, "missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 4
# Validate Signal Column Non Numeric
# --------------------------------------------------


def test_validate_signal_column_non_numeric() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL])
    df["signal"] = df["signal"].astype(str)

    try:
        validate_signal_column(df, "signal")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 5
# Invert Signal
# --------------------------------------------------


def test_invert_signal() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL])
    result = invert_signal(df, "signal")

    expected = pd.Series(
        [SHORT_SIGNAL, FLAT_SIGNAL, LONG_SIGNAL],
        dtype="int64",
    )
    pdt.assert_series_equal(
        result["signal_inverted"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Invert Signal Default Output
# --------------------------------------------------


def test_invert_signal_default_output() -> None:
    df = _make_signal_frame(
        [LONG_SIGNAL, FLAT_SIGNAL],
        signal_column="zscore_signal",
    )
    result = invert_signal(df, "zscore_signal", output_column=None)

    assert "zscore_signal_inverted" in result.columns


# --------------------------------------------------
# TEST 7
# Invert Signal Custom Output
# --------------------------------------------------


def test_invert_signal_custom_output() -> None:
    df = _make_signal_frame([LONG_SIGNAL, SHORT_SIGNAL])
    result = invert_signal(
        df,
        "signal",
        output_column="custom_inverted",
    )

    assert "custom_inverted" in result.columns
    assert "signal_inverted" not in result.columns


# --------------------------------------------------
# TEST 8
# Invert Signal Immutability
# --------------------------------------------------


def test_invert_signal_immutability() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL])
    df_before = copy.deepcopy(df)

    invert_signal(df, "signal")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 9
# Count Signal Changes
# --------------------------------------------------


def test_count_signal_changes() -> None:
    df = _make_signal_frame(
        [
            FLAT_SIGNAL,
            FLAT_SIGNAL,
            LONG_SIGNAL,
            LONG_SIGNAL,
            SHORT_SIGNAL,
            SHORT_SIGNAL,
            FLAT_SIGNAL,
        ]
    )

    assert count_signal_changes(df, "signal") == 3


# --------------------------------------------------
# TEST 10
# Count Signal Changes No Changes
# --------------------------------------------------


def test_count_signal_changes_no_changes() -> None:
    df = _make_signal_frame(
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL]
    )

    assert count_signal_changes(df, "signal") == 0


# --------------------------------------------------
# TEST 11
# Summarize Signal
# --------------------------------------------------


def test_summarize_signal() -> None:
    df = _make_signal_frame(
        [
            LONG_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
            FLAT_SIGNAL,
            FLAT_SIGNAL,
            SHORT_SIGNAL,
        ]
    )
    summary = summarize_signal(df, "signal")

    expected = pd.Series(
        {
            "long": 2,
            "flat": 3,
            "short": 1,
        },
        dtype="int64",
    )
    pdt.assert_series_equal(summary, expected, check_names=False)


# --------------------------------------------------
# TEST 12
# Summarize Signal Missing Categories
# --------------------------------------------------


def test_summarize_signal_missing_categories() -> None:
    df = _make_signal_frame([FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL])
    summary = summarize_signal(df, "signal")

    assert summary["long"] == 0
    assert summary["flat"] == 3
    assert summary["short"] == 0


# --------------------------------------------------
# TEST 13
# Summarize Signal Index Order
# --------------------------------------------------


def test_summarize_signal_index_order() -> None:
    df = _make_signal_frame(
        [SHORT_SIGNAL, LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL]
    )
    summary = summarize_signal(df, "signal")

    assert list(summary.index) == ["long", "flat", "short"]


# --------------------------------------------------
# TEST 14
# Output Schema
# --------------------------------------------------


def test_output_schema() -> None:
    df = _make_signal_frame([LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL])
    result = invert_signal(df, "signal")

    expected_columns = list(df.columns) + ["signal_inverted"]
    assert list(result.columns) == expected_columns


def main() -> None:
    test_validate_signal_column_valid()
    test_validate_signal_column_invalid_values()
    test_validate_signal_column_missing_column()
    test_validate_signal_column_non_numeric()
    test_invert_signal()
    test_invert_signal_default_output()
    test_invert_signal_custom_output()
    test_invert_signal_immutability()
    test_count_signal_changes()
    test_count_signal_changes_no_changes()
    test_summarize_signal()
    test_summarize_signal_missing_categories()
    test_summarize_signal_index_order()
    test_output_schema()

    print("🎉 ALL SIGNAL UTILS TESTS PASSED")


if __name__ == "__main__":
    main()
