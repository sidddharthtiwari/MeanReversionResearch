import copy

import pandas as pd
import pandas.testing as pdt

from src.portfolio.exposure import generate_exposure
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL


def _make_position_frame(
    positions: list[int],
    position_column: str = "position",
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a small deterministic single-column position frame."""
    return pd.DataFrame(
        {position_column: positions},
        dtype="int64",
        index=index,
    )


# --------------------------------------------------
# TEST 1
# Default Output Columns
# --------------------------------------------------


def test_default_output_columns() -> None:
    df = _make_position_frame([LONG_SIGNAL, FLAT_SIGNAL, SHORT_SIGNAL])
    result = generate_exposure(df, position_column="position")

    assert "position_net_exposure" in result.columns
    assert "position_gross_exposure" in result.columns


# --------------------------------------------------
# TEST 2
# Custom Output Columns
# --------------------------------------------------


def test_custom_output_columns() -> None:
    df = _make_position_frame([LONG_SIGNAL, SHORT_SIGNAL])
    result = generate_exposure(
        df,
        position_column="position",
        net_exposure_column="net_exp",
        gross_exposure_column="gross_exp",
    )

    assert "net_exp" in result.columns
    assert "gross_exp" in result.columns
    assert "position_net_exposure" not in result.columns
    assert "position_gross_exposure" not in result.columns


# --------------------------------------------------
# TEST 3
# Long Positions
# --------------------------------------------------


def test_long_positions() -> None:
    df = _make_position_frame([LONG_SIGNAL, LONG_SIGNAL])
    result = generate_exposure(df, position_column="position")

    pdt.assert_series_equal(
        result["position_net_exposure"],
        pd.Series([LONG_SIGNAL, LONG_SIGNAL], dtype="int64"),
        check_names=False,
    )
    pdt.assert_series_equal(
        result["position_gross_exposure"],
        pd.Series([1, 1], dtype="int64"),
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Short Positions
# --------------------------------------------------


def test_short_positions() -> None:
    df = _make_position_frame([SHORT_SIGNAL, SHORT_SIGNAL])
    result = generate_exposure(df, position_column="position")

    pdt.assert_series_equal(
        result["position_net_exposure"],
        pd.Series([SHORT_SIGNAL, SHORT_SIGNAL], dtype="int64"),
        check_names=False,
    )
    pdt.assert_series_equal(
        result["position_gross_exposure"],
        pd.Series([1, 1], dtype="int64"),
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Flat Positions
# --------------------------------------------------


def test_flat_positions() -> None:
    df = _make_position_frame([FLAT_SIGNAL, FLAT_SIGNAL])
    result = generate_exposure(df, position_column="position")

    pdt.assert_series_equal(
        result["position_net_exposure"],
        pd.Series([FLAT_SIGNAL, FLAT_SIGNAL], dtype="int64"),
        check_names=False,
    )
    pdt.assert_series_equal(
        result["position_gross_exposure"],
        pd.Series([0, 0], dtype="int64"),
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Mixed Positions
# --------------------------------------------------


def test_mixed_positions() -> None:
    df = _make_position_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, LONG_SIGNAL]
    )
    result = generate_exposure(df, position_column="position")

    pdt.assert_series_equal(
        result["position_net_exposure"],
        pd.Series(
            [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, LONG_SIGNAL],
            dtype="int64",
        ),
        check_names=False,
    )
    pdt.assert_series_equal(
        result["position_gross_exposure"],
        pd.Series([0, 1, 1, 1], dtype="int64"),
        check_names=False,
    )


# --------------------------------------------------
# TEST 7
# Gross Equals Absolute Position
# --------------------------------------------------


def test_gross_equals_absolute_position() -> None:
    df = _make_position_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, SHORT_SIGNAL]
    )
    result = generate_exposure(df, position_column="position")

    pdt.assert_series_equal(
        result["position_gross_exposure"],
        result["position"].abs(),
        check_names=False,
    )


# --------------------------------------------------
# TEST 8
# Net Equals Position
# --------------------------------------------------


def test_net_equals_position() -> None:
    df = _make_position_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, LONG_SIGNAL]
    )
    result = generate_exposure(df, position_column="position")

    pdt.assert_series_equal(
        result["position_net_exposure"],
        result["position"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 9
# Input DataFrame Is Immutable
# --------------------------------------------------


def test_input_dataframe_is_immutable() -> None:
    df = _make_position_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL]
    )
    df_before = copy.deepcopy(df)

    generate_exposure(df, position_column="position")

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 10
# Missing Position Column
# --------------------------------------------------


def test_missing_position_column() -> None:
    df = _make_position_frame([LONG_SIGNAL, FLAT_SIGNAL])

    try:
        generate_exposure(df, position_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Invalid Position Values
# --------------------------------------------------


def test_invalid_position_values() -> None:
    df = _make_position_frame([LONG_SIGNAL, 2, FLAT_SIGNAL])

    try:
        generate_exposure(df, position_column="position")
        raised = False
    except ValueError as error:
        raised = True
        assert "invalid signal values" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Non-Numeric Position Column
# --------------------------------------------------


def test_non_numeric_position_column() -> None:
    df = _make_position_frame([LONG_SIGNAL, FLAT_SIGNAL])
    df["position"] = df["position"].astype(str)

    try:
        generate_exposure(df, position_column="position")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 13
# Index Is Preserved
# --------------------------------------------------


def test_index_is_preserved() -> None:
    index = pd.Index([10, 20, 30], name="row_id")
    df = _make_position_frame(
        [LONG_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
        index=index,
    )
    result = generate_exposure(df, position_column="position")

    pdt.assert_index_equal(result.index, df.index)


# --------------------------------------------------
# TEST 14
# Generated Columns Contain No Missing Values
# --------------------------------------------------


def test_generated_columns_contain_no_missing_values() -> None:
    df = _make_position_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, LONG_SIGNAL]
    )
    result = generate_exposure(df, position_column="position")

    assert result["position_net_exposure"].notna().all()
    assert result["position_gross_exposure"].notna().all()


def main() -> None:
    test_default_output_columns()
    test_custom_output_columns()
    test_long_positions()
    test_short_positions()
    test_flat_positions()
    test_mixed_positions()
    test_gross_equals_absolute_position()
    test_net_equals_position()
    test_input_dataframe_is_immutable()
    test_missing_position_column()
    test_invalid_position_values()
    test_non_numeric_position_column()
    test_index_is_preserved()
    test_generated_columns_contain_no_missing_values()

    print("🎉 ALL EXPOSURE TESTS PASSED")


if __name__ == "__main__":
    main()
