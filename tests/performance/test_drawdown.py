import copy

import pandas as pd
import pandas.testing as pdt

from src.performance.constants import DEFAULT_DRAWDOWN_COLUMN
from src.performance.drawdown import compute_drawdown_series


# --------------------------------------------------
# TEST 1
# Basic Drawdown-Series Computation
# --------------------------------------------------


def test_basic_drawdown_series_computation() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.10, 0.99, 1.05]})
    result = compute_drawdown_series(df, equity_column="equity")

    expected = (df["equity"] / df["equity"].cummax()) - 1.0
    pdt.assert_series_equal(
        result[DEFAULT_DRAWDOWN_COLUMN],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Default Output Column Name
# --------------------------------------------------


def test_default_output_column_name() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.05, 0.95]})
    result = compute_drawdown_series(df, equity_column="equity")

    assert DEFAULT_DRAWDOWN_COLUMN in result.columns
    assert DEFAULT_DRAWDOWN_COLUMN == "drawdown"


# --------------------------------------------------
# TEST 3
# Custom Output Column Name
# --------------------------------------------------


def test_custom_output_column_name() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.05, 0.95]})
    result = compute_drawdown_series(
        df,
        equity_column="equity",
        output_column="dd",
    )

    assert "dd" in result.columns
    assert DEFAULT_DRAWDOWN_COLUMN not in result.columns


# --------------------------------------------------
# TEST 4
# Input DataFrame Remains Unchanged
# --------------------------------------------------


def test_input_dataframe_is_immutable() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.10, 0.99]})
    original = copy.deepcopy(df)

    compute_drawdown_series(df, equity_column="equity")

    pdt.assert_frame_equal(df, original)


# --------------------------------------------------
# TEST 5
# Missing Equity Column Raises KeyError
# --------------------------------------------------


def test_missing_equity_column() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.05]})

    try:
        compute_drawdown_series(df, equity_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Non-Numeric Equity Column Raises TypeError
# --------------------------------------------------


def test_non_numeric_equity_column() -> None:
    df = pd.DataFrame({"equity": ["1.0", "1.05"]})

    try:
        compute_drawdown_series(df, equity_column="equity")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Empty Equity Series Raises ValueError
# --------------------------------------------------


def test_empty_equity_series() -> None:
    df = pd.DataFrame({"equity": pd.Series(dtype="float64")})

    try:
        compute_drawdown_series(df, equity_column="equity")
        raised = False
    except ValueError as error:
        raised = True
        assert "must not be empty" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# output_column Equal to Equity Column Raises ValueError
# --------------------------------------------------


def test_output_column_equals_equity_column() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.05]})

    try:
        compute_drawdown_series(
            df,
            equity_column="equity",
            output_column="equity",
        )
        raised = False
    except ValueError as error:
        raised = True
        assert (
            str(error)
            == "output_column must be different from equity_column."
        )

    assert raised


def main() -> None:
    test_basic_drawdown_series_computation()
    test_default_output_column_name()
    test_custom_output_column_name()
    test_input_dataframe_is_immutable()
    test_missing_equity_column()
    test_non_numeric_equity_column()
    test_empty_equity_series()
    test_output_column_equals_equity_column()

    print("ALL DRAWDOWN SERIES TESTS PASSED")


if __name__ == "__main__":
    main()
