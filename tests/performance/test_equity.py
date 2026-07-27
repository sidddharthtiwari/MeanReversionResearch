import copy

import pandas as pd
import pandas.testing as pdt

from src.performance.constants import DEFAULT_EQUITY_COLUMN
from src.performance.equity import compute_equity_curve


# --------------------------------------------------
# TEST 1
# Basic Equity-Curve Computation
# --------------------------------------------------


def test_basic_equity_curve_computation() -> None:
    df = pd.DataFrame({"returns": [0.01, 0.02, -0.01]})
    result = compute_equity_curve(df, return_column="returns")

    expected = (1.0 + df["returns"]).cumprod()
    pdt.assert_series_equal(
        result[DEFAULT_EQUITY_COLUMN],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Default Output Column Name
# --------------------------------------------------


def test_default_output_column_name() -> None:
    df = pd.DataFrame({"returns": [0.01, -0.02]})
    result = compute_equity_curve(df, return_column="returns")

    assert DEFAULT_EQUITY_COLUMN in result.columns
    assert DEFAULT_EQUITY_COLUMN == "equity"


# --------------------------------------------------
# TEST 3
# Custom Output Column Name
# --------------------------------------------------


def test_custom_output_column_name() -> None:
    df = pd.DataFrame({"returns": [0.01, -0.02]})
    result = compute_equity_curve(
        df,
        return_column="returns",
        output_column="equity_curve",
    )

    assert "equity_curve" in result.columns
    assert DEFAULT_EQUITY_COLUMN not in result.columns


# --------------------------------------------------
# TEST 4
# Input DataFrame Remains Unchanged
# --------------------------------------------------


def test_input_dataframe_is_immutable() -> None:
    df = pd.DataFrame({"returns": [0.01, 0.02, -0.01]})
    original = copy.deepcopy(df)

    compute_equity_curve(df, return_column="returns")

    pdt.assert_frame_equal(df, original)


# --------------------------------------------------
# TEST 5
# Missing Return Column Raises KeyError
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = pd.DataFrame({"returns": [0.01, -0.02]})

    try:
        compute_equity_curve(df, return_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Non-Numeric Return Column Raises TypeError
# --------------------------------------------------


def test_non_numeric_return_column() -> None:
    df = pd.DataFrame({"returns": ["0.01", "-0.02"]})

    try:
        compute_equity_curve(df, return_column="returns")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Empty Return Series Raises ValueError
# --------------------------------------------------


def test_empty_return_series() -> None:
    df = pd.DataFrame({"returns": pd.Series(dtype="float64")})

    try:
        compute_equity_curve(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert "must not be empty" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# output_column Equal to Return Column Raises ValueError
# --------------------------------------------------


def test_output_column_equals_return_column() -> None:
    df = pd.DataFrame({"returns": [0.01, -0.02]})

    try:
        compute_equity_curve(
            df,
            return_column="returns",
            output_column="returns",
        )
        raised = False
    except ValueError as error:
        raised = True
        assert (
            str(error)
            == "output_column must be different from return_column."
        )

    assert raised


def main() -> None:
    test_basic_equity_curve_computation()
    test_default_output_column_name()
    test_custom_output_column_name()
    test_input_dataframe_is_immutable()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_output_column_equals_return_column()

    print("ALL EQUITY CURVE TESTS PASSED")


if __name__ == "__main__":
    main()
