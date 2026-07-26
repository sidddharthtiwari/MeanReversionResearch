import copy

import pandas as pd
import pandas.testing as pdt

from src.analytics.cumulative import compute_cumulative_returns


def _make_return_frame(
    returns: list[float],
    return_column: str = "returns",
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a small deterministic period-return frame."""
    return pd.DataFrame(
        {return_column: returns},
        index=index,
    )


# --------------------------------------------------
# TEST 1
# Basic Cumulative-Return Computation
# --------------------------------------------------


def test_basic_cumulative_return_computation() -> None:
    df = _make_return_frame([0.01, 0.02, -0.01])
    result = compute_cumulative_returns(df, return_column="returns")

    expected = (1.0 + df["returns"]).cumprod() - 1.0
    pdt.assert_series_equal(
        result["returns_cumulative_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 2
# Default Output Column Name
# --------------------------------------------------


def test_default_output_column_name() -> None:
    df = _make_return_frame([0.01, -0.02])
    result = compute_cumulative_returns(df, return_column="returns")

    assert "returns_cumulative_return" in result.columns


# --------------------------------------------------
# TEST 3
# Custom Output Column Name
# --------------------------------------------------


def test_custom_output_column_name() -> None:
    df = _make_return_frame([0.01, -0.02])
    result = compute_cumulative_returns(
        df,
        return_column="returns",
        output_column="cum_ret",
    )

    assert "cum_ret" in result.columns
    assert "returns_cumulative_return" not in result.columns


# --------------------------------------------------
# TEST 4
# Input DataFrame Remains Unchanged
# --------------------------------------------------


def test_input_dataframe_is_immutable() -> None:
    df = _make_return_frame([0.01, 0.02, -0.01])
    original = copy.deepcopy(df)

    compute_cumulative_returns(df, return_column="returns")

    pdt.assert_frame_equal(df, original)


# --------------------------------------------------
# TEST 5
# Missing Return Column Raises KeyError
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        compute_cumulative_returns(df, return_column="missing")
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
    df = _make_return_frame([0.01, -0.02])
    df["returns"] = df["returns"].astype(str)

    try:
        compute_cumulative_returns(df, return_column="returns")
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
        compute_cumulative_returns(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert "must not be empty" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Invalid output_column Type Raises TypeError
# --------------------------------------------------


def test_invalid_output_column_type() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        compute_cumulative_returns(
            df,
            return_column="returns",
            output_column=123,  # type: ignore[arg-type]
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "output_column" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# output_column == return_column Raises ValueError
# --------------------------------------------------


def test_output_column_equals_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        compute_cumulative_returns(
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


# --------------------------------------------------
# TEST 10
# Returned DataFrame Is a New Object
# --------------------------------------------------


def test_returned_dataframe_is_new_object() -> None:
    df = _make_return_frame([0.01, -0.02])
    result = compute_cumulative_returns(df, return_column="returns")

    assert result is not df


def main() -> None:
    test_basic_cumulative_return_computation()
    test_default_output_column_name()
    test_custom_output_column_name()
    test_input_dataframe_is_immutable()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_invalid_output_column_type()
    test_output_column_equals_return_column()
    test_returned_dataframe_is_new_object()

    print("🎉 ALL CUMULATIVE RETURN TESTS PASSED")


if __name__ == "__main__":
    main()
