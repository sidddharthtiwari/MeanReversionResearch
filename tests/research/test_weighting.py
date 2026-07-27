import math

import pandas as pd
import pandas.testing as pdt
import pytest

from src.research.weighting import compute_equal_weights


@pytest.fixture
def multi_symbol_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AUBANK": [0.01, -0.02],
            "AXISBANK": [0.0, 0.01],
            "HDFCBANK": [0.02, -0.01],
        },
        index=pd.Index(["2024-01-01", "2024-01-02"], name="date"),
    )


@pytest.fixture
def single_symbol_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {"AUBANK": [0.01, -0.02, 0.015]},
        index=pd.Index(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            name="date",
        ),
    )


@pytest.fixture
def empty_column_returns() -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index(["2024-01-01", "2024-01-02"], name="date"))


@pytest.fixture
def duplicate_column_returns() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.01, 0.02], [-0.01, 0.0]],
        columns=["AUBANK", "AUBANK"],
        index=pd.Index(["2024-01-01", "2024-01-02"], name="date"),
    )


# --------------------------------------------------
# TEST 1
# Equal Weights For Multiple Symbols
# --------------------------------------------------


def test_equal_weights_multiple_symbols(
    multi_symbol_returns: pd.DataFrame,
) -> None:
    result = compute_equal_weights(multi_symbol_returns)

    assert isinstance(result, pd.Series)
    pdt.assert_index_equal(result.index, multi_symbol_returns.columns)
    pdt.assert_series_equal(
        result,
        pd.Series(
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            index=multi_symbol_returns.columns,
            dtype="float64",
        ),
    )
    assert math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=0.0)
    assert result.dtype == "float64"


# --------------------------------------------------
# TEST 2
# Equal Weights For A Single Symbol
# --------------------------------------------------


def test_equal_weights_single_symbol(
    single_symbol_returns: pd.DataFrame,
) -> None:
    result = compute_equal_weights(single_symbol_returns)

    assert math.isclose(float(result.iloc[0]), 1.0, rel_tol=0.0, abs_tol=0.0)
    pdt.assert_index_equal(result.index, single_symbol_returns.columns)


# --------------------------------------------------
# TEST 3
# Empty DataFrame Raises ValueError
# --------------------------------------------------


def test_empty_dataframe_raises_value_error(
    empty_column_returns: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        compute_equal_weights(empty_column_returns)


# --------------------------------------------------
# TEST 4
# Duplicate Columns Raise ValueError
# --------------------------------------------------


def test_duplicate_columns_raise_value_error(
    duplicate_column_returns: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        compute_equal_weights(duplicate_column_returns)


# --------------------------------------------------
# TEST 5
# Invalid Input Type
# --------------------------------------------------


def test_invalid_input_type() -> None:
    with pytest.raises(TypeError, match="aligned_returns must be"):
        compute_equal_weights([[0.01, 0.02], [-0.01, 0.0]])  # type: ignore[arg-type]
