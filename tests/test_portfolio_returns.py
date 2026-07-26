import copy

import pandas as pd
import pandas.testing as pdt

from src.portfolio.returns import generate_strategy_returns
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL


def _make_returns_frame(
    positions: list[int],
    asset_returns: list[float],
    position_column: str = "position",
    asset_return_column: str = "asset_return",
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a small deterministic position and asset-return frame."""
    return pd.DataFrame(
        {
            position_column: positions,
            asset_return_column: asset_returns,
        },
        index=index,
    )


# --------------------------------------------------
# TEST 1
# Default Output Columns
# --------------------------------------------------


def test_default_output_columns() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    assert "position_strategy_return" in result.columns
    assert "position_cumulative_return" in result.columns
    assert "position_equity_curve" in result.columns


# --------------------------------------------------
# TEST 2
# Custom Output Columns
# --------------------------------------------------


def test_custom_output_columns() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
        strategy_return_column="strat_ret",
        cumulative_return_column="cum_ret",
        equity_curve_column="equity",
    )

    assert "strat_ret" in result.columns
    assert "cum_ret" in result.columns
    assert "equity" in result.columns
    assert "position_strategy_return" not in result.columns
    assert "position_cumulative_return" not in result.columns
    assert "position_equity_curve" not in result.columns


# --------------------------------------------------
# TEST 3
# Long Position Returns
# --------------------------------------------------


def test_long_position_returns() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, LONG_SIGNAL],
        [0.02, -0.01],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    expected = pd.Series([0.02, -0.01], dtype="float64")
    pdt.assert_series_equal(
        result["position_strategy_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Short Position Returns
# --------------------------------------------------


def test_short_position_returns() -> None:
    df = _make_returns_frame(
        [SHORT_SIGNAL, SHORT_SIGNAL],
        [0.02, -0.01],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    expected = pd.Series([-0.02, 0.01], dtype="float64")
    pdt.assert_series_equal(
        result["position_strategy_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Flat Position Returns
# --------------------------------------------------


def test_flat_position_returns() -> None:
    df = _make_returns_frame(
        [FLAT_SIGNAL, FLAT_SIGNAL],
        [0.05, -0.03],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    expected = pd.Series([0.0, 0.0], dtype="float64")
    pdt.assert_series_equal(
        result["position_strategy_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Mixed Positions
# --------------------------------------------------


def test_mixed_positions() -> None:
    df = _make_returns_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
        [0.01, 0.02, 0.03, -0.04],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    expected = pd.Series([0.0, 0.02, -0.03, 0.0], dtype="float64")
    pdt.assert_series_equal(
        result["position_strategy_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 7
# Cumulative Return Formula
# --------------------------------------------------


def test_cumulative_return_formula() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
        [0.10, -0.05, 0.20],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    strategy_returns = result["position_strategy_return"]
    expected = (1.0 + strategy_returns).cumprod() - 1.0
    pdt.assert_series_equal(
        result["position_cumulative_return"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 8
# Equity Curve Formula
# --------------------------------------------------


def test_equity_curve_formula() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
        [0.10, 0.05, -0.02],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    expected = 1.0 + result["position_cumulative_return"]
    pdt.assert_series_equal(
        result["position_equity_curve"],
        expected,
        check_names=False,
    )


# --------------------------------------------------
# TEST 9
# Input DataFrame Is Immutable
# --------------------------------------------------


def test_input_dataframe_is_immutable() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02, 0.03],
    )
    df_before = copy.deepcopy(df)

    generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 10
# Missing Position Column
# --------------------------------------------------


def test_missing_position_column() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02],
    )

    try:
        generate_strategy_returns(
            df,
            position_column="missing",
            asset_return_column="asset_return",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Missing Return Column
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02],
    )

    try:
        generate_strategy_returns(
            df,
            position_column="position",
            asset_return_column="missing",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Invalid Position Values
# --------------------------------------------------


def test_invalid_position_values() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, 2, FLAT_SIGNAL],
        [0.01, -0.02, 0.03],
    )

    try:
        generate_strategy_returns(
            df,
            position_column="position",
            asset_return_column="asset_return",
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "invalid signal values" in str(error)

    assert raised


# --------------------------------------------------
# TEST 13
# Non-Numeric Return Column
# --------------------------------------------------


def test_non_numeric_return_column() -> None:
    df = _make_returns_frame(
        [LONG_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02],
    )
    df["asset_return"] = df["asset_return"].astype(str)

    try:
        generate_strategy_returns(
            df,
            position_column="position",
            asset_return_column="asset_return",
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 14
# Index Is Preserved
# --------------------------------------------------


def test_index_is_preserved() -> None:
    index = pd.Index([10, 20, 30], name="row_id")
    df = _make_returns_frame(
        [LONG_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
        [0.01, -0.02, 0.03],
        index=index,
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    pdt.assert_index_equal(result.index, df.index)


# --------------------------------------------------
# TEST 15
# Generated Columns Contain No Missing Values
# --------------------------------------------------


def test_generated_columns_contain_no_missing_values() -> None:
    df = _make_returns_frame(
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, LONG_SIGNAL],
        [0.01, 0.02, -0.03, 0.04],
    )
    result = generate_strategy_returns(
        df,
        position_column="position",
        asset_return_column="asset_return",
    )

    assert result["position_strategy_return"].notna().all()
    assert result["position_cumulative_return"].notna().all()
    assert result["position_equity_curve"].notna().all()


def main() -> None:
    test_default_output_columns()
    test_custom_output_columns()
    test_long_position_returns()
    test_short_position_returns()
    test_flat_position_returns()
    test_mixed_positions()
    test_cumulative_return_formula()
    test_equity_curve_formula()
    test_input_dataframe_is_immutable()
    test_missing_position_column()
    test_missing_return_column()
    test_invalid_position_values()
    test_non_numeric_return_column()
    test_index_is_preserved()
    test_generated_columns_contain_no_missing_values()

    print("🎉 ALL PORTFOLIO RETURN TESTS PASSED")


if __name__ == "__main__":
    main()
