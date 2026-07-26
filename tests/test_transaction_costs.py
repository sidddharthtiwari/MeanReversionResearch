import copy

import pandas as pd
import pandas.testing as pdt

from src.backtest.transaction_costs import apply_transaction_costs
from src.signals.constants import FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL


def _make_frame(
    strategy_returns: list[float],
    positions: list[int],
    strategy_return_column: str = "strategy_return",
    position_column: str = "position",
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a small deterministic strategy-return and position frame."""
    return pd.DataFrame(
        {
            strategy_return_column: strategy_returns,
            position_column: positions,
        },
        index=index,
    )


# --------------------------------------------------
# TEST 1
# Default Output Columns
# --------------------------------------------------


def test_default_output_columns() -> None:
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
    )

    assert "strategy_return_transaction_cost" in result.columns
    assert "strategy_return_net_return" in result.columns


# --------------------------------------------------
# TEST 2
# Custom Output Columns
# --------------------------------------------------


def test_custom_output_columns() -> None:
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
        transaction_cost_column="fees",
        output_column="net",
    )

    assert "fees" in result.columns
    assert "net" in result.columns
    assert "strategy_return_transaction_cost" not in result.columns
    assert "strategy_return_net_return" not in result.columns


# --------------------------------------------------
# TEST 3
# No Trades
# --------------------------------------------------


def test_no_trades() -> None:
    df = _make_frame(
        [0.01, -0.02, 0.03, 0.04],
        [FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL, FLAT_SIGNAL],
    )
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.01,
    )

    expected_costs = pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64")
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )
    pdt.assert_series_equal(
        result["strategy_return_net_return"],
        result["strategy_return"],
        check_names=False,
    )


# --------------------------------------------------
# TEST 4
# Single Long Entry
# --------------------------------------------------


def test_single_long_entry() -> None:
    cost = 0.01
    df = _make_frame(
        [0.01, 0.02, 0.03, 0.04],
        [FLAT_SIGNAL, LONG_SIGNAL, LONG_SIGNAL, LONG_SIGNAL],
    )
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=cost,
    )

    expected_costs = pd.Series([0.0, cost, 0.0, 0.0], dtype="float64")
    expected_net = pd.Series(
        [0.01, 0.02 - cost, 0.03, 0.04],
        dtype="float64",
    )
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )
    pdt.assert_series_equal(
        result["strategy_return_net_return"],
        expected_net,
        check_names=False,
    )


# --------------------------------------------------
# TEST 5
# Single Short Entry
# --------------------------------------------------


def test_single_short_entry() -> None:
    cost = 0.01
    df = _make_frame(
        [0.01, 0.02, 0.03],
        [FLAT_SIGNAL, SHORT_SIGNAL, SHORT_SIGNAL],
    )
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=cost,
    )

    expected_costs = pd.Series([0.0, cost, 0.0], dtype="float64")
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )


# --------------------------------------------------
# TEST 6
# Position Exit
# --------------------------------------------------


def test_position_exit() -> None:
    cost = 0.01
    df = _make_frame([0.02, -0.01], [LONG_SIGNAL, FLAT_SIGNAL])
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=cost,
    )

    # Initial long is one trade; exiting to flat is a second unit trade.
    expected_costs = pd.Series([cost, cost], dtype="float64")
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )
    assert result["strategy_return_transaction_cost"].iloc[1] == cost


# --------------------------------------------------
# TEST 7
# Long To Short Reversal
# --------------------------------------------------


def test_long_to_short_reversal() -> None:
    cost = 0.01
    df = _make_frame([0.05, -0.02], [LONG_SIGNAL, SHORT_SIGNAL])
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=cost,
    )

    expected_costs = pd.Series([cost, 2 * cost], dtype="float64")
    expected_net = pd.Series([0.05 - cost, -0.02 - 2 * cost], dtype="float64")
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )
    pdt.assert_series_equal(
        result["strategy_return_net_return"],
        expected_net,
        check_names=False,
    )


# --------------------------------------------------
# TEST 8
# Short To Long Reversal
# --------------------------------------------------


def test_short_to_long_reversal() -> None:
    cost = 0.01
    df = _make_frame([-0.03, 0.04], [SHORT_SIGNAL, LONG_SIGNAL])
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=cost,
    )

    expected_costs = pd.Series([cost, 2 * cost], dtype="float64")
    expected_net = pd.Series([-0.03 - cost, 0.04 - 2 * cost], dtype="float64")
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )
    pdt.assert_series_equal(
        result["strategy_return_net_return"],
        expected_net,
        check_names=False,
    )


# --------------------------------------------------
# TEST 9
# Mixed Positions
# --------------------------------------------------


def test_mixed_positions() -> None:
    cost = 0.01
    df = _make_frame(
        [0.01, 0.02, 0.03, -0.01, 0.04, -0.02, 0.05, 0.01],
        [
            FLAT_SIGNAL,
            LONG_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
            SHORT_SIGNAL,
            SHORT_SIGNAL,
            LONG_SIGNAL,
            FLAT_SIGNAL,
        ],
    )
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=cost,
    )

    # Trade sizes: 0, 1, 0, 1, 1, 0, 2, 1
    expected_costs = pd.Series(
        [0.0, cost, 0.0, cost, cost, 0.0, 2 * cost, cost],
        dtype="float64",
    )
    expected_net = (
        pd.Series(
            [0.01, 0.02, 0.03, -0.01, 0.04, -0.02, 0.05, 0.01],
            dtype="float64",
        )
        - expected_costs
    )
    pdt.assert_series_equal(
        result["strategy_return_transaction_cost"],
        expected_costs,
        check_names=False,
    )
    pdt.assert_series_equal(
        result["strategy_return_net_return"],
        expected_net,
        check_names=False,
    )


# --------------------------------------------------
# TEST 10
# Missing Strategy Return Column
# --------------------------------------------------


def test_missing_strategy_return_column() -> None:
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])

    try:
        apply_transaction_costs(
            df,
            strategy_return_column="missing",
            position_column="position",
        )
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Missing Position Column
# --------------------------------------------------


def test_missing_position_column() -> None:
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])

    try:
        apply_transaction_costs(
            df,
            strategy_return_column="strategy_return",
            position_column="missing",
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
    df = _make_frame(
        [0.01, 0.02, 0.03],
        [LONG_SIGNAL, 2, FLAT_SIGNAL],
    )

    try:
        apply_transaction_costs(
            df,
            strategy_return_column="strategy_return",
            position_column="position",
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
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])
    df["strategy_return"] = df["strategy_return"].astype(str)

    try:
        apply_transaction_costs(
            df,
            strategy_return_column="strategy_return",
            position_column="position",
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 14
# Invalid Transaction Cost Type
# --------------------------------------------------


def test_invalid_transaction_cost_type() -> None:
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])

    try:
        apply_transaction_costs(
            df,
            strategy_return_column="strategy_return",
            position_column="position",
            transaction_cost="0.01",  # type: ignore[arg-type]
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "transaction_cost" in str(error)

    assert raised


# --------------------------------------------------
# TEST 15
# Negative Transaction Cost
# --------------------------------------------------


def test_negative_transaction_cost() -> None:
    df = _make_frame([0.01, 0.02], [FLAT_SIGNAL, LONG_SIGNAL])

    try:
        apply_transaction_costs(
            df,
            strategy_return_column="strategy_return",
            position_column="position",
            transaction_cost=-0.01,
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "transaction_cost" in str(error)

    assert raised


# --------------------------------------------------
# TEST 16
# Input DataFrame Not Modified
# --------------------------------------------------


def test_input_dataframe_not_modified() -> None:
    df = _make_frame(
        [0.01, 0.02, 0.03],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
    )
    df_before = copy.deepcopy(df)

    apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.01,
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 17
# Index Preserved
# --------------------------------------------------


def test_index_preserved() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    df = _make_frame(
        [0.01, 0.02, 0.03],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
        index=index,
    )
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.01,
    )

    pdt.assert_index_equal(result.index, df.index)


# --------------------------------------------------
# TEST 18
# No Missing Values
# --------------------------------------------------


def test_no_missing_values() -> None:
    df = _make_frame(
        [0.01, 0.02, -0.01, 0.03],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL, FLAT_SIGNAL],
    )
    result = apply_transaction_costs(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.01,
    )

    assert result["strategy_return_transaction_cost"].notna().all()
    assert result["strategy_return_net_return"].notna().all()


def main() -> None:
    test_default_output_columns()
    test_custom_output_columns()
    test_no_trades()
    test_single_long_entry()
    test_single_short_entry()
    test_position_exit()
    test_long_to_short_reversal()
    test_short_to_long_reversal()
    test_mixed_positions()
    test_missing_strategy_return_column()
    test_missing_position_column()
    test_invalid_position_values()
    test_non_numeric_return_column()
    test_invalid_transaction_cost_type()
    test_negative_transaction_cost()
    test_input_dataframe_not_modified()
    test_index_preserved()
    test_no_missing_values()

    print("🎉 ALL TRANSACTION COST TESTS PASSED")


if __name__ == "__main__":
    main()
