import copy

import pandas as pd
import pandas.testing as pdt

from src.backtest.runner import run_backtest
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
# No Execution Costs
# --------------------------------------------------


def test_no_execution_costs() -> None:
    df = _make_frame(
        [0.01, 0.02, -0.01],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0,
        slippage=0,
    )

    assert result is df
    pdt.assert_frame_equal(result, df)
    assert list(result.columns) == list(df.columns)


# --------------------------------------------------
# TEST 2
# Transaction Cost Only
# --------------------------------------------------


def test_transaction_cost_only() -> None:
    df = _make_frame(
        [0.01, 0.02, -0.01],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
        slippage=0,
    )

    assert "strategy_return_transaction_cost" in result.columns
    assert "strategy_return_net_return" in result.columns
    assert "strategy_return_slippage" not in result.columns
    assert "strategy_return_net_return_slippage" not in result.columns


# --------------------------------------------------
# TEST 3
# Slippage Only
# --------------------------------------------------


def test_slippage_only() -> None:
    df = _make_frame(
        [0.01, 0.02, -0.01],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0,
        slippage=0.001,
    )

    assert "strategy_return_slippage" in result.columns
    assert "strategy_return_net_return" in result.columns
    assert "strategy_return_transaction_cost" not in result.columns


# --------------------------------------------------
# TEST 4
# Transaction Cost And Slippage
# --------------------------------------------------


def test_transaction_cost_and_slippage() -> None:
    df = _make_frame(
        [0.01, 0.02, -0.01, 0.03],
        [FLAT_SIGNAL, LONG_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
        slippage=0.001,
    )

    first_stage_net = "strategy_return_net_return"
    slippage_column = f"{first_stage_net}_slippage"
    final_net = f"{first_stage_net}_net_return"

    assert "strategy_return_transaction_cost" in result.columns
    assert first_stage_net in result.columns
    assert slippage_column in result.columns
    assert final_net in result.columns

    # Slippage stage is keyed off the transaction-cost net-return column.
    assert "strategy_return_slippage" not in result.columns
    assert slippage_column.startswith(first_stage_net)


# --------------------------------------------------
# TEST 5
# Custom Transaction Cost Column
# --------------------------------------------------


def test_custom_transaction_cost_column() -> None:
    df = _make_frame(
        [0.01, 0.02],
        [FLAT_SIGNAL, LONG_SIGNAL],
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
        transaction_cost_column="custom_fees",
    )

    assert "custom_fees" in result.columns
    assert "strategy_return_transaction_cost" not in result.columns


# --------------------------------------------------
# TEST 6
# Custom Slippage Column
# --------------------------------------------------


def test_custom_slippage_column() -> None:
    df = _make_frame(
        [0.01, 0.02],
        [FLAT_SIGNAL, LONG_SIGNAL],
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        slippage=0.001,
        slippage_column="custom_slip",
    )

    assert "custom_slip" in result.columns
    assert "strategy_return_slippage" not in result.columns


# --------------------------------------------------
# TEST 7
# Shared Output Column Rejected
# --------------------------------------------------


def test_shared_output_column_rejected() -> None:
    df = _make_frame(
        [0.01, 0.02],
        [FLAT_SIGNAL, LONG_SIGNAL],
    )

    try:
        run_backtest(
            df,
            strategy_return_column="strategy_return",
            position_column="position",
            transaction_cost=0.001,
            slippage=0.001,
            output_column="shared_net",
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "output_column" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Input DataFrame Not Modified
# --------------------------------------------------


def test_input_dataframe_not_modified() -> None:
    df = _make_frame(
        [0.01, 0.02, -0.01],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
    )
    df_before = copy.deepcopy(df)

    run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
        slippage=0.001,
    )

    pdt.assert_frame_equal(df, df_before)


# --------------------------------------------------
# TEST 9
# Index Preserved
# --------------------------------------------------


def test_index_preserved() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    df = _make_frame(
        [0.01, 0.02, -0.01],
        [FLAT_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL],
        index=index,
    )
    result = run_backtest(
        df,
        strategy_return_column="strategy_return",
        position_column="position",
        transaction_cost=0.001,
        slippage=0.001,
    )

    pdt.assert_index_equal(result.index, df.index)


def main() -> None:
    test_no_execution_costs()
    test_transaction_cost_only()
    test_slippage_only()
    test_transaction_cost_and_slippage()
    test_custom_transaction_cost_column()
    test_custom_slippage_column()
    test_shared_output_column_rejected()
    test_input_dataframe_not_modified()
    test_index_preserved()

    print("🎉 ALL RUNNER TESTS PASSED")


if __name__ == "__main__":
    main()
