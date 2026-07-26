import copy

import pandas as pd
import pandas.testing as pdt

from src.backtest.transaction_costs import apply_transaction_costs
from src.features.returns import compute_simple_returns
from src.features.rolling import compute_rolling_mean, compute_rolling_std
from src.features.zscore import compute_zscore
from src.portfolio.positions import generate_positions
from src.portfolio.returns import generate_strategy_returns
from src.signals.mean_reversion import generate_mean_reversion_signal


# --------------------------------------------------
# INTEGRATION TEST
# Transaction Cost Pipeline
# --------------------------------------------------


def test_transaction_cost_pipeline() -> None:
    window = 3
    entry_threshold = 1.0
    transaction_cost = 0.001

    position_column = f"zscore_{window}_signal_position"
    strategy_return_column = f"{position_column}_strategy_return"
    transaction_cost_column = f"{strategy_return_column}_transaction_cost"
    net_return_column = f"{strategy_return_column}_net_return"

    df = pd.DataFrame(
        {
            "symbol": ["A"] * 20,
            "close": [
                100.0,
                101.0,
                100.5,
                102.0,
                101.5,
                103.0,
                90.0,
                88.0,
                87.0,
                89.0,
                120.0,
                125.0,
                124.0,
                126.0,
                80.0,
                78.0,
                79.0,
                110.0,
                112.0,
                111.0,
            ],
        }
    )
    df_before = copy.deepcopy(df)

    with_returns = compute_simple_returns(df)
    with_mean = compute_rolling_mean(
        with_returns,
        window=window,
        column="simple_return",
    )
    with_std = compute_rolling_std(
        with_mean,
        window=window,
        column="simple_return",
    )
    with_zscore = compute_zscore(
        with_std,
        window=window,
        column="simple_return",
    )
    with_signal = generate_mean_reversion_signal(
        with_zscore,
        feature_column=f"zscore_{window}",
        entry_threshold=entry_threshold,
    )
    with_positions = generate_positions(
        with_signal,
        signal_column=f"zscore_{window}_signal",
    )
    prepared = with_positions.copy()
    prepared["simple_return"] = prepared["simple_return"].fillna(0.0)
    with_strategy_returns = generate_strategy_returns(
        prepared,
        position_column=position_column,
        asset_return_column="simple_return",
    )
    result = apply_transaction_costs(
        with_strategy_returns,
        strategy_return_column=strategy_return_column,
        position_column=position_column,
        transaction_cost=transaction_cost,
    )

    # 1. Pipeline executes successfully.
    assert result is not None

    # 2. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 3. Transaction-cost column exists.
    assert transaction_cost_column in result.columns

    # 4. Net-return column exists.
    assert net_return_column in result.columns

    # 5. Transaction costs are non-negative.
    assert (result[transaction_cost_column] >= 0).all()

    # 6. Net returns equal strategy_return - transaction_cost.
    expected_net_returns = (
        result[strategy_return_column] - result[transaction_cost_column]
    )
    pdt.assert_series_equal(
        result[net_return_column],
        expected_net_returns,
        check_names=False,
    )

    # 7. No NaN values in generated cost and net-return columns.
    assert result[transaction_cost_column].notna().all()
    assert result[net_return_column].notna().all()

    # 8. Index is preserved.
    pdt.assert_index_equal(result.index, df.index)

    # 9. Returned DataFrame length matches input length.
    assert len(result) == len(df)


def main() -> None:
    test_transaction_cost_pipeline()

    print("🎉 TRANSACTION COST PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
