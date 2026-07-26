import copy

import pandas as pd
import pandas.testing as pdt

from src.analytics.constants import TRADING_DAYS_PER_YEAR
from src.analytics.returns import generate_return_summary


# --------------------------------------------------
# INTEGRATION TEST
# Return Summary Pipeline
# --------------------------------------------------


def test_return_summary_pipeline() -> None:
    return_column = "returns"

    df = pd.DataFrame(
        {
            return_column: [0.01, 0.02, -0.01, 0.03],
        },
        index=pd.Index([10, 20, 30, 40], name="row_id"),
    )
    df_before = copy.deepcopy(df)

    result = generate_return_summary(
        df,
        return_column=return_column,
        frequency="D",
    )

    # 1. Returned object is a dictionary.
    assert isinstance(result, dict)

    # 2. Dictionary contains the required metrics.
    assert set(result) == {
        "total_return",
        "average_period_return",
        "annualised_return",
        "cagr",
    }

    # 3. Computed values are numerically correct.
    returns = df[return_column]
    total_return = float((1.0 + returns).prod() - 1.0)
    average_period_return = float(returns.mean())
    annualised_return = float(
        average_period_return * TRADING_DAYS_PER_YEAR
    )
    cagr = float(
        (1.0 + total_return)
        ** (TRADING_DAYS_PER_YEAR / len(returns))
        - 1.0
    )
    assert abs(result["total_return"] - total_return) < 1e-12
    assert abs(result["average_period_return"] - average_period_return) < 1e-12
    assert abs(result["annualised_return"] - annualised_return) < 1e-12
    assert abs(result["cagr"] - cagr) < 1e-12

    # 4. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 5. Number of rows remains unchanged.
    assert len(df) == len(df_before)

    # 6. Index remains unchanged.
    pdt.assert_index_equal(df.index, df_before.index)


def main() -> None:
    test_return_summary_pipeline()

    print("🎉 RETURN SUMMARY PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
