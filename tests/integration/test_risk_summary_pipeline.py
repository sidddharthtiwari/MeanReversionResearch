import copy
import math

import pandas as pd
import pandas.testing as pdt

from src.analytics.constants import TRADING_DAYS_PER_YEAR
from src.analytics.risk import generate_risk_summary


# --------------------------------------------------
# INTEGRATION TEST
# Risk Summary Pipeline
# --------------------------------------------------


def test_risk_summary_pipeline() -> None:
    return_column = "returns"

    df = pd.DataFrame(
        {
            return_column: [0.01, -0.02, 0.03, -0.01],
        },
        index=pd.Index([10, 20, 30, 40], name="row_id"),
    )
    df_before = copy.deepcopy(df)

    result = generate_risk_summary(
        df,
        return_column=return_column,
        frequency="D",
    )

    # 1. Returned object is a dictionary.
    assert isinstance(result, dict)

    # 2. Dictionary contains the required metrics.
    assert set(result) == {
        "volatility",
        "annualised_volatility",
        "downside_deviation",
    }

    # 3. Computed values are numerically correct.
    returns = df[return_column]
    volatility = float(returns.std(ddof=1))
    annualised_volatility = float(
        volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    negative_returns = returns.clip(upper=0)
    downside_deviation = float(math.sqrt((negative_returns ** 2).mean()))
    assert math.isclose(
        result["volatility"],
        volatility,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["annualised_volatility"],
        annualised_volatility,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["downside_deviation"],
        downside_deviation,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    # 4. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 5. Number of rows remains unchanged.
    assert len(df) == len(df_before)

    # 6. Index remains unchanged.
    pdt.assert_index_equal(df.index, df_before.index)

    # 7. Dictionary contains exactly three metrics.
    assert len(result) == 3


def main() -> None:
    test_risk_summary_pipeline()

    print("🎉 RISK SUMMARY PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
