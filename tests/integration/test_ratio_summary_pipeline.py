import copy
import math

import pandas as pd
import pandas.testing as pdt

from src.analytics.drawdown import generate_drawdown_summary
from src.analytics.ratios import generate_ratio_summary
from src.analytics.returns import generate_return_summary
from src.analytics.risk import generate_risk_summary


# --------------------------------------------------
# INTEGRATION TEST
# Ratio Summary Pipeline
# --------------------------------------------------


def test_ratio_summary_pipeline() -> None:
    return_column = "returns"
    risk_free_rate = 0.0

    df = pd.DataFrame(
        {
            return_column: [0.01, -0.02, 0.03, -0.01, 0.02],
        },
        index=pd.Index([10, 20, 30, 40, 50], name="row_id"),
    )
    df_before = copy.deepcopy(df)

    result = generate_ratio_summary(
        df,
        return_column=return_column,
        frequency="D",
        risk_free_rate=risk_free_rate,
    )

    # 1. Returned object is a dictionary.
    assert isinstance(result, dict)

    # 2. Dictionary contains exactly the required metrics.
    assert set(result) == {
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
    }
    assert len(result) == 3

    # 3. Computed values are numerically correct.
    # 7. Pipeline integrates return, risk, and drawdown summaries.
    return_summary = generate_return_summary(
        df,
        return_column=return_column,
        frequency="D",
    )
    risk_summary = generate_risk_summary(
        df,
        return_column=return_column,
        frequency="D",
    )
    drawdown_summary = generate_drawdown_summary(
        df,
        return_column=return_column,
    )

    expected_sharpe = (
        return_summary["annualised_return"] - risk_free_rate
    ) / risk_summary["annualised_volatility"]
    expected_sortino = (
        return_summary["annualised_return"] - risk_free_rate
    ) / risk_summary["downside_deviation"]
    expected_calmar = return_summary["cagr"] / abs(
        drawdown_summary["max_drawdown"]
    )

    assert math.isclose(
        result["sharpe_ratio"],
        expected_sharpe,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["sortino_ratio"],
        expected_sortino,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["calmar_ratio"],
        expected_calmar,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    # 4. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 5. Number of rows remains unchanged.
    assert len(df) == len(df_before)

    # 6. Index remains unchanged.
    pdt.assert_index_equal(df.index, df_before.index)


def main() -> None:
    test_ratio_summary_pipeline()

    print("🎉 RATIO SUMMARY PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
