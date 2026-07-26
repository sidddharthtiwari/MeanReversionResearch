import copy
import math

import pandas as pd
import pandas.testing as pdt

from src.analytics.drawdown import generate_drawdown_summary
from src.analytics.ratios import generate_ratio_summary
from src.analytics.returns import generate_return_summary
from src.analytics.risk import generate_risk_summary
from src.analytics.summary import generate_summary

_EXPECTED_SUMMARY_KEYS = {
    "total_return",
    "average_period_return",
    "annualised_return",
    "cagr",
    "volatility",
    "annualised_volatility",
    "downside_deviation",
    "max_drawdown",
    "drawdown_duration",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
}


# --------------------------------------------------
# INTEGRATION TEST
# Summary Pipeline
# --------------------------------------------------


def test_summary_pipeline() -> None:
    return_column = "returns"
    risk_free_rate = 0.0

    df = pd.DataFrame(
        {
            return_column: [0.01, -0.02, 0.03, -0.01, 0.02],
        },
        index=pd.Index([10, 20, 30, 40, 50], name="row_id"),
    )
    df_before = copy.deepcopy(df)

    result = generate_summary(
        df,
        return_column=return_column,
        frequency="D",
        risk_free_rate=risk_free_rate,
    )

    # 1. End-to-end pipeline produces a flat summary dictionary.
    assert isinstance(result, dict)

    # 2. Final summary contains exactly 12 metrics.
    assert set(result) == _EXPECTED_SUMMARY_KEYS
    assert len(result) == 12

    # 3. Summary matches the combined outputs from the four summary modules.
    expected: dict[str, float | int] = {}
    expected.update(
        generate_return_summary(
            df,
            return_column=return_column,
            frequency="D",
        )
    )
    expected.update(
        generate_risk_summary(
            df,
            return_column=return_column,
            frequency="D",
        )
    )
    expected.update(
        generate_drawdown_summary(
            df,
            return_column=return_column,
        )
    )
    expected.update(
        generate_ratio_summary(
            df,
            return_column=return_column,
            frequency="D",
            risk_free_rate=risk_free_rate,
        )
    )

    assert set(result) == set(expected)
    for key, value in expected.items():
        if isinstance(value, int) and not isinstance(value, bool):
            assert result[key] == value
        else:
            assert math.isclose(
                float(result[key]),
                float(value),
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
    test_summary_pipeline()

    print("🎉 SUMMARY PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
