import copy
import math

import pandas as pd
import pandas.testing as pdt

from src.analytics.drawdown import generate_drawdown_summary


# --------------------------------------------------
# INTEGRATION TEST
# Drawdown Summary Pipeline
# --------------------------------------------------


def test_drawdown_summary_pipeline() -> None:
    return_column = "returns"

    df = pd.DataFrame(
        {
            return_column: [0.10, -0.05, -0.05, 0.20],
        },
        index=pd.Index([10, 20, 30, 40], name="row_id"),
    )
    df_before = copy.deepcopy(df)

    result = generate_drawdown_summary(df, return_column=return_column)

    # 1. Returned object is a dictionary.
    assert isinstance(result, dict)

    # 2. Dictionary contains the required metrics.
    assert set(result) == {
        "max_drawdown",
        "drawdown_duration",
    }

    # 3. Computed values are numerically correct.
    returns = df[return_column]
    equity_curve = (1.0 + returns).cumprod()
    drawdown_series = (equity_curve / equity_curve.cummax()) - 1.0
    expected_max_drawdown = float(drawdown_series.min())

    current_duration = 0
    expected_drawdown_duration = 0
    for value in drawdown_series:
        if value < 0:
            current_duration += 1
            if current_duration > expected_drawdown_duration:
                expected_drawdown_duration = current_duration
        else:
            current_duration = 0

    assert math.isclose(
        result["max_drawdown"],
        expected_max_drawdown,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["drawdown_duration"] == expected_drawdown_duration

    # 4. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 5. Number of rows remains unchanged.
    assert len(df) == len(df_before)

    # 6. Index remains unchanged.
    pdt.assert_index_equal(df.index, df_before.index)

    # 7. Dictionary contains exactly two metrics.
    assert len(result) == 2


def main() -> None:
    test_drawdown_summary_pipeline()

    print("🎉 DRAWDOWN SUMMARY PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
