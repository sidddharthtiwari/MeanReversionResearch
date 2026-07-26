import copy

import pandas as pd
import pandas.testing as pdt

from src.analytics.cumulative import compute_cumulative_returns


# --------------------------------------------------
# INTEGRATION TEST
# Cumulative Returns Pipeline
# --------------------------------------------------


def test_cumulative_pipeline() -> None:
    return_column = "returns"
    cumulative_return_column = f"{return_column}_cumulative_return"

    df = pd.DataFrame(
        {
            return_column: [0.01, 0.02, -0.01, 0.03],
        },
        index=pd.Index([10, 20, 30, 40], name="row_id"),
    )
    df_before = copy.deepcopy(df)

    result = compute_cumulative_returns(df, return_column=return_column)

    # 1. Original return column is preserved.
    assert return_column in result.columns
    pdt.assert_series_equal(result[return_column], df[return_column])

    # 2. Default cumulative-return column is created.
    assert cumulative_return_column in result.columns

    # 3. Computed cumulative-return values are correct.
    expected = (1.0 + df[return_column]).cumprod() - 1.0
    pdt.assert_series_equal(
        result[cumulative_return_column],
        expected,
        check_names=False,
    )

    # 4. Original DataFrame remains unchanged.
    pdt.assert_frame_equal(df, df_before)

    # 5. Returned DataFrame has one additional column.
    assert len(result.columns) == len(df.columns) + 1

    # 6. Row count remains unchanged.
    assert len(result) == len(df)

    # 7. Index is preserved.
    pdt.assert_index_equal(result.index, df.index)


def main() -> None:
    test_cumulative_pipeline()

    print("🎉 CUMULATIVE RETURNS PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
