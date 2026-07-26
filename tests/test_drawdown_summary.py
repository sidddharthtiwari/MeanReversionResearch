import math

import pandas as pd

from src.analytics.drawdown import generate_drawdown_summary


def _make_return_frame(
    returns: list[float],
    return_column: str = "returns",
) -> pd.DataFrame:
    """Build a small deterministic period-return frame."""
    return pd.DataFrame({return_column: returns})


def _expected_summary(returns: pd.Series) -> dict[str, float | int]:
    """Compute expected drawdown-summary metrics for assertions."""
    equity_curve = (1.0 + returns).cumprod()
    drawdown_series = (equity_curve / equity_curve.cummax()) - 1.0
    max_drawdown = float(drawdown_series.min())

    current_duration = 0
    maximum_duration = 0
    for value in drawdown_series:
        if value < 0:
            current_duration += 1
            if current_duration > maximum_duration:
                maximum_duration = current_duration
        else:
            current_duration = 0

    return {
        "max_drawdown": max_drawdown,
        "drawdown_duration": maximum_duration,
    }


# --------------------------------------------------
# TEST 1
# Basic Drawdown Summary Computation
# --------------------------------------------------


def test_basic_drawdown_summary_computation() -> None:
    df = _make_return_frame([0.10, -0.05, -0.05, 0.20])
    result = generate_drawdown_summary(df, return_column="returns")
    expected = _expected_summary(df["returns"])

    assert math.isclose(
        result["max_drawdown"],
        expected["max_drawdown"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["drawdown_duration"] == expected["drawdown_duration"]


# --------------------------------------------------
# TEST 2
# Monotonically Increasing Equity
# --------------------------------------------------


def test_monotonically_increasing_equity() -> None:
    df = _make_return_frame([0.01, 0.02, 0.03, 0.04])
    result = generate_drawdown_summary(df, return_column="returns")

    assert math.isclose(
        result["max_drawdown"],
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["drawdown_duration"] == 0


# --------------------------------------------------
# TEST 3
# Single Drawdown Period
# --------------------------------------------------


def test_single_drawdown_period() -> None:
    df = _make_return_frame([0.10, -0.05, -0.05, 0.20])
    result = generate_drawdown_summary(df, return_column="returns")
    expected = _expected_summary(df["returns"])

    assert math.isclose(
        result["max_drawdown"],
        expected["max_drawdown"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["drawdown_duration"] == 2


# --------------------------------------------------
# TEST 4
# Multiple Drawdown Periods
# --------------------------------------------------


def test_multiple_drawdown_periods() -> None:
    # Short deep drawdown, then a longer shallower drawdown.
    df = _make_return_frame(
        [0.10, -0.20, 0.25, 0.10, -0.05, -0.05, -0.05, 0.20]
    )
    result = generate_drawdown_summary(df, return_column="returns")
    expected = _expected_summary(df["returns"])

    assert math.isclose(
        result["max_drawdown"],
        expected["max_drawdown"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["drawdown_duration"] == expected["drawdown_duration"]
    assert result["drawdown_duration"] == 3
    assert result["max_drawdown"] < -0.15


# --------------------------------------------------
# TEST 5
# Immediate Recovery
# --------------------------------------------------


def test_immediate_recovery() -> None:
    df = _make_return_frame([0.10, -0.05, 0.10])
    result = generate_drawdown_summary(df, return_column="returns")
    expected = _expected_summary(df["returns"])

    assert math.isclose(
        result["max_drawdown"],
        expected["max_drawdown"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["drawdown_duration"] == 1


# --------------------------------------------------
# TEST 6
# Missing Return Column Raises KeyError
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_drawdown_summary(df, return_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Non-Numeric Return Column Raises TypeError
# --------------------------------------------------


def test_non_numeric_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])
    df["returns"] = df["returns"].astype(str)

    try:
        generate_drawdown_summary(df, return_column="returns")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Empty Return Series Raises ValueError
# --------------------------------------------------


def test_empty_return_series() -> None:
    df = pd.DataFrame({"returns": pd.Series(dtype="float64")})

    try:
        generate_drawdown_summary(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert "must not be empty" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Returned Object Structure
# --------------------------------------------------


def test_returned_object_structure() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03])
    result = generate_drawdown_summary(df, return_column="returns")

    assert isinstance(result, dict)
    assert set(result) == {
        "max_drawdown",
        "drawdown_duration",
    }


def main() -> None:
    test_basic_drawdown_summary_computation()
    test_monotonically_increasing_equity()
    test_single_drawdown_period()
    test_multiple_drawdown_periods()
    test_immediate_recovery()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_returned_object_structure()

    print("🎉 ALL DRAWDOWN SUMMARY TESTS PASSED")


if __name__ == "__main__":
    main()
