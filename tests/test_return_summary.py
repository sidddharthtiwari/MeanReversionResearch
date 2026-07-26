import pandas as pd

from src.analytics.constants import (
    TRADING_DAYS_PER_YEAR,
    TRADING_MONTHS_PER_YEAR,
    TRADING_WEEKS_PER_YEAR,
)
from src.analytics.returns import generate_return_summary


def _make_return_frame(
    returns: list[float],
    return_column: str = "returns",
) -> pd.DataFrame:
    """Build a small deterministic period-return frame."""
    return pd.DataFrame({return_column: returns})


def _expected_summary(
    returns: pd.Series,
    periods_per_year: int,
) -> dict[str, float]:
    """Compute expected return-summary metrics for assertions."""
    total_return = float((1.0 + returns).prod() - 1.0)
    average_period_return = float(returns.mean())
    annualised_return = float(average_period_return * periods_per_year)
    cagr = float(
        (1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0
    )
    return {
        "total_return": total_return,
        "average_period_return": average_period_return,
        "annualised_return": annualised_return,
        "cagr": cagr,
    }


def _assert_summaries_equal(
    actual: dict[str, float],
    expected: dict[str, float],
) -> None:
    """Assert two return-summary dictionaries match within float tolerance."""
    assert set(actual) == set(expected)
    for key in expected:
        assert abs(actual[key] - expected[key]) < 1e-12


# --------------------------------------------------
# TEST 1
# Basic Return Summary Computation
# --------------------------------------------------


def test_basic_return_summary_computation() -> None:
    df = _make_return_frame([0.01, 0.02, -0.01])
    result = generate_return_summary(df, return_column="returns")

    expected = _expected_summary(df["returns"], TRADING_DAYS_PER_YEAR)
    _assert_summaries_equal(result, expected)


# --------------------------------------------------
# TEST 2
# Daily Frequency Annualisation
# --------------------------------------------------


def test_daily_frequency_annualisation() -> None:
    df = _make_return_frame([0.01, 0.02, -0.01])
    result = generate_return_summary(
        df,
        return_column="returns",
        frequency="D",
    )

    expected = _expected_summary(df["returns"], TRADING_DAYS_PER_YEAR)
    _assert_summaries_equal(result, expected)


# --------------------------------------------------
# TEST 3
# Weekly Frequency Annualisation
# --------------------------------------------------


def test_weekly_frequency_annualisation() -> None:
    df = _make_return_frame([0.01, 0.02, -0.01])
    result = generate_return_summary(
        df,
        return_column="returns",
        frequency="W",
    )

    expected = _expected_summary(df["returns"], TRADING_WEEKS_PER_YEAR)
    _assert_summaries_equal(result, expected)


# --------------------------------------------------
# TEST 4
# Monthly Frequency Annualisation
# --------------------------------------------------


def test_monthly_frequency_annualisation() -> None:
    df = _make_return_frame([0.01, 0.02, -0.01])
    result = generate_return_summary(
        df,
        return_column="returns",
        frequency="M",
    )

    expected = _expected_summary(df["returns"], TRADING_MONTHS_PER_YEAR)
    _assert_summaries_equal(result, expected)


# --------------------------------------------------
# TEST 5
# Missing Return Column Raises KeyError
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_return_summary(df, return_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Non-Numeric Return Column Raises TypeError
# --------------------------------------------------


def test_non_numeric_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])
    df["returns"] = df["returns"].astype(str)

    try:
        generate_return_summary(df, return_column="returns")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Empty Return Series Raises ValueError
# --------------------------------------------------


def test_empty_return_series() -> None:
    df = pd.DataFrame({"returns": pd.Series(dtype="float64")})

    try:
        generate_return_summary(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert "must not be empty" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Invalid Frequency Type Raises TypeError
# --------------------------------------------------


def test_invalid_frequency_type() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_return_summary(
            df,
            return_column="returns",
            frequency=252,  # type: ignore[arg-type]
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "frequency" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Unsupported Frequency Raises ValueError
# --------------------------------------------------


def test_unsupported_frequency() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_return_summary(
            df,
            return_column="returns",
            frequency="Y",
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "frequency" in str(error)

    assert raised


# --------------------------------------------------
# TEST 10
# CAGR Undefined for Non-Positive Wealth
# --------------------------------------------------


def test_cagr_undefined_for_non_positive_wealth() -> None:
    df = _make_return_frame([-1.0])

    try:
        generate_return_summary(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert (
            str(error)
            == "CAGR is undefined when cumulative wealth "
            "is zero or negative."
        )

    assert raised


def main() -> None:
    test_basic_return_summary_computation()
    test_daily_frequency_annualisation()
    test_weekly_frequency_annualisation()
    test_monthly_frequency_annualisation()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_invalid_frequency_type()
    test_unsupported_frequency()
    test_cagr_undefined_for_non_positive_wealth()

    print("🎉 ALL RETURN SUMMARY TESTS PASSED")


if __name__ == "__main__":
    main()
