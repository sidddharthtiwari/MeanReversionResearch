import math

import pandas as pd

from src.analytics.constants import (
    TRADING_DAYS_PER_YEAR,
    TRADING_MONTHS_PER_YEAR,
    TRADING_WEEKS_PER_YEAR,
)
from src.analytics.risk import generate_risk_summary


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
    """Compute expected risk-summary metrics for assertions."""
    volatility = float(returns.std(ddof=1))
    annualised_volatility = float(volatility * math.sqrt(periods_per_year))
    negative_returns = returns.clip(upper=0)
    downside_deviation = float(math.sqrt((negative_returns ** 2).mean()))
    return {
        "volatility": volatility,
        "annualised_volatility": annualised_volatility,
        "downside_deviation": downside_deviation,
    }


def _assert_summaries_equal(
    actual: dict[str, float],
    expected: dict[str, float],
) -> None:
    """Assert two risk-summary dictionaries match within float tolerance."""
    assert set(actual) == set(expected)
    for key in expected:
        assert math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=1e-12)


# --------------------------------------------------
# TEST 1
# Basic Risk Summary Computation
# --------------------------------------------------


def test_basic_risk_summary_computation() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01])
    result = generate_risk_summary(df, return_column="returns")

    expected = _expected_summary(df["returns"], TRADING_DAYS_PER_YEAR)
    _assert_summaries_equal(result, expected)


# --------------------------------------------------
# TEST 2
# Daily Frequency Annualisation
# --------------------------------------------------


def test_daily_frequency_annualisation() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01])
    result = generate_risk_summary(
        df,
        return_column="returns",
        frequency="D",
    )

    volatility = float(df["returns"].std(ddof=1))
    expected_annualised = volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isclose(
        result["annualised_volatility"],
        expected_annualised,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 3
# Weekly Frequency Annualisation
# --------------------------------------------------


def test_weekly_frequency_annualisation() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01])
    result = generate_risk_summary(
        df,
        return_column="returns",
        frequency="W",
    )

    volatility = float(df["returns"].std(ddof=1))
    expected_annualised = volatility * math.sqrt(TRADING_WEEKS_PER_YEAR)
    assert math.isclose(
        result["annualised_volatility"],
        expected_annualised,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 4
# Monthly Frequency Annualisation
# --------------------------------------------------


def test_monthly_frequency_annualisation() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01])
    result = generate_risk_summary(
        df,
        return_column="returns",
        frequency="M",
    )

    volatility = float(df["returns"].std(ddof=1))
    expected_annualised = volatility * math.sqrt(TRADING_MONTHS_PER_YEAR)
    assert math.isclose(
        result["annualised_volatility"],
        expected_annualised,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 5
# Missing Return Column Raises KeyError
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_risk_summary(df, return_column="missing")
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
        generate_risk_summary(df, return_column="returns")
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
        generate_risk_summary(df, return_column="returns")
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
        generate_risk_summary(
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
        generate_risk_summary(
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
# All Positive Returns
# --------------------------------------------------


def test_all_positive_returns_downside_deviation_is_zero() -> None:
    df = _make_return_frame([0.01, 0.02, 0.03])
    result = generate_risk_summary(df, return_column="returns")

    assert math.isclose(
        result["downside_deviation"],
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 11
# Mixed Positive and Negative Returns
# --------------------------------------------------


def test_mixed_returns_downside_deviation() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01])
    result = generate_risk_summary(df, return_column="returns")

    negative_returns = df["returns"].clip(upper=0)
    expected = float(math.sqrt((negative_returns ** 2).mean()))
    assert math.isclose(
        result["downside_deviation"],
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 12
# Returned Object Structure
# --------------------------------------------------


def test_returned_object_structure() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03])
    result = generate_risk_summary(df, return_column="returns")

    assert isinstance(result, dict)
    assert set(result) == {
        "volatility",
        "annualised_volatility",
        "downside_deviation",
    }


def main() -> None:
    test_basic_risk_summary_computation()
    test_daily_frequency_annualisation()
    test_weekly_frequency_annualisation()
    test_monthly_frequency_annualisation()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_invalid_frequency_type()
    test_unsupported_frequency()
    test_all_positive_returns_downside_deviation_is_zero()
    test_mixed_returns_downside_deviation()
    test_returned_object_structure()

    print("🎉 ALL RISK SUMMARY TESTS PASSED")


if __name__ == "__main__":
    main()
