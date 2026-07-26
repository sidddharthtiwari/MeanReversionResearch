import math

import pandas as pd

from src.analytics.constants import TRADING_DAYS_PER_YEAR
from src.analytics.ratios import generate_ratio_summary


def _make_return_frame(
    returns: list[float],
    return_column: str = "returns",
) -> pd.DataFrame:
    """Build a small deterministic period-return frame."""
    return pd.DataFrame({return_column: returns})


def _expected_ratios(
    returns: pd.Series,
    frequency_periods: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Manually compute expected ratio-summary metrics."""
    total_return = float((1.0 + returns).prod() - 1.0)
    average_period_return = float(returns.mean())
    annualised_return = float(average_period_return * frequency_periods)
    cagr = float(
        (1.0 + total_return) ** (frequency_periods / len(returns)) - 1.0
    )

    volatility = float(returns.std(ddof=1))
    annualised_volatility = float(volatility * math.sqrt(frequency_periods))
    negative_returns = returns.clip(upper=0)
    downside_deviation = float(math.sqrt((negative_returns ** 2).mean()))

    equity_curve = (1.0 + returns).cumprod()
    max_drawdown = float((equity_curve / equity_curve.cummax() - 1.0).min())

    if annualised_volatility <= 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(
            (annualised_return - risk_free_rate) / annualised_volatility
        )

    if downside_deviation <= 0:
        sortino_ratio = 0.0
    else:
        sortino_ratio = float(
            (annualised_return - risk_free_rate) / downside_deviation
        )

    absolute_max_drawdown = abs(max_drawdown)
    if absolute_max_drawdown <= 0:
        calmar_ratio = 0.0
    else:
        calmar_ratio = float(cagr / absolute_max_drawdown)

    return {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
    }


# --------------------------------------------------
# TEST 1
# Basic Ratio Computation
# --------------------------------------------------


def test_basic_ratio_computation() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01, 0.02])
    result = generate_ratio_summary(df, return_column="returns")
    expected = _expected_ratios(df["returns"])

    assert math.isclose(
        result["sharpe_ratio"],
        expected["sharpe_ratio"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["sortino_ratio"],
        expected["sortino_ratio"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["calmar_ratio"],
        expected["calmar_ratio"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 2
# Zero Annualised Volatility
# --------------------------------------------------


def test_zero_annualised_volatility() -> None:
    df = _make_return_frame([0.01, 0.01, 0.01, 0.01])
    result = generate_ratio_summary(df, return_column="returns")

    assert math.isclose(
        result["sharpe_ratio"],
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 3
# Zero Downside Deviation
# --------------------------------------------------


def test_zero_downside_deviation() -> None:
    df = _make_return_frame([0.0, 0.01, 0.02, 0.03])
    result = generate_ratio_summary(df, return_column="returns")

    assert math.isclose(
        result["sortino_ratio"],
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --------------------------------------------------
# TEST 4
# Zero Maximum Drawdown
# --------------------------------------------------


def test_zero_maximum_drawdown() -> None:
    df = _make_return_frame([0.01, 0.02, 0.03, 0.04])
    result = generate_ratio_summary(df, return_column="returns")

    assert math.isclose(
        result["calmar_ratio"],
        0.0,
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
        generate_ratio_summary(df, return_column="missing")
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
        generate_ratio_summary(df, return_column="returns")
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
        generate_ratio_summary(df, return_column="returns")
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
        generate_ratio_summary(
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
        generate_ratio_summary(
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
# Invalid Risk-Free Rate Type Raises TypeError
# --------------------------------------------------


def test_invalid_risk_free_rate_type() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_ratio_summary(
            df,
            return_column="returns",
            risk_free_rate="0.01",  # type: ignore[arg-type]
        )
        raised = False
    except TypeError as error:
        raised = True
        assert "risk_free_rate" in str(error)

    assert raised


# --------------------------------------------------
# TEST 11
# Non-Finite Risk-Free Rate Raises ValueError
# --------------------------------------------------


def test_non_finite_risk_free_rate() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_ratio_summary(
            df,
            return_column="returns",
            risk_free_rate=math.nan,
        )
        raised = False
    except ValueError as error:
        raised = True
        assert "risk_free_rate" in str(error)

    assert raised


# --------------------------------------------------
# TEST 12
# Returned Object Structure
# --------------------------------------------------


def test_returned_object_structure() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03])
    result = generate_ratio_summary(df, return_column="returns")

    assert isinstance(result, dict)
    assert set(result) == {
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
    }


def main() -> None:
    test_basic_ratio_computation()
    test_zero_annualised_volatility()
    test_zero_downside_deviation()
    test_zero_maximum_drawdown()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_invalid_frequency_type()
    test_unsupported_frequency()
    test_invalid_risk_free_rate_type()
    test_non_finite_risk_free_rate()
    test_returned_object_structure()

    print("🎉 ALL RATIO SUMMARY TESTS PASSED")


if __name__ == "__main__":
    main()
