import math

import pandas as pd

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


def _make_return_frame(
    returns: list[float],
    return_column: str = "returns",
) -> pd.DataFrame:
    """Build a small deterministic period-return frame."""
    return pd.DataFrame({return_column: returns})


# --------------------------------------------------
# TEST 1
# Happy Path
# --------------------------------------------------


def test_happy_path() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01, 0.02])
    result = generate_summary(df, return_column="returns")

    expected: dict[str, float | int] = {}
    expected.update(
        generate_return_summary(df, return_column="returns")
    )
    expected.update(generate_risk_summary(df, return_column="returns"))
    expected.update(
        generate_drawdown_summary(df, return_column="returns")
    )
    expected.update(generate_ratio_summary(df, return_column="returns"))

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


# --------------------------------------------------
# TEST 2
# Missing Return Column Raises KeyError
# --------------------------------------------------


def test_missing_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_summary(df, return_column="missing")
        raised = False
    except KeyError as error:
        raised = True
        assert "missing" in str(error)

    assert raised


# --------------------------------------------------
# TEST 3
# Non-Numeric Return Column Raises TypeError
# --------------------------------------------------


def test_non_numeric_return_column() -> None:
    df = _make_return_frame([0.01, -0.02])
    df["returns"] = df["returns"].astype(str)

    try:
        generate_summary(df, return_column="returns")
        raised = False
    except TypeError as error:
        raised = True
        assert "must be numeric" in str(error)

    assert raised


# --------------------------------------------------
# TEST 4
# Empty Return Series Raises ValueError
# --------------------------------------------------


def test_empty_return_series() -> None:
    df = pd.DataFrame({"returns": pd.Series(dtype="float64")})

    try:
        generate_summary(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert "must not be empty" in str(error)

    assert raised


# --------------------------------------------------
# TEST 5
# Invalid Frequency Type Raises TypeError
# --------------------------------------------------


def test_invalid_frequency_type() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_summary(
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
# TEST 6
# Unsupported Frequency Raises ValueError
# --------------------------------------------------


def test_unsupported_frequency() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_summary(
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
# TEST 7
# Invalid Risk-Free Rate Type Raises TypeError
# --------------------------------------------------


def test_invalid_risk_free_rate_type() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_summary(
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
# TEST 8
# Non-Finite Risk-Free Rate Raises ValueError
# --------------------------------------------------


def test_non_finite_risk_free_rate() -> None:
    df = _make_return_frame([0.01, -0.02])

    try:
        generate_summary(
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
# TEST 9
# CAGR Undefined Propagates ValueError
# --------------------------------------------------


def test_cagr_undefined_propagates_value_error() -> None:
    df = _make_return_frame([-1.0])

    try:
        generate_summary(df, return_column="returns")
        raised = False
    except ValueError as error:
        raised = True
        assert (
            str(error)
            == "CAGR is undefined when cumulative wealth "
            "is zero or negative."
        )

    assert raised


# --------------------------------------------------
# TEST 10
# Returned Dictionary Contains Exactly 12 Flat Metrics
# --------------------------------------------------


def test_returned_dictionary_contains_exactly_12_flat_metrics() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01, 0.02])
    result = generate_summary(df, return_column="returns")

    assert isinstance(result, dict)
    assert set(result) == _EXPECTED_SUMMARY_KEYS
    assert len(result) == 12


# --------------------------------------------------
# TEST 11
# No Nested Dictionaries
# --------------------------------------------------


def test_no_nested_dictionaries() -> None:
    df = _make_return_frame([0.01, -0.02, 0.03, -0.01, 0.02])
    result = generate_summary(df, return_column="returns")

    for value in result.values():
        assert not isinstance(value, dict)
        assert isinstance(value, (int, float))


def main() -> None:
    test_happy_path()
    test_missing_return_column()
    test_non_numeric_return_column()
    test_empty_return_series()
    test_invalid_frequency_type()
    test_unsupported_frequency()
    test_invalid_risk_free_rate_type()
    test_non_finite_risk_free_rate()
    test_cagr_undefined_propagates_value_error()
    test_returned_dictionary_contains_exactly_12_flat_metrics()
    test_no_nested_dictionaries()

    print("🎉 ALL SUMMARY TESTS PASSED")


if __name__ == "__main__":
    main()
