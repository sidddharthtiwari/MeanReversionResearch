import pandas as pd

from src.data.loader import load_sector, load_sector_metadata
from src.data.validator import ValidationReport, validate_dataset


def load_bank_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared bank OHLC and metadata fixtures once."""
    return load_sector("bank"), load_sector_metadata("bank")


def print_report(title: str, report: ValidationReport) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(report.summary())


# --------------------------------------------------
# TEST 1
# Valid Dataset
# Should PASS
# --------------------------------------------------


def test_valid_dataset(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    report = validate_dataset(df, metadata)

    print_report("VALID DATASET", report)

    assert report.is_valid is True
    assert report.total_checks == 10
    assert report.passed_checks == 10
    assert report.failed_checks == 0
    assert report.errors == []
    assert isinstance(report.summary(), str)


# --------------------------------------------------
# TEST 2
# Duplicate Rows
# Should FAIL
# --------------------------------------------------


def test_duplicate_rows(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    duplicate_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = validate_dataset(duplicate_df, metadata)

    print_report("DUPLICATE", report)

    assert report.is_valid is False
    assert report.total_checks == 10
    assert report.failed_checks >= 1
    assert any("duplicate" in error.lower() for error in report.errors)


# --------------------------------------------------
# TEST 3
# Negative Price
# Should FAIL
# --------------------------------------------------


def test_negative_price(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    negative_price_df = df.copy()
    negative_price_df.loc[0, "close"] = -100

    report = validate_dataset(negative_price_df, metadata)

    print_report("NEGATIVE PRICE", report)

    assert report.is_valid is False
    assert report.failed_checks >= 1
    assert any("close" in error and "not > 0" in error for error in report.errors)


# --------------------------------------------------
# TEST 4
# Negative Volume
# Should FAIL
# --------------------------------------------------


def test_negative_volume(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    negative_volume_df = df.copy()
    negative_volume_df.loc[0, "volume"] = -50

    report = validate_dataset(negative_volume_df, metadata)

    print_report("NEGATIVE VOLUME", report)

    assert report.is_valid is False
    assert report.failed_checks == 1
    assert any("volume" in error and "< 0" in error for error in report.errors)


# --------------------------------------------------
# TEST 5
# Invalid OHLC Relationship
# Should FAIL
# --------------------------------------------------


def test_invalid_ohlc(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    ohlc_df = df.copy()
    ohlc_df.loc[0, "high"] = 1
    ohlc_df.loc[0, "close"] = 100

    report = validate_dataset(ohlc_df, metadata)

    print_report("OHLC", report)

    assert report.is_valid is False
    assert report.failed_checks == 1
    assert any("ohlc_relationship" in error for error in report.errors)
    assert any("high >= open" in error for error in report.errors)


# --------------------------------------------------
# TEST 6
# Unknown Symbol
# Should FAIL
# --------------------------------------------------


def test_unknown_symbol(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    symbol_df = df.copy()
    symbol_df["symbol"] = symbol_df["symbol"].astype(str)
    symbol_df.loc[0, "symbol"] = "FAKEBANK"

    report = validate_dataset(symbol_df, metadata)

    print_report("UNKNOWN SYMBOL", report)

    assert report.is_valid is False
    assert report.failed_checks == 1
    assert any("FAKEBANK" in error for error in report.errors)
    assert any("symbol" in warning for warning in report.warnings)


# --------------------------------------------------
# TEST 7
# Missing Values
# Should FAIL
# --------------------------------------------------


def test_missing_values(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    missing_df = df.copy()
    missing_df.loc[0, "close"] = None

    report = validate_dataset(missing_df, metadata)

    print_report("MISSING VALUES", report)

    assert report.is_valid is False
    assert report.failed_checks >= 1
    assert any("missing value" in error for error in report.errors)
    assert any("close" in error for error in report.errors)


# --------------------------------------------------
# TEST 8
# Missing Required Column
# Should FAIL
# --------------------------------------------------


def test_missing_required_column(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    missing_column_df = df.drop(columns=["volume"])

    report = validate_dataset(missing_column_df, metadata)

    print_report("MISSING COLUMN", report)

    assert report.is_valid is False
    assert report.total_checks == 10
    assert report.failed_checks == 4
    assert any("Missing required columns" in error for error in report.errors)
    assert any("volume" in error for error in report.errors)


# --------------------------------------------------
# TEST 9
# Empty Dataset
# Should FAIL
# --------------------------------------------------


def test_empty_dataset(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    empty_df = df.iloc[0:0].copy()

    report = validate_dataset(empty_df, metadata)

    print_report("EMPTY DATASET", report)

    assert report.is_valid is False
    assert report.failed_checks == 1
    assert any("empty" in error.lower() for error in report.errors)


# --------------------------------------------------
# TEST 10
# ValidationReport API
# --------------------------------------------------


def test_validation_report_api(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    report = validate_dataset(df, metadata)

    print_report("VALIDATION REPORT API", report)

    assert isinstance(report, ValidationReport)
    assert isinstance(report.is_valid, bool)
    assert report.total_checks == 10
    assert isinstance(report.failed_checks, int)
    assert isinstance(report.passed_checks, int)
    assert report.passed_checks + report.failed_checks == report.total_checks
    assert isinstance(report.errors, list)
    assert isinstance(report.warnings, list)
    assert isinstance(report.summary(), str)
    assert "Validation Report" in report.summary()


def main() -> None:
    df, metadata = load_bank_data()

    test_valid_dataset(df, metadata)
    test_duplicate_rows(df, metadata)
    test_negative_price(df, metadata)
    test_negative_volume(df, metadata)
    test_invalid_ohlc(df, metadata)
    test_unknown_symbol(df, metadata)
    test_missing_values(df, metadata)
    test_missing_required_column(df, metadata)
    test_empty_dataset(df, metadata)
    test_validation_report_api(df, metadata)


if __name__ == "__main__":
    main()
