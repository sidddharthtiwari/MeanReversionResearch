"""Dataset validation for quantitative research safety checks.

This module inspects OHLC and sector-basket metadata DataFrames and reports
issues. It never modifies the input data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass
class ValidationResult:
    """Outcome of a single validation check.

    Attributes:
        name: Identifier of the validation check.
        passed: Whether the check passed.
        errors: Blocking issues found by the check.
        warnings: Non-blocking issues found by the check.
    """

    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregate report across all validation checks.

    Attributes:
        results: Ordered list of individual validation results.
    """

    results: list[ValidationResult]

    @property
    def is_valid(self) -> bool:
        """Return True if every validation check passed."""
        return all(result.passed for result in self.results)

    @property
    def total_checks(self) -> int:
        """Return the number of validation checks executed."""
        return len(self.results)

    @property
    def passed_checks(self) -> int:
        """Return the number of validation checks that passed."""
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_checks(self) -> int:
        """Return the number of validation checks that failed."""
        return sum(1 for result in self.results if not result.passed)

    @property
    def errors(self) -> list[str]:
        """Return all error messages across every check."""
        return [
            f"[{result.name}] {message}"
            for result in self.results
            for message in result.errors
        ]

    @property
    def warnings(self) -> list[str]:
        """Return all warning messages across every check."""
        return [
            f"[{result.name}] {message}"
            for result in self.results
            for message in result.warnings
        ]

    def summary(self) -> str:
        """Return a human-readable summary of the validation report.

        Returns:
            Multi-line string describing overall status, check counts, errors,
            and warnings. Does not print.
        """
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            "Validation Report",
            "────────────────────────────────",
            "",
            f"Status        : {status}",
            "",
            f"Total Checks  : {self.total_checks}",
            f"Passed        : {self.passed_checks}",
            f"Failed        : {self.failed_checks}",
            "",
            "Errors",
            "------",
            "",
        ]

        if self.errors:
            lines.extend(f"- {message}" for message in self.errors)
        else:
            lines.append("None")

        lines.extend(
            [
                "",
                "Warnings",
                "--------",
                "",
            ]
        )

        if self.warnings:
            lines.extend(f"- {message}" for message in self.warnings)
        else:
            lines.append("None")

        return "\n".join(lines)


def _missing_required_columns(df: pd.DataFrame) -> list[str]:
    """Return required column names absent from ``df``."""
    return [column for column in _REQUIRED_COLUMNS if column not in df.columns]


def _metadata_symbol_column(metadata: pd.DataFrame) -> str | None:
    """Return the metadata symbol column name, if present."""
    for column in ("Symbol", "symbol"):
        if column in metadata.columns:
            return column
    return None


def _validate_columns(df: pd.DataFrame) -> ValidationResult:
    """Validate that all required OHLC columns are present."""
    missing = _missing_required_columns(df)
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {missing}")
    return ValidationResult(
        name="columns",
        passed=not errors,
        errors=errors,
    )


def _validate_empty(df: pd.DataFrame) -> ValidationResult:
    """Validate that the dataset is not empty."""
    errors: list[str] = []
    if df.empty:
        errors.append("Dataset is empty.")
    return ValidationResult(
        name="empty",
        passed=not errors,
        errors=errors,
    )


def _validate_duplicates(df: pd.DataFrame) -> ValidationResult:
    """Validate that the (symbol, date) primary key is unique."""
    errors: list[str] = []
    required = ["symbol", "date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        errors.append(
            f"Cannot validate duplicates; missing columns: {missing}"
        )
        return ValidationResult(name="duplicates", passed=False, errors=errors)

    duplicate_count = int(df.duplicated(subset=required).sum())
    if duplicate_count > 0:
        errors.append(
            f"Found {duplicate_count} duplicate (symbol, date) primary key(s)."
        )
    return ValidationResult(
        name="duplicates",
        passed=not errors,
        errors=errors,
    )


def _validate_missing_values(df: pd.DataFrame) -> ValidationResult:
    """Validate that required columns contain no missing values."""
    errors: list[str] = []
    missing_cols = _missing_required_columns(df)
    if missing_cols:
        errors.append(
            f"Cannot validate missing values; missing columns: {missing_cols}"
        )
        return ValidationResult(
            name="missing_values",
            passed=False,
            errors=errors,
        )

    null_counts = df.loc[:, list(_REQUIRED_COLUMNS)].isna().sum()
    for column, count in null_counts.items():
        if count > 0:
            errors.append(
                f"Column '{column}' has {int(count)} missing value(s)."
            )
    return ValidationResult(
        name="missing_values",
        passed=not errors,
        errors=errors,
    )


def _validate_prices(df: pd.DataFrame) -> ValidationResult:
    """Validate that all price columns are strictly positive."""
    errors: list[str] = []
    missing = [column for column in _PRICE_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Cannot validate prices; missing columns: {missing}")
        return ValidationResult(name="prices", passed=False, errors=errors)

    prices = df.loc[:, list(_PRICE_COLUMNS)]
    non_positive = (prices <= 0) | prices.isna()
    for column in _PRICE_COLUMNS:
        count = int(non_positive[column].sum())
        if count > 0:
            errors.append(
                f"Column '{column}' has {count} value(s) that are not > 0."
            )
    return ValidationResult(
        name="prices",
        passed=not errors,
        errors=errors,
    )


def _validate_volume(df: pd.DataFrame) -> ValidationResult:
    """Validate that volume values are non-negative."""
    errors: list[str] = []
    if "volume" not in df.columns:
        errors.append("Cannot validate volume; missing column: 'volume'")
        return ValidationResult(name="volume", passed=False, errors=errors)

    invalid = (df["volume"] < 0) | df["volume"].isna()
    count = int(invalid.sum())
    if count > 0:
        errors.append(f"Column 'volume' has {count} value(s) that are < 0.")
    return ValidationResult(
        name="volume",
        passed=not errors,
        errors=errors,
    )


def _validate_ohlc_relationship(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLC inequality constraints."""
    errors: list[str] = []
    required = list(_PRICE_COLUMNS)
    missing = [column for column in required if column not in df.columns]
    if missing:
        errors.append(
            f"Cannot validate OHLC relationships; missing columns: {missing}"
        )
        return ValidationResult(
            name="ohlc_relationship",
            passed=False,
            errors=errors,
        )

    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]

    checks = {
        "high >= open": high < open_,
        "high >= close": high < close,
        "high >= low": high < low,
        "low <= open": low > open_,
        "low <= close": low > close,
    }
    for rule, violated in checks.items():
        count = int(violated.fillna(True).sum())
        if count > 0:
            errors.append(
                f"OHLC relationship '{rule}' violated in {count} row(s)."
            )
    return ValidationResult(
        name="ohlc_relationship",
        passed=not errors,
        errors=errors,
    )


def _validate_dates(df: pd.DataFrame) -> ValidationResult:
    """Validate that dates are monotonically increasing within each symbol."""
    errors: list[str] = []
    required = ["symbol", "date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        errors.append(f"Cannot validate dates; missing columns: {missing}")
        return ValidationResult(name="dates", passed=False, errors=errors)

    if df.empty:
        return ValidationResult(name="dates", passed=True, errors=errors)

    bad_symbols = [
        symbol
        for symbol, dates in df.groupby("symbol", sort=False)["date"]
        if not dates.is_monotonic_increasing
    ]
    if bad_symbols:
        preview = bad_symbols[:10]
        suffix = "" if len(bad_symbols) <= 10 else f" (and {len(bad_symbols) - 10} more)"
        errors.append(
            "Dates are not monotonically increasing within symbol(s): "
            f"{preview}{suffix}."
        )
    return ValidationResult(
        name="dates",
        passed=not errors,
        errors=errors,
    )


def _validate_symbols(df: pd.DataFrame, metadata: pd.DataFrame) -> ValidationResult:
    """Validate that every OHLC symbol exists in the metadata basket."""
    errors: list[str] = []
    if "symbol" not in df.columns:
        errors.append("Cannot validate symbols; missing column: 'symbol'")
        return ValidationResult(name="symbols", passed=False, errors=errors)

    symbol_column = _metadata_symbol_column(metadata)
    if symbol_column is None:
        errors.append(
            "Cannot validate symbols; metadata has no 'Symbol' or 'symbol' column."
        )
        return ValidationResult(name="symbols", passed=False, errors=errors)

    ohlc_symbols = set(df["symbol"].dropna().astype(str).unique())
    metadata_symbols = set(metadata[symbol_column].dropna().astype(str).unique())
    missing = sorted(ohlc_symbols - metadata_symbols)
    if missing:
        preview = missing[:10]
        suffix = "" if len(missing) <= 10 else f" (and {len(missing) - 10} more)"
        errors.append(
            f"OHLC symbol(s) missing from metadata: {preview}{suffix}."
        )
    return ValidationResult(
        name="symbols",
        passed=not errors,
        errors=errors,
    )


def _validate_dtypes(df: pd.DataFrame) -> ValidationResult:
    """Validate column dtypes; compatible mismatches warn, incompatible fail."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = _missing_required_columns(df)
    if missing:
        errors.append(f"Cannot validate dtypes; missing columns: {missing}")
        return ValidationResult(
            name="dtypes",
            passed=False,
            errors=errors,
            warnings=warnings,
        )

    symbol_dtype = df["symbol"].dtype
    if isinstance(symbol_dtype, pd.CategoricalDtype) or str(symbol_dtype) == "category":
        pass
    elif pd.api.types.is_string_dtype(symbol_dtype) or pd.api.types.is_object_dtype(
        symbol_dtype
    ):
        warnings.append(
            f"Column 'symbol' has dtype '{symbol_dtype}' "
            "(compatible with category; category is preferred)."
        )
    else:
        errors.append(
            f"Column 'symbol' has incompatible dtype '{symbol_dtype}' "
            "(expected category)."
        )

    date_dtype = df["date"].dtype
    if pd.api.types.is_datetime64_any_dtype(date_dtype):
        pass
    else:
        errors.append(
            f"Column 'date' has incompatible dtype '{date_dtype}' "
            "(expected datetime64)."
        )

    for column in _PRICE_COLUMNS:
        dtype = df[column].dtype
        if pd.api.types.is_float_dtype(dtype):
            continue
        if pd.api.types.is_integer_dtype(dtype):
            warnings.append(
                f"Column '{column}' has integer dtype '{dtype}' "
                "(compatible with float)."
            )
        elif pd.api.types.is_numeric_dtype(dtype):
            warnings.append(
                f"Column '{column}' has numeric dtype '{dtype}' "
                "(compatible with float)."
            )
        else:
            errors.append(
                f"Column '{column}' has incompatible dtype '{dtype}' "
                "(expected float)."
            )

    volume_dtype = df["volume"].dtype
    if pd.api.types.is_integer_dtype(volume_dtype):
        pass
    elif pd.api.types.is_float_dtype(volume_dtype):
        warnings.append(
            f"Column 'volume' has float dtype '{volume_dtype}' "
            "(compatible with integer)."
        )
    elif pd.api.types.is_numeric_dtype(volume_dtype):
        warnings.append(
            f"Column 'volume' has numeric dtype '{volume_dtype}' "
            "(compatible with integer)."
        )
    else:
        errors.append(
            f"Column 'volume' has incompatible dtype '{volume_dtype}' "
            "(expected integer)."
        )

    return ValidationResult(
        name="dtypes",
        passed=not errors,
        errors=errors,
        warnings=warnings,
    )


def validate_dataset(df: pd.DataFrame, metadata: pd.DataFrame) -> ValidationReport:
    """Validate whether an OHLC dataset is safe for quantitative research.

    Runs every validation check against ``df`` and ``metadata``, collects all
    results, and returns a single report. Never modifies either DataFrame and
    never fails fast.

    Args:
        df: OHLC price DataFrame expected to contain the required columns.
        metadata: Sector basket metadata used to verify symbol membership.

    Returns:
        ValidationReport containing one ValidationResult per check.
    """
    results = [
        _validate_columns(df),
        _validate_empty(df),
        _validate_duplicates(df),
        _validate_missing_values(df),
        _validate_prices(df),
        _validate_volume(df),
        _validate_ohlc_relationship(df),
        _validate_dates(df),
        _validate_symbols(df, metadata),
        _validate_dtypes(df),
    ]
    return ValidationReport(results=results)
