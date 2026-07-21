"""Dataset auditing for quantitative research descriptive analysis.

This module inspects OHLC and sector-basket metadata DataFrames and reports
descriptive statistics. It never validates, cleans, transforms, or modifies
input data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_BYTES_PER_MEGABYTE = 1024 * 1024
_PRIMARY_KEY_COLUMNS: tuple[str, ...] = ("symbol", "date")
_NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass
class NumericSummary:
    """Descriptive statistics for a single numeric column.

    Attributes:
        minimum: Minimum observed value.
        maximum: Maximum observed value.
        mean: Arithmetic mean.
        median: Median value.
        std: Sample standard deviation (pandas default, ddof=1).
    """

    minimum: float
    maximum: float
    mean: float
    median: float
    std: float


@dataclass
class AuditReport:
    """Descriptive audit of an OHLC dataset and its basket metadata.

    Attributes:
        rows: Number of rows in the OHLC DataFrame.
        columns: Number of columns in the OHLC DataFrame.
        memory_usage_mb: Deep memory usage of the OHLC DataFrame in megabytes.
        total_symbols: Number of unique symbols in the OHLC DataFrame.
        symbols: Sorted list of unique OHLC symbols.
        start_date: Earliest date in the OHLC DataFrame.
        end_date: Latest date in the OHLC DataFrame.
        trading_days: Number of unique dates in the OHLC DataFrame.
        missing_by_column: Missing-value count for each OHLC column.
        duplicate_primary_keys: Count of duplicated (symbol, date) rows.
        open_stats: Numeric summary for the open column.
        high_stats: Numeric summary for the high column.
        low_stats: Numeric summary for the low column.
        close_stats: Numeric summary for the close column.
        volume_stats: Numeric summary for the volume column.
        metadata_symbols: Number of unique symbols in metadata.
        matched_symbols: Count of symbols present in both OHLC and metadata.
        missing_in_metadata: OHLC symbols absent from metadata.
        unused_metadata_symbols: Metadata symbols absent from OHLC.
    """

    rows: int
    columns: int
    memory_usage_mb: float
    total_symbols: int
    symbols: list[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trading_days: int
    missing_by_column: dict[str, int]
    duplicate_primary_keys: int
    open_stats: NumericSummary
    high_stats: NumericSummary
    low_stats: NumericSummary
    close_stats: NumericSummary
    volume_stats: NumericSummary
    metadata_symbols: int
    matched_symbols: int
    missing_in_metadata: list[str]
    unused_metadata_symbols: list[str]

    def summary(self) -> str:
        """Return a human-readable summary of the audit report.

        Returns:
            Multi-line string suitable for terminal display. Does not print.
        """
        lines = [
            "Audit Report",
            "────────────────────────────────",
            "",
            "Shape",
            "-----",
            "",
            f"Rows              : {self.rows}",
            f"Columns           : {self.columns}",
            f"Memory Usage (MB) : {self.memory_usage_mb:.4f}",
            "",
            "Symbols",
            "-------",
            "",
            f"Total Symbols     : {self.total_symbols}",
            f"Symbols           : {self._format_symbol_preview(self.symbols)}",
            "",
            "Dates",
            "-----",
            "",
            f"Start Date        : {self.start_date.strftime('%Y-%m-%d')}",
            f"End Date          : {self.end_date.strftime('%Y-%m-%d')}",
            f"Trading Days      : {self.trading_days}",
            "",
            "Data Quality Snapshot",
            "---------------------",
            "",
            f"Duplicate Keys    : {self.duplicate_primary_keys}",
            "Missing by Column :",
        ]

        if self.missing_by_column:
            lines.extend(
                f"  - {column}: {count}"
                for column, count in self.missing_by_column.items()
            )
        else:
            lines.append("  None")

        lines.extend(
            [
                "",
                "Price Statistics",
                "----------------",
                "",
                self._format_numeric_summary("open", self.open_stats),
                self._format_numeric_summary("high", self.high_stats),
                self._format_numeric_summary("low", self.low_stats),
                self._format_numeric_summary("close", self.close_stats),
                "",
                "Volume Statistics",
                "-----------------",
                "",
                self._format_numeric_summary("volume", self.volume_stats),
                "",
                "Metadata Coverage",
                "-----------------",
                "",
                f"Metadata Symbols         : {self.metadata_symbols}",
                f"Matched Symbols          : {self.matched_symbols}",
                f"Missing in Metadata      : {self.missing_in_metadata or 'None'}",
                f"Unused Metadata Symbols  : {self.unused_metadata_symbols or 'None'}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_symbol_preview(symbols: list[str], limit: int = 10) -> str:
        """Return a truncated, comma-separated symbol preview for display."""
        if not symbols:
            return "None"
        if len(symbols) <= limit:
            return ", ".join(symbols)
        preview = ", ".join(symbols[:limit])
        remaining = len(symbols) - limit
        return f"{preview}, ... (+{remaining} more)"

    @staticmethod
    def _format_numeric_summary(name: str, stats: NumericSummary) -> str:
        """Format one NumericSummary as a compact multi-field line."""
        return (
            f"{name:6} | "
            f"min={stats.minimum:.6g}  "
            f"max={stats.maximum:.6g}  "
            f"mean={stats.mean:.6g}  "
            f"median={stats.median:.6g}  "
            f"std={stats.std:.6g}"
        )


def _metadata_symbol_column(metadata: pd.DataFrame) -> str | None:
    """Return the metadata symbol column name, if present."""
    for column in ("Symbol", "symbol"):
        if column in metadata.columns:
            return column
    return None


def _audit_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return row and column counts for the dataset."""
    return int(len(df)), int(df.shape[1])


def _audit_memory(df: pd.DataFrame) -> float:
    """Return deep memory usage of the DataFrame in megabytes."""
    return float(df.memory_usage(deep=True).sum() / _BYTES_PER_MEGABYTE)


def _audit_symbols(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Return unique symbol count and sorted symbol list."""
    symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    return len(symbols), symbols


def _audit_dates(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Return start date, end date, and unique trading-day count."""
    start_date = pd.Timestamp(df["date"].min())
    end_date = pd.Timestamp(df["date"].max())
    trading_days = int(df["date"].nunique(dropna=True))
    return start_date, end_date, trading_days


def _audit_missing(df: pd.DataFrame) -> dict[str, int]:
    """Return missing-value counts keyed by column name."""
    null_counts = df.isna().sum()
    return {column: int(count) for column, count in null_counts.items()}


def _audit_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicated (symbol, date) primary keys."""
    return int(df.duplicated(subset=list(_PRIMARY_KEY_COLUMNS)).sum())


def _summarize_numeric(series: pd.Series) -> NumericSummary:
    """Compute descriptive statistics for a numeric Series."""
    return NumericSummary(
        minimum=float(series.min()),
        maximum=float(series.max()),
        mean=float(series.mean()),
        median=float(series.median()),
        std=float(series.std()),
    )


def _audit_metadata(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[int, int, list[str], list[str]]:
    """Compare OHLC symbols against metadata basket symbols.

    Returns:
        A tuple of:
        - metadata symbol count
        - matched symbol count
        - OHLC symbols missing from metadata
        - metadata symbols unused by OHLC
    """
    symbol_column = _metadata_symbol_column(metadata)
    ohlc_symbols = set(df["symbol"].dropna().astype(str).unique())

    if symbol_column is None:
        metadata_symbol_set: set[str] = set()
    else:
        metadata_symbol_set = set(
            metadata[symbol_column].dropna().astype(str).unique()
        )

    matched = ohlc_symbols & metadata_symbol_set
    missing_in_metadata = sorted(ohlc_symbols - metadata_symbol_set)
    unused_metadata_symbols = sorted(metadata_symbol_set - ohlc_symbols)

    return (
        len(metadata_symbol_set),
        len(matched),
        missing_in_metadata,
        unused_metadata_symbols,
    )


def audit_dataset(df: pd.DataFrame, metadata: pd.DataFrame) -> AuditReport:
    """Audit an OHLC dataset and its sector-basket metadata.

    Produces a read-only descriptive report. Does not validate, clean,
    transform, reject, or repair either DataFrame.

    Args:
        df: OHLC price DataFrame containing symbol, date, and OHLC columns.
        metadata: Sector basket metadata used for symbol coverage statistics.

    Returns:
        AuditReport summarizing shape, coverage, and numeric distributions.
    """
    rows, columns = _audit_shape(df)
    memory_usage_mb = _audit_memory(df)
    total_symbols, symbols = _audit_symbols(df)
    start_date, end_date, trading_days = _audit_dates(df)
    missing_by_column = _audit_missing(df)
    duplicate_primary_keys = _audit_duplicates(df)

    missing_columns = [
        column for column in _NUMERIC_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing required numeric columns for audit: {missing_columns}"
        )

    numeric_summaries = {
        column: _summarize_numeric(df[column]) for column in _NUMERIC_COLUMNS
    }

    (
        metadata_symbols,
        matched_symbols,
        missing_in_metadata,
        unused_metadata_symbols,
    ) = _audit_metadata(df, metadata)

    return AuditReport(
        rows=rows,
        columns=columns,
        memory_usage_mb=memory_usage_mb,
        total_symbols=total_symbols,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        missing_by_column=missing_by_column,
        duplicate_primary_keys=duplicate_primary_keys,
        open_stats=numeric_summaries["open"],
        high_stats=numeric_summaries["high"],
        low_stats=numeric_summaries["low"],
        close_stats=numeric_summaries["close"],
        volume_stats=numeric_summaries["volume"],
        metadata_symbols=metadata_symbols,
        matched_symbols=matched_symbols,
        missing_in_metadata=missing_in_metadata,
        unused_metadata_symbols=unused_metadata_symbols,
    )
