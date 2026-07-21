import copy

import pandas as pd

from src.data.audit import audit_dataset
from src.data.loader import load_sector, load_sector_metadata


def load_bank_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared bank OHLC and metadata fixtures once."""
    return load_sector("bank"), load_sector_metadata("bank")


# --------------------------------------------------
# TEST 1
# Valid Dataset
# --------------------------------------------------


def test_valid_dataset(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    report = audit_dataset(df, metadata)

    assert report.rows == len(df)
    assert report.columns == df.shape[1]
    assert report.total_symbols == df["symbol"].nunique()
    assert report.trading_days == df["date"].nunique()
    assert report.duplicate_primary_keys == 0
    assert report.metadata_symbols == metadata["Symbol"].nunique()
    assert report.matched_symbols == report.total_symbols
    assert report.missing_in_metadata == []
    assert report.unused_metadata_symbols == []


# --------------------------------------------------
# TEST 2
# Missing Values
# --------------------------------------------------


def test_missing_values(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    missing_df = df.copy()
    missing_df.loc[0, "close"] = None
    missing_df.loc[1, "volume"] = None
    missing_df.loc[2, "open"] = None

    report = audit_dataset(missing_df, metadata)

    assert report.missing_by_column["close"] == 1
    assert report.missing_by_column["volume"] == 1
    assert report.missing_by_column["open"] == 1
    assert report.missing_by_column["high"] == 0
    assert report.missing_by_column["low"] == 0
    assert report.missing_by_column["symbol"] == 0
    assert report.missing_by_column["date"] == 0


# --------------------------------------------------
# TEST 3
# Duplicate Primary Keys
# --------------------------------------------------


def test_duplicate_primary_keys(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    duplicate_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = audit_dataset(duplicate_df, metadata)

    assert report.duplicate_primary_keys == 1


# --------------------------------------------------
# TEST 4
# Metadata Mismatch
# --------------------------------------------------


def test_metadata_mismatch(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    mismatch_df = df.copy()
    mismatch_df["symbol"] = mismatch_df["symbol"].astype(str)
    mismatch_df.loc[0, "symbol"] = "FAKEBANK"

    report = audit_dataset(mismatch_df, metadata)

    assert "FAKEBANK" in report.missing_in_metadata


# --------------------------------------------------
# TEST 5
# Unused Metadata Symbols
# --------------------------------------------------


def test_unused_metadata_symbols(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    unused_metadata = pd.concat(
        [metadata, metadata.iloc[[0]].assign(Symbol="UNUSED_SYM")],
        ignore_index=True,
    )

    report = audit_dataset(df, unused_metadata)

    assert "UNUSED_SYM" in report.unused_metadata_symbols


# --------------------------------------------------
# TEST 6
# Numeric Summaries
# --------------------------------------------------


def test_numeric_summaries(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    report = audit_dataset(df, metadata)
    close = df["close"]

    assert report.close_stats.minimum == float(close.min())
    assert report.close_stats.maximum == float(close.max())
    assert report.close_stats.mean == float(close.mean())
    assert report.close_stats.median == float(close.median())
    assert report.close_stats.std == float(close.std())


# --------------------------------------------------
# TEST 7
# Summary Method
# --------------------------------------------------


def test_summary_method(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    report = audit_dataset(df, metadata)
    summary = report.summary()

    assert isinstance(summary, str)
    assert "Audit Report" in summary


# --------------------------------------------------
# TEST 8
# Read-Only Guarantee
# --------------------------------------------------


def test_read_only_guarantee(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    df_before = copy.deepcopy(df)
    metadata_before = copy.deepcopy(metadata)

    audit_dataset(df, metadata)

    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_frame_equal(metadata, metadata_before)


# --------------------------------------------------
# TEST 9
# Missing Numeric Columns
# --------------------------------------------------


def test_missing_numeric_columns(df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    missing_column_df = df.drop(columns=["volume"])

    try:
        audit_dataset(missing_column_df, metadata)
        raised = False
    except KeyError as error:
        raised = True
        assert "Missing required numeric columns for audit:" in str(error)
        assert "volume" in str(error)

    assert raised


def main() -> None:
    df, metadata = load_bank_data()

    test_valid_dataset(df, metadata)
    test_missing_values(df, metadata)
    test_duplicate_primary_keys(df, metadata)
    test_metadata_mismatch(df, metadata)
    test_unused_metadata_symbols(df, metadata)
    test_numeric_summaries(df, metadata)
    test_summary_method(df, metadata)
    test_read_only_guarantee(df, metadata)
    test_missing_numeric_columns(df, metadata)
    
    print("\n🎉 ALL AUDIT TESTS PASSED")

if __name__ == "__main__":
    main()
