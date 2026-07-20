import pandas as pd

from src.data.loader import load_sector, load_sector_metadata
from src.data.validator import validate_dataset


def print_report(title, report):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(report.summary())


# --------------------------------------------------
# TEST 1
# Original Bank Dataset
# Should PASS
# --------------------------------------------------

print("\nTEST 1 - VALID DATASET")

df = load_sector("bank")
metadata = load_sector_metadata("bank")

report = validate_dataset(df, metadata)

print_report("VALID DATASET", report)


# --------------------------------------------------
# TEST 2
# Duplicate Row
# Should FAIL
# --------------------------------------------------

print("\nTEST 2 - DUPLICATE")

duplicate_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

report = validate_dataset(duplicate_df, metadata)

print_report("DUPLICATE", report)


# --------------------------------------------------
# TEST 3
# Negative Price
# Should FAIL
# --------------------------------------------------

print("\nTEST 3 - NEGATIVE PRICE")

negative_price_df = df.copy()

negative_price_df.loc[0, "close"] = -100

report = validate_dataset(negative_price_df, metadata)

print_report("NEGATIVE PRICE", report)


# --------------------------------------------------
# TEST 4
# Negative Volume
# Should FAIL
# --------------------------------------------------

print("\nTEST 4 - NEGATIVE VOLUME")

negative_volume_df = df.copy()

negative_volume_df.loc[0, "volume"] = -50

report = validate_dataset(negative_volume_df, metadata)

print_report("NEGATIVE VOLUME", report)


# --------------------------------------------------
# TEST 5
# OHLC Relationship
# Should FAIL
# --------------------------------------------------

print("\nTEST 5 - OHLC")

ohlc_df = df.copy()

ohlc_df.loc[0, "high"] = 1

ohlc_df.loc[0, "close"] = 100

report = validate_dataset(ohlc_df, metadata)

print_report("OHLC", report)


# --------------------------------------------------
# TEST 6
# Unknown Symbol
# Should FAIL
# --------------------------------------------------

print("\nTEST 6 - SYMBOL")

symbol_df = df.copy()

symbol_df["symbol"] = symbol_df["symbol"].astype(str)

symbol_df.loc[0, "symbol"] = "FAKEBANK"

report = validate_dataset(symbol_df, metadata)

print_report("UNKNOWN SYMBOL", report)


# --------------------------------------------------
# TEST 7
# Missing Values
# Should FAIL
# --------------------------------------------------

print("\nTEST 7 - MISSING VALUES")

missing_df = df.copy()

missing_df.loc[0, "close"] = None

report = validate_dataset(missing_df, metadata)

print_report("MISSING VALUES", report)

# --------------------------------------------------
# TEST 8

print("\nTEST 8 - MISSING COLUMN")

missing_column_df = df.drop(columns=["volume"])

report = validate_dataset(missing_column_df, metadata)

print_report("MISSING COLUMN", report)

# --------------------------------------------------
# TEST 9

print("\nTEST 9 - EMPTY DATASET")

empty_df = df.iloc[0:0].copy()

report = validate_dataset(empty_df, metadata)

print_report("EMPTY DATASET", report)

# --------------------------------------------------
# TEST 10

print(type(report))

print(report.is_valid)

print(report.total_checks)

print(report.failed_checks)

print(report.errors)

print(report.warnings)