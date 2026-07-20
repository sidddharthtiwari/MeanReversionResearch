"""Data loading utilities for sector-wise OHLC and basket metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "Data"
_OHLC_DIR = _DATA_ROOT / "OHLC data"
_METADATA_DIR = _DATA_ROOT / "Sector Baskets Info"


def _ensure_ohlc_dir() -> None:
    """Raise FileNotFoundError if the OHLC data directory is missing."""
    if not _OHLC_DIR.is_dir():
        raise FileNotFoundError(f"OHLC data directory not found at {_OHLC_DIR}")


def _ensure_metadata_dir() -> None:
    """Raise FileNotFoundError if the sector basket metadata directory is missing."""
    if not _METADATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Sector basket metadata directory not found at {_METADATA_DIR}"
        )


def _discover_sectors() -> list[str]:
    """Return sorted sector names discovered from OHLC parquet files."""
    _ensure_ohlc_dir()
    return sorted(path.stem for path in _OHLC_DIR.glob("*.parquet"))


def _ohlc_path(sector: str) -> Path:
    """Return the filesystem path to a sector's OHLC parquet file."""
    return _OHLC_DIR / f"{sector}.parquet"


def _metadata_path(sector: str) -> Path:
    """Return the filesystem path to a sector's basket metadata CSV file."""
    return _METADATA_DIR / f"{sector}.csv"


def available_sectors() -> list[str]:
    """Return the list of sector names available for loading.

    Sector names are discovered dynamically from ``*.parquet`` files in the
    ``Data/OHLC data/`` directory. Names are returned in sorted order.

    Returns:
        Sorted list of sector identifiers, e.g. ``["auto", "bank", ...]``.
    """
    return _discover_sectors()


def load_sector(sector: str) -> pd.DataFrame:
    """Load OHLC data for a single sector.

    Args:
        sector: Sector identifier matching a ``{sector}.parquet`` file in
            ``Data/OHLC data/``.

    Returns:
        DataFrame containing the sector's OHLC data exactly as stored on disk.

    Raises:
        FileNotFoundError: If the OHLC data directory or the requested sector
            parquet file does not exist.
    """
    sector = sector.strip().lower()
    if not sector:
        raise ValueError("Sector name cannot be empty.")
    path = _ohlc_path(sector)
    if not path.is_file():
        available = available_sectors()
        raise FileNotFoundError(
            f"OHLC data for sector '{sector}' not found at {path}. "
            f"Available sectors: {available}"
        )
    return pd.read_parquet(path)


def load_all_sectors() -> dict[str, pd.DataFrame]:
    """Load OHLC data for every discovered sector.

    Returns:
        Mapping of sector name to DataFrame for each sector found in
        ``Data/OHLC data/``.
    """
    return {sector: load_sector(sector) for sector in available_sectors()}


def load_sector_metadata(sector: str) -> pd.DataFrame:
    """Load basket metadata for a single sector.

    Args:
        sector: Sector identifier matching a ``{sector}.csv`` file in
            ``Data/Sector Baskets Info/``.

    Returns:
        DataFrame containing the sector's basket metadata exactly as stored
        on disk.

    Raises:
        FileNotFoundError: If the metadata directory or the requested sector
            CSV file does not exist.
    """
    sector = sector.strip().lower()
    if not sector:
        raise ValueError("Sector name cannot be empty.")
    _ensure_metadata_dir()
    path = _metadata_path(sector)
    if not path.is_file():
        available = available_sectors()
        raise FileNotFoundError(
            f"Metadata for sector '{sector}' not found at {path}. "
            f"Available sectors: {available}"
        )
    return pd.read_csv(path)
