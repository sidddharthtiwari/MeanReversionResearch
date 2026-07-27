"""Reusable presentation helpers for example programs.

Contains formatting and CSV loading utilities only. This module does not
contain framework logic or financial calculations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_HEADER_WIDTH = 60
_LABEL_WIDTH = 22


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame loaded from ``path``.
    """
    return pd.read_csv(path)


def print_header(title: str) -> None:
    """Print a formatted title block.

    Args:
        title: Header text to display.
    """
    border = "=" * _HEADER_WIDTH
    print(border)
    print(title)
    print(border)


def print_section(title: str) -> None:
    """Print a formatted section heading.

    Args:
        title: Section title to display.
    """
    border = "-" * _HEADER_WIDTH
    print(border)
    print(title)
    print(border)


def print_key_value(label: str, value: object) -> None:
    """Pretty-print a single aligned key/value pair.

    Args:
        label: Display label.
        value: Value to print.
    """
    print(f"{label:<{_LABEL_WIDTH}} : {value}")


def print_footer(message: str) -> None:
    """Print a formatted closing message.

    Args:
        message: Footer text to display.
    """
    border = "=" * _HEADER_WIDTH
    print(border)
    print(message)
    print(border)
