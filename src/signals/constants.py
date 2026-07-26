"""Shared constants for the signal layer.

This module defines discrete signal values and naming conventions used by
signal generators. It contains constants only.
"""

from __future__ import annotations

__all__ = [
    "LONG_SIGNAL",
    "FLAT_SIGNAL",
    "SHORT_SIGNAL",
    "SIGNAL_SUFFIX",
    "CROSSOVER_SIGNAL_SUFFIX",
]

# Discrete position direction encoded by signal generators.
LONG_SIGNAL = 1
FLAT_SIGNAL = 0
SHORT_SIGNAL = -1

# Appended to feature names when deriving default signal column names.
SIGNAL_SUFFIX = "_signal"
CROSSOVER_SIGNAL_SUFFIX = "_cross_signal"
