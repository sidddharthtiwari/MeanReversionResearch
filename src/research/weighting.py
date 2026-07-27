"""Portfolio weighting helpers for basket-level research aggregation.

Computes symbol weights from an aligned return matrix. Weighting algorithms
are kept here and injected into aggregation so aggregation can remain closed
for modification while still supporting additional weighting methodologies.

This module does not aggregate returns, run pipelines, or produce analytics.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "compute_equal_weights",
]


def compute_equal_weights(aligned_returns: pd.DataFrame) -> pd.Series:
    """Compute equal portfolio weights for every column in ``aligned_returns``.

    Args:
        aligned_returns: Date-aligned return matrix with symbols as columns.

    Returns:
        Series of equal weights indexed by symbol. Each weight is
        ``1 / n_symbols`` and stored as ``float64``.

    Raises:
        TypeError: If ``aligned_returns`` is not a DataFrame.
        ValueError: If ``aligned_returns`` has no columns, or if its column
            labels are not unique.
    """
    if not isinstance(aligned_returns, pd.DataFrame):
        raise TypeError(
            "aligned_returns must be a pandas DataFrame, "
            f"got {type(aligned_returns).__name__}."
        )
    # Duplicate symbols would collapse into ambiguous weight labels and make
    # portfolio construction non-deterministic.
    if not aligned_returns.columns.is_unique:
        raise ValueError(
            "aligned_returns column labels must be unique; "
            "duplicate symbols cannot produce unambiguous weights."
        )
    n_symbols = aligned_returns.shape[1]
    if n_symbols == 0:
        raise ValueError("aligned_returns must contain at least one symbol.")
    return pd.Series(
        1.0 / n_symbols,
        index=aligned_returns.columns,
        dtype="float64",
    )
