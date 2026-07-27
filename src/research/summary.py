"""Summary-row construction for basket-level research outputs.

Builds a single flat summary dictionary from a completed research run. This
module does not compute analytics, persist files, generate plots, or modify
``ResearchResult``.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.research.result import ResearchResult

__all__ = [
    "create_summary_row",
]


def create_summary_row(
    basket_name: str,
    result: ResearchResult,
    analytics: Mapping[str, float | int],
) -> dict[str, object]:
    """Create one summary row for a completed basket research run.

    Combines basket identity, processed/skipped symbol counts, and the
    supplied analytics mapping into a single flat dictionary. Analytics
    metric names are taken from ``analytics`` and are not hardcoded.

    Args:
        basket_name: Display name of the researched basket.
        result: Immutable basket-level research outputs.
        analytics: Mapping of scalar analytics summary metrics.

    Returns:
        Flat dictionary containing ``Basket``, ``Processed Symbols``,
        ``Skipped Symbols``, and every entry from ``analytics``.

    Raises:
        TypeError: If ``basket_name`` is not a string, ``result`` is not a
            ``ResearchResult``, or ``analytics`` is not a ``Mapping``.
    """
    if not isinstance(basket_name, str):
        raise TypeError(
            f"basket_name must be a string, got {type(basket_name).__name__}."
        )
    if not isinstance(result, ResearchResult):
        raise TypeError(
            f"result must be a ResearchResult, got {type(result).__name__}."
        )
    if not isinstance(analytics, Mapping):
        raise TypeError(
            f"analytics must be a Mapping, got {type(analytics).__name__}."
        )

    return {
        "Basket": basket_name,
        "Processed Symbols": len(result.processed_symbols),
        "Skipped Symbols": len(result.skipped_symbols),
        **analytics,
    }
