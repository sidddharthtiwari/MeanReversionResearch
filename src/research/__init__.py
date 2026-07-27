"""Public API for basket-level research orchestration."""

from __future__ import annotations

from src.research.aggregation import aggregate
from src.research.pipeline import run_research
from src.research.result import ResearchResult
from src.research.runner import run_basket
from src.research.weighting import compute_equal_weights

__all__ = [
    "run_research",
    "run_basket",
    "aggregate",
    "compute_equal_weights",
    "ResearchResult",
]
