"""Immutable result object for research pipeline outputs.

Stores the DataFrame and analytics outputs produced by a complete research
run. This module contains result structure only and does not execute pipeline
logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "PipelineResult",
]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Immutable outputs from a complete research pipeline run.

    Attributes:
        signals: DataFrame containing generated trading signals.
        portfolio: DataFrame containing portfolio construction outputs.
        backtest: DataFrame containing backtest outputs.
        analytics: Mapping of scalar analytics summary metrics.
    """

    signals: pd.DataFrame
    portfolio: pd.DataFrame
    backtest: pd.DataFrame
    analytics: Mapping[str, float | int]

    def __post_init__(self) -> None:
        """Validate result field types after initialisation.

        Raises:
            TypeError: If any field has an unexpected type.
        """
        if not isinstance(self.signals, pd.DataFrame):
            raise TypeError(
                "signals must be a pandas DataFrame, "
                f"got {type(self.signals).__name__}."
            )
        if not isinstance(self.portfolio, pd.DataFrame):
            raise TypeError(
                "portfolio must be a pandas DataFrame, "
                f"got {type(self.portfolio).__name__}."
            )
        if not isinstance(self.backtest, pd.DataFrame):
            raise TypeError(
                "backtest must be a pandas DataFrame, "
                f"got {type(self.backtest).__name__}."
            )
        if not isinstance(self.analytics, Mapping):
            raise TypeError(
                "analytics must be a dictionary, "
                f"got {type(self.analytics).__name__}."
            )
