"""Immutable configuration for the research pipeline.

Defines parameters required to execute a complete research run. This module
contains configuration only and does not execute pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PipelineConfig",
]

SUPPORTED_REBALANCE_FREQUENCIES = frozenset({"D", "W", "M"})


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable configuration for a complete research pipeline run.

    Attributes:
        lookback: Lookback window length used by research features.
        entry_zscore: Absolute z-score threshold for trade entry.
        exit_zscore: Absolute z-score threshold for trade exit.
        transaction_cost: Proportional transaction-cost rate.
        slippage: Proportional slippage rate.
        rebalance_frequency: Rebalance frequency code (``"D"``, ``"W"``,
            or ``"M"``).
        output_directory: Directory path for research outputs.
    """

    lookback: int = 20
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    rebalance_frequency: str = "D"
    output_directory: str = "outputs"

    def __post_init__(self) -> None:
        """Validate configuration fields after initialisation.

        Raises:
            ValueError: If any configuration field violates its constraints.
        """
        if self.lookback <= 0:
            raise ValueError(
                f"lookback must be greater than 0, got {self.lookback}."
            )
        if self.entry_zscore <= self.exit_zscore:
            raise ValueError(
                "entry_zscore must be greater than exit_zscore, "
                f"got entry_zscore={self.entry_zscore}, "
                f"exit_zscore={self.exit_zscore}."
            )
        if self.transaction_cost < 0:
            raise ValueError(
                "transaction_cost must be greater than or equal to 0, "
                f"got {self.transaction_cost}."
            )
        if self.slippage < 0:
            raise ValueError(
                "slippage must be greater than or equal to 0, "
                f"got {self.slippage}."
            )
        if self.rebalance_frequency not in SUPPORTED_REBALANCE_FREQUENCIES:
            raise ValueError(
                "rebalance_frequency must be one of "
                f"{sorted(SUPPORTED_REBALANCE_FREQUENCIES)}, "
                f"got '{self.rebalance_frequency}'."
            )
        if not self.output_directory:
            raise ValueError("output_directory must not be empty.")
