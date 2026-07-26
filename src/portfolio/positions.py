"""Position generation for quantitative research.

Converts discrete trading signals into portfolio positions. Flat signals
represent no new trading instruction rather than an exit. The previous
non-flat position is carried forward until a new long or short signal is
generated. This module does not compute returns, manage exposure, or perform
backtesting.
"""

from __future__ import annotations

import pandas as pd

from src.portfolio.constants import POSITION_SUFFIX
from src.signals.constants import FLAT_SIGNAL
from src.signals.signal_utils import _validate_signal_column

__all__ = [
    "generate_positions",
]


def _generate_output_column_name(
    signal_column: str,
    output_column: str | None,
) -> str:
    """Resolve the position output column name.

    Args:
        signal_column: Source signal column used in the default name.
        output_column: Explicit output name, or ``None`` for the default.

    Returns:
        ``output_column`` when provided, otherwise
        ``{signal_column}{POSITION_SUFFIX}``.
    """
    if output_column is not None:
        return output_column
    return f"{signal_column}{POSITION_SUFFIX}"


def _validate_position_inputs(df: pd.DataFrame, signal_column: str) -> None:
    """Validate inputs required to generate positions from signals.

    Args:
        df: Input DataFrame containing the signal column.
        signal_column: Discrete signal column used to derive positions.

    Raises:
        KeyError: If ``signal_column`` is not present in ``df``.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If ``signal_column`` contains invalid signal values.
    """
    _validate_signal_column(df, signal_column)


def _apply_position_logic(signal: pd.Series) -> pd.Series:
    """Map discrete signals into carried-forward portfolio positions.

    Rules:
        - ``LONG_SIGNAL`` -> long position
        - ``SHORT_SIGNAL`` -> short position
        - ``FLAT_SIGNAL`` -> carry forward the previous position
        - Leading ``FLAT_SIGNAL`` values remain flat

    Args:
        signal: Discrete signal series containing ``LONG_SIGNAL``,
            ``FLAT_SIGNAL``, and ``SHORT_SIGNAL``.

    Returns:
        Integer position series aligned to ``signal.index``.
    """
    positions = signal.replace(FLAT_SIGNAL, pd.NA)
    positions = positions.ffill()
    positions = positions.fillna(FLAT_SIGNAL)
    return positions.astype("int64")


def generate_positions(
    df: pd.DataFrame,
    signal_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Generate portfolio positions from a discrete signal column.

    Non-flat signals set the active position. Flat signals carry the previous
    position forward. Leading flats remain flat.

    Args:
        df: Input DataFrame containing ``signal_column``.
        signal_column: Discrete trading-signal column.
        output_column: Optional output column name. Defaults to
            ``{signal_column}_position``.

    Returns:
        A new DataFrame with the position column appended. The input is
        unchanged.

    Raises:
        KeyError: If ``signal_column`` is missing.
        TypeError: If ``signal_column`` is not numeric.
        ValueError: If ``signal_column`` contains invalid signal values.
    """
    _validate_position_inputs(df, signal_column)

    name = _generate_output_column_name(signal_column, output_column)
    result = df.copy()
    result[name] = _apply_position_logic(result[signal_column])
    return result
