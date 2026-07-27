from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.config import PipelineConfig
from src.research.pipeline import run_research
from src.research.weighting import compute_equal_weights


# --------------------------------------------------
# TEST 1
# Run Research Success
# --------------------------------------------------


def test_run_research_success() -> None:
    sector_data = pd.DataFrame({"symbol": ["AUBANK"], "close": [100.0]})
    metadata = pd.DataFrame({"Symbol": ["AUBANK"]})
    config = PipelineConfig(lookback=20)
    basket_results = {"AUBANK": object()}
    research_result = MagicMock(name="ResearchResult")

    with (
        patch(
            "src.research.pipeline.run_basket",
            return_value=basket_results,
        ) as mock_run_basket,
        patch(
            "src.research.pipeline.aggregate",
            return_value=research_result,
        ) as mock_aggregate,
    ):
        result = run_research(
            sector_data,
            metadata,
            config,
            weighting_function=compute_equal_weights,
        )

    mock_run_basket.assert_called_once_with(sector_data, metadata, config)
    mock_aggregate.assert_called_once_with(
        results=basket_results,
        weighting_function=compute_equal_weights,
    )
    assert result is research_result


# --------------------------------------------------
# TEST 2
# Custom Weighting Function Forwarded
# --------------------------------------------------


def test_custom_weighting_function_forwarded() -> None:
    sector_data = pd.DataFrame({"symbol": ["AUBANK"], "close": [100.0]})
    metadata = pd.DataFrame({"Symbol": ["AUBANK"]})
    config = PipelineConfig(lookback=20)
    basket_results = {"AUBANK": object()}

    def custom_weights(aligned_returns: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=aligned_returns.columns)

    with (
        patch(
            "src.research.pipeline.run_basket",
            return_value=basket_results,
        ),
        patch(
            "src.research.pipeline.aggregate",
            return_value=MagicMock(name="ResearchResult"),
        ) as mock_aggregate,
    ):
        run_research(
            sector_data,
            metadata,
            config,
            weighting_function=custom_weights,
        )

    assert mock_aggregate.call_args.kwargs["weighting_function"] is custom_weights


# --------------------------------------------------
# TEST 3
# Run Basket Failure Propagates
# --------------------------------------------------


def test_run_basket_failure_propagates() -> None:
    sector_data = pd.DataFrame({"symbol": ["AUBANK"], "close": [100.0]})
    metadata = pd.DataFrame({"Symbol": ["AUBANK"]})
    config = PipelineConfig(lookback=20)

    with (
        patch(
            "src.research.pipeline.run_basket",
            side_effect=RuntimeError("basket failed"),
        ),
        patch("src.research.pipeline.aggregate") as mock_aggregate,
    ):
        with pytest.raises(RuntimeError, match="basket failed"):
            run_research(sector_data, metadata, config)

    mock_aggregate.assert_not_called()


# --------------------------------------------------
# TEST 4
# Aggregate Failure Propagates
# --------------------------------------------------


def test_aggregate_failure_propagates() -> None:
    sector_data = pd.DataFrame({"symbol": ["AUBANK"], "close": [100.0]})
    metadata = pd.DataFrame({"Symbol": ["AUBANK"]})
    config = PipelineConfig(lookback=20)

    with (
        patch(
            "src.research.pipeline.run_basket",
            return_value={"AUBANK": object()},
        ),
        patch(
            "src.research.pipeline.aggregate",
            side_effect=ValueError("aggregate failed"),
        ),
    ):
        with pytest.raises(ValueError, match="aggregate failed"):
            run_research(sector_data, metadata, config)
