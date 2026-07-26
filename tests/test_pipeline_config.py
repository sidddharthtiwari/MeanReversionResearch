from dataclasses import FrozenInstanceError

from src.pipeline.config import PipelineConfig


# --------------------------------------------------
# TEST 1
# Default Configuration
# --------------------------------------------------


def test_default_configuration() -> None:
    config = PipelineConfig()

    assert config.lookback == 20
    assert config.entry_zscore == 2.0
    assert config.exit_zscore == 0.5
    assert config.transaction_cost == 0.001
    assert config.slippage == 0.0005
    assert config.rebalance_frequency == "D"
    assert config.output_directory == "outputs"


# --------------------------------------------------
# TEST 2
# Custom Configuration
# --------------------------------------------------


def test_custom_configuration() -> None:
    config = PipelineConfig(
        lookback=40,
        entry_zscore=1.5,
        exit_zscore=0.25,
        transaction_cost=0.002,
        slippage=0.001,
        rebalance_frequency="W",
        output_directory="results",
    )

    assert config.lookback == 40
    assert config.entry_zscore == 1.5
    assert config.exit_zscore == 0.25
    assert config.transaction_cost == 0.002
    assert config.slippage == 0.001
    assert config.rebalance_frequency == "W"
    assert config.output_directory == "results"


# --------------------------------------------------
# TEST 3
# Invalid Lookback
# --------------------------------------------------


def test_invalid_lookback() -> None:
    try:
        PipelineConfig(lookback=0)
        raised = False
    except ValueError as error:
        raised = True
        assert "lookback" in str(error)

    assert raised


# --------------------------------------------------
# TEST 4
# Entry Z-Score Less Than or Equal Exit
# --------------------------------------------------


def test_entry_zscore_less_than_exit() -> None:
    try:
        PipelineConfig(entry_zscore=0.5, exit_zscore=0.5)
        raised = False
    except ValueError as error:
        raised = True
        assert "entry_zscore" in str(error)
        assert "exit_zscore" in str(error)

    assert raised


# --------------------------------------------------
# TEST 5
# Negative Transaction Cost
# --------------------------------------------------


def test_negative_transaction_cost() -> None:
    try:
        PipelineConfig(transaction_cost=-0.001)
        raised = False
    except ValueError as error:
        raised = True
        assert "transaction_cost" in str(error)

    assert raised


# --------------------------------------------------
# TEST 6
# Negative Slippage
# --------------------------------------------------


def test_negative_slippage() -> None:
    try:
        PipelineConfig(slippage=-0.0005)
        raised = False
    except ValueError as error:
        raised = True
        assert "slippage" in str(error)

    assert raised


# --------------------------------------------------
# TEST 7
# Invalid Rebalance Frequency
# --------------------------------------------------


def test_invalid_rebalance_frequency() -> None:
    try:
        PipelineConfig(rebalance_frequency="Y")
        raised = False
    except ValueError as error:
        raised = True
        assert "rebalance_frequency" in str(error)

    assert raised


# --------------------------------------------------
# TEST 8
# Empty Output Directory
# --------------------------------------------------


def test_empty_output_directory() -> None:
    try:
        PipelineConfig(output_directory="")
        raised = False
    except ValueError as error:
        raised = True
        assert "output_directory" in str(error)

    assert raised


# --------------------------------------------------
# TEST 9
# Configuration Is Frozen
# --------------------------------------------------


def test_configuration_is_frozen() -> None:
    config = PipelineConfig()

    try:
        config.lookback = 40  # type: ignore[misc]
        raised = False
    except FrozenInstanceError:
        raised = True

    assert raised


def main() -> None:
    test_default_configuration()
    test_custom_configuration()
    test_invalid_lookback()
    test_entry_zscore_less_than_exit()
    test_negative_transaction_cost()
    test_negative_slippage()
    test_invalid_rebalance_frequency()
    test_empty_output_directory()
    test_configuration_is_frozen()

    print("🎉 ALL PIPELINE CONFIG TESTS PASSED")


if __name__ == "__main__":
    main()
