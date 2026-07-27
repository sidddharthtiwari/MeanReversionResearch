# MeanReversionResearch

A modular framework for researching mean-reversion strategies on sector equity
baskets.

It provides a full path from OHLC data to signals, portfolios, cost-aware
backtests, analytics, and visualizations. Each stage is implemented as a
separate package with a small public API, so research workflows stay
reproducible and easy to extend.

---

## Key Features

- Basket research over sector OHLC parquet files and membership metadata
- Single-asset and multi-symbol orchestration through public APIs
- Feature library: returns, z-scores, volatility, ATR, spreads, rolling statistics
- Signal library: mean reversion, threshold, crossover, breakout
- Portfolio construction with positions, strategy returns, and exposure
- Transaction costs and slippage in a dedicated backtest stage
- Analytics summaries: returns, risk, drawdown, Sharpe, Sortino, Calmar
- Equity and drawdown time series with matplotlib plots
- Equal-weight basket aggregation with injectable weighting functions
- Frozen configuration and result objects
- Unit and integration tests
- Example scripts for common research workflows

---

## Architecture

Higher layers orchestrate. Lower layers compute.

```text
                    main.py
                       │
                       ▼
                   research/
                       │
                       ▼
                   pipeline/
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
     features/     signals/     portfolio/
                                     │
                                     ▼
                                 backtest/
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    analytics/   performance/  visualization/
                       │
                       ▼
                     data/
```

- `main.py` and the pipeline/research runners compose existing APIs. They do
  not reimplement formulas.
- Domain packages own one concern each and expose a narrow public surface.
- Private helpers stay internal. Callers depend on `__all__`, not on
  implementation details.

This keeps modules independently testable and allows new features, signals, or
weighting methods without rewriting the workflow.

---

## Repository Structure

```text
MeanReversionResearch/
├── main.py              Batch research entry point for all sector baskets
├── src/
│   ├── data/            Load, validate, and audit sector datasets
│   ├── features/        Compute research features from OHLC data
│   ├── signals/         Map features to discrete trading signals
│   ├── portfolio/       Build positions, strategy returns, and exposure
│   ├── backtest/        Apply transaction costs and slippage
│   ├── pipeline/        Single-asset research orchestration
│   ├── research/        Basket research, aggregation, and persistence
│   ├── analytics/       Scalar performance statistics
│   ├── performance/     Equity and drawdown time series
│   └── visualization/   Equity and drawdown plots
├── examples/            Usage examples built on public APIs
├── tests/               Unit and integration tests
├── Data/                Sector OHLC parquet files and basket metadata
├── outputs/             Generated research artefacts
└── docs/                Project charter
```

---

## Workflow

For each sector basket discovered under `Data/OHLC data/`:

```text
Load OHLC + metadata
        │
        ▼
run_research()
  • run the single-asset pipeline per symbol
  • aggregate returns with equal weights
        │
        ▼
generate_summary()
        │
        ▼
compute_equity_curve() / compute_drawdown_series()
        │
        ▼
save_research_outputs() / save_research_visualizations()
        │
        ▼
create_summary_row()
        │
        ▼
basket_comparison.csv
```

Baskets that fail are logged and skipped. Remaining baskets continue.

Single-asset research uses `run_pipeline()` and follows the same
features → signals → portfolio → backtest → analytics path for one symbol.

---

## Requirements

- Python 3.10+ (tested on 3.13)
- Dependencies listed in `requirements.txt` (including `pandas`, `numpy`, and `matplotlib`)
- Sector OHLC files in `Data/OHLC data/*.parquet`
- Matching metadata in `Data/Sector Baskets Info/*.csv`

---

## Installation

```bash
git clone https://github.com/sidddharthtiwari/MeanReversionResearch.git
cd MeanReversionResearch

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install pytest          # required for the test suite
```

---

## Usage

Run research across every available basket:

```bash
python main.py
```

Default settings come from `PipelineConfig`
(`lookback=20`, `entry_zscore=2.0`, `transaction_cost=0.001`,
`slippage=0.0005`, `rebalance_frequency="D"`).

Focused examples:

```bash
python -m examples.single_asset
python -m examples.basket_research
python -m examples.configuration_examples
```

---

## Outputs

```text
outputs/
├── bank/
│   ├── portfolio_returns.csv
│   ├── aligned_returns.csv
│   ├── weights.csv
│   ├── analytics.json
│   ├── equity_curve.png
│   └── drawdown_curve.png
├── auto/
│   └── ...
└── basket_comparison.csv
```

| File | Contents |
| --- | --- |
| `portfolio_returns.csv` | Aggregated basket return series |
| `aligned_returns.csv` | Date-aligned per-symbol strategy returns |
| `weights.csv` | Aggregation weights by symbol |
| `analytics.json` | Return, risk, drawdown, and ratio metrics |
| `equity_curve.png` | Equity curve |
| `drawdown_curve.png` | Drawdown curve |
| `basket_comparison.csv` | Cross-basket summary of successful runs |

`basket_comparison.csv` is written only when at least one basket completes.

---

## Testing

```bash
pytest
```

Unit tests cover domain modules. Integration tests cover single-asset and basket
research workflows. Both target public APIs.

---

## Design Principles

1. **Orchestration is not computation.**  
   Entry points call public APIs. Formulas live in the packages that own them.

2. **One concern per module.**  
   Loading, features, signals, portfolio logic, costs, metrics, and plots are
   separated.

3. **Small public surfaces.**  
   Callers use documented exports. Underscore-prefixed helpers are internal.

4. **Immutable contracts.**  
   `PipelineConfig`, `PipelineResult`, and `ResearchResult` are frozen.

5. **Validate early.**  
   Type and schema checks fail fast instead of leaking silent `NaN`s.

6. **Extend by injection.**  
   Aggregation accepts a weighting function. Equal weight is the default.

7. **Deterministic artefacts.**  
   Output filenames and JSON formatting are fixed for reproducible runs.

---

## Future Work

- Additional weighting models (for example inverse volatility)
- Walk-forward evaluation
- Parallel basket execution
- Broader signal and factor libraries
- Dashboards over `outputs/`

---

## License

The repository includes a `LICENSE` file. Licensing terms have not been
specified yet.
