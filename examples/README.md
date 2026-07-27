# Examples

Usage examples for the MeanReversionResearch framework.

## Datasets

Example datasets live under the top-level `data/` directory:

```text
data/
├── single_asset/
│   ├── AUBANK.csv
│   └── AXISBANK.csv
├── sectors/
│   └── bank_sector.csv
└── metadata/
    └── bank_metadata.csv
```

- `data/single_asset/` — single-stock OHLC CSVs used by `single_asset.py`
- `data/sectors/` — multi-symbol sector OHLC CSVs for basket research
- `data/metadata/` — basket membership metadata (includes a `Symbol` column)

To analyse your own data, replace the relevant CSV or change the path
constants at the top of an example script.

## Scripts

| Script | Purpose |
| --- | --- |
| `single_asset.py` | Analyse one stock with `run_pipeline()` |
| `basket_research.py` | Run sector-basket research with `run_research()` |
| `configuration_examples.py` | Display common `PipelineConfig` presets |
| `utils.py` | Shared presentation helpers for example output |

## Running

From the repository root:

```bash
python -m examples.single_asset
python -m examples.basket_research
python -m examples.configuration_examples
```
