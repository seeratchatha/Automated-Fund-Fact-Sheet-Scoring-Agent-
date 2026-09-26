# Fact-Sheet Scoring Agent: AI-Powered Mutual Fund Valuation

An LLM-based system for extracting behavioral and holdings-quality metrics from mutual fund fact sheets, validated as statistically independent from traditional return-based ratings.

## Quick Summary

- **30 funds** analyzed across 5 Indian AMCs (ABSL, HDFC, SBI, ICICI, Axis)
- **4-dimensional scoring framework**: concentration, style consistency, sector alignment, turnover stability
- **Validation**: Pearson correlation r = -0.519 with Morningstar ratings (p < 0.05)
- **Key finding**: Holdings-based metrics capture fund risk dimensions invisible to traditional ratings

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from fact_sheet_scoring_agent_v2 import run_batch
import json

config = json.load(open("scheme_config_HDFC_AXIS_ICICI_combined.json"))
run_batch(batch_dir="data/factsheets", config_dict=config, out_csv="results.csv")
```

## Results

- Correlation with Morningstar: **r = -0.519** (p < 0.05)
- Top-10 selection divergence: **90% unique funds**
- Concentration scoring reliability: **0% missing data**

See `all_scores_FINAL_30funds.csv` for complete results.

## Files

- `fact_sheet_scoring_agent_v2.py` — Main scoring engine
- `scheme_config_HDFC_AXIS_ICICI_combined.json` — Fund configuration
- `all_scores_FINAL_30funds.csv` — Validation dataset
- `TECHNICAL_OVERVIEW.md` — Full methodology

## Author

Seerat Chatha | [GitHub](https://github.com/seeratchatha) | [LinkedIn](https://linkedin.com/in/seeratchatha)

## License

MIT
