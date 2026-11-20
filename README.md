# Wine Recommendation System

Exploratory analysis and rating prediction with X-Wines. The experiments compare user/wine identities, wine attributes, and temporal features using chronological, random, and unseen-wine splits.

Start with the [analysis notebook](notebooks/xwines_analysis.ipynb), [findings](reports/findings.md), and [evaluation protocol](docs/evaluation_protocol.md).

## Reproduce

Python 3.11 was used. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/execute_notebook.py
python -m unittest discover -s tests -v
```

The notebook can also run in Jupyter or VS Code with this environment. `scripts/build_notebook.py` regenerates its source and clears saved outputs; execute the notebook afterward to regenerate results.

## Data and experiments

The downloader retrieves the official Slim edition (150,000 ratings) and Test example edition (1,000 ratings). URLs and SHA-256 checksums are recorded in `data/sources.json`. If Google Drive returns an HTML page, download the named CSVs from the linked official folder into `data/raw/`; the same checksum checks apply.

Experiments include rating and activity distributions, origin/type/grape comparisons, within-user type differences, taste-concentration permutations, sparse ridge feature comparisons, RMSE/MAE, bootstrap confidence intervals, and error slices. Raw data, generated split CSVs, prediction rows, and standalone figure PNGs stay local. The notebook embeds results and figures; reports/ contains compact summaries and split hashes.

Prices and consumer locations are unavailable. Catalogue metadata is retrospective, so chronological splitting does not establish historical attribute availability. Metadata and time features are compared separately to distinguish their predictive contributions.

Dataset citation: de Azambuja, R. X., Morais, A. J., & Filipe, V. (2023). [X-Wines: A Wine Dataset for Recommender Systems and Machine Learning](https://doi.org/10.3390/bdcc7010020). *Big Data and Cognitive Computing*, 7(1), 20. [Official repository](https://github.com/rogerioxavier/X-Wines).
