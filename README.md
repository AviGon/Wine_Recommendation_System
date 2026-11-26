# Wine Recommendation System

Exploratory analysis and rating prediction with X-Wines. The experiments compare user/wine identities, wine attributes, and temporal features using chronological, random, and unseen-wine splits.

Start with the [cumulative notebook](notebooks/wine_recommendation.ipynb) or its [readable HTML report](output/workbook.html). It consolidates the exploration, model comparisons, evaluation and related research. The detailed notebooks and original reports remain available below.

## Cumulative report and recording outline

The cumulative notebook contains embedded, checksum-verified saved reports and code excerpts from the merged experiment revision. It runs in Colab without downloading the dataset or installing the compiled model libraries. Running it rebuilds the figures and tables; full model reproduction is optional and disabled by default.

[Open in Colab](https://colab.research.google.com/github/AviGon/Wine_Recommendation_System/blob/ksheer/notebooks/wine_recommendation.ipynb)

To build and export the cumulative report locally, without refitting models:

```sh
python -m pip install -r requirements-report.txt
python scripts/build_cumulative_notebook.py
python scripts/execute_notebook.py notebooks/wine_recommendation.ipynb
python scripts/export_workbook.py
```

The export is `output/workbook.html`, with figures and outputs included and the embedded data-loader input hidden. The [recording outline](docs/presentation_flow.md) proposes a twenty-minute flow for four speakers, a shared deck of about ten slides, and focused notebook/code walkthroughs.

## Reproduce the detailed experiments

Python 3.11 was used. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/execute_notebook.py
python -m unittest discover -s tests -v
```

The notebook can also run in Jupyter or VS Code with this environment. `scripts/build_notebook.py` regenerates its source and clears saved outputs; execute the notebook afterward to regenerate results.

## Reuse the data and scoring pipeline

`python scripts/prepare_data.py` prepares all three published partitions without running EDA or fitting models. It verifies raw-file checksums and the tracked split manifest before writing local files. It never replaces the reference hashes with newly generated ones.

```python
from pathlib import Path
from src.pipeline import load_prepared_splits
from src.experiments import join_metadata

wines, partitions = load_prepared_splits(Path.cwd(), "chronological")
train = join_metadata(partitions["train"], wines)
validation = join_metadata(partitions["validation"], wines).drop(columns="Rating")
```

Fit preprocessing and model parameters using `train` only. Write one `RatingID,prediction` row for every validation event, then run:

```sh
python scripts/evaluate_predictions.py outputs/validation_predictions.csv
```

The scorer aligns IDs, rejects duplicates, missing/extra IDs and non-finite scores, clips to [1,5], and reports RMSE/MAE. Its default partition is validation. After choices are frozen, use `--partition test`; select another protocol with `--protocol random` or `--protocol cold_item`. Full input/output rules are in the [evaluation protocol](docs/evaluation_protocol.md).

## Collaborative filtering

[`notebooks/collaborative_filtering.ipynb`](notebooks/collaborative_filtering.ipynb) compares a regularized bias model, item-item CF (Jaccard/cosine/Pearson), matrix factorization (our ALS and Surprise SVD) and SVD++ on the same chronological split, with tuning, learning curves, scaling, error slices and parameter interpretation. Code: [`src/cf.py`](src/cf.py); results: [`reports/cf_findings.md`](reports/cf_findings.md) and `reports/cf_*.csv`. It needs `scikit-surprise` (pinned in `requirements.txt`). Rebuild and run with:

```sh
python scripts/build_cf_notebook.py
python scripts/execute_notebook.py notebooks/collaborative_filtering.ipynb
```

Prediction files (`RatingID,prediction`) for validation and test are written to `outputs/` (not tracked) for model comparison. Run this notebook before the factorization-machine notebook, which reads `reports/cf_test_metrics.csv`.

## Factorization machines, content and hybrid models

[`notebooks/factorization_machines.ipynb`](notebooks/factorization_machines.ipynb) fits factorization machines on user/wine IDs, wine metadata (type, grapes, country, region, winery, ABV) and rating context (time, vintage age). It covers the hyperparameter grid and learning curves, the feature ablation, time vs vintage age, the extra features (Body, Acidity, Harmonize), the cold-start split, the final comparison with the ridge and collaborative-filtering models, error slices, feature weights, Top-10 ranking against matrix factorization, and diversity. Code: [`src/fm.py`](src/fm.py) (own L-BFGS FM plus a myFM Bayesian FM wrapper; fastFM does not build on Python 3.11 / NumPy 2); results: [`reports/fm_findings.md`](reports/fm_findings.md) and `reports/fm_*.csv`. It imports `src/cf.py` for the bias and ALS references. Rebuild and run with:

```sh
python scripts/build_fm_notebook.py
python scripts/execute_notebook.py notebooks/factorization_machines.ipynb
```

Prediction files: `outputs/fm_validation.csv`, `outputs/fm_test.csv`, `outputs/fm_cold_item_test.csv`.

## Data and experiments

The downloader retrieves the official Slim edition (150,000 ratings) and Test example edition (1,000 ratings). URLs and SHA-256 checksums are recorded in `data/sources.json`. If Google Drive returns an HTML page, download the named CSVs from the linked official folder into `data/raw/`; the same checksum checks apply.

Experiments include rating and activity distributions, origin/type/grape comparisons, within-user type differences, taste-concentration permutations, sparse ridge feature comparisons, RMSE/MAE, bootstrap confidence intervals, and error slices. Raw data, generated split CSVs, prediction rows, and standalone figure PNGs stay local. The notebook embeds results and figures; reports/ contains compact summaries and split hashes.

Prices and consumer locations are unavailable. Catalogue metadata is retrospective, so chronological splitting does not establish historical attribute availability. [Matched validation contrasts](reports/chronological_feature_contrasts.csv) separate static-attribute additions from rating-context additions at the same regularization. The original ridge context features combine calendar time and rated-vintage age; the FM extension separates those components in `reports/fm_time_vs_age.csv` and the cumulative notebook.

Dataset citation: de Azambuja, R. X., Morais, A. J., & Filipe, V. (2023). [X-Wines: A Wine Dataset for Recommender Systems and Machine Learning](https://doi.org/10.3390/bdcc7010020). *Big Data and Cognitive Computing*, 7(1), 20. [Official repository](https://github.com/rogerioxavier/X-Wines).
