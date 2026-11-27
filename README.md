# Wine Recommendation System

Rating prediction with X-Wines, including data validation, exploratory analysis, collaborative filtering, and factorization machines.

[Analysis report](notebooks/wine_recommendation.ipynb) · [HTML report](output/workbook.html) · [Evaluation protocol](docs/evaluation_protocol.md)

## Environment

Python 3.11 is used for the recorded experiments.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Report rendering uses a smaller dependency set:

```sh
python -m pip install -r requirements-report.txt
```

## Data preparation

```sh
python scripts/download_data.py
python scripts/prepare_data.py
```

Download URLs and SHA-256 checksums are defined in `data/sources.json`. Prepared partitions are verified against `reports/split_manifests.json`; mismatches raise an error. Raw data and generated prediction files are excluded from version control.

```python
from pathlib import Path
from src.pipeline import load_prepared_splits
from src.experiments import join_metadata

wines, partitions = load_prepared_splits(Path.cwd(), "chronological")
train = join_metadata(partitions["train"], wines)
validation = join_metadata(partitions["validation"], wines).drop(columns="Rating")
```

Encoders, imputation values, aggregates and model parameters are fitted on training rows only. The supported evaluation protocols are `chronological`, `random` and `cold_item`.

## Models and analysis

- Rating analysis: data quality, rating distributions, feature comparisons and uncertainty. Implementation: [experiments.py](src/experiments.py); notebook generator: [build_rating_analysis.py](scripts/build_rating_analysis.py).
- Collaborative filtering: regularized biases, item similarities, ALS, Surprise SVD and SVD++. Implementation: [collaborative_filtering.py](src/collaborative_filtering.py); notebook generator: [build_collaborative_filtering.py](scripts/build_collaborative_filtering.py).
- Factorization machines: sparse feature interactions, matched linear references, context ablations and unseen-wine evaluation. Implementation: [factorization_machines.py](src/factorization_machines.py); notebook generator: [build_factorization_machines.py](scripts/build_factorization_machines.py).

Generate and execute the detailed notebooks in this order:

```sh
python scripts/build_rating_analysis.py
python scripts/execute_notebook.py notebooks/xwines_analysis.ipynb
python scripts/build_collaborative_filtering.py
python scripts/execute_notebook.py notebooks/collaborative_filtering.ipynb
python scripts/build_factorization_machines.py
python scripts/execute_notebook.py notebooks/factorization_machines.ipynb
```

Detailed experiment notebooks are generated locally and excluded from version control. Rebuilding one clears its saved outputs. The published notebook is `notebooks/wine_recommendation.ipynb`.

## Report rendering

The published notebook reads the saved results in `reports/` (and three figures in `reports/figures/`) and draws its tables and charts from them; it does not train models or download data, and runs in a few seconds. Edit it directly, then re-run and export:

```sh
python scripts/execute_notebook.py
python scripts/export_html.py
```

The notebook runner defaults to the published report. The exporter accepts a notebook path and `--output`; its default destination is `output/workbook.html`. Model retraining uses the detailed notebooks above and remains separate from report rendering. `scripts/build_recommendation_report.py` generates an earlier self-contained variant that embeds the results inside the notebook; running it overwrites `notebooks/wine_recommendation.ipynb`.

## Prediction scoring

Write a CSV containing exactly one `RatingID,prediction` row per evaluation event.

```sh
python scripts/evaluate_predictions.py outputs/validation_predictions.csv
```

Scoring defaults to chronological validation. Use `--protocol` to select another protocol and `--partition test` for held-out evaluation after model selection. The scorer aligns IDs, rejects duplicate/missing/extra IDs and non-finite predictions, clips to [1,5], and reports RMSE/MAE. The Python API is `src.pipeline.score_predictions`.

## Tests

```sh
python -m unittest discover -s tests -v
```

Full model tests require the dependencies in `requirements.txt`, including `scikit-surprise`. Numerical results and runtime measurements can vary with native libraries and hardware.

## Interpretation

Prices and consumer locations are unavailable. Wine metadata is a retrospective catalogue, so chronological splitting does not establish historical attribute availability. Rating accuracy, unseen-wine generalization and recommendation ranking are evaluated separately. The [feature contrasts](reports/chronological_feature_contrasts.csv) and `reports/fm_time_vs_age.csv` distinguish static attributes, calendar time and rated-vintage age.

Dataset citation: de Azambuja, R. X., Morais, A. J., & Filipe, V. (2023). [X-Wines: A Wine Dataset for Recommender Systems and Machine Learning](https://doi.org/10.3390/bdcc7010020). *Big Data and Cognitive Computing*, 7(1), 20. [Official repository](https://github.com/rogerioxavier/X-Wines).
