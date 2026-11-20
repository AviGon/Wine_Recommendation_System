# Wine_Recommendation_System

CSE 258R course project: X-Wines rating prediction. Person A's contribution covers **Task 1 (exploration)** and **Task 2 (predictive task, features and evaluation)**, with lightweight feature probes for the team handoff.

Start with [the executed notebook](notebooks/person_a_tasks_1_2.ipynb), [short measured findings](reports/findings.md), [assignment/plan review](docs/plan_review.md), and [shared data contract](docs/data_contract.md).

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

You can also open `notebooks/person_a_tasks_1_2.ipynb` in Jupyter or VS Code and run all cells using the environment above. `scripts/build_notebook.py` regenerates the source notebook and clears outputs; run it only when changing its authored content, then execute again. Execution may take several minutes (69 main ridge fits, three time-diagnostic fits, EDA and clustered bootstraps).

The downloader uses the official **Slim 150,000-rating** files plus the **Test 1,000-rating** example edition. `data/sources.json` records exact URLs/hashes. If Drive serves an HTML page instead of data, download the named CSVs from the linked official folder into `data/raw/`; the same checksums must pass. Never silently substitute Test for Slim. Raw data, regenerated split CSVs, prediction rows and standalone figure PNGs are ignored; the notebook embeds results and figures, and tracked reports contain compact summaries and split hashes.

## Coverage and limits

The notebook maps all five assignment sections and the restaurant example to actual X-Wines fields. It includes activity/category/origin/time EDA, within-user type comparisons, taste-concentration permutations, chronological/random/cold-item splits, coverage, shared output files, validation feature/regularization experiments, RMSE/MAE, paired user-bootstrap intervals and error slices.

No price or user-location fields exist. Catalogue metadata is a retrospective snapshot; the future-prediction split does not prove historical availability of attributes. A combined metadata/time improvement is not isolated proof of static metadata value. Full, KNN/MF/final hybrids, literature synthesis and Top-K remain team work; the notebook labels these boundaries.

The assignment PDF states **early week 10**, without an exact deadline. Confirm Gradescope/course announcements. See the review for a relative integration schedule and the 20-minute presentation plan.

Dataset citation: de Azambuja, R. X., Morais, A. J., & Filipe, V. (2023). [X-Wines: A Wine Dataset for Recommender Systems and Machine Learning](https://doi.org/10.3390/bdcc7010020). *Big Data and Cognitive Computing*, 7(1), 20. [Official repository](https://github.com/rogerioxavier/X-Wines).
