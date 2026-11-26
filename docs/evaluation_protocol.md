# Data and evaluation protocol

## Inputs and reproducibility

Use **Slim**, not upstream Test, for reported core results. `data/sources.json` pins download URLs, byte counts and SHA-256 checksums. Download with `python scripts/download_data.py`. Four inputs cover the Slim experiment and the Test-edition schema/example comparison. Keep raw data local; regenerate instead of committing large CSVs.

Run `python scripts/prepare_data.py` to create `data/processed/Slim/{chronological,random,cold_item}/{train,validation,test}.csv`. Every row retains **RatingID, UserID, WineID, Vintage, Rating, Date**. `reports/split_manifests.json` records checksums and row counts of the exported CSVs. No dense user×wine matrix is materialized. Global random seed: **258**.

For reuse without rerunning experiments, run `python scripts/download_data.py` and `python scripts/prepare_data.py`. Preparation uses the tracked manifest as a read-only reference, verifies all generated CSV hashes before publishing any of them, and writes rows in RatingID order. Changing input, seed, membership or serialization fails verification rather than silently changing the benchmark. Missing local partitions must be prepared explicitly.

`src.pipeline.load_prepared_splits(root, protocol)` returns `(wines, partitions)` after checking input hashes, partition hashes, row counts, full event coverage, unique IDs and the protocol's grouping/time rules. Dates are parsed and Vintage remains a string, including `N.V.`. `src.experiments.join_metadata(ratings, wines)` performs a checked many-to-one join; missing catalogue entries raise an error instead of silently dropping ratings.

Primary = chronological. Random = sensitivity with whole user/wine pairs grouped. Cold item = disjoint wine groups; event counts need not be 80/10/10 because wines have unequal popularity. The upstream **Test edition** is unrelated to any held-out **test partition**. Chronology keeps timestamp ties together and uses strict time ordering. We retain valid repeated events/vintages in Full; no blind pair deduplication.

## Target and query assumptions

Predict the continuous 1–5 rating of an observed event. Query IDs, wine metadata, and optional rated Vintage/Date must be known at prediction time. The data contains ratings rather than purchases or recommendation exposures; consumer home locations and prices are unavailable. Labels exist in partition files for reproducibility, but test labels are not allowed in fitting, feature engineering or selection.

The catalogue is a 2022 snapshot: historical metadata availability is unverified. This is a retrospective sampled-catalogue experiment. Cold-item analysis assumes item attributes exist even when training interactions do not; it does not reconstruct launch dates.

## Features

- IDs, type, elaborate, country, region and winery are categories. RegionID/WineryID values are not ordered magnitudes.
- Parse Grapes safely as a list; use fractional multi-hot values (1/k per grape). Keep vocabulary from training interactions only.
- ABV: unique training-wine median and standard deviation; missing indicator. No validation/test fitting of imputation or scaling.
- Optional time: known rating year and cyclical month. Optional age uses rated Vintage, with N.V./invalid/future values handled as missing. State any timestamp assumption.
- Primary probes exclude price, consumer location, WineName, Website, whole Vintages lists, Body, Acidity and Harmonize. Body/Acidity have partly review-derived provenance; any later experiment using them must be explicitly labeled and justified.
- User/item means, counts and profiles must use training data only. A training row's target-derived features require out-of-fold or strictly prior construction. A global mean computed from all labels is leakage.

Training parameters in an ordinary supervised ID regression are not target-aggregate features; fitting the model to training labels is expected. `FeatureProbe` learns a global training offset plus L2-regularized ID/content coefficients, with no target aggregates in the input.

The probe accepts only the declared feature fields. Labels and undeclared raw columns cannot be added as predictors accidentally. Prediction queries do not need a Rating column. For another implementation, explicitly separate target labels from query features and fit every encoder, imputer and aggregate on training data only.

## Static attributes versus rating context

`reports/chronological_feature_contrasts.csv` derives four validation comparisons from the existing grids at each identical alpha: IDs → static attributes; IDs → rating context; context → context plus static attributes; static attributes → static attributes plus context. Negative ΔRMSE favors the candidate. No models are refitted and no test errors select these contrasts.

At the original ID-reference alpha of 10, adding static attributes changes validation RMSE from about 0.4544 to 0.4539, while adding context changes it to 0.4456. Static attributes plus context reach 0.4440. This supports keeping the two feature groups distinct; it does not establish causal effects.

The historical name `ids_time_only` includes calendar features **and rated-vintage age**. It is not a calendar-only ablation. The `selected_metadata` prediction column can likewise represent a model with context. These names are retained for compatibility with saved results. Their combined test improvement is not an isolated static-metadata improvement, and calendar versus vintage-age contributions are not separated in those original ridge probes. The later FM study adds separate Time/Age validation ablations in `reports/fm_time_vs_age.csv`; these are integrated in `notebooks/wine_recommendation.ipynb`.

## Evaluation and predictions

Use the original training partition throughout these comparisons; this notebook does **not** refit on train+validation after tuning. This keeps history coverage and rare-user/item slices consistent. If the fitting policy changes, rerun every reference under the same policy and publish a new manifest/protocol version.

Tune on validation only. Primary metric **micro RMSE**, secondary **MAE**; also show the mean of per-user RMSE values (macro-user RMSE). Clip predictions to [1,5], do not round. Mean baselines use shrinkage 10 and unknown-ID fallback to training global mean. Ridge unknown categorical terms contribute zero; content remains usable for unknown wines. Report coverage and fallback behavior for every final model.

Return `RatingID,prediction` for **all and only** the corresponding test RatingIDs, exactly once. Validate IDs as sets, merge one-to-one, check finite values and order independence. Any model unable to score a row uses its stated fallback; no selective row dropping. Save validation grid, selected settings, seed, fit/predict runtime and train/validation error.

Use `python scripts/evaluate_predictions.py path/to/predictions.csv` for chronological validation scoring. It loads the verified partition, aligns by RatingID regardless of CSV order, and rejects missing/extra IDs, duplicate IDs and non-finite scores. Predictions are clipped to [1,5]; the output includes the number clipped, evaluated rows, RMSE and MAE. `--protocol` chooses the split and `--partition test` explicitly opts into final evaluation. The same checks are available as `src.pipeline.score_predictions(reference, predictions)`.

Preparation and loading retain labels for reproducibility and scoring; this is an access convention, not a hidden-label service. Never tune on test labels or use test coverage summaries to select a model. An inspected test set remains exploratory if it influences later choices.

The notebook selects the best metadata representation and alpha by validation RMSE within each protocol, then evaluates that selected probe plus prespecified global/user/item/ID references on test. Test metrics are never used to select a feature representation. Reusing an inspected holdout during further model design makes it exploratory; use a fresh final holdout if test results influence the next choices.

Paired 95% user-cluster bootstrap: 2,000 resamples, seed 258, ΔRMSE = candidate − reference. Negative means improvement. User resampling preserves within-user dependence but does not capture retraining uncertainty, temporal shocks or all dependence through shared wines. Sensitivity protocols are exploratory, not multiple independent confirmations.

Report warm/new-user/new-item/new-both errors and training-count bins 0, 1–5, 6–20, 21–100, 101+. Slice counts and user counts accompany metrics. Do not report rating-zero for an unrated wine or equate sample review frequency with market demand.

Optional Top-K is specified in the notebook but **not executed**. It requires a fixed relevance threshold, candidate catalogue, training-item exclusion, identical eligible users/candidates for every model, explicit zero-positive exclusions and a caveat about unknown historical item availability.
