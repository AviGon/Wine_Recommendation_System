# Shared data and evaluation contract

## Inputs and reproducibility

Use **Slim**, not upstream Test, for reported core results. `data/sources.json` pins download URLs, byte counts and SHA-256 checksums. Download with `python scripts/download_data.py`. Four inputs cover the Slim experiment and the Test-edition schema/example comparison. Keep raw data local; regenerate instead of committing large CSVs.

Run the notebook to create `data/processed/Slim/{chronological,random,cold_item}/{train,validation,test}.csv`. Every row retains **RatingID, UserID, WineID, Vintage, Rating, Date**. `reports/split_manifests.json` records checksums and row counts of the exported CSVs. No dense user×wine matrix is materialized. Global random seed: **258**.

Primary = chronological. Random = sensitivity with whole user/wine pairs grouped. Cold item = disjoint wine groups; event counts need not be 80/10/10 because wines have unequal popularity. The upstream **Test edition** is unrelated to any held-out **test partition**. Chronology keeps timestamp ties together and uses strict time ordering. We retain valid repeated events/vintages in Full; no blind pair deduplication.

## Target and query assumptions

Predict the continuous 1–5 rating of an observed event. Query IDs, wine metadata, and optional rated Vintage/Date must be known at prediction time. No task implies purchase, recommendation exposure, consumer home location, or price. Labels exist in partition files for reproducibility, but test labels are not allowed in fitting, feature engineering or selection.

The catalogue is a 2022 snapshot: historical metadata availability is unverified. Call this a retrospective sampled-catalogue experiment. Cold-item analysis assumes item attributes exist even when training interactions do not; it does not reconstruct launch dates.

## Features

- IDs, type, elaborate, country, region and winery are categories. RegionID/WineryID values are not ordered magnitudes.
- Parse Grapes safely as a list; use fractional multi-hot values (1/k per grape). Keep vocabulary from training interactions only.
- ABV: unique training-wine median and standard deviation; missing indicator. No validation/test fitting of imputation or scaling.
- Optional time: known rating year and cyclical month. Optional age uses rated Vintage, with N.V./invalid/future values handled as missing. State any timestamp assumption.
- Primary probes exclude price, consumer location, WineName, Website, whole Vintages lists, Body, Acidity and Harmonize. Body/Acidity have partly review-derived provenance; any later experiment using them must be explicitly labeled and justified.
- Train-only user/item means, counts and profiles are allowed as team extensions. A training row's target-derived features require out-of-fold or strictly prior construction. A global mean computed from all labels is leakage.

Training parameters in an ordinary supervised ID regression are not target-aggregate features; fitting the model to training labels is expected. `FeatureProbe` learns a global training offset plus L2-regularized ID/content coefficients, with no target aggregates in the input.

## Evaluation and predictions

Use the original training partition throughout these comparisons; this notebook does **not** refit on train+validation after tuning. This keeps history coverage and rare-user/item slices consistent. If the team changes that policy, rerun every reference under the same policy and publish a new manifest/protocol version.

Tune on validation only. Primary metric **micro RMSE**, secondary **MAE**; also show the mean of per-user RMSE values (macro-user RMSE). Clip predictions to [1,5], do not round. Mean baselines use shrinkage 10 and unknown-ID fallback to training global mean. Ridge unknown categorical terms contribute zero; content remains usable for unknown wines. Report coverage and fallback behavior for every final model.

Return `RatingID,prediction` for **all and only** the corresponding test RatingIDs, exactly once. Validate IDs as sets, merge one-to-one, check finite values and order independence. Any model unable to score a row uses its stated fallback; no selective row dropping. Save validation grid, selected settings, seed, fit/predict runtime and train/validation error.

The notebook selects the best metadata representation and alpha by validation RMSE within each protocol, then evaluates that selected probe plus prespecified global/user/item/ID references on test. Test metrics are never used to select a feature representation. Reusing an inspected holdout during further model design makes it exploratory; use a fresh final holdout if test results influence the next choices.

Paired 95% user-cluster bootstrap: 2,000 resamples, seed 258, ΔRMSE = candidate − reference. Negative means improvement. User resampling preserves within-user dependence but does not capture retraining uncertainty, temporal shocks or all dependence through shared wines. Sensitivity protocols are exploratory, not multiple independent confirmations.

Report warm/new-user/new-item/new-both errors and training-count bins 0, 1–5, 6–20, 21–100, 101+. Slice counts and user counts accompany metrics. Do not report rating-zero for an unrated wine or equate sample review frequency with market demand.

## Person B/C/D integration

- **B:** compare task, data, features, objective, metrics and split definitions in the literature. A paper's score under a different split/candidate set is not directly comparable.
- **C:** add KNN/MF, tuned latent dimension/regularization, convergence, runtime and baseline comparisons.
- **D:** add content/hybrid models, feature removal tests, representation alternatives and unseen-item fallback. The ridge probes here are preliminary, not a complete hybrid study.
- **Everyone:** interpret fitted parameters/embeddings and limitations in Tasks 3/5. Metadata non-improvement is a valid result; do not search the test set for a better story.

Optional Top-K is specified in the notebook but **not executed**. It requires a fixed relevance threshold, candidate catalogue, training-item exclusion, identical eligible users/candidates for every model, explicit zero-positive exclusions and a caveat about unknown historical item availability.
