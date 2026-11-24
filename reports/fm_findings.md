# Findings: factorization machines, content and hybrid models

Notebook: [`notebooks/factorization_machines.ipynb`](../notebooks/factorization_machines.ipynb). Code: [`src/fm.py`](../src/fm.py). Tests: [`tests/test_fm.py`](../tests/test_fm.py).
Slim edition, chronological split (train 120,000 / validation 15,000 / test 15,000), seed 258. Every configuration was selected on validation, and test was scored once per selected model. The shared scorer (`scripts/evaluate_predictions.py --partition test`) reproduces the FM's test RMSE exactly.

## Models

### Implementation (fastFM replaced)
- **fastFM does not build** on Python 3.11 / NumPy 2: both the PyPI release and the GitHub source fail at the wheel-build step. So we use two other FMs:
  1. **Our own FM** (`FactorizationMachine`), the main model. It is the standard second-order FM, with the pairwise term computed in O(k·nnz). It minimizes ½·SSE + ½λ_w‖w‖² + ½λ_V‖V‖² by **full-batch L-BFGS** with an analytic gradient, checked against finite differences in the tests. With k = 0 it reproduces the closed-form ridge solution (also tested), and on the repo's ridge features it matches the repo's `FeatureProbe` validation RMSE (0.4440 vs 0.44397 at λ = 10).
  2. **A library FM, myFM** (`BayesianFM`, Gibbs-sampling Bayesian FM) with one hyperprior group per field. myFM 0.4.0 needs `setuptools<81` (it imports `pkg_resources`); both are pinned in `requirements.txt`.
- **Features:** one-hot user, wine, type, elaborate, country, region and winery; fractional multi-hot grapes (1/k for a k-grape blend); ABV z-score + missing flag. Context features: month sin/cos, rating year, and rated-vintage age + missing flag. All of these are fitted on training rows only. An unseen user, wine or category gets no column, so it contributes zero; that is the FM's fallback.

### Hyperparameters and overfitting
Grid on the full feature set: k ∈ {0, 2, 4, 8, 16}, λ_w ∈ {2…10}, λ_V ∈ {100, 300, 1000}, 44 fits in about 3 minutes using 8 processes.

| | Validation RMSE |
|---|---|
| Linear (k = 0), λ_w = 5 | 0.4419 |
| **FM, k = 8, λ_w = 5, λ_V = 300 (selected)** | **0.4416** |
| Bayesian FM (myFM) k = 0 / 4 / 8 | 0.4428 / 0.4462 / 0.4479 |

- **k barely matters** once λ_V is large enough. k = 2…16 at λ_V = 300 are all 0.4416. At λ_V = 1000 the factors are shrunk to zero and the FM equals the linear model (0.4419).
- **λ_w matters more than k.** λ_w = 5 is better than the repo ridge's fixed α = 10, which comes from its coarse {1, 10, 100} grid.
- **Overfitting:**
  - With weak factor regularization (λ_V = 10, k = 16), training RMSE falls to 0.23 while validation rises to 0.557. Validation is best (0.452) at iteration 79 and then degrades.
  - The selected model converges in 580 iterations (26 s), ending at train 0.4457 / validation 0.4416.
  - The Bayesian FM learns per-field priors but still overfits with k > 0 (train 0.33 at k = 8). It is worse than our tuned FM, so it stays a cross-check.

### Feature ablation (validation RMSE, each step tuned)

| Step | Linear | FM |
|---|---|---|
| IDs | 0.4513 | 0.4513 |
| +type | 0.4512 | 0.4512 |
| +country | 0.4513 | 0.4513 |
| +grape | 0.4514 | 0.4514 |
| +region | 0.4510 | 0.4510 |
| all (+elaborate, winery, ABV) | 0.4506 | 0.4506 |
| **+time (rating time + vintage age)** | 0.4419 | **0.4416** |

- **Static metadata adds 0.0007 in total; context adds 0.009.**
- The FM's interactions help only once context is present (−0.0003), and only through context × metadata terms (below).

### Time vs vintage age (separated for the first time; the repo left this unresolved)

| | Without static metadata | With static metadata |
|---|---|---|
| +Time | −0.0082 | −0.0089 |
| +Age | −0.0008 | −0.0008 |
| +both | −0.0090 | −0.0091 |

- **Calendar time is about 90% of the context gain.** It captures the upward rating drift also seen in the collaborative-filtering analysis. Vintage age adds a small, separate gain.

### Extra features
- Body, Acidity and Harmonize change the FM's validation RMSE by ≤ 0.0001, so they don't help on warm wines.
- Body and Acidity are partly derived from reviews, so a gain from them would have needed a leakage caveat anyway. We do not use them in the main model.

### Cold-start wines (`cold_item` split; no test wine has a training rating)

Validation (8,500 rows, 101 wines):

| Content used | Linear | FM |
|---|---|---|
| users only | 0.6008 | 0.6008 |
| users + type | 0.5963 | 0.5956 |
| users + country | 0.6040 | 0.6040 |
| users + grape | 0.6067 | 0.6067 |
| users + region | 0.5887 | 0.5878 |
| all static | 0.5583 | 0.5446 |
| all + time (main) | 0.5480 | **0.5402** |
| main + extras | **0.5472** | 0.5475 |

Test (14,536 rows, 101 wines, scored once):

| Model | RMSE | MAE | ΔRMSE vs repo ridge [95% CI] |
|---|---|---|---|
| User mean (repo) | 0.6397 | 0.4982 | +0.065 |
| **Ridge IDs + metadata + time (repo)** | **0.5744** | 0.4490 | reference |
| Linear, main + extras (ours) | 0.5788 | 0.4516 | +0.004 [+0.004, +0.005] |
| FM, all + time (validation-selected) | 0.5909 | 0.4638 | +0.017 [+0.014, +0.019] |

- **Metadata matters a lot for new wines.** Content takes the error from 0.640 (user mean) to 0.57–0.59, and the combination of attributes matters most: no single field gets below 0.589.
- **The FM's cold-start interactions did not generalize.** It was best on validation (−0.008 vs linear) but worst on test (+0.017 vs ridge).
  - Validation holds only 101 wines, so the FM's lower regularization (λ_w = 3, λ_V = 100) and its wine-side attribute interactions (type × country × grape × region) fit those particular wines.
  - The lesson: with few cold wines, choose the simpler, more regularized model, or tune with grouped cross-validation over wines. We report the failure rather than re-select on test.

## Results

### Final comparison (chronological test, same rows and scorer)

| Family | Model | Settings | Test RMSE | Test MAE | Fit (s) |
|---|---|---|---|---|---|
| Factorization machine | **Linear, IDs + metadata + time** | k = 0, λ_w = 5 | **0.4315** | **0.3165** | 2.5 |
| Factorization machine | **FM, IDs + metadata + time** | k = 8, λ_w = 5, λ_V = 300 | **0.4315** | 0.3174 | 19.5 |
| Repository baseline | Ridge IDs + metadata + time | α = 10 | 0.4332 | 0.3177 | 1.4 |
| Collaborative filtering | Item CF | cosine, k = 10, d = 1 | 0.4446 | 0.3342 | 0.8 |
| Collaborative filtering | Surprise SVD / SVD++ | k = 4 | 0.4457 | 0.3350 | 1.2 / 2.5 |
| Collaborative filtering | Bias model | λ_u = 5, λ_i = 2 | 0.4457 | 0.3356 | 0.02 |
| Collaborative filtering | ALS MF | k = 8, λ = 11 | 0.4460 | 0.3358 | 0.9 |
| Factorization machine | FM, IDs only | k = 8, λ_w = 5, λ_V = 300 | 0.4469 | 0.3370 | 1.4 |
| Repository baseline | Ridge IDs | α = 10 | 0.4493 | 0.3401 | 0.5 |
| Repository baseline | Item mean / user mean / global | | 0.4785 / 0.5878 / 0.6123 | | |

- **FM vs ridge:** −0.0016 RMSE [−0.0030, −0.0002]. **FM vs bias model:** −0.0142 [−0.0163, −0.0120].
- The best models are the ones that see rating **context**. The FM ties the linear model on the same features, so the gain over the repo ridge comes from better regularization (λ_w = 5), not from interactions.
- **Did not beat simpler models:**
  - FM interactions over linear (0.4315 vs 0.4315).
  - Static metadata on warm wines (+0.0007 on validation).
  - The extra features.
  - The Bayesian FM.
  - The FM on the cold-item test.
  - On IDs alone, the FM (0.4469) is slightly worse than the bias model in `src/cf.py` (0.4457), which has separate user and wine regularization.

### Does metadata help warm wines, or only new ones?
- **Mostly new wines.** On warm rows the gain over IDs comes almost entirely from context. For example, adding static metadata to IDs changes validation RMSE by only −0.0007, while time alone gives −0.008.
- On test the full FM is 0.015–0.033 better than FM-IDs in every wine-popularity bin with history (1–5 ratings: −0.033, 6–20: −0.020, 21–100 and 101+: −0.015). Mostly this is the time feature, which fixes the drift.
- For the 16 test rows on wines unseen in training, the chronological FM does worse than FM-IDs (0.72 vs 0.63). The sample is tiny, and the cold-item split above is the reliable test of new wines.
- **Frequent users are easier:** users with no history 0.507, 1–5 ratings 0.438, 6–20 0.427, 21–100 0.391.

### Feature weights (FM linear weights, in stars, on top of user and wine IDs)
- **Type:** Rosé −0.21, White −0.09, Red −0.03, Dessert +0.07, Sparkling +0.08, Dessert/Port +0.17.
- **Country** (≥ 5 training wines): United States +0.26, France +0.21, New Zealand +0.18, Australia +0.15, Germany +0.13; Russia −0.20, Brazil −0.17, Chile −0.16, Uruguay −0.12.
- **Grapes** (≥ 5 training wines): Petit Verdot +0.29, Cabernet Franc +0.24, Malvasia +0.19, Furmint +0.18; Gamay −0.18, Touriga Franca −0.13, Tannat −0.12, Merlot −0.11.
- **Context:** rating year +0.21 per 5 years (the drift), vintage age +0.04 per decade, non-vintage (N.V.) −0.12, ABV +0.06 per SD.
- The FM's and the repo ridge's weights correlate at 0.987, so the interpretation does not depend on the model.
- These weights are shrunk because the wine ID absorbs most of each wine's quality.

### Country or grape: which is more informative?
- **Neither does much alone.** On warm wines, country and grape each change validation RMSE by ≤ 0.0001.
- On cold wines, country alone (0.604) and grape alone (0.607) are both *worse* than users only (0.601): too many sparse levels, each with few wines. Type (0.596) and region (0.588) help more.
- **Country's coefficients are larger and better supported** (31 countries vs 187 grapes; 106 grapes have fewer than 3 training wines). It is the more interpretable single attribute, but the value of metadata comes from combining fields.

### Time and vintage age vs static metadata
- **Context matters more than static metadata:** −0.009 vs −0.0007 on validation, consistently with and without the other group. Most of it is calendar time.
- Caveat: the catalogue is a 2022 snapshot, so static attributes may not reflect what was known at rating time. Time is a rating-event feature and has no such problem.

### What the FM's interactions learned
- **User × attribute interactions shrink to zero** at the selected λ_V (spread over users of ⟨v_u, v_type⟩ < 0.0001). The model found no repeatable personal taste for types, countries or grapes.
- This matches the collaborative-filtering results: the split-half test (personal Red−White preference r = 0.03) and matrix factorization performing no better than the bias model (see [`cf_findings.md`](cf_findings.md)).
- The small interaction gain sits in **context × metadata** terms. Mean factor norm by field: Age 0.16, Time 0.16, ABV 0.06, Type 0.02, users 0.001.
- Example: rating year × type adds −0.007 stars per 5 years for White and +0.011 for Rosé, on top of the +0.21 drift.

### Top-K ranking (4,684 eligible users, 1,007 candidate wines, relevant = test rating ≥ 4, training wines excluded)

| Model | P@10 | R@10 | MRR | AUC |
|---|---|---|---|---|
| FM (IDs + static metadata) | 0.0079 | 0.035 | 0.027 | 0.714 |
| ALS MF | 0.0082 | 0.036 | 0.041 | 0.713 |
| Bias model | 0.0080 | 0.035 | 0.041 | 0.713 |
| **Popularity** | **0.0183** | **0.091** | **0.070** | **0.850** |

- **Popularity wins clearly.** The rating models rank by predicted *rating*, which favours rarely-rated but highly rated wines. Relevance, though, requires the user to have *chosen* the wine, and that is driven by popularity (selection bias: users rate what they drink).
- FM, MF and the bias model have almost the same precision, recall and AUC. The FM's MRR is lower (0.027 vs 0.041), so its first hit tends to come later in the list.
- Precision values are low because unrated wines count as non-relevant.

### Diversity of the top-10 lists
- **FM, MF and bias all recommend almost the same ~16–21 wines to everyone:** catalogue coverage 1.6–2.1%, Gini 0.99, US share 42%, 70% reds.
- Catalogue: 50% red, France 18% top country. Training ratings: 66% red, France 26%.
- Popularity is just as concentrated but in a different way: 98% red and Argentina-heavy, with 1.2 wine types per list vs 4.0 for FM.
- **The personalized models are bias-dominated:** a wine's global quality decides the list, so they do not personalize either. Calibration or a diversity re-ranking would be needed before deployment.

### Noise and missing data in the content features
- **Non-vintage wines:** 9,643 Slim ratings (7,381 in train) have Vintage "N.V." They get the median training age plus an `age_missing` flag, so the FM learns their offset (−0.12 stars; N.V. train mean 3.68 vs 3.79 for vintage rows) instead of an impossible age. All 7,381 missing training ages are N.V. rows (no future or > 100-year vintages).
- **Blends:** 288 of 1,000 training wines have more than one grape (up to 9). Fractional 1/k encoding keeps each wine's total grape weight at 1, so blends are not over-counted.
- **Rare categories:** 220 of 321 regions and 106 of 187 grapes have fewer than 3 training wines. L2 regularization shrinks them: mean |weight| is 0.074 for categories with ≤ 2 wines vs 0.108 for > 20. Their coefficients are still noisy, so we only interpret categories with ≥ 5 wines.
- **Catalogue snapshot:** metadata is from 2022 and is used retrospectively for older ratings (see `docs/evaluation_protocol.md`). ABV has no missing values in Slim, but the missing flag is kept for Full.
