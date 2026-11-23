# Findings: collaborative filtering (Person C)

Notebook: [`notebooks/C_collaborative_filtering.ipynb`](../notebooks/C_collaborative_filtering.ipynb). Code: [`src/cf.py`](../src/cf.py).
Slim edition, chronological split (train 120,000 / validation 15,000 / test 15,000), seed 258. Every configuration was selected on validation; test was scored once per selected model.

## Task 3: models

| Model | Selected on validation | Test RMSE | Test MAE | ΔRMSE vs bias, 95% CI |
|---|---|---|---|---|
| Global mean (repo) | – | 0.6123 | 0.4932 | |
| User mean (repo) | shrinkage 10 | 0.5878 | 0.4649 | |
| Item mean (repo) | shrinkage 10 | 0.4785 | 0.3682 | |
| Ridge on IDs (repo) | alpha 10 | 0.4493 | 0.3401 | |
| **Bias model** μ + b_u + b_i | λ_user 5, λ_wine 2 | **0.4457** | 0.3356 | reference |
| **Item-item CF** | cosine, k = 10, damping 1 | **0.4446** | 0.3342 | −0.0011 [−0.0017, −0.0005] |
| Matrix factorization, our ALS | k = 8, λ = 11 | 0.4460 | 0.3358 | +0.0003 [−0.0001, +0.0007] |
| Matrix factorization, Surprise SVD (SGD) | k = 4, reg 0.02, 20 epochs | 0.4457 | 0.3350 | −0.0001 [−0.0010, +0.0009] |
| SVD++ (implicit feedback), Surprise | k = 4, reg 0.02, 20 epochs | 0.4457 | 0.3351 | −0.0000 [−0.0010, +0.0009] |

Random-split sensitivity check (same settings, not re-tuned): bias 0.4706, item CF 0.4687 (−0.0019 [−0.0025, −0.0013]), ALS 0.4707, Surprise SVD 0.4760, SVD++ 0.4758.

- **Evaluation.** RMSE (primary) and MAE on the chronological split; paired user-cluster bootstrap against the bias model. The predictions agree exactly with the team scorer (`scripts/evaluate_predictions.py`).
- **Why these models.** The task is explicit-rating prediction, so the relevant class models are the bias model, similarity-based CF and latent-factor models (plus the implicit-feedback extension). Classifiers ignore that the stars are ordered, and BPR/one-class models are for clicks or purchases. See `Class_Models_vs_XWines.txt`.
- **Optimization.** Bias model: exact alternating updates. Item CF: closed form. Our MF: alternating least squares, where each half-step is an exact ridge solve, so there is no learning rate and training error falls monotonically. Surprise: SGD with a fixed learning rate of 0.005.
- **Overfitting.** This is the central issue:
  - With weak regularization, MF memorizes training ratings. For example, k = 64 and λ = 5 drives training RMSE down to 0.29 while validation gets worse than the bias model.
  - With λ ≥ 14 the factors shrink to zero and MF becomes exactly the bias model.
  - The useful window in between gains at most 0.0003 RMSE on validation and nothing on test.
  - Surprise SGD shows the same thing over epochs: the best is 20 epochs; more epochs overfit.
  - Item CF also overfits without damping: the textbook weighted average (d = 0) gives 0.494 on validation vs 0.450 for the bias model. A damping term of 1 fixes it.
- **Scaling.**
  - On Slim every model fits in under about 5 seconds.
  - For Full (21M ratings, 1.06M users, 100K wines), the bias model and Surprise SGD scale with the number of ratings (minutes), and our ALS scales with users + wines (its Python loop would need to be vectorized).
  - Item CF is the hard one: a dense wine × wine similarity matrix would need 81 GB, so it would need a sparse top-neighbour index.
  - A dense user × wine matrix (850 TB) is impossible, which is why everything uses sparse data structures.
- **Noise and missing data.**
  - Ratings are half-stars, 73.5% of them between 3.5 and 4.5, and the bias model's residual RMSE is 0.45 even on training data.
  - The median user has 10 training ratings.
  - 6.6% of test rows come from users unseen in training; every model falls back to μ + b_i for them.
  - Ratings drift upward over time: the bias model under-predicts every validation/test month by 0.06–0.12 stars, and CF cannot fix that.

## Task 5: results

- **What worked:** the regularized bias model. It beats the repo's ridge-on-IDs reference (0.4457 vs 0.4493) because users and wines get separate regularization. Item CF with damping gives a small but reliable further gain, on both splits.
- **What did not work:** matrix factorization and SVD++. Once properly regularized, they are statistically indistinguishable from the bias model on the chronological test. With the chronological settings reused on the random split, Surprise SVD/SVD++ are worse (+0.005).
- **How many latent dimensions?** The number of dimensions barely matters. The best k at each λ is within 0.0003 RMSE of the bias model.
- **Who is easier to predict?** Frequent users are predicted much more accurately: test RMSE is 0.525 for users with no training history and 0.405 for users with 21–100 ratings. Wine popularity matters less.
- **Interpreting the bias parameters:**
  - **Wine bias b_i** spans about ±1 star. The highest are iconic wines: Penfolds Grange, Château Lafite and Mouton Rothschild (Pauillac Premier Grand Cru), Dom Pérignon Oenothèque, and Napa Valley reds (Colgin, Eisele Vineyard, Ovid). The lowest are inexpensive varietals from Chile, Argentina, Brazil and Spain, and Retsina.
  - **By type:** Rosé (−0.33) and White (−0.16) sit lowest, and Dessert/Port highest (+0.10).
  - **Popularity:** wine bias is almost unrelated to how often a wine is rated (Spearman 0.08).
  - **User bias b_u** has a standard deviation of 0.20 stars (90% of users within −0.34 to +0.28).
- **Do the latent factors group wines?** No. A wine's 10 nearest neighbours in factor space share its type 38.0% of the time, against 37.5% for random neighbours, and its country 13.9% vs 11.7%. The PCA plot shows no type clusters.
- **Why CF has nothing to learn here:** personal taste is not stable in this data. A user's personal Red-minus-White preference, after removing biases, measured on one random half of their ratings, has a correlation of only r = 0.03 with the same measure on the other half (1,536 users, 20 random splits). Users differ in *how generously* they rate, and wines differ in *how good* they are, but on Slim there is almost no repeatable "this user likes this kind of wine" signal for latent factors to capture.
- **Implication for the project:** remaining error comes from rating noise and time drift, not missing personalization. The gains will come from time/context features and wine attributes for unseen wines (the repo's ridge model and Person D's FM), not from more complex collaborative filtering.
