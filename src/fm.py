"""Factorization machine (FM) and content/hybrid helpers for X-Wines.

fastFM does not build on Python 3.11 / NumPy 2 (its wheel build fails), so this
module implements the standard second-order FM directly:

    y_hat(x) = w0 + sum_j w_j x_j + sum_{j<l} <v_j, v_l> x_j x_l
             = w0 + x.w + 0.5 * sum_f [ (x V)_f^2 - (x^2 V^2)_f ]

Every encoder vocabulary, imputation value and parameter is fitted on the
training frame only. Query labels are never read. Unknown categories (an
unseen user, wine, region, ...) get no column, so they contribute zero and the
remaining known features (for example a new wine's type, grapes, country)
still score it. Predictions are clipped to [1, 5] (the shared scoring rule).

Fitting minimizes
    0.5 * sum (y_hat - y)^2 + 0.5 * reg_linear * |w|^2 + 0.5 * reg_factors * |V|^2
(w0 unregularized) with full-batch L-BFGS. Every loss/gradient evaluation is a
few sparse matrix products, so no per-row Python loop and no learning rate.
With n_factors=0 the model is ordinary ridge regression on the same features.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from .experiments import SAFE_META, metrics

SEED = 258
CONTEXT = ['Time', 'Age']
EXTRA = ['Body', 'Acidity', 'Harmonize']  # Body/Acidity partly derived from reviews
LIST_FIELDS = {'Grapes': 'GrapeList', 'Harmonize': 'HarmonizeList'}
ALLOWED = frozenset(['UserID', 'WineID'] + SAFE_META + CONTEXT + EXTRA)
ID_FIELDS = ['UserID', 'WineID']
ALL_STATIC = ID_FIELDS + SAFE_META

# Feature ablation order: identities, then one metadata field at a time, then rating context.
ABLATION = [
    ('ids', ID_FIELDS),
    ('+type', ID_FIELDS + ['Type']),
    ('+country', ID_FIELDS + ['Type', 'Country']),
    ('+grape', ID_FIELDS + ['Type', 'Country', 'Grapes']),
    ('+region', ID_FIELDS + ['Type', 'Country', 'Grapes', 'RegionID']),
    ('all', ALL_STATIC),                       # adds Elaborate, WineryID, ABV
    ('+time', ALL_STATIC + CONTEXT),
]


def clip(values):
    return np.clip(np.asarray(values, dtype=float), 1, 5)


def prediction_frame(query, values):
    """Shared output contract: one RatingID,prediction row per query event."""
    return pd.DataFrame({'RatingID': query.RatingID.to_numpy(), 'prediction': clip(values)})


def _age(frame):
    """Rated-vintage age in years; N.V., invalid and implausible values -> NaN."""
    age = frame.Date.dt.year.to_numpy(dtype=float) - pd.to_numeric(frame.Vintage, errors='coerce').to_numpy(dtype=float)
    return np.where((age >= 0) & (age <= 100), age, np.nan)


class FMEncoder:
    """Sparse design matrix for declared fields; vocabulary from training rows only.

    Categorical fields: one-hot 'Field=value' (missing -> '__MISSING__').
    Grapes, Harmonize: fractional multi-hot, 1/k for a list of k entries.
    ABV: z-score with unique-training-wine median/sd, plus a missing indicator.
    Time: month sin/cos and (year - 2017) / 5 (same as the repo's ridge probe).
    Age: rated-vintage age in decades, median-imputed, plus a missing indicator
         (non-vintage 'N.V.' wines are the main source of missing ages).
    """

    def __init__(self, fields):
        if not fields or set(fields) - ALLOWED:
            raise ValueError(f'Unknown fields: {sorted(set(fields) - ALLOWED)}')
        self.fields = list(fields)

    def _entries(self, frame):
        """Yield (field, row index, column name, value) arrays for every field."""
        n = len(frame); rows = np.arange(n)
        for field in self.fields:
            if field in LIST_FIELDS:
                lists = frame[LIST_FIELDS[field]].tolist()
                lengths = np.array([len(x) for x in lists])
                r = np.repeat(rows, lengths)
                names = np.array([f'{field}={g}' for x in lists for g in x], dtype=object)
                values = np.repeat(1 / np.maximum(lengths, 1), lengths)
                yield field, r, names, values
            elif field == 'ABV':
                abv = pd.to_numeric(frame.ABV, errors='coerce').to_numpy(dtype=float)
                missing = np.isnan(abv)
                z = (np.where(missing, self.abv_median, abv) - self.abv_median) / self.abv_sd
                yield field, rows, np.full(n, 'ABV_z', object), z
                yield field, rows, np.full(n, 'ABV_missing', object), missing.astype(float)
            elif field == 'Time':
                month, year = frame.Date.dt.month.to_numpy(), frame.Date.dt.year.to_numpy()
                yield field, rows, np.full(n, 'month_sin', object), np.sin(2 * np.pi * month / 12)
                yield field, rows, np.full(n, 'month_cos', object), np.cos(2 * np.pi * month / 12)
                yield field, rows, np.full(n, 'year_centered', object), (year - 2017) / 5
            elif field == 'Age':
                age = _age(frame); missing = np.isnan(age)
                yield field, rows, np.full(n, 'age_decades', object), np.where(missing, self.age_median, age) / 10
                yield field, rows, np.full(n, 'age_missing', object), missing.astype(float)
            else:
                values = frame[field].astype(object).where(frame[field].notna(), '__MISSING__')
                yield field, rows, (field + '=' + values.astype(str)).to_numpy(dtype=object), np.ones(n)

    def fit(self, frame):
        wines = frame.drop_duplicates('WineID')
        abv = pd.to_numeric(wines.ABV, errors='coerce') if 'ABV' in frame else pd.Series(dtype=float)
        self.abv_median = float(abv.median()) if abv.notna().any() else 0.
        self.abv_sd = float(abv.std()) if abv.notna().sum() > 1 and abv.std() > 0 else 1.
        age = _age(frame) if {'Date', 'Vintage'} <= set(frame) else np.array([np.nan])
        self.age_median = float(np.nanmedian(age)) if np.isfinite(age).any() else 0.
        names, fields = [], []
        for field, _, col, _ in self._entries(frame):
            new = pd.unique(col)
            names.extend(new); fields.extend([field] * len(new))
        order = pd.Index(names)
        keep = ~order.duplicated()
        self.columns = order[keep]
        self.column_field = np.array(fields, dtype=object)[keep]
        self.index = pd.Series(np.arange(len(self.columns)), index=self.columns)
        return self

    def transform(self, frame):
        rows, cols, vals = [], [], []
        for _, r, col, v in self._entries(frame):
            c = pd.Series(col).map(self.index).to_numpy()
            known = ~np.isnan(c)
            rows.append(r[known]); cols.append(c[known].astype(int)); vals.append(np.asarray(v, float)[known])
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(len(frame), len(self.columns)))

    @property
    def n_features(self):
        return len(self.columns)


class FactorizationMachine:
    """Second-order FM on a sparse design matrix, fitted with L-BFGS.

    n_factors=0 gives a linear (ridge) model. reg_linear and reg_factors may each
    be a float or a {field: value} dict (unlisted fields use the dict's
    'default' entry, else 10), so for example user and wine IDs can be shrunk
    differently from metadata.
    """

    def __init__(self, n_factors=8, reg_linear=10., reg_factors=10., max_iter=1000,
                 init_std=0.01, seed=SEED):
        self.n_factors, self.reg_linear, self.reg_factors = n_factors, reg_linear, reg_factors
        self.max_iter, self.init_std, self.seed = max_iter, init_std, seed

    @staticmethod
    def _reg_vector(reg, column_field):
        if not isinstance(reg, dict):
            return np.full(len(column_field), float(reg))
        default = reg.get('default', 10.)
        return np.array([reg.get(f, default) for f in column_field], dtype=float)

    def _unpack(self, theta):
        p, k = self.p, self.n_factors
        return theta[0], theta[1:p + 1], theta[p + 1:].reshape(p, k)

    def _raw(self, x, x2, w0, w, v):
        out = w0 + x @ w
        if self.n_factors:
            xv = x @ v
            out = out + 0.5 * ((xv ** 2).sum(1) - (x2 @ (v ** 2)).sum(1))
        return out

    def fit(self, x, y, column_field, track=None):
        """track: optional (x_val, y_val) to record train/validation RMSE per iteration."""
        start = time.perf_counter()
        x = sparse.csr_matrix(x, dtype=float); y = np.asarray(y, dtype=float)
        x2 = x.multiply(x).tocsr(); xt, x2t = x.T.tocsr(), x2.T.tocsr()
        self.p = x.shape[1]; k = self.n_factors
        lam_w = self._reg_vector(self.reg_linear, column_field)
        lam_v = self._reg_vector(self.reg_factors, column_field)[:, None]
        rng = np.random.default_rng(self.seed)
        theta0 = np.concatenate([[y.mean()], np.zeros(self.p), rng.normal(0, self.init_std, self.p * k)])

        def loss_grad(theta):
            w0, w, v = self._unpack(theta)
            e = self._raw(x, x2, w0, w, v) - y
            loss = 0.5 * (e @ e + lam_w @ (w ** 2) + (lam_v * v ** 2).sum())
            g_w = xt @ e + lam_w * w
            parts = [[e.sum()], g_w]
            if k:
                g_v = xt @ (e[:, None] * (x @ v)) - v * (x2t @ e)[:, None] + lam_v * v
                parts.append(g_v.ravel())
            return loss, np.concatenate(parts)

        self.curve = []
        def callback(theta):
            if track is None:
                return
            w0, w, v = self._unpack(theta)
            xv_, yv_ = track
            self.curve.append(dict(
                iteration=len(self.curve) + 1,
                train_rmse=metrics(y, clip(self._raw(x, x2, w0, w, v)))['rmse'],
                val_rmse=metrics(yv_, clip(self._raw(xv_, xv_.multiply(xv_), w0, w, v)))['rmse']))

        result = minimize(loss_grad, theta0, jac=True, method='L-BFGS-B', callback=callback,
                          options=dict(maxiter=self.max_iter, maxcor=20, ftol=1e-10, gtol=1e-6))
        self.w0, self.w, self.v = self._unpack(result.x)
        self.converged, self.n_iter = bool(result.success), int(result.nit)
        self.fit_seconds = time.perf_counter() - start
        return self

    def raw(self, x):
        x = sparse.csr_matrix(x, dtype=float)
        return self._raw(x, x.multiply(x), self.w0, self.w, self.v)

    def predict(self, x):
        return clip(self.raw(x))


class FMRecommender:
    """Encoder + FM on rating frames (already joined with wine metadata)."""

    def __init__(self, fields, **fm_params):
        self.fields = list(fields)
        self.encoder = FMEncoder(fields)
        self.fm = FactorizationMachine(**fm_params)

    def fit(self, train, track=None):
        start = time.perf_counter()
        self.encoder.fit(train)
        x = self.encoder.transform(train)
        if track is not None:
            track = (self.encoder.transform(track), track.Rating.to_numpy(dtype=float))
        self.fm.fit(x, train.Rating.to_numpy(dtype=float), self.encoder.column_field, track)
        self.fit_seconds = time.perf_counter() - start
        return self

    def raw(self, frame):
        return self.fm.raw(self.encoder.transform(frame))

    def predict(self, frame):
        return clip(self.raw(frame))

    def weights(self):
        """Linear weight, field and factor norm of every training feature."""
        out = pd.DataFrame({'feature': self.encoder.columns, 'field': self.encoder.column_field,
                            'weight': self.fm.w})
        out['factor_norm'] = np.linalg.norm(self.fm.v, axis=1) if self.fm.n_factors else 0.
        return out

    def factors(self):
        return pd.DataFrame(self.fm.v, index=self.encoder.columns)


class BayesianFM:
    """Library FM (myFM, Gibbs-sampling Bayesian FM) on the same FMEncoder features.

    Replaces the unbuildable fastFM. Each field gets its own hyperprior group, so
    the sampler learns a separate regularization strength for users, wines, each
    metadata field and the context features instead of one hand-tuned lambda.
    Only n_factors and the number of sweeps are chosen on validation.
    """

    def __init__(self, fields, n_factors=8, n_iter=200, n_kept=150, seed=SEED):
        self.fields = list(fields)
        self.encoder = FMEncoder(fields)
        self.n_factors, self.n_iter, self.n_kept, self.seed = n_factors, n_iter, n_kept, seed

    def fit(self, train):
        import myfm  # optional dependency, pinned in requirements.txt
        start = time.perf_counter()
        x = self.encoder.fit(train).transform(train)
        fields = self.encoder.column_field
        # Encoder emits each field's columns contiguously: one group per run of a field.
        cuts = np.flatnonzero(fields[1:] != fields[:-1]) + 1
        groups = np.diff(np.concatenate([[0], cuts, [len(fields)]])).tolist()
        self.mu = float(train.Rating.mean())
        self.model = myfm.MyFMRegressor(rank=self.n_factors, random_seed=self.seed)
        self.model.fit(x, train.Rating.to_numpy(dtype=float) - self.mu, n_iter=self.n_iter,
                       n_kept_samples=self.n_kept, group_shapes=groups)
        self.fit_seconds = time.perf_counter() - start
        return self

    def raw(self, frame):
        return self.mu + self.model.predict(self.encoder.transform(frame))

    def predict(self, frame):
        return clip(self.raw(frame))


def evaluate_fm(train, validation, fields, name=None, **params):
    """Fit on train, score train and validation; one grid row."""
    model = FMRecommender(fields, **params).fit(train)
    return dict(name=name or '+'.join(fields), n_factors=model.fm.n_factors,
                reg_linear=params.get('reg_linear', 10.), reg_factors=params.get('reg_factors', 10.),
                features=model.encoder.n_features, iterations=model.fm.n_iter,
                converged=model.fm.converged, fit_seconds=model.fit_seconds,
                train_rmse=metrics(train.Rating, model.predict(train))['rmse'],
                **{('val_' + k): v for k, v in metrics(validation.Rating, model.predict(validation)).items()})


def evaluate_bayes(train, validation, fields, name=None, **params):
    model = BayesianFM(fields, **params).fit(train)
    return dict(name=name or '+'.join(fields), model='bayesian_fm (myFM)', n_factors=model.n_factors,
                sweeps=model.n_iter, features=model.encoder.n_features, fit_seconds=model.fit_seconds,
                train_rmse=metrics(train.Rating, model.predict(train))['rmse'],
                **{('val_' + k): v for k, v in metrics(validation.Rating, model.predict(validation)).items()})


def grid_search(train, validation, configs, n_jobs=8, evaluator=evaluate_fm):
    """configs: list of (name, fields, params). Fits run in parallel processes;
    each fit is deterministic (fixed seed), so results do not depend on n_jobs."""
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=n_jobs)(delayed(evaluator)(train, validation, fields, name, **params)
                                   for name, fields, params in configs)
    return pd.DataFrame(rows)


def best_params(grid_table):
    """Lowest validation RMSE; ties go to fewer factors."""
    row = grid_table.sort_values(['val_rmse', 'n_factors']).iloc[0]
    return dict(n_factors=int(row.n_factors), reg_linear=float(row.reg_linear),
                reg_factors=float(row.reg_factors))


def interaction_spread(model, field_a, field_b):
    """Std over users of the pairwise term <v_user, v_b> for each category b of field_b.

    Measures how much the FM lets users differ in their preference for, e.g.,
    each wine type (0 means everyone gets the same shift).
    """
    f = model.factors(); fields = model.encoder.column_field
    va = f[fields == field_a].to_numpy(); vb = f[fields == field_b]
    if not len(va) or not len(vb):
        return pd.DataFrame(columns=['feature', 'std_over_users'])
    terms = va @ vb.to_numpy().T
    return pd.DataFrame({'feature': vb.index, 'std_over_users': terms.std(0),
                         'mean_abs': np.abs(terms).mean(0)}).sort_values('std_over_users', ascending=False)


# ---------------------------------------------------------------- Top-K ranking

def fm_score_matrix(model, users, wines_frame):
    """FM scores for every (user, wine) pair without materializing user x wine rows.

    Requires a model with user-side field UserID only and wine-side static fields
    (no Time/Age, which are not defined for a candidate the user never rated):
    score = w0 + w_u + [item linear + item self-interaction] + <v_u, sum_item x_j v_j>.
    """
    if set(model.fields) & set(CONTEXT):
        raise ValueError('Ranking needs a model without rating-context fields')
    enc, fm = model.encoder, model.fm
    item_x = enc.transform(wines_frame.assign(UserID=-10**9))  # dummy user, never in vocabulary
    item_part = fm.raw(item_x) - fm.w0                         # wine-side linear + self-interactions
    item_vec = item_x @ fm.v if fm.n_factors else np.zeros((len(wines_frame), 0))
    ucol = pd.Series(['UserID=' + str(u) for u in users]).map(enc.index).to_numpy()
    known = ~np.isnan(ucol); ucol = np.where(known, ucol, 0).astype(int)
    wu = np.where(known, fm.w[ucol], 0.)
    vu = np.where(known[:, None], fm.v[ucol], 0.) if fm.n_factors else np.zeros((len(users), 0))
    return fm.w0 + wu[:, None] + item_part[None, :] + vu @ np.asarray(item_vec).T


def frame_score_matrix(predict_raw, users, wine_ids, chunk=200):
    """Scores from any model with raw(frame) on a user x wine grid, in user chunks."""
    out = np.empty((len(users), len(wine_ids)))
    for s in range(0, len(users), chunk):
        block = users[s:s + chunk]
        q = pd.DataFrame({'UserID': np.repeat(block, len(wine_ids)), 'WineID': np.tile(wine_ids, len(block))})
        out[s:s + chunk] = predict_raw(q).reshape(len(block), len(wine_ids))
    return out


def topk_setup(train, test, wine_ids, threshold=4.):
    """Eligible users: seen in train and with >=1 relevant (rating >= threshold) test wine
    that is not already in their training history. Candidates: all catalogue wines
    minus the user's training wines."""
    seen = train.groupby('UserID').WineID.agg(set)
    rel = test[test.Rating >= threshold].groupby('UserID').WineID.agg(set)
    users = np.array(sorted(u for u in rel.index if u in seen.index and rel[u] - seen[u]))
    col = pd.Series(np.arange(len(wine_ids)), index=wine_ids)
    exclude = sparse.lil_matrix((len(users), len(wine_ids)), dtype=bool)
    relevant = sparse.lil_matrix((len(users), len(wine_ids)), dtype=bool)
    for r, u in enumerate(users):
        exclude[r, col.reindex(list(seen[u])).dropna().astype(int).to_numpy()] = True
        relevant[r, col.reindex(list(rel[u] - seen[u])).dropna().astype(int).to_numpy()] = True
    return users, exclude.toarray(), relevant.toarray()


def topk_metrics(scores, exclude, relevant, k=10):
    """Precision@k, Recall@k, MRR, AUC per eligible user (averaged), plus the top-k lists."""
    s = np.where(exclude, -np.inf, scores)
    order = np.argsort(-s, axis=1, kind='stable')
    hits = np.take_along_axis(relevant, order, axis=1)
    n_rel = relevant.sum(1); n_cand = (~exclude).sum(1)
    first = np.where(hits.any(1), hits.argmax(1) + 1, np.inf)
    # AUC: fraction of (relevant, non-relevant candidate) pairs ranked correctly.
    ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(s.shape[1])[None, :], axis=1)
    rel_rank_sum = np.where(relevant, ranks, 0).sum(1)          # 0-based ranks of relevant items
    n_neg = n_cand - n_rel
    wrong = rel_rank_sum - n_rel * (n_rel - 1) / 2              # negatives ranked above each positive
    auc = 1 - wrong / np.maximum(n_rel * n_neg, 1)
    result = dict(users=len(s), precision_at_k=float(hits[:, :k].sum(1).mean() / k),
                  recall_at_k=float((hits[:, :k].sum(1) / n_rel).mean()),
                  mrr=float((1 / first).mean()), auc=float(auc.mean()))
    return result, order[:, :k]


def diversity(top_lists, wine_meta):
    """Concentration of top-k lists: type/country shares, distinct countries per list,
    catalogue coverage and Gini coefficient of recommendation counts."""
    flat = top_lists.ravel()
    meta = wine_meta.iloc[flat]
    counts = np.bincount(flat, minlength=len(wine_meta))
    c = np.sort(counts); n = len(c)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ c / (n * c.sum()))
    per_list = wine_meta.Country.to_numpy()[top_lists]
    return dict(red_share=float(meta.Type.eq('Red').mean()),
                top_country=str(meta.Country.value_counts().index[0]),
                top_country_share=float(meta.Country.value_counts(normalize=True).iloc[0]),
                countries_per_list=float(np.mean([len(set(r)) for r in per_list])),
                types_per_list=float(np.mean([len(set(r)) for r in wine_meta.Type.to_numpy()[top_lists]])),
                catalogue_coverage=float((counts > 0).mean()), gini=gini)
