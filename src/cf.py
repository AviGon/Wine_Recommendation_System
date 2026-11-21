"""Person C: collaborative-filtering models for X-Wines rating prediction.

Every model is fitted on a training frame only and predicts for any query frame
with UserID and WineID columns. Query labels are never read. Predictions are
clipped to [1, 5] (the shared scoring rule) and returned in query row order.

Models (all from the course recommender-systems slides):
- BiasBaseline: mu + b_u + b_i, regularized, fitted by alternating updates.
- ItemCF: item-to-item collaborative filtering with Jaccard, cosine or Pearson
  similarity, predicting deviations from the bias baseline.
- SurpriseMF: latent-factor model mu + b_u + b_i + p_u . q_i (Surprise SVD,
  stochastic gradient descent) and its implicit-feedback extension (SVD++).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse

from .experiments import metrics

SEED = 258
SIMILARITIES = ('jaccard', 'cosine', 'pearson')


def clip(values):
    return np.clip(np.asarray(values, dtype=float), 1, 5)


def prediction_frame(query, values):
    """Shared output contract: one RatingID,prediction row per query event."""
    return pd.DataFrame({'RatingID': query.RatingID.to_numpy(), 'prediction': clip(values)})


class BiasBaseline:
    """Regularized bias model r = mu + b_u + b_i.

    Minimizes sum (r - mu - b_u - b_i)^2 + reg_user*sum b_u^2 + reg_item*sum b_i^2
    by alternating closed-form updates (each step is exact for one block).
    Unknown users or wines contribute a zero bias.
    """

    def __init__(self, reg_user=5., reg_item=5., n_iter=15):
        self.reg_user, self.reg_item, self.n_iter = reg_user, reg_item, n_iter

    def fit(self, train):
        users = train.UserID.to_numpy(); items = train.WineID.to_numpy()
        r = train.Rating.to_numpy(dtype=float)
        self.mu = float(r.mean())
        self.user_ids, u = np.unique(users, return_inverse=True)
        self.item_ids, i = np.unique(items, return_inverse=True)
        n_u = np.bincount(u, minlength=len(self.user_ids))
        n_i = np.bincount(i, minlength=len(self.item_ids))
        bu = np.zeros(len(self.user_ids)); bi = np.zeros(len(self.item_ids))
        for _ in range(self.n_iter):
            bi = np.bincount(i, r - self.mu - bu[u], len(bi)) / (self.reg_item + n_i)
            bu = np.bincount(u, r - self.mu - bi[i], len(bu)) / (self.reg_user + n_u)
        self.bu = pd.Series(bu, index=self.user_ids)
        self.bi = pd.Series(bi, index=self.item_ids)
        self.user_count = pd.Series(n_u, index=self.user_ids)
        self.item_count = pd.Series(n_i, index=self.item_ids)
        return self

    def raw(self, frame):
        return (self.mu + frame.UserID.map(self.bu).fillna(0).to_numpy()
                + frame.WineID.map(self.bi).fillna(0).to_numpy())

    def predict(self, frame):
        return clip(self.raw(frame))


def item_similarity(train, kind, shrink=0.):
    """Dense wine x wine similarity from training ratings only.

    jaccard: |U_i & U_j| / |U_i | U_j| on the sets of users who rated each wine.
    cosine:  cosine of the two wines' rating vectors (unrated = 0).
    pearson: Pearson correlation over co-rating users only.
    shrink:  multiply by n_ij / (n_ij + shrink) to distrust small overlaps.
    """
    if kind not in SIMILARITIES:
        raise ValueError(f'Unknown similarity: {kind}')
    item_ids, i = np.unique(train.WineID.to_numpy(), return_inverse=True)
    _, u = np.unique(train.UserID.to_numpy(), return_inverse=True)
    r = train.Rating.to_numpy(dtype=float)
    shape = (u.max() + 1, len(item_ids))
    # Repeated (user, wine) events are summed by COO->CSR; average them instead.
    counts = sparse.csr_matrix((np.ones_like(r), (u, i)), shape=shape)
    R = sparse.csr_matrix((r, (u, i)), shape=shape)
    if not (np.array_equal(R.indptr, counts.indptr) and np.array_equal(R.indices, counts.indices)):
        raise AssertionError('Rating and count matrices must share one sparsity pattern')
    R.data = R.data / counts.data
    B = counts.copy(); B.data[:] = 1.
    n = (B.T @ B).toarray()
    if kind == 'jaccard':
        size = np.diag(n)
        union = size[:, None] + size[None, :] - n
        sim = np.divide(n, union, out=np.zeros_like(n), where=union > 0)
    elif kind == 'cosine':
        dot = (R.T @ R).toarray()
        norm = np.sqrt(np.diag(dot))
        denom = norm[:, None] * norm[None, :]
        sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    else:
        sx = (R.T @ B).toarray()            # sum of x_i over users who rated i and j
        sxy = (R.T @ R).toarray()
        sxx = (R.multiply(R).T @ B).toarray()
        safe = np.where(n > 0, n, 1)
        cov = sxy - sx * sx.T / safe
        var = sxx - sx ** 2 / safe
        denom = np.sqrt(np.clip(var * var.T, 0, None))
        sim = np.divide(cov, denom, out=np.zeros_like(cov), where=(denom > 1e-12) & (n >= 2))
    if shrink:
        sim = sim * n / (n + shrink)
    np.fill_diagonal(sim, 0.)
    return item_ids, sim


class ItemCF:
    """Item-to-item CF on top of the bias baseline.

    prediction = baseline(u, i) + sum_j s_ij (r_uj - baseline(u, j)) / (sum_j s_ij + damping)
    over the k wines j that user u rated in training with the largest positive
    similarity to i. Unknown users/wines, or no positive neighbour, fall back to
    the baseline. damping > 0 shrinks the neighbour correction toward zero when
    the neighbours are few or weakly similar; damping=0 is the textbook average.
    """

    def __init__(self, similarity='pearson', k=20, shrink=50., damping=5., baseline=None):
        self.similarity, self.k, self.shrink, self.damping = similarity, k, shrink, damping
        self.baseline = baseline if baseline is not None else BiasBaseline()

    def fit(self, train):
        self.baseline.fit(train)  # always refit: a baseline from another split would leak
        self.item_ids, self.sim = item_similarity(train, self.similarity, self.shrink)
        self.item_index = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        work = pd.DataFrame({'UserID': train.UserID.to_numpy(),
                             'item': train.WineID.map(self.item_index).to_numpy(),
                             'res': train.Rating.to_numpy(dtype=float) - self.baseline.raw(train)})
        work = work.groupby(['UserID', 'item'], as_index=False).res.mean()
        self.history = {user: (g.item.to_numpy(), g.res.to_numpy())
                        for user, g in work.groupby('UserID')}
        return self

    def raw(self, frame):
        base = self.baseline.raw(frame)
        items = frame.WineID.map(self.item_index).to_numpy()
        out = base.copy()
        self.neighbour_rows = 0
        for row, (user, item) in enumerate(zip(frame.UserID.to_numpy(), items)):
            if np.isnan(item) or user not in self.history:
                continue
            js, res = self.history[user]
            s = self.sim[int(item), js]
            keep = s > 0
            if not keep.any():
                continue
            s, res = s[keep], res[keep]
            if len(s) > self.k:
                top = np.argpartition(-s, self.k)[:self.k]
                s, res = s[top], res[top]
            out[row] += s @ res / (s.sum() + self.damping)
            self.neighbour_rows += 1
        return out

    def predict(self, frame):
        return clip(self.raw(frame))


class ALSMF:
    """Latent-factor model r = mu + b_u + b_i + p_u . q_i fitted by alternating
    least squares (our own implementation).

    Objective: sum (r - mu - b_u - b_i - p_u.q_i)^2
               + reg * (sum_u |p_u|^2 + sum_i |q_i|^2)
               + reg_user_bias * sum_u b_u^2 + reg_item_bias * sum_i b_i^2
    With wines fixed, each user's (b_u, p_u) is an exact ridge regression; then
    the roles swap. Every half-step cannot increase the objective, so training
    error decreases monotonically. n_factors=0 gives the bias-only model.
    """

    def __init__(self, n_factors=8, reg=10., reg_user_bias=5., reg_item_bias=5., n_iter=15,
                 init_std=0.1, seed=SEED, track=None):
        self.n_factors, self.reg = n_factors, reg
        self.reg_user_bias, self.reg_item_bias = reg_user_bias, reg_item_bias
        self.n_iter, self.init_std, self.seed = n_iter, init_std, seed
        self.track = track  # optional (validation_frame) to record a learning curve

    @staticmethod
    def _solve(groups, other_bias, other_vec, reg, reg_bias):
        """For each row group: ridge of target - other_bias on [1, other_vec]."""
        k = other_vec.shape[1]
        penalty = np.diag([reg_bias] + [reg] * k)
        bias = np.zeros(len(groups)); vec = np.zeros((len(groups), k))
        for g, (idx, y) in enumerate(groups):
            x = np.hstack([np.ones((len(idx), 1)), other_vec[idx]])
            w = np.linalg.solve(x.T @ x + penalty, x.T @ (y - other_bias[idx]))
            bias[g], vec[g] = w[0], w[1:]
        return bias, vec

    def fit(self, train):
        r = train.Rating.to_numpy(dtype=float)
        self.mu = float(r.mean())
        self.user_ids, u = np.unique(train.UserID.to_numpy(), return_inverse=True)
        self.item_ids, i = np.unique(train.WineID.to_numpy(), return_inverse=True)
        y = r - self.mu
        by_user = [(i[idx], y[idx]) for idx in _group_indices(u, len(self.user_ids))]
        by_item = [(u[idx], y[idx]) for idx in _group_indices(i, len(self.item_ids))]
        rng = np.random.default_rng(self.seed)
        k = self.n_factors
        self.pu = rng.normal(0, self.init_std, (len(self.user_ids), k))
        self.qi = rng.normal(0, self.init_std, (len(self.item_ids), k))
        self.bu = np.zeros(len(self.user_ids)); self.bi = np.zeros(len(self.item_ids))
        self._index()
        self.curve = []
        start = time.perf_counter()
        for it in range(1, self.n_iter + 1):
            self.bu, self.pu = self._solve(by_user, self.bi, self.qi, self.reg, self.reg_user_bias)
            self.bi, self.qi = self._solve(by_item, self.bu, self.pu, self.reg, self.reg_item_bias)
            if self.track is not None:
                self.curve.append(dict(iteration=it,
                                       train_rmse=metrics(r, self.predict(train))['rmse'],
                                       val_rmse=metrics(self.track.Rating, self.predict(self.track))['rmse']))
        self.fit_seconds = time.perf_counter() - start
        return self

    def _index(self):
        self.user_index = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        self.item_index = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)

    def raw(self, frame):
        u = frame.UserID.map(self.user_index).to_numpy()
        i = frame.WineID.map(self.item_index).to_numpy()
        ku, ki = ~np.isnan(u), ~np.isnan(i)
        out = np.full(len(frame), self.mu, dtype=float)
        out[ku] += self.bu[u[ku].astype(int)]
        out[ki] += self.bi[i[ki].astype(int)]
        both = ku & ki
        out[both] += np.einsum('nk,nk->n', self.pu[u[both].astype(int)], self.qi[i[both].astype(int)])
        return out

    def predict(self, frame):
        return clip(self.raw(frame))

    def wine_factors(self):
        return (pd.DataFrame(self.qi, index=self.item_ids), pd.Series(self.bi, index=self.item_ids))

    def user_factors(self):
        return (pd.DataFrame(self.pu, index=self.user_ids), pd.Series(self.bu, index=self.user_ids))


def _group_indices(codes, n_groups):
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    return [order[bounds[g]:bounds[g + 1]] for g in range(n_groups)]


class SurpriseMF:
    """Latent-factor model fitted with Surprise (SGD on regularized squared error).

    algo='svd':   r = mu + b_u + b_i + q_i . p_u
    algo='svdpp': r = mu + b_u + b_i + q_i . (p_u + |N(u)|^-1/2 sum_{j in N(u)} y_j)
    where N(u) are the wines u rated in training (implicit feedback).
    Unknown users/wines drop their bias and factor terms, as Surprise does.
    """

    def __init__(self, n_factors=16, reg=0.05, lr=0.005, n_epochs=20, algo='svd', seed=SEED):
        if algo not in {'svd', 'svdpp'}:
            raise ValueError(algo)
        self.n_factors, self.reg, self.lr = n_factors, reg, lr
        self.n_epochs, self.algo, self.seed = n_epochs, algo, seed

    def fit(self, train):
        from surprise import SVD, SVDpp, Dataset, Reader
        data = Dataset.load_from_df(train[['UserID', 'WineID', 'Rating']], Reader(rating_scale=(1, 5)))
        self.trainset = trainset = data.build_full_trainset()
        cls = SVD if self.algo == 'svd' else SVDpp
        kwargs = dict(n_factors=self.n_factors, n_epochs=self.n_epochs, lr_all=self.lr,
                      reg_all=self.reg, random_state=self.seed)
        if self.algo == 'svdpp':
            kwargs['cache_ratings'] = True
        start = time.perf_counter()
        self.model = cls(**kwargs).fit(trainset)
        self.fit_seconds = time.perf_counter() - start
        self.mu = trainset.global_mean
        self.user_index = pd.Series({trainset.to_raw_uid(x): x for x in trainset.all_users()})
        self.item_index = pd.Series({trainset.to_raw_iid(x): x for x in trainset.all_items()})
        m = self.model
        self.bu, self.bi, self.pu, self.qi = m.bu, m.bi, m.pu, m.qi
        if self.algo == 'svdpp':
            implicit = np.zeros_like(self.pu)
            for u in trainset.all_users():
                js = [j for j, _ in trainset.ur[u]]
                implicit[u] = m.yj[js].sum(axis=0) / np.sqrt(len(js))
            self.user_vector = self.pu + implicit
        else:
            self.user_vector = self.pu
        return self

    def raw(self, frame):
        u = frame.UserID.map(self.user_index).to_numpy()
        i = frame.WineID.map(self.item_index).to_numpy()
        ku, ki = ~np.isnan(u), ~np.isnan(i)
        out = np.full(len(frame), self.mu, dtype=float)
        out[ku] += self.bu[u[ku].astype(int)]
        out[ki] += self.bi[i[ki].astype(int)]
        both = ku & ki
        uu, ii = u[both].astype(int), i[both].astype(int)
        out[both] += np.einsum('nk,nk->n', self.user_vector[uu], self.qi[ii])
        return out

    def predict(self, frame):
        return clip(self.raw(frame))

    def wine_factors(self):
        """Wine latent vectors q_i and biases b_i indexed by WineID."""
        ids = self.item_index.sort_values().index
        return (pd.DataFrame(self.qi, index=ids), pd.Series(self.bi, index=ids))


def evaluate(model, train, validation, name, **params):
    """Fit on train, score train and validation; return one grid row."""
    start = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    val_pred = model.predict(validation)
    predict_seconds = time.perf_counter() - start
    return dict(model=name, **params,
                train_rmse=metrics(train.Rating, model.predict(train))['rmse'],
                **{f'val_{k}': v for k, v in metrics(validation.Rating, val_pred).items()},
                fit_seconds=fit_seconds, predict_seconds=predict_seconds)


def mf_epoch_curve(train, validation, epochs, **params):
    """Train/validation RMSE after each number of epochs.

    Surprise visits ratings in a fixed order from a seeded initialization, so a
    fit with e epochs equals the first e epochs of a longer fit.
    """
    rows = []
    for e in epochs:
        model = SurpriseMF(n_epochs=e, **params).fit(train)
        rows.append(dict(epochs=e, **params,
                         train_rmse=metrics(train.Rating, model.predict(train))['rmse'],
                         val_rmse=metrics(validation.Rating, model.predict(validation))['rmse']))
    return pd.DataFrame(rows)


def neighbour_type_agreement(vectors, labels, k=10):
    """Share of each wine's k nearest wines (cosine, latent space) with the same label,
    compared with the share expected if neighbours were random."""
    x = vectors.to_numpy(dtype=float)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    lab = labels.reindex(vectors.index).to_numpy()
    observed = float((lab[nn] == lab[:, None]).mean())
    share = pd.Series(lab).value_counts(normalize=True)
    n = len(lab)
    counts = pd.Series(lab).map(pd.Series(lab).value_counts()).to_numpy()
    expected = float(((counts - 1) / (n - 1)).mean())
    return dict(k=k, observed_same_label=observed, random_expected=expected,
                largest_class_share=float(share.iloc[0]))
