"""Checks for the factorization machine and ranking helpers; synthetic data only."""
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import check_grad

from src import fm as fm_module
from src.fm import (FactorizationMachine, FMEncoder, FMRecommender, fm_score_matrix, topk_metrics)


def synthetic(n_users=30, n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(u, 100 + i) for u in range(n_users) for i in range(n_items) if rng.random() < 0.5]
    frame = pd.DataFrame(rows, columns=['UserID', 'WineID'])
    types = np.array(['Red', 'White', 'Rosé'])[np.arange(n_items) % 3]
    grapes = [['Merlot'], ['Chardonnay', 'Viognier'], ['Syrah', 'Grenache', 'Mourvedre']]
    meta = pd.DataFrame({'WineID': 100 + np.arange(n_items), 'Type': types,
                         'GrapeList': [grapes[i % 3] for i in range(n_items)],
                         'Country': np.where(np.arange(n_items) < 6, 'France', 'Chile'),
                         'ABV': np.where(np.arange(n_items) == 0, np.nan, 12 + np.arange(n_items) / 4)})
    frame = frame.merge(meta, on='WineID')
    frame['Rating'] = np.clip(3.7 + (frame.Type == 'Red') * .3 + rng.normal(0, .3, len(frame)), 1, 5)
    frame['RatingID'] = np.arange(len(frame))
    frame['Date'] = pd.Timestamp('2018-01-01') + pd.to_timedelta(rng.integers(0, 900, len(frame)), 'D')
    frame['Vintage'] = np.where(rng.random(len(frame)) < .2, 'N.V.', '2015')
    return frame, meta


class FMTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.meta = synthetic()

    def test_gradient_matches_finite_differences(self):
        x = sparse.random(40, 9, density=.4, random_state=1, format='csr')
        y = np.random.default_rng(1).normal(3.5, 1, 40)
        model = FactorizationMachine(n_factors=3, reg_linear={'a': 2., 'default': 4.}, reg_factors=3., max_iter=1)
        captured = {}
        original = fm_module.minimize
        def spy(fun, theta, **kw):
            captured['fun'] = fun; return original(fun, theta, **kw)
        fm_module.minimize = spy
        try:
            model.fit(x, y, np.array(['a'] * 4 + ['b'] * 5, dtype=object))
        finally:
            fm_module.minimize = original
        theta = np.random.default_rng(2).normal(0, .5, 1 + 9 + 27)
        fun = captured['fun']
        self.assertLess(check_grad(lambda t: fun(t)[0], lambda t: fun(t)[1], theta), 1e-4)

    def test_pairwise_term_equals_explicit_sum(self):
        rng = np.random.default_rng(3)
        model = FactorizationMachine(n_factors=2)
        model.p, model.n_factors = 5, 2
        x = sparse.csr_matrix(rng.random((4, 5)))
        w0, w, v = 0.5, rng.normal(size=5), rng.normal(size=(5, 2))
        dense = x.toarray()
        explicit = [w0 + d @ w + sum(v[j] @ v[l] * d[j] * d[l] for j in range(5) for l in range(j + 1, 5))
                    for d in dense]
        np.testing.assert_allclose(model._raw(x, x.multiply(x), w0, w, v), explicit)

    def test_encoder_is_fitted_on_training_only(self):
        train, query = self.frame.iloc[:80], self.frame.iloc[80:].copy()
        enc = FMEncoder(['UserID', 'WineID', 'Type', 'Grapes', 'ABV', 'Time', 'Age']).fit(train)
        query['UserID'] = -1                       # unseen users get no column
        x = enc.transform(query)
        self.assertEqual(x.shape[1], enc.n_features)
        self.assertFalse(any(c.startswith('UserID=-1') for c in enc.columns))
        blend = enc.transform(train[train.GrapeList.map(len) == 3].head(1)).toarray()[0]
        grape_cols = [i for i, c in enumerate(enc.columns) if c.startswith('Grapes=')]
        self.assertAlmostEqual(blend[grape_cols].sum(), 1.0)
        abv_missing = enc.transform(train[train.ABV.isna()].head(1)).toarray()[0][enc.index['ABV_missing']]
        self.assertEqual(abv_missing, 1.0)

    def test_linear_fm_matches_ridge_closed_form(self):
        x = sparse.random(60, 6, density=.5, random_state=4, format='csr')
        y = np.random.default_rng(4).normal(3, 1, 60)
        model = FactorizationMachine(n_factors=0, reg_linear=2.).fit(x, y, np.array(['f'] * 6, dtype=object))
        a = np.hstack([np.ones((60, 1)), x.toarray()])
        penalty = np.diag([0.] + [2.] * 6)
        theta = np.linalg.solve(a.T @ a + penalty, a.T @ y)
        np.testing.assert_allclose(np.r_[model.w0, model.w], theta, atol=1e-5)

    def test_fast_score_matrix_matches_predict(self):
        fields = ['UserID', 'WineID', 'Type', 'Grapes', 'Country', 'ABV']
        model = FMRecommender(fields, n_factors=3, reg_linear=1., reg_factors=1.).fit(self.frame)
        users = np.array([0, 1, 2, 999])            # 999 is unseen
        scores = fm_score_matrix(model, users, self.meta)
        grid = pd.DataFrame({'UserID': np.repeat(users, len(self.meta)),
                             'WineID': np.tile(self.meta.WineID, len(users))}).merge(self.meta, on='WineID', sort=False)
        np.testing.assert_allclose(model.raw(grid).reshape(len(users), -1), scores)

    def test_topk_metrics_on_hand_example(self):
        scores = np.array([[4., 3., 2., 1.]])
        exclude = np.array([[True, False, False, False]])
        relevant = np.array([[False, False, True, False]])
        result, top = topk_metrics(scores, exclude, relevant, k=2)
        self.assertEqual(top.tolist(), [[1, 2]])
        self.assertAlmostEqual(result['precision_at_k'], 0.5)
        self.assertAlmostEqual(result['recall_at_k'], 1.0)
        self.assertAlmostEqual(result['mrr'], 0.5)
        self.assertAlmostEqual(result['auc'], 0.5)   # one negative above, one below


if __name__ == '__main__':
    unittest.main()
