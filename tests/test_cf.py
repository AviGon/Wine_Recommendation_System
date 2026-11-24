"""Checks for the collaborative-filtering models; synthetic data only."""
import unittest

import numpy as np
import pandas as pd

from src.cf import ALSMF, BiasBaseline, ItemCF, SurpriseMF, item_similarity


def synthetic(n_users=40, n_items=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(u, i) for u in range(n_users) for i in range(n_items) if rng.random() < 0.5]
    frame = pd.DataFrame(rows, columns=['UserID', 'WineID'])
    bu, bi = rng.normal(0, .4, n_users), rng.normal(0, .4, n_items)
    frame['Rating'] = np.clip(np.round(2 * (3.6 + bu[frame.UserID] + bi[frame.WineID]
                                            + rng.normal(0, .3, len(frame)))) / 2, 1, 5)
    frame['RatingID'] = np.arange(len(frame))
    return frame


class CFTests(unittest.TestCase):
    def setUp(self):
        self.train = synthetic()

    def test_pearson_matches_numpy_on_co_rating_users(self):
        ids, sim = item_similarity(self.train, 'pearson')
        wide = self.train.pivot(index='UserID', columns='WineID', values='Rating')
        a, b = ids[0], ids[1]
        both = wide[[a, b]].dropna()
        self.assertAlmostEqual(sim[0, 1], np.corrcoef(both[a], both[b])[0, 1], places=10)
        self.assertEqual(sim[0, 0], 0.)

    def test_jaccard_and_cosine_are_bounded_and_symmetric(self):
        for kind in ['jaccard', 'cosine']:
            _, sim = item_similarity(self.train, kind)
            self.assertTrue(np.allclose(sim, sim.T))
            self.assertTrue((sim >= 0).all() and (sim <= 1 + 1e-12).all())

    def test_unknown_users_and_wines_fall_back_and_labels_are_ignored(self):
        query = pd.DataFrame({'UserID': [999, 0, 999], 'WineID': [0, 999, 999], 'RatingID': [1, 2, 3]})
        for model in [BiasBaseline(), ItemCF('cosine'), ALSMF(3, 5.), SurpriseMF(3, n_epochs=5)]:
            model.fit(self.train)
            first = model.predict(query)
            second = model.predict(query.assign(Rating=[5., 1., 3.]))
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_array_equal(first, second)
            self.assertAlmostEqual(first[2], np.clip(model.raw(query)[2], 1, 5))

    def test_als_without_factors_equals_bias_baseline(self):
        als = ALSMF(0, 1., reg_user_bias=5., reg_item_bias=2., n_iter=60).fit(self.train)
        bias = BiasBaseline(5., 2., n_iter=60).fit(self.train)
        np.testing.assert_allclose(als.predict(self.train), bias.predict(self.train), atol=1e-6)

    def test_als_training_error_does_not_increase(self):
        model = ALSMF(4, 1., n_iter=8, track=self.train).fit(self.train)
        errors = [row['train_rmse'] for row in model.curve]
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(errors, errors[1:])))

    def test_vectorised_surprise_matches_library_predictions(self):
        for algo in ['svd', 'svdpp']:
            model = SurpriseMF(4, n_epochs=10, algo=algo).fit(self.train)
            query = self.train.sample(30, random_state=1)
            expected = [model.model.predict(u, i).est for u, i in zip(query.UserID, query.WineID)]
            np.testing.assert_allclose(model.predict(query), expected, atol=1e-10)

    def test_item_cf_refits_its_baseline_on_new_training_data(self):
        model = ItemCF('cosine').fit(self.train)
        shifted = self.train.assign(Rating=np.clip(self.train.Rating + 1, 1, 5))
        model.fit(shifted)
        self.assertAlmostEqual(model.baseline.mu, shifted.Rating.mean())


if __name__ == '__main__':
    unittest.main()
