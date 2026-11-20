"""Meaningful checks for split and feature leakage boundaries; no external data."""
import unittest
import numpy as np
import pandas as pd
from src.experiments import make_splits, FeatureProbe, parse_list


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'RatingID':np.arange(200), 'UserID':np.arange(200)%13,
            'WineID':np.arange(200)%20, 'Rating':2+np.arange(200)%7/2,
            'Date':pd.to_datetime('2015-01-01')+pd.to_timedelta(np.arange(200)//2,unit='D'),
            'Vintage':['2012']*200, 'ABV':12+np.arange(200)%5/2,
            'GrapeList':[['A','B']]*200,'Type':['Red']*200})

    def test_splits_are_disjoint_complete_and_label_independent(self):
        for protocol in ['chronological','random','cold_item']:
            first = make_splits(self.frame,protocol)
            altered = self.frame.sample(frac=1,random_state=19).copy()
            altered['Rating'] = 1.
            second = make_splits(altered,protocol)
            self.assertEqual(set().union(*(set(f.RatingID) for f in first.values())),set(self.frame.RatingID))
            for part in first:
                self.assertEqual(set(first[part].RatingID),set(second[part].RatingID))

    def test_random_repeated_pairs_stay_together(self):
        repeat = pd.concat([self.frame,self.frame.assign(RatingID=lambda f:f.RatingID+1000)])
        parts = make_splits(repeat,'random')
        tagged = pd.concat([f.assign(part=p) for p,f in parts.items()])
        self.assertTrue(tagged.groupby(['UserID','WineID']).part.nunique().eq(1).all())

    def test_unknown_ids_fallback_and_query_labels_ignored(self):
        tr = self.frame.iloc[:150]
        model = FeatureProbe(['UserID','WineID'],10).fit(tr)
        query = self.frame.iloc[150:].copy()
        p1 = model.predict(query)
        query['Rating'] = 1000
        np.testing.assert_allclose(p1,model.predict(query))
        query['UserID'] = 99999; query['WineID'] = 99999
        np.testing.assert_allclose(model.predict(query),tr.Rating.mean())

    def test_scaling_fits_only_training_and_lists_are_safe(self):
        tr = self.frame.iloc[:150]
        model = FeatureProbe(['UserID','WineID','ABV','Grapes'],10).fit(tr)
        median = model.abv_median
        query = self.frame.iloc[150:].copy(); query['ABV'] = 999
        model.predict(query)
        self.assertEqual(model.abv_median,median)
        self.assertEqual(parse_list("['Merlot', 'Syrah']"),['Merlot','Syrah'])
        with self.assertRaises((ValueError,SyntaxError)):
            parse_list("__import__('os').system('false')")


if __name__ == '__main__':
    unittest.main()
