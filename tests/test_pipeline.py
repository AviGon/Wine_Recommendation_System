"""Regression checks for fixed partitions, ID-aligned evaluation, and feature guards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import (FeatureProbe, PROTOCOLS, export_splits, join_metadata, make_splits,
                             metrics, paired_type_effect, run_protocol, sha256, validate_splits)
from src.pipeline import (feature_contrasts, load_prepared_splits, prepare_splits,
                          score_predictions)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'RatingID': np.arange(200), 'UserID': np.arange(200) % 13,
            'WineID': np.arange(200) % 20, 'Vintage': ['2012']*200,
            'Rating': 2+np.arange(200) % 7/2,
            'Date': pd.Timestamp('2015-01-01')+pd.to_timedelta(np.arange(200)//2, unit='D'),
        })

    def test_invalid_partitions_are_rejected(self):
        splits = make_splits(self.frame, 'random')
        missing = {p: f.copy() for p, f in splits.items()}
        missing['test'] = missing['test'].iloc[1:]
        with self.assertRaisesRegex(ValueError, 'cover every'):
            validate_splits(missing, 'random', self.frame)
        repeated = {p: f.copy() for p, f in splits.items()}
        repeated['train'] = pd.concat([repeated['train'], repeated['train'].iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_splits(repeated, 'random')
        leaked = {p: f.copy() for p, f in splits.items()}
        leaked['test'].loc[leaked['test'].index[0], ['UserID', 'WineID']] = splits['train'].iloc[0][['UserID', 'WineID']].to_numpy()
        with self.assertRaisesRegex(ValueError, 'pairs cross'):
            validate_splits(leaked, 'random')

    def test_timestamp_ties_and_item_overlap_are_rejected(self):
        chronological = make_splits(self.frame, 'chronological')
        chronological['validation'].loc[:, 'Date'] = chronological['train'].Date.max()
        with self.assertRaisesRegex(ValueError, 'timestamp ties'):
            validate_splits(chronological, 'chronological')
        cold = make_splits(self.frame, 'cold_item')
        cold['test'].loc[:, 'WineID'] = cold['train'].WineID.iloc[0]
        with self.assertRaisesRegex(ValueError, 'WineID overlaps'):
            validate_splits(cold, 'cold_item')

    def test_preparation_preserves_manifest_and_checks_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'reports').mkdir()
            expected = {p: export_splits(make_splits(self.frame, p), p, root/'initial') for p in PROTOCOLS}
            manifest = root/'reports/split_manifests.json'
            manifest.write_text(json.dumps(expected))
            original = manifest.read_bytes()
            shuffled = self.frame.sample(frac=1, random_state=5)
            with patch('src.pipeline.verified_data', return_value=(pd.DataFrame(), shuffled)):
                prepare_splits(root)
                self.assertEqual(manifest.read_bytes(), original)
                _, loaded = load_prepared_splits(root, 'chronological')
                self.assertEqual(sum(map(len, loaded.values())), len(self.frame))
                csv = root/'data/processed/Slim/chronological/train.csv'
                previous_hash = sha256(csv)
                expected['random']['test']['sha256'] = 'invalid'
                manifest.write_text(json.dumps(expected))
                with self.assertRaisesRegex(ValueError, 'published manifest'):
                    prepare_splits(root)
                self.assertEqual(sha256(csv), previous_hash)
                manifest.write_bytes(original)
                csv.write_bytes(csv.read_bytes()+b'\n')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    load_prepared_splits(root)

    def test_prediction_scoring_is_order_independent(self):
        reference = self.frame[['RatingID', 'Rating']]
        predictions = reference.rename(columns={'Rating': 'prediction'}).sample(frac=1, random_state=2)
        score = score_predictions(reference, predictions)
        self.assertEqual(score, dict(rows=200, clipped=0, rmse=0., mae=0.))

    def test_metadata_join_cannot_silently_drop_rating_rows(self):
        wines = pd.DataFrame({'WineID': sorted(self.frame.WineID.unique()), 'Type': 'Red'})
        self.assertEqual(len(join_metadata(self.frame, wines)), len(self.frame))
        with self.assertRaisesRegex(ValueError, 'absent from the catalogue'):
            join_metadata(self.frame, wines.iloc[1:])

    def test_scoring_rejects_missing_extra_duplicate_and_nonfinite(self):
        reference = self.frame[['RatingID', 'Rating']]
        predictions = reference.rename(columns={'Rating': 'prediction'})
        extra = pd.concat([predictions, pd.DataFrame({'RatingID': [999], 'prediction': [4.]})])
        duplicate = pd.concat([predictions, predictions.iloc[:1]])
        nonfinite = predictions.copy(); nonfinite.loc[0, 'prediction'] = np.nan
        for invalid in [predictions.iloc[1:], extra, duplicate, nonfinite]:
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                score_predictions(reference, invalid)
        with self.assertRaises(ValueError):
            metrics([3., 4.], [[3.], [4.]])

    def test_scoring_applies_shared_clipping(self):
        reference = pd.DataFrame({'RatingID': [1, 2], 'Rating': [1., 5.]})
        predictions = pd.DataFrame({'RatingID': [2, 1], 'prediction': [7., -2.]})
        self.assertEqual(score_predictions(reference, predictions)['rmse'], 0.)
        self.assertEqual(score_predictions(reference, predictions)['clipped'], 2)

    def test_target_and_undeclared_features_are_rejected(self):
        for fields in [['Rating'], ['WineID', 'Date'], []]:
            with self.assertRaises(ValueError):
                FeatureProbe(fields, 10)

    def test_context_only_grid_is_not_a_metadata_comparison(self):
        splits = make_splits(self.frame, 'chronological')
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'metadata representation'):
                run_protocol(pd.DataFrame(), splits, 'chronological', temporary,
                             specs={'ids': ['UserID', 'WineID'],
                                    'context': ['UserID', 'WineID', 'Time', 'Age']})

    def test_type_comparison_handles_absent_category(self):
        one_type = self.frame.assign(Type='Red')
        self.assertEqual(paired_type_effect(one_type)['eligible_users'], 0)

    def test_feature_contrasts_match_regularization_without_test_labels(self):
        grid = pd.DataFrame([
            {'representation': name, 'alpha': alpha, 'rmse': score}
            for alpha in [1., 10.] for name, score in [('ids', .5), ('ids_metadata', .49), ('ids_metadata_time', .45)]])
        context = pd.DataFrame({'representation': ['ids_time_only']*2, 'alpha': [1., 10.], 'rmse': [.46, .46]})
        result = feature_contrasts(grid, context)
        self.assertEqual(len(result), 8)
        np.testing.assert_allclose(result.query("reference == 'ids_time_only'").delta_validation_rmse, -.01)
        with self.assertRaises(ValueError):
            feature_contrasts(grid, context.iloc[:1])


if __name__ == '__main__':
    unittest.main()
