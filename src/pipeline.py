"""Prepare pinned data partitions and evaluate predictions by event ID."""
from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
import pandas as pd

from .experiments import (PARTITIONS, PROTOCOLS, export_splits, load_data,
                          make_splits, metrics, sha256, validate_splits)


def verified_data(root):
    root = Path(root)
    sources = json.loads((root/'data/sources.json').read_text())['files']
    for name in ['XWines_Slim_1K_wines.csv', 'XWines_Slim_150K_ratings.csv']:
        path = root/'data/raw'/name
        if sha256(path) != sources[name]['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    return load_data(root/'data/raw', 'Slim')


def prepare_splits(root):
    """Generate the published partitions without fitting models or changing references.

    Verify every exported byte against the tracked manifest before replacing local
    partition files. A changed seed, input, or serialization cannot redefine it.
    """
    root = Path(root)
    _, ratings = verified_data(root)
    expected = json.loads((root/'reports/split_manifests.json').read_text())
    partitions = {p: make_splits(ratings, p) for p in PROTOCOLS}
    destination = root/'data/processed/Slim'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        for protocol in PROTOCOLS:
            actual = export_splits(partitions[protocol], protocol, temporary)
            if actual != expected[protocol]:
                raise ValueError(f'{protocol}: generated splits differ from the published manifest')
        for protocol in PROTOCOLS:
            folder = destination/protocol
            folder.mkdir(parents=True, exist_ok=True)
            for path in (Path(temporary)/protocol).iterdir():
                shutil.move(str(path), str(folder/path.name))
    return partitions


def load_prepared_splits(root, protocol='chronological'):
    """Return catalogue and checked partitions; never regenerate missing files silently."""
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown protocol: {protocol}')
    root = Path(root)
    wines, ratings = verified_data(root)
    expected = json.loads((root/'reports/split_manifests.json').read_text())[protocol]
    partitions = {}
    for part in PARTITIONS:
        path = root/'data/processed/Slim'/protocol/f'{part}.csv'
        if sha256(path) != expected[part]['sha256']:
            raise ValueError(f'{protocol}/{part}: split checksum mismatch')
        frame = pd.read_csv(path, dtype={'Vintage': 'string'}, parse_dates=['Date'])
        if len(frame) != expected[part]['rows']:
            raise ValueError(f'{protocol}/{part}: split row count mismatch')
        partitions[part] = frame
    validate_splits(partitions, protocol, ratings)
    return wines, partitions


def score_predictions(reference, predictions):
    """Align by RatingID, require full coverage, then apply the shared clipping rule."""
    for frame, required in [(reference, ['RatingID', 'Rating']),
                            (predictions, ['RatingID', 'prediction'])]:
        if not set(required).issubset(frame.columns):
            raise ValueError(f'Required columns: {required}')
        if frame.RatingID.isna().any() or not frame.RatingID.is_unique:
            raise ValueError('RatingID must be non-null and unique')
    if set(reference.RatingID) != set(predictions.RatingID):
        raise ValueError('Prediction IDs must exactly match the evaluation partition')
    aligned = reference[['RatingID', 'Rating']].merge(
        predictions[['RatingID', 'prediction']], on='RatingID', validate='one_to_one')
    values = aligned.prediction.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Predictions must be finite; missing scores cannot be dropped')
    return dict(rows=len(aligned), clipped=int(((values < 1) | (values > 5)).sum()),
                **metrics(aligned.Rating, np.clip(values, 1, 5)))


def feature_contrasts(validation_grid, time_grid):
    """Compare existing validation errors at matched alpha; no new fits or test access.

    The context probe contains calendar features AND rated-vintage age, so its
    contribution must not be described as a calendar-time-only effect.
    """
    grid = pd.concat([validation_grid, time_grid], ignore_index=True)
    values = grid.pivot(index='alpha', columns='representation', values='rmse')
    rows = []
    for reference, candidate in [('ids', 'ids_metadata'), ('ids', 'ids_time_only'),
                                  ('ids_time_only', 'ids_metadata_time'),
                                  ('ids_metadata', 'ids_metadata_time')]:
        for alpha, errors in values.iterrows():
            if pd.isna(errors[reference]) or pd.isna(errors[candidate]):
                raise ValueError('Every contrast requires both representations at the same alpha')
            rows.append(dict(reference=reference, candidate=candidate, alpha=alpha,
                             reference_rmse=errors[reference], candidate_rmse=errors[candidate],
                             delta_validation_rmse=errors[candidate]-errors[reference]))
    return pd.DataFrame(rows)
