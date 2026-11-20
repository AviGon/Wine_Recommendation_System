"""X-Wines data validation, fixed splits, and feature experiments.

Sparse ridge probes compare identity, content, and temporal features.
All fitted encoders and parameters see training rows only. Validation selects alpha.
"""
from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

SEED = 258
PARTITIONS = ('train', 'validation', 'test')
PROTOCOLS = ('chronological', 'random', 'cold_item')
SAFE_META = ['Type', 'Elaborate', 'Country', 'RegionID', 'WineryID', 'Grapes', 'ABV']
SPECS = {
    'ids': ['UserID', 'WineID'],
    'content': SAFE_META,
    'ids_type': ['UserID', 'WineID', 'Type'],
    'ids_country': ['UserID', 'WineID', 'Country'],
    'ids_grapes': ['UserID', 'WineID', 'Grapes'],
    'ids_region': ['UserID', 'WineID', 'RegionID'],
    'ids_metadata': ['UserID', 'WineID'] + SAFE_META,
    'ids_metadata_time': ['UserID', 'WineID'] + SAFE_META + ['Time', 'Age'],
    'user_taste': ['UserID', 'WineID'] + SAFE_META + ['UserType'],
}
ALLOWED_FIELDS = frozenset(['UserID', 'WineID', 'Time', 'Age', 'UserType'] + SAFE_META)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_list(value):
    """Parse Python-list CSV fields without evaluating code."""
    if pd.isna(value) or not str(value).strip():
        return []
    result = ast.literal_eval(str(value))
    if not isinstance(result, list):
        raise ValueError(f'Expected a list, got {type(result).__name__}')
    return [str(x).strip() for x in result if str(x).strip()]


def load_data(root, version='Slim'):
    root = Path(root)
    if version not in {'Slim', 'Test'}:
        raise ValueError('Supported editions: Slim, Test')
    suffix = ('1K', '150K') if version == 'Slim' else ('100', '1K')
    wines = pd.read_csv(root / f'XWines_{version}_{suffix[0]}_wines.csv')
    ratings = pd.read_csv(root / f'XWines_{version}_{suffix[1]}_ratings.csv',
                          dtype={'Vintage': 'string'}, low_memory=False)
    ratings['Date'] = pd.to_datetime(ratings.Date, errors='raise')
    if wines.WineID.isna().any() or not wines.WineID.is_unique:
        raise ValueError('WineID must be non-null and unique in the catalogue')
    if ratings.RatingID.isna().any() or not ratings.RatingID.is_unique:
        raise ValueError('RatingID must be non-null and unique')
    if ratings[['UserID', 'WineID', 'Rating', 'Date']].isna().any().any():
        raise ValueError('Missing rating identifiers, target, or timestamp')
    if not ratings.WineID.isin(wines.WineID).all():
        raise ValueError('Orphan wine foreign key')
    if not ratings.Rating.isin(np.arange(1, 5.01, .5)).all():
        raise ValueError('Ratings must be half-star values from 1 to 5')
    wines['GrapeList'] = wines.Grapes.map(parse_list)
    wines['HarmonizeList'] = wines.Harmonize.map(parse_list)
    return wines, ratings.sort_values('RatingID').reset_index(drop=True)


def audit(wines, ratings):
    vintage = pd.to_numeric(ratings.Vintage, errors='coerce')
    n_pairs = ratings[['UserID', 'WineID']].drop_duplicates().shape[0]
    return pd.Series({
        'ratings': len(ratings), 'users': ratings.UserID.nunique(),
        'catalog_wines': len(wines), 'rated_wines': ratings.WineID.nunique(),
        'countries': wines.Country.nunique(), 'regions': wines.RegionID.nunique(),
        'wine_types': wines.Type.nunique(), 'date_min': str(ratings.Date.min()),
        'date_max': str(ratings.Date.max()),
        'sparsity': 1 - n_pairs / (ratings.UserID.nunique() * len(wines)),
        'repeated_user_wine_rows': len(ratings) - n_pairs,
        'repeated_user_wine_vintage_rows': int(ratings.duplicated(['UserID','WineID','Vintage']).sum()),
        'exact_duplicate_rows': int(ratings.duplicated().sum()),
        'future_vintage_rows': int((vintage > ratings.Date.dt.year).sum()),
        'non_numeric_vintage_rows': int(vintage.isna().sum()),
        'empty_grape_lists': int(wines.GrapeList.map(len).eq(0).sum()),
        'price_available': any('price' in c.lower() for c in wines.columns),
        'user_location_available': False,
        'singleton_user_fraction': float(ratings.UserID.value_counts().eq(1).mean()),
        'mean_rating': float(ratings.Rating.mean()),
        'fraction_rating_ge_4': float(ratings.Rating.ge(4).mean()),
    }, name='value')


def make_splits(ratings, protocol, seed=SEED):
    """Partition without labels; chronology never splits timestamp ties."""
    rng = np.random.default_rng(seed)
    if protocol == 'chronological':
        ordered = ratings.sort_values(['Date','RatingID'])
        t1 = ordered.Date.iloc[int(.8 * len(ordered))]
        t2 = ordered.Date.iloc[int(.9 * len(ordered))]
        part = np.where(ratings.Date < t1, 'train',
                        np.where(ratings.Date < t2, 'validation', 'test'))
    elif protocol == 'random':
        # Keep repeated user-wine events together if running on Full later.
        keys = ratings[['UserID','WineID']].drop_duplicates().sort_values(['UserID','WineID'])
        keys = keys.iloc[rng.permutation(len(keys))].copy()
        keys['partition'] = (['train'] * int(.8*len(keys)) +
                             ['validation'] * (int(.9*len(keys))-int(.8*len(keys))) +
                             ['test'] * (len(keys)-int(.9*len(keys))))
        assigned = ratings.merge(keys, on=['UserID','WineID'], validate='many_to_one')
        part = ratings.RatingID.map(assigned.set_index('RatingID').partition).to_numpy()
    elif protocol == 'cold_item':
        ids = np.sort(ratings.WineID.unique())
        ids = ids[rng.permutation(len(ids))]
        a, b = int(.8*len(ids)), int(.9*len(ids))
        part = np.where(ratings.WineID.isin(ids[:a]), 'train',
                        np.where(ratings.WineID.isin(ids[a:b]), 'validation', 'test'))
    else:
        raise ValueError(protocol)
    splits = {p: ratings.loc[part == p].copy() for p in ['train','validation','test']}
    validate_splits(splits, protocol, ratings)
    return splits


def validate_splits(splits, protocol, source=None):
    """Reject missing/duplicate events, temporal overlap, and grouped-split leakage."""
    if protocol not in PROTOCOLS or set(splits) != set(PARTITIONS):
        raise ValueError('Expected a known protocol and train/validation/test partitions')
    for part in PARTITIONS:
        frame = splits[part]
        if frame.empty or frame[['RatingID', 'UserID', 'WineID', 'Date']].isna().any().any():
            raise ValueError(f'{part}: empty partition or missing identifiers/timestamps')
        if not frame.RatingID.is_unique:
            raise ValueError(f'{part}: duplicate RatingID')
    ids = [set(splits[p].RatingID) for p in PARTITIONS]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError('RatingID overlaps across partitions')
    if source is not None and (not source.RatingID.is_unique or set.union(*ids) != set(source.RatingID)):
        raise ValueError('Partitions must cover every source RatingID exactly once')
    if protocol == 'chronological':
        if not (splits['train'].Date.max() < splits['validation'].Date.min() and
                splits['validation'].Date.max() < splits['test'].Date.min()):
            raise ValueError('Chronological partitions overlap or split timestamp ties')
    if protocol == 'random':
        pairs = pd.concat([splits[p][['UserID', 'WineID']].assign(partition=p) for p in PARTITIONS])
        if pairs.groupby(['UserID', 'WineID']).partition.nunique().gt(1).any():
            raise ValueError('Repeated user-wine pairs cross partitions')
    if protocol == 'cold_item':
        items = [set(splits[p].WineID) for p in PARTITIONS]
        if items[0] & items[1] or items[0] & items[2] or items[1] & items[2]:
            raise ValueError('WineID overlaps across disjoint-wine partitions')


def coverage_table(splits):
    tr = splits['train']
    rows = []
    for name, frame in splits.items():
        u, i = frame.UserID.isin(tr.UserID), frame.WineID.isin(tr.WineID)
        rows.append(dict(partition=name, n=len(frame), users=frame.UserID.nunique(),
                         wines=frame.WineID.nunique(), start=str(frame.Date.min()),
                         end=str(frame.Date.max()), warm_both=float((u&i).mean()),
                         new_user_only=float((~u&i).mean()), new_item_only=float((u&~i).mean()),
                         new_both=float((~u&~i).mean()), mean_rating=frame.Rating.mean()))
    return pd.DataFrame(rows)


def group_summary(frame, group):
    """Both interaction-weighted and equal-wine means, plus sample support."""
    base = frame.groupby(group).agg(ratings=('Rating','size'), users=('UserID','nunique'),
                                   wines=('WineID','nunique'), mean_rating=('Rating','mean'))
    wine_means = frame.groupby([group,'WineID']).Rating.mean().groupby(level=0).mean()
    base['equal_wine_mean'] = wine_means
    return base.sort_values('ratings', ascending=False)


def join_metadata(ratings, wines):
    """Preserve rating rows and reject missing catalogue entries instead of dropping them."""
    joined = ratings.merge(wines, on='WineID', how='left', validate='many_to_one', indicator=True)
    if joined['_merge'].ne('both').any():
        raise ValueError('Ratings reference wines absent from the catalogue')
    return joined.drop(columns='_merge')


def paired_type_effect(train_joined, seed=SEED, n_boot=2000):
    """Within-user Red minus White mean; each retained user has >=2 of each."""
    t = train_joined[train_joined.Type.isin(['Red','White'])]
    g = t.groupby(['UserID','Type']).Rating.agg(['mean','size'])
    means = g['mean'].unstack().reindex(columns=['Red', 'White'])
    counts = g['size'].unstack(fill_value=0)
    eligible = counts.reindex(columns=['Red','White'], fill_value=0).ge(2).all(axis=1)
    deltas = (means.Red - means.White).loc[eligible].dropna().to_numpy()
    if len(deltas) < 2:
        return {'eligible_users': len(deltas), 'mean_delta': None, 'ci_low': None, 'ci_high': None}
    rng = np.random.default_rng(seed)
    bs = [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(n_boot)]
    return dict(eligible_users=len(deltas), mean_delta=float(deltas.mean()),
                ci_low=float(np.quantile(bs,.025)), ci_high=float(np.quantile(bs,.975)))


class FeatureProbe:
    """Sparse ridge regression; unique training wines determine ABV scaling."""
    def __init__(self, fields, alpha):
        if not fields or set(fields) - ALLOWED_FIELDS:
            raise ValueError('Use declared prediction features; labels and undeclared fields are forbidden')
        self.fields, self.alpha = list(fields), alpha

    def fit(self, frame):
        abv = pd.to_numeric(frame.drop_duplicates('WineID').ABV, errors='coerce')
        self.abv_median = float(abv.median()) if abv.notna().any() else 0.
        self.abv_sd = float(abv.std()) if abv.std() > 0 else 1.
        age = frame.Date.dt.year - pd.to_numeric(frame.Vintage, errors='coerce')
        age = age.where(age.between(0,100))
        self.age_median = float(age.median()) if age.notna().any() else 0.
        self.mu = float(frame.Rating.mean())
        self.vectorizer = DictVectorizer(sparse=True)
        x = self.vectorizer.fit_transform(self.records(frame))
        self.model = Ridge(alpha=self.alpha, fit_intercept=False, solver='lsqr', tol=1e-5)
        self.model.fit(x, frame.Rating.to_numpy() - self.mu)
        self.n_features = x.shape[1]
        return self

    def records(self, frame):
        out = []
        for row in frame.itertuples():
            d = {}
            for field in self.fields:
                if field == 'Grapes':
                    grapes = row.GrapeList
                    for grape in grapes:
                        d['grape=' + grape] = 1 / max(len(grapes),1)
                elif field == 'ABV':
                    value = float(row.ABV) if pd.notna(row.ABV) else self.abv_median
                    d['ABV_z'] = (value-self.abv_median)/self.abv_sd
                    d['ABV_missing'] = float(pd.isna(row.ABV))
                elif field == 'Time':
                    d['month_sin'] = np.sin(2*np.pi*row.Date.month/12)
                    d['month_cos'] = np.cos(2*np.pi*row.Date.month/12)
                    d['year_centered'] = (row.Date.year-2017)/5
                elif field == 'Age':
                    try:
                        age = row.Date.year - float(row.Vintage)
                    except (ValueError, TypeError):
                        age = np.nan
                    valid = np.isfinite(age) and 0 <= age <= 100
                    d['age_decades'] = (age if valid else self.age_median)/10
                    d['age_missing'] = float(not valid)
                elif field == 'UserType':
                    d[f'user_type={row.UserID}|{row.Type}'] = 1.
                else:
                    value = getattr(row, field)
                    d[field+'='+str(value if pd.notna(value) else '__MISSING__')] = 1.
            out.append(d)
        return out

    def predict(self, frame):
        return np.clip(self.mu + self.model.predict(self.vectorizer.transform(self.records(frame))),1,5)


def baseline_predictions(train, query, shrinkage=10):
    mu = train.Rating.mean()
    def estimate(key):
        stats = train.groupby(key).Rating.agg(['sum','size'])
        values = (stats['sum'] + shrinkage*mu)/(stats['size']+shrinkage)
        return query[key].map(values).fillna(mu).to_numpy()
    return {'global': np.full(len(query),mu), 'user_mean': estimate('UserID'), 'item_mean': estimate('WineID')}


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError('Targets and predictions must be non-empty, equally sized 1-D arrays')
    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
        raise ValueError('Targets and predictions must be finite')
    error = y - prediction
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(abs(error)))}


def cluster_bootstrap(frame, reference, candidate, n_boot=2000, seed=SEED):
    """Paired user-cluster bootstrap for micro-RMSE difference; negative favors candidate."""
    errors = pd.DataFrame({'u':frame.UserID.to_numpy(),
                           'ref':(frame.Rating.to_numpy()-reference)**2,
                           'new':(frame.Rating.to_numpy()-candidate)**2})
    by = errors.groupby('u').agg(ref=('ref','sum'),new=('new','sum'),n=('ref','size')).to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        totals = by[rng.integers(0,len(by),len(by))].sum(axis=0)
        samples[b] = np.sqrt(totals[1]/totals[2])-np.sqrt(totals[0]/totals[2])
    return {'delta_rmse': metrics(frame.Rating,candidate)['rmse']-metrics(frame.Rating,reference)['rmse'],
            'ci_low':float(np.quantile(samples,.025)), 'ci_high':float(np.quantile(samples,.975)),
            'clusters':len(by), 'bootstrap_replicates':n_boot}


def error_slices(train, test, prediction, model):
    work = test[['UserID','WineID','Rating']].copy()
    work['prediction'] = prediction
    u = work.UserID.isin(train.UserID); i = work.WineID.isin(train.WineID)
    work['regime'] = np.select([u&i,~u&i,u&~i],['warm_both','new_user','new_item'],default='new_both')
    for key in ['UserID','WineID']:
        counts = work[key].map(train[key].value_counts()).fillna(0)
        work[key+'_history'] = pd.cut(counts,[-1,0,5,20,100,np.inf],labels=['0','1-5','6-20','21-100','101+'])
    rows = []
    for field in ['regime','UserID_history','WineID_history']:
        for group, g in work.groupby(field,observed=True):
            rows.append(dict(model=model, slice=field, group=str(group), n=len(g),
                             users=g.UserID.nunique(), **metrics(g.Rating,g.prediction)))
    return pd.DataFrame(rows)


def run_protocol(wines, splits, protocol, out_dir, specs=None):
    """Tune on validation; test selected representation and prespecified references once.

    Do not refit on validation here: errors stay aligned to the exported train-only
    histories. Model comparisons must use the same fitting policy.
    """
    validate_splits(splits, protocol)
    specs = SPECS if specs is None else specs
    candidates = [name for name, fields in specs.items() if set(fields) & set(SAFE_META)]
    if set(specs.get('ids', [])) != {'UserID', 'WineID'} or not candidates:
        raise ValueError('Provide the ID reference and at least one metadata representation')
    out_dir = Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    joined = {p: join_metadata(f, wines) for p, f in splits.items()}
    tr, va, te = (joined[p] for p in ['train','validation','test'])
    rows, fitted, chosen = [], {}, {}
    for name, fields in specs.items():
        options = []
        for alpha in [1.,10.,100.]:
            start = time.perf_counter()
            probe = FeatureProbe(fields,alpha).fit(tr)
            pred = probe.predict(va)
            row = dict(protocol=protocol, representation=name,alpha=alpha,
                       features=probe.n_features,seconds=time.perf_counter()-start,
                       train_rmse=metrics(tr.Rating,probe.predict(tr))['rmse'],**metrics(va.Rating,pred))
            rows.append(row); options.append((row['rmse'],alpha,probe))
        _,alpha,probe = min(options,key=lambda t:(t[0],t[1]))
        fitted[name] = probe; chosen[name] = alpha
        print(f'{protocol}: {name} validation RMSE {min(x[0] for x in options):.4f}',flush=True)
    grid = pd.DataFrame(rows)
    # Content-bearing candidates only; identities are the reference.
    best = min(candidates,key=lambda name: min(grid.loc[grid.representation.eq(name),'rmse']))
    predictions = baseline_predictions(tr,te)
    predictions['ids'] = fitted['ids'].predict(te)
    predictions['selected_metadata'] = fitted[best].predict(te)
    scores = pd.DataFrame([dict(protocol=protocol,model=k,
                              representation=best if k=='selected_metadata' else k,
                              **metrics(te.Rating,v)) for k,v in predictions.items()])
    bootstrap = cluster_bootstrap(te,predictions['ids'],predictions['selected_metadata'])
    bootstrap.update(protocol=protocol,selected=best,alpha=chosen[best])
    slices = pd.concat([error_slices(tr,te,v,k) for k,v in predictions.items()],ignore_index=True)
    pred_frame = te[['RatingID','UserID','WineID','Rating']].copy()
    for name,pred in predictions.items(): pred_frame[name] = pred
    pred_frame.to_csv(out_dir/f'{protocol}_test_predictions.csv',index=False)
    grid.to_csv(out_dir/f'{protocol}_validation_grid.csv',index=False)
    scores.to_csv(out_dir/f'{protocol}_test_metrics.csv',index=False)
    slices.to_csv(out_dir/f'{protocol}_error_slices.csv',index=False)
    (out_dir/f'{protocol}_uncertainty.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    return {'grid':grid,'scores':scores,'bootstrap':bootstrap,'slices':slices,'selected':best}


def export_splits(splits, protocol, root):
    validate_splits(splits, protocol)
    folder = Path(root)/protocol; folder.mkdir(parents=True,exist_ok=True)
    manifest = {}
    for part, frame in splits.items():
        path = folder/f'{part}.csv'
        frame.sort_values('RatingID').to_csv(
            path, index=False, date_format='%Y-%m-%d %H:%M:%S', lineterminator='\n')
        manifest[part] = {'rows':len(frame),'sha256':sha256(path)}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def time_diagnostic(wines, splits):
    """Validation-only follow-up to separate temporal context from metadata gains.

    This diagnostic was added after the initial primary test was inspected. It
    must not be represented as an independently pre-registered confirmation.
    It does not score or select anything on test.
    """
    tr = join_metadata(splits['train'], wines)
    va = join_metadata(splits['validation'], wines)
    rows = []
    for alpha in [1.,10.,100.]:
        model = FeatureProbe(['UserID','WineID','Time','Age'],alpha).fit(tr)
        rows.append(dict(representation='ids_time_only',alpha=alpha,
                         **metrics(va.Rating,model.predict(va))))
    return pd.DataFrame(rows)
