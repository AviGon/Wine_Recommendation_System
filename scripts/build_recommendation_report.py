"""Build the wine recommendation report from versioned experiment data."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import json
import subprocess
import tempfile
import textwrap
import zlib

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_REVISION = '9bba3f2'
REPORT_NAMES = [
    'dataset_summary.csv', 'metadata_quality.csv', 'activity_quantiles.csv',
    'split_coverage.csv', 'train_type_summary.csv', 'train_country_summary.csv',
    'train_year_summary.csv', 'eda_effects.json', 'taste_concentration.json',
    'chronological_test_metrics.csv', 'chronological_validation_grid.csv',
    'chronological_feature_contrasts.csv', 'cf_test_metrics.csv', 'cf_bias_grid.csv',
    'cf_itemcf_grid.csv', 'cf_als_grid.csv', 'cf_surprise_grid.csv',
    'cf_als_learning_curves.csv', 'cf_time_drift.csv', 'cf_scaling.csv',
    'cf_taste_reliability.csv', 'fm_final_comparison.csv', 'fm_test_metrics.csv',
    'fm_grid.csv', 'fm_learning_curves.csv', 'fm_ablation.csv', 'fm_time_vs_age.csv',
    'fm_cold_item_test.csv', 'fm_error_slices.csv', 'fm_feature_weights.csv',
    'fm_topk.csv', 'fm_diversity.csv', 'split_manifests.json',
]
CHART_REPORT_NAMES = (
    'dataset_summary.csv', 'activity_quantiles.csv', 'train_year_summary.csv',
    'train_type_summary.csv', 'train_country_summary.csv',
)
CHART_NAMES = ('rating_activity', 'categories_origins')


def pinned(path, revision):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)


def snippet(path, symbol, revision):
    source = pinned(path, revision).decode()
    nodes = ast.parse(source).body
    node = None
    for part in symbol.split('.'):
        node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
        nodes = node.body
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return dict(path=path, symbol=symbol, line=first,
                code=textwrap.dedent('\n'.join(source.splitlines()[first-1:node.end_lineno])))


def embedded_data(notebook):
    """Read the literal report archive without executing notebook code."""
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.parse(cell.source).body:
            if (isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == '_ARCHIVE' for target in node.targets)):
                archive = ast.literal_eval(node.value)
                return json.loads(zlib.decompress(base64.b64decode(archive, validate=True)))
    raise ValueError('Published notebook has no embedded report data')


def chart_images(revision, available_paths, reports):
    analysis_path = 'notebooks/xwines_analysis.ipynb'
    if analysis_path in available_paths:
        analysis = nbf.reads(pinned(analysis_path, revision).decode(), as_version=4)
        images = {}
        for name in CHART_NAMES:
            cell = next(c for c in analysis.cells if c.cell_type == 'code' and f'{name}.png' in c.source)
            images[name] = next(o.data['image/png'] for o in cell.outputs if 'image/png' in o.get('data', {}))
        return images

    report_path = 'notebooks/wine_recommendation.ipynb'
    if report_path not in available_paths:
        raise ValueError('Data revision contains no saved analysis figures')
    data = embedded_data(nbf.reads(pinned(report_path, revision).decode(), as_version=4))
    for name in CHART_REPORT_NAMES:
        if data['reports'].get(name) != reports[name]:
            raise ValueError(f'Saved charts do not match the data revision: {name}')
    return {name: data['images'][name] for name in CHART_NAMES}


def build(revision=DEFAULT_DATA_REVISION, output=None):
    source_commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', f'{revision}^{{commit}}'],
        cwd=ROOT, text=True).strip()
    reports = {name: pinned('reports/'+name, source_commit).decode() for name in REPORT_NAMES}
    available_paths = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', source_commit], cwd=ROOT, text=True).splitlines())
    legacy_paths = {'src/collaborative_filtering.py': 'src/cf.py',
                    'src/factorization_machines.py': 'src/fm.py'}
    sources = {name: snippet(path if path in available_paths else legacy_paths.get(path, path),
                             symbol, source_commit) for name, path, symbol in [
        ('split', 'src/experiments.py', 'make_splits'),
        ('score', 'src/pipeline.py', 'score_predictions'),
        ('metrics', 'src/experiments.py', 'metrics'),
        ('bias', 'src/collaborative_filtering.py', 'BiasBaseline.fit'),
        ('neighbours', 'src/collaborative_filtering.py', 'ItemCF.raw'),
        ('als', 'src/collaborative_filtering.py', 'ALSMF._solve'),
        ('fm', 'src/factorization_machines.py', 'FactorizationMachine._raw'),
        ('fm_fit', 'src/factorization_machines.py', 'FactorizationMachine.fit'),
        ('encoder', 'src/factorization_machines.py', 'FMEncoder.fit'),
    ]}
    images = chart_images(source_commit, available_paths, reports)
    training_sources = []
    for label, current, archived in [
        ('Rating analysis', 'scripts/build_rating_analysis.py', 'scripts/build_notebook.py'),
        ('Collaborative filtering', 'scripts/build_collaborative_filtering.py', 'scripts/build_cf_notebook.py'),
        ('Factorization machines', 'scripts/build_factorization_machines.py', 'scripts/build_fm_notebook.py'),
    ]:
        path = current if current in available_paths else archived
        if path not in available_paths:
            raise ValueError(f'Missing experiment generator: {current}')
        training_sources.append(dict(label=label, path=path))
    payload = dict(reports=reports, sources=sources, images=images,
                   hashes={name: hashlib.sha256(value.encode()).hexdigest() for name, value in reports.items()},
                   source_commit=source_commit, training_sources=training_sources)
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    cells = []
    def md(value): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(value).strip()))
    def code(value, **metadata): cells.append(nbf.v4.new_code_cell(textwrap.dedent(value).strip(), metadata=metadata))

    md('''
    # X-Wines: Predicting Wine Ratings

    **Research question:** How much do historical ratings, wine attributes, and rating context contribute to predicting a later wine rating?

    Historical ratings, wine attributes, and rating context are evaluated on a shared chronological split, with separate checks for unseen wines and recommendation ranking.

    [Prediction and study design](#prediction) · [Related research](#research) · [Data and exploration](#data) · [Models and implementation](#models) · [Evaluation and interpretation](#evaluation) · [Conclusions](#conclusions) · [Reproduction](#reproduction)
    ''')
    code('#@title Load report data\nimport base64, hashlib, io, json, zlib\n'
         f'_ARCHIVE = {packed!r}\n'
         'SNAPSHOT = json.loads(zlib.decompress(base64.b64decode(_ARCHIVE)))\n'
         'for _name, _text in SNAPSHOT["reports"].items():\n'
         '    if hashlib.sha256(_text.encode()).hexdigest() != SNAPSHOT["hashes"][_name]:\n'
         '        raise ValueError(f"Changed embedded report: {_name}")\n',
         tags=['report-payload'], cellView='form', jupyter={'source_hidden': True})
    code('''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Code, Image, Markdown, display
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    WINE, GRAY = '#713e50', '#7a858c'
    def report(name, **kwargs):
        return pd.read_csv(io.StringIO(SNAPSHOT['reports'][name]), **kwargs)
    def figure(name):
        display(Image(data=base64.b64decode(SNAPSHOT['images'][name])))
    def show_code(name):
        source = SNAPSHOT['sources'][name]
        url = ('https://github.com/AviGon/Wine_Recommendation_System/blob/'
               + SNAPSHOT['source_commit'] + '/' + source['path'] + '#L' + str(source['line']))
        display(Markdown(f"[{source['path']} — {source['symbol']}]({url})"))
        display(Code(source['code'], language='python'))
    def finish(fig, caption):
        plt.show()
        display(Markdown('*'+caption+'*'))
    print('Data revision:', SNAPSHOT['source_commit'][:7])
    ''')
    md(r'''
    <a id="prediction"></a>
    ## Prediction objective and study design

    Given user $u$, wine $i$, and declared query context, predict the continuous rating $r_{ui}\in[1,5]$. User and wine identity, catalogue attributes, rating date, and rated vintage are possible inputs. **Rating date and vintage must be known for context-aware predictions.** We predict ratings conditional on an observed rating event, not purchases, sales, or satisfaction across all unobserved pairs.

    The primary question is future-rating prediction. A global chronological split uses 120,000 training, 15,000 validation, and 15,000 test events. Timestamp ties stay together. Training fits every vocabulary, imputer, aggregate, and parameter; validation selects settings; test evaluates the selected configurations. Predictions are clipped to [1,5], without half-star rounding.

    **RMSE** is primary because large rating errors receive a larger penalty. **MAE** gives a more direct typical absolute error in stars. Neither measures list quality, so ranking is a separate, secondary experiment. Global/user/item means, additive user–wine biases, and a linear model with the same features provide comparison points before attributing gains to model complexity.
    ''')
    code('''
    coverage = report('split_coverage.csv')
    primary = coverage[coverage.protocol.eq('chronological')].copy()
    display(primary[['partition', 'n', 'users', 'wines', 'start', 'end', 'warm_both', 'new_user_only']])
    ''')
    md('''
    **Sensitivity protocols.** Random splitting keeps repeated user–wine pairs together; it asks a different interpolation question. The disjoint-wine protocol puts complete wines in separate partitions and tests prediction without an item's rating history. It simulates missing history, not actual launch dates. The same saved membership hashes underpin the model notebooks. Scores across protocols do not share identical test distributions.
    ''')
    md('''
    <a id="research"></a>
    ## Related research and model choices

    **Dataset origin.** [de Azambuja, Morais and Filipe (2023), X-Wines](https://doi.org/10.3390/bdcc7010020) introduce an open wine-rating dataset assembled from web sources in 2022. Their examples include neighbourhood recommendation and comparisons of recommender algorithms. We use the smaller Slim edition to compare identity, static attributes, and context under fixed partitions; our protocols and scores are not a direct replication of their published experiments.

    **Neighbourhood recommendation.** [Sarwar et al. (2001), Item-Based Collaborative Filtering Recommendation Algorithms](https://files.grouplens.org/papers/www10_sarwar.pdf) motivate prediction from item similarities. Our implementation compares Jaccard, cosine and Pearson similarities and shrinks neighbour corrections toward an additive bias prediction. The resulting gain is small on this sample, so similarity alone does not explain the larger gains of context-aware models.

    **Latent factors.** [Koren, Bell and Volinsky (2009), Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263) describe bias terms, regularized latent factors and extensions of collaborative filtering. Those ideas motivate our bias, ALS, SVD and SVD++ comparisons. Their success in other rating domains is motivation rather than evidence of a gain on Slim; our tuned factors do not clearly outperform the additive reference.

    **Sparse feature interactions.** [Rendle (2010), Factorization Machines](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf) describes factorized interactions among general sparse features, unifying several factorization models. We borrow the second-order prediction equation and efficient interaction calculation. Here, a matched linear model has almost the same test error as the FM, so additional interactions do not show a material point-estimate gain.

    **Temporal drift.** [Koren (2010), Collaborative Filtering with Temporal Dynamics](https://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/p89-koren.pdf) models changing rating behaviour on Netflix. Our simpler year/month features are motivated by that concern, but are not timeSVD++. Their validation improvement agrees with the importance of drift; different datasets and protocols prevent a numerical comparison with Netflix results.

    **Similar review data.** [McAuley and Leskovec (2013), Hidden Factors and Hidden Topics](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys13.pdf) study rating prediction with review text, including wine/CellarTracker data. Their text-informed factors can help sparse cases. These X-Wines CSVs do not expose the corresponding review text, so we examine structured attributes instead and cannot claim to reproduce the text-based method.

    **Strong references.** [Rendle, Zhang and Koren (2019), On the Difficulty of Evaluating Baselines](https://arxiv.org/abs/1905.01395) show how careful baseline tuning can change apparent model advantages. This motivates comparing FM against a linear model with identical features and tuned regularization. Their results do not imply that simple models always win; our conclusion is limited to these experiments.
    ''')
    md('''
    <a id="data"></a>
    ## Dataset, collection and preprocessing

    The authors collected and validated ratings from 2012–2021. Their Full edition reports 100,646 wines and about 21 million ratings; those are published counts, not a Full-edition experiment here. Slim is the measured study sample. User identifiers are anonymized; production country and region are wine attributes, not consumer locations. Prices and purchase/exposure records are unavailable.

    The upstream `Test` edition is a small example dataset, distinct from our held-out test partitions. Catalogue attributes are a retrospective 2022 snapshot. A chronological split cannot establish that every attribute was available at each historical rating time.
    ''')
    code('''
    summary = report('dataset_summary.csv', index_col=0)
    display(summary.loc[['ratings', 'users', 'catalog_wines', 'countries', 'wine_types',
                         'date_min', 'date_max', 'sparsity', 'repeated_user_wine_rows']])
    ''')
    md('''
    ### Ratings, activity and sampling

    Rating mass is concentrated above the midpoint and activity is uneven across users and wines. Review frequency describes the sampled rating process; it is not a direct measure of demand. Sparse histories motivate regularization and explicit fallbacks for unseen IDs.
    ''')
    code("figure('rating_activity')")
    md('''
    ### Wine attributes and observed preferences

    Review exposure and catalogue composition differ: a country can contain many wines yet account for fewer reviews. Type-level means below show both event weighting and equal weight per wine; neither removes selection bias. Interpretations should concern associations, not intrinsic country or grape quality.
    ''')
    code('''
    figure('categories_origins')
    display(report('train_type_summary.csv'))
    effects = json.loads(SNAPSHOT['reports']['eda_effects.json'])
    display(pd.Series(effects['red_white'], name='Within-user Red minus White rating'))
    ''')
    md('''
    The Red–White comparison averages within-user differences among users with at least two ratings of each type. It does not isolate a causal type effect. The later split-half analysis asks a different question: whether residual preferences repeat after accounting for user/wine biases. A positive mean type difference can coexist with weak residual preference stability.

    ### Feature representation and data quality

    IDs and origin/type fields are categorical. Grape blends use fractional multi-hot weights of 1/k. ABV is scaled and imputed using unique training wines. Non-vintage or invalid ages use a training median and a missing indicator. Rating month uses sine/cosine; rating year and vintage age have declared scales. Entire catalogue vintage lists are excluded because they can contain later availability information.

    Every metadata join preserves rating rows or raises an error. Unknown categorical levels contribute no learned coefficient; known bias/content terms provide fallbacks. Body/Acidity are partly review-derived and appear only in a caveated sensitivity experiment. Prices are never imputed from unrelated sources.
    ''')
    code('''
    quality = report('metadata_quality.csv', index_col=0)
    display(quality[quality['missing'].gt(0)])
    show_code('encoder')
    ''')
    md(r'''
    <a id="models"></a>
    ## Models, objectives and implementation

    The progression tests whether complexity contributes beyond strong references:

    - **Additive bias:** $\hat r_{ui}=\mu+b_u+b_i$, squared loss with separate user/item L2 penalties; alternating closed-form updates.
    - **Item-based CF:** bias prediction plus a damped weighted mean of neighbours' training residuals. Similarity choice, neighbourhood size and damping are selected on validation.
    - **Matrix factorization:** $\mu+b_u+b_i+p_u^Tq_i$, with regularized squared loss. Our solver uses alternating ridge solves; Surprise SVD uses SGD; SVD++ includes an implicit-feedback component.
    - **Linear content/context:** a sparse encoded feature vector with L2 regularization; a matched linear reference has the same inputs as the FM.
    - **Factorization machine:** $w_0+\sum_jw_jx_j+\sum_{j<l}\langle v_j,v_l\rangle x_jx_l$. Full-batch L-BFGS optimizes squared error with separate linear/factor penalties. At k=0 it reduces to a linear model.

    Regression matches the rating target and loss. Classification accuracy is not the same objective; a classification approach would need an explicitly ordered loss or conversion to a comparable expected rating. Pairwise implicit-feedback objectives answer the separate ranking question.
    ''')
    code("show_code('bias')\nshow_code('neighbours')")
    md('''
    ### Sparse latent factors and feature interactions

    ALS solves one regularized least-squares block at a time. The FM calculates all second-order interactions using sparse matrix products instead of enumerating every pair. Both require regularization: adding dimensions can reduce training error without improving validation. User/item factors can collapse toward the additive model when penalized strongly.
    ''')
    code("show_code('als')\nshow_code('fm')")
    md('''
    ### Validation selection and optimization

    All displayed settings are from saved validation experiments. The main comparisons fit training only after selection, preserving the shared history cutoff. Different model families have different search budgets; a coarse ridge grid versus finer FM linear penalties is a potential explanation for small differences. A library Bayesian FM is an additional comparison, not the selected primary model.
    ''')
    code('''
    choices = []
    for name, filename in [('Bias', 'cf_bias_grid.csv'), ('Item CF', 'cf_itemcf_grid.csv'),
                           ('ALS MF', 'cf_als_grid.csv'), ('Surprise', 'cf_surprise_grid.csv'),
                           ('FM', 'fm_grid.csv')]:
        grid = report(filename)
        best = grid.sort_values('val_rmse').head(1).copy()
        row = best.iloc[0]
        settings = ', '.join(f'{key}={row[key]}' for key in grid.columns
                             if key not in ['model','val_rmse','val_mae','train_rmse','fit_seconds','predict_seconds']
                             and pd.notna(row[key]))
        choices.append(dict(family=name, settings=settings, validation_rmse=row.val_rmse,
                            training_rmse=row.get('train_rmse',np.nan)))
    display(pd.DataFrame(choices).round(6))
    ''')
    code('''
    curve = report('fm_learning_curves.csv')
    configs = curve['config'].unique()
    fig, axes = plt.subplots(1, len(configs), figsize=(6*len(configs), 3.8), squeeze=False,
                             layout='constrained')
    for axis, name in zip(axes.flat, configs):
        rows = curve[curve.config.eq(name)]
        axis.plot(rows.iteration, rows.train_rmse, label='Training', color=GRAY)
        axis.plot(rows.iteration, rows.val_rmse, label='Validation', color=WINE)
        axis.set(title=str(name), xlabel='L-BFGS iteration', ylabel='RMSE (stars)')
        axis.legend()
    finish(fig, 'Source: fm_learning_curves.csv; chronological training and validation events.')
    ''')
    md('''
    The weakly regularized run illustrates overfitting; the selected run is judged by validation, not by its lower training error. Sparse encodings avoid materializing a dense user–wine matrix. Item CF still uses a dense wine similarity matrix on Slim; extending that implementation to the Full catalogue requires a sparse neighbour index. Reported Slim scaling measurements and Full projections are distinct: Full was not trained in this study.
    ''')
    code("display(report('cf_scaling.csv').round(4))")
    md('''
    <a id="evaluation"></a>
    ## Evaluation protocol and prediction validity

    Event IDs, not row order, align predictions and labels. The shared scorer rejects missing/extra/duplicate IDs and non-finite scores, applies clipping, and reports RMSE/MAE. Published input and split hashes are immutable references; a rerun cannot silently redefine the benchmark.

    The primary uncertainty analysis resamples users in a paired bootstrap, keeping both models' predictions aligned. These intervals condition on the fitted models and fixed split. Shared wines, retraining variation and time shifts are not fully captured. For the disjoint-wine experiment, user-bootstrap intervals do not measure uncertainty over newly sampled wines. Existing inspected test results remain exploratory if they influence further model choices.
    ''')
    code("show_code('score')")
    md('''
    ### Chronological rating prediction

    The combined table below uses the original shared references, the collaborative-filtering results and the new linear/FM results. Each row identifies its source. It does not replace earlier reports. Minor numerical differences in reference refits across notebooks are retained in their original files; they do not change the four-decimal conclusions. Timing entries from separate runs are not a controlled cross-machine speed benchmark.
    ''')
    code('''
    base = report('chronological_test_metrics.csv')
    cf = report('cf_test_metrics.csv')
    fm = report('fm_test_metrics.csv')
    records = []
    def add(frame, key, label, inputs, source):
        row = frame[frame.model.eq(key)].iloc[0]
        records.append(dict(model=label, inputs=inputs, rmse=row.rmse, mae=row.mae,
                            source=source))
    for key, label in [('global','Global mean'), ('user_mean','User mean'),
                       ('item_mean','Item mean'), ('ids','Ridge: IDs'),
                       ('selected_metadata','Ridge: attributes + context')]:
        add(base, key, label, 'IDs / reference' if key!='selected_metadata' else 'IDs + attributes + context',
            'chronological_test_metrics.csv')
    for key, label in [('bias','Regularized bias'), ('item_cf','Item CF'), ('als_mf','ALS MF'),
                       ('surprise_svd','Surprise SVD'), ('surprise_svdpp','SVD++')]:
        add(cf, key, label, 'Training rating histories', 'cf_test_metrics.csv')
    for key, label in [('linear same features','Linear: matched FM features'),
                       ('fm','FM: attributes + context'), ('fm ids only','FM: IDs')]:
        add(fm, key, label, 'IDs' if key=='fm ids only' else 'IDs + attributes + context', 'fm_test_metrics.csv')
    comparison = pd.DataFrame(records).sort_values('rmse').reset_index(drop=True)
    display(comparison.round(6))
    fig, axis = plt.subplots(figsize=(9, 5), layout='constrained')
    axis.barh(comparison.model, comparison.rmse, color=WINE)
    axis.invert_yaxis()
    axis.set(xlabel='Chronological test RMSE (stars; lower is better)', ylabel='Model',
             title='Same 15,000 held-out rating events')
    finish(fig, 'Sources: chronological_test_metrics.csv, cf_test_metrics.csv, fm_test_metrics.csv.')
    ''')
    md('''
    Context-aware linear and FM predictions have nearly identical point estimates, around 0.4315 RMSE. Item CF makes a small improvement over the bias model, while the tested MF configurations do not show a clear gain over that reference. These results do not establish that personalized preferences are absent, that MF is generally ineffective, or that the linear/FM errors are statistically equivalent.
    ''')
    code('''
    cf_intervals = cf[cf.vs_bias_ci_low.notna() & ~cf.model.eq('bias')]
    fm_intervals = fm[fm.vs_ridge_ci_low.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout='constrained')
    for axis, frame, delta, low, high, title in [
        (axes[0], cf_intervals, 'vs_bias_delta_rmse', 'vs_bias_ci_low', 'vs_bias_ci_high',
         'CF minus regularized bias'),
        (axes[1], fm_intervals, 'vs_ridge_delta', 'vs_ridge_ci_low', 'vs_ridge_ci_high',
         'FM-family models minus context ridge refit')]:
        y = np.arange(len(frame))
        axis.hlines(y, frame[low], frame[high], color=WINE)
        axis.scatter(frame[delta], y, color=WINE)
        axis.axvline(0, color=GRAY, linewidth=1)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        axis.set(yticks=y, yticklabels=frame.model, xlabel='Delta test RMSE (stars; negative favors candidate)',
                 ylabel='Model', title=title)
    finish(fig, 'Paired 95% user-bootstrap intervals from each source report; references differ by panel.')
    ''')
    md('''
    ### Static attributes, calendar time and vintage age

    The original ridge context probe combined time and age. The merged FM study adds separate calendar-only and age-only ablations. Compare these **validation** results, rather than attributing a combined test improvement to metadata. Calendar time produces the larger reduction; vintage age contributes a smaller change. The combined change is not an additive causal decomposition, and the FM/linear entries may have different selected settings.
    ''')
    code('''
    display(report('fm_ablation.csv').round(6))
    context = report('fm_time_vs_age.csv')
    display(context.round(6))
    fig, axis = plt.subplots(figsize=(9, 3.8), layout='constrained')
    axis.barh(context.name, context.vs_same_without_context_fm, color=WINE)
    axis.axvline(0, color=GRAY, linewidth=1)
    axis.set(xlabel='Change in validation RMSE versus same static inputs (stars)', ylabel='Feature set',
             title='Calendar time and vintage age ablations')
    finish(fig, 'Source: fm_time_vs_age.csv; validation only. Time is calendar context, Age uses rated vintage.')
    ''')
    md('''
    ### Unseen wines and activity groups

    The chronological test contains only 16 events for wines without training history, so it is insufficient for a strong cold-item conclusion. The separate disjoint-wine test contains 14,536 events across 101 held-out wines. Its FM wins on validation but underperforms its ridge reference on test. We keep that failure, without using test to reselect a winner. Metadata is useful for scoring an unseen item, but larger interaction models do not automatically generalize to new wines.
    ''')
    code('''
    display(report('fm_cold_item_test.csv').round(6))
    slices = report('fm_error_slices.csv')
    display(slices[slices.model.eq('fm') & slices['slice'].isin(['regime','UserID_history'])].round(5))
    ''')
    md('''
    ### Parameter interpretation

    Coefficients describe conditional associations in a regularized, correlated feature representation. They are not causal effects or intrinsic wine quality. User/wine identity can absorb much of an attribute's signal; scale matters when reading numeric coefficients.
    ''')
    code('''
    weights = report('fm_feature_weights.csv')
    display(weights[weights.field.isin(['Type','Time','Age','ABV'])]
            [['feature','weight','factor_norm','train_wines']].round(5))
    display(report('cf_taste_reliability.csv').round(5))
    ''')
    md('''
    ### Ranking and recommendation diversity

    Ranking is secondary to predicting observed ratings. The saved experiment uses 4,684 training-seen users with at least one eligible held-out rating of at least four stars, a 1,007-wine catalogue, and excludes each user's training wines. Unobserved wines are unlabeled; treating them as non-relevant introduces exposure/selection bias. Catalogue availability at each historical query time is unknown.

    The ranked FM uses **static features**, because a never-rated candidate has no known rated vintage. It is not the context-aware model from the main rating table. Popularity has higher reported Recall@10 here; this does not mean it best predicts rating values or necessarily wins in an online experiment.
    ''')
    code('''
    ranking = report('fm_topk.csv')
    diversity = report('fm_diversity.csv')
    display(ranking.round(5))
    display(diversity[['model','catalogue_coverage','gini','types_per_list','red_share']].round(5))
    ''')
    md('''
    <a id="conclusions"></a>
    ## Conclusions and limitations

    - Strong additive references explain much of the predictable signal. Item CF adds a small gain; the tested latent-factor models do not clearly outperform the bias model.
    - Rating context accounts for more validation improvement than static attributes on warm wines. The calendar/age ablation identifies calendar time as the larger component in this experiment.
    - The matched linear model and FM have nearly equal test RMSE. Small gains over the original ridge can reflect regularization and search effort, not just interactions.
    - Unseen-item prediction and ranking remain harder questions. The cold-item FM fails to retain its validation advantage, and the ranking objective rewards a different behaviour from predicting rating values.
    - Conclusions are limited to the sampled catalogue, observed rating events, fixed partitions and tested searches. Retrospective metadata, user/item dependence, rating selection and missing historical availability limit causal and deployment claims.

    This evidence connects the literature to the experiments: item similarity remains a useful small correction; factorized interactions are flexible but need not outperform a well-tuned linear reference; temporal drift deserves explicit modeling. The next experiment should preserve an untouched holdout and focus on the intended recommendation setting before increasing model complexity.
    ''')
    md('''
    <a id="reproduction"></a>
    ## Data provenance and reproducibility

    Experiment inputs and source excerpts are pinned to the revision below and verified by checksum. Figures and tables are generated from those recorded results. Training configuration, dependency installation and execution commands are documented in the repository README. Numerical results and runtimes can vary with native libraries and hardware.
    ''')
    code('''
    print('Repository:', 'https://github.com/AviGon/Wine_Recommendation_System')
    print('Experiment revision:', SNAPSHOT['source_commit'])
    for source in SNAPSHOT['training_sources']:
        display(Markdown('['+source['label']+' source](https://github.com/AviGon/Wine_Recommendation_System/blob/'
                         + SNAPSHOT['source_commit'] + '/' + source['path'] + ')'))
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.11'},
        'colab': {'name':'wine_recommendation.ipynb','provenance':[]},
        'experiment_revision': source_commit,
    })
    path = Path(output) if output is not None else ROOT/'notebooks/wine_recommendation.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(notebook)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.ipynb', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            nbf.write(notebook, temporary)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f'Built {path}: {len(cells)} cells')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default=DEFAULT_DATA_REVISION, help='Git revision containing experiment data')
    parser.add_argument('--output', type=Path, help='Destination notebook')
    arguments = parser.parse_args()
    build(arguments.revision, arguments.output)
