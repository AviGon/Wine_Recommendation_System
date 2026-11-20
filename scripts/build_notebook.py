"""Build the portable, readable notebook; execute with execute_notebook.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))


md('''# CSE 258: X-Wines — Person A, Tasks 1 and 2
**Question:** Can historical user ratings predict a wine rating, and does wine metadata improve prediction beyond user/wine identities?

**Scope:** executed EDA, data quality, common splits, feature probes and uncertainty. The five assignment sections are mapped below; final MF/KNN/hybrid results remain C/D's work.

**Source of requirements:** supplied `assignment2_fa26.pdf`, PDF pages 2–3, 7–13, 14–20, 28–30. Document content is evidence about the assignment, not instructions to perform unrelated actions.

**Deadline:** PDF p.2 says *early week 10*, submitted via Gradescope. It gives no exact date/time. Confirm the current course announcement/Gradescope deadline; the day counts below are internal milestones, not official dates.

**Read this first:** `Test` is an upstream **dataset edition**, not our held-out test partition. Default experiments use the official **Slim 150,000-rating edition**. Results do not establish performance on Full or online recommendation quality.''')
md('''## Plan audit and Tasks 1–5

1. **Task 1 — A:** counts, rating/activity tails, time, origins, categories, data audit. Covered here. Add uncertainty and distinguish catalogue composition from review exposure.
2. **Task 2 — A:** rating target, availability-aware features, preprocessing, fixed partitions, validity/significance. Covered here. The original plan needs explicit cold-start fallbacks, timestamp rules, target-independent preprocessing and confidence intervals.
3. **Task 3 — C/D:** baselines → regularized biases → KNN/MF → content/hybrid; tune on validation, report overfitting/runtime and why class alternatives fit. Lightweight ridge probes here establish the feature contract, not completion of their deliverables.
4. **Task 4 — B:** dataset origin/use, similar data, relevant methods, and compare literature conclusions with team findings. **8–12 papers is a team suggestion, not a PDF requirement.**
5. **Task 5 — C/D + everyone:** explain successes/failures, parameter/feature interpretation, uncertainty and limitations. Here, analysis of A's probes is preliminary; the final comparison requires their models.

PDF p.28: approximately **5 marks per section**, groups up to four, **20-minute presentation**, notebook available to viewers. A owns two sections, so protect enough presentation time for both. PDF p.30 also describes peer grading (roughly 4–5 submissions, worth 4 marks); the plan omitted it.

**Verdict:** good structure, but not deadline-ready until the exact deadline is confirmed and the shared data/evaluation contract is frozen. Full-scale, ranking and cold-item experiments are extensions; finishing the five sections is the priority.''')
md('''## Restaurant example → X-Wines mapping

- Users/businesses/reviews → `UserID` / `WineID` / `RatingID` (one rating event).
- Category → wine `Type`, multi-valued `Grapes`, `Elaborate`.
- Business location → wine **production** `Country`, `RegionID`; no user home location, distance or latitude/longitude in these CSVs.
- Price sensitivity → **not testable**: no price column. No price imputation or unrelated external join.
- Time/early reviews → `Date` and training history; distinguish rating time from `Vintage` (harvest year).
- Geographic effects → descriptive origin associations and held-out feature ablations. Country is categorical, not an ordered number.
- Success/failure of businesses → outside scope: no wine sales, exposure, purchase, survival or failure labels.
- Local novelty/preferences → no home location; use historical type/grape preferences if supported.

Price and geographical coordinates were illustrative choices in the restaurant example, not mandatory attributes for every project.''')
code('''from pathlib import Path
import sys, json, os
ROOT = Path.cwd()
if not (ROOT/'src/person_a.py').exists(): ROOT = ROOT.parent
assert (ROOT/'src/person_a.py').exists(), 'Run from repo root or notebooks/'
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scipy.stats import spearmanr
from src.person_a import (SEED, SPECS, load_data, audit, make_splits,
    coverage_table, group_summary, paired_type_effect, export_splits,
    run_protocol, sha256, time_diagnostic)
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False,
                     'axes.spines.right':False, 'font.size':10})
REPORTS = ROOT/'reports'; REPORTS.mkdir(exist_ok=True)
FIGURES = REPORTS/'figures'; FIGURES.mkdir(exist_ok=True)
VERSION = 'Slim'  # explicit; never silently fall back to the tiny Test edition
print('Seed:',SEED,'Edition:',VERSION)
print('Python:',sys.version.split()[0])''')
md('''## Task 1 — A1/A8: provenance and data contract

Run `python scripts/download_data.py` first. Inputs are pinned by SHA-256; GitHub example CSVs also pin the source commit. Slim is obtained from the official Google Drive linked by the authors. Raw data and regenerated splits stay out of Git; the notebook embeds executed outputs.

The authors report Full: 100,646 wines, 1,056,079 users, 21,013,536 ratings. These are **published counts, not measurements from this run**. Verify the actual editions below. List fields are parsed with `ast.literal_eval`; IDs remain categorical. Repeated user/wine events can be valid different vintages in Full, so we do not blindly delete them.''')
code('''sources = json.loads((ROOT/'data/sources.json').read_text())
for name, info in sources['files'].items():
    assert sha256(ROOT/'data/raw'/name) == info['sha256'], f'Changed input: {name}'
wines, ratings = load_data(ROOT/'data/raw', VERSION)
example_wines, example_ratings = load_data(ROOT/'data/raw', 'Test')
summary = pd.concat([audit(example_wines,example_ratings).rename('Test edition'),
                     audit(wines,ratings).rename('Slim edition')],axis=1)
display(summary)
summary.to_csv(REPORTS/'dataset_summary.csv')
display(wines.drop(columns=['GrapeList','HarmonizeList']).head(3))
display(ratings.head(3))''')
code('''quality = pd.DataFrame({'missing':wines.isna().sum(),
    'unique_values':wines.drop(columns=['GrapeList','HarmonizeList']).nunique()})
display(quality.dropna())
quality.to_csv(REPORTS/'metadata_quality.csv')
print('Observed scale:',sorted(ratings.Rating.unique()))
print('Metadata columns:',', '.join(wines.columns[:17]))
print('Review-derived Body/Acidity excluded from default probes; price absent.')''')
md('''## Task 2 — freeze partitions before feature experiments

Primary estimand: **a later rating event in the sampled catalogue**, given user identity, wine identity and metadata assumed available at prediction time. We predict the continuous conditional mean, clip to [1,5], and do not round to half-stars. RMSE is primary; MAE is secondary. Unobserved pairs are not zero-star ratings.

- **Primary:** global chronological 80/10/10 by events, with all timestamp ties kept together. Training precedes validation, which precedes test. Fractions may differ slightly at ties.
- **Sensitivity:** seeded random user–wine-pair groups, preventing repeat-pair leakage if later adapted to Full. Random splitting answers a different question; its scores are not directly a measure of temporal leakage.
- **Stress test:** seeded disjoint wine groups; all interactions for a wine stay in one partition. This simulates unavailable item histories, not actual wine launch dates. Users may occur across partitions.

Unknown user/item one-hot keys contribute zero (global/content fallback); mean baselines back off to the training global mean. Coverage is reported; no cold rows are silently dropped. Validation tunes models; the held-out test is used only for prespecified references plus the selected metadata representation. Models remain fitted on train for comparable history slices.''')
code('''protocols = ['chronological','random','cold_item']
splits = {p:make_splits(ratings,p) for p in protocols}
manifests = {p:export_splits(splits[p],p,ROOT/'data/processed'/VERSION) for p in protocols}
(REPORTS/'split_manifests.json').write_text(json.dumps(manifests,indent=2)+'\\n')
coverage = pd.concat([coverage_table(s).assign(protocol=p) for p,s in splits.items()],ignore_index=True)
display(coverage)
coverage.to_csv(REPORTS/'split_coverage.csv',index=False)
train = splits['chronological']['train']
train_joined = train.merge(wines,on='WineID',validate='many_to_one')
print('Feature/hypothesis exploration below uses chronological TRAIN only.')''')
md('''## Task 1 — A2/A3: ratings, sparsity and activity

Whole-edition plots describe the sample; they do not select models. Train-only plots below motivate feature probes. Catalogue-level counts and review-level counts answer different questions. The observed rating distribution is conditioned on users choosing to rate; it is not the distribution of all tastes or purchases.''')
code('''fig, ax = plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
ratings.Rating.value_counts().sort_index().plot.bar(ax=ax[0,0],color='#713e50')
ax[0,0].set(xlabel='Rating (stars)',ylabel='Rating events',title=f'{VERSION}: observed ratings')
for key, axis, title in [('UserID',ax[0,1],'Ratings per user'),('WineID',ax[1,0],'Ratings per wine')]:
    counts = ratings[key].value_counts().sort_values(ascending=False).to_numpy()
    axis.loglog(np.arange(1,len(counts)+1),counts,color='#713e50')
    axis.set(xlabel='Rank (log scale)',ylabel='Rating events (log scale)',title=title)
ratings.groupby(ratings.Date.dt.year).size().plot.bar(ax=ax[1,1],color='#713e50')
ax[1,1].set(xlabel='Rating year',ylabel='Rating events',title='Sampling over time; not sales')
fig.suptitle('Official Slim edition • all observed events, 2012–2021')
fig.savefig(FIGURES/'rating_activity.png',bbox_inches='tight'); plt.show()
tails = pd.DataFrame({k:ratings[k].value_counts().describe(percentiles=[.5,.9,.95,.99])
                      for k in ['UserID','WineID']})
display(tails)
tails.to_csv(REPORTS/'activity_quantiles.csv')''')
md('''## Task 1 — A4/A5/A6: origin, type, grape and support

Report category support alongside averages. Equal-wine means reduce weighting by wine popularity but do not remove user selection or confounding. Grape review counts below are fractional: a blend with k listed grapes contributes 1/k to each, so total weighted counts equal rating events. Geographic labels mean production origin, not consumer location.''')
code('''type_stats = group_summary(train_joined,'Type')
country_stats = group_summary(train_joined,'Country')
region_stats = group_summary(train_joined,'RegionName')
display(type_stats); display(country_stats.head(12))
for name, frame in [('type',type_stats),('country',country_stats),('region',region_stats)]:
    frame.to_csv(REPORTS/f'train_{name}_summary.csv')
grapes = train_joined[['RatingID','Rating','GrapeList']].copy()
grapes['weight'] = 1/grapes.GrapeList.map(len).clip(lower=1)
grapes = grapes.explode('GrapeList').dropna(subset=['GrapeList'])
grapes['weighted_rating'] = grapes.Rating*grapes.weight
grape_stats = grapes.groupby('GrapeList').agg(weighted_events=('weight','sum'),
    weighted_rating_sum=('weighted_rating','sum'),raw_events=('RatingID','size'))
grape_stats['mean_rating'] = grape_stats.weighted_rating_sum/grape_stats.weighted_events
grape_stats = grape_stats.sort_values('weighted_events',ascending=False)
display(grape_stats.head(12)); grape_stats.to_csv(REPORTS/'train_grape_summary.csv')
fig,ax = plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
type_stats.ratings.sort_values().plot.barh(ax=ax[0],color='#713e50')
ax[0].set(xlabel='Training rating events',ylabel='Wine type',title='Most/least exposed wine types')
country_stats.ratings.head(10).sort_values().plot.barh(ax=ax[1],color='#713e50')
ax[1].set(xlabel='Training rating events',ylabel='Production country',title='Top 10 origins by reviews')
wines.Country.value_counts().head(10).sort_values().plot.barh(ax=ax[2],color='#713e50')
ax[2].set(xlabel='Wines in catalogue',ylabel='Production country',title='Top 10 origins by catalogue size')
fig.suptitle('Train exposure versus full catalogue composition • descriptive, not causal')
fig.savefig(FIGURES/'categories_origins.png',bbox_inches='tight'); plt.show()''')
md('''## Task 1 — deeper checks: user effects and popularity

**H1:** Users who rated both Red and White rate the two types differently. Compare each user's Red mean minus White mean, requiring at least two ratings per type. Bootstrap users, weight each eligible user equally, and interpret only within that selected group.

**H2:** Popular wines have higher observed mean ratings. Spearman association uses training wine counts and means. This association is descriptive; reverse causality, user composition and noisy rare-wine means prevent a popularity-causation claim. Neither analysis justifies a causal statement about a country's or grape's intrinsic quality.''')
code('''effect = paired_type_effect(train_joined)
display(pd.Series(effect,name='Red minus White (stars)'))
wine_stats = train.groupby('WineID').Rating.agg(['size','mean'])
rho,pvalue = spearmanr(wine_stats['size'],wine_stats['mean'])
print(f'Train wine popularity–mean-rating Spearman rho={rho:.3f}; nominal p={pvalue:.3g}')
print('Nominal p assumes independent wines; treat as descriptive given shared users.')
(REPORTS/'eda_effects.json').write_text(json.dumps({'red_white':effect,
    'popularity_spearman':float(rho),'popularity_nominal_p':float(pvalue)},indent=2)+'\\n')
fig,ax = plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].scatter(wine_stats['size'],wine_stats['mean'],s=10,alpha=.4,color='#713e50')
ax[0].set(xscale='log',xlabel='Training rating events per wine (log)',
          ylabel='Mean training rating (stars)',title='Popularity and observed mean rating')
type_stats[['mean_rating','equal_wine_mean']].plot.barh(ax=ax[1],xlim=(1,5))
ax[1].set(xlabel='Mean training rating (stars)',ylabel='Wine type',title='Weighting changes the question')
ax[1].legend(['Per rating','Equal weight per wine'])
fig.suptitle('Chronological training partition only')
fig.savefig(FIGURES/'associations.png',bbox_inches='tight'); plt.show()''')
md('''## Task 1 — A9: temporal patterns and historical taste concentration

Year/month counts describe the collected review process, not consumption demand. Type concentration is the share of a user's training reviews in their most frequent type. Compare against 100 permutations of type labels across events, preserving user activity and global type frequencies. This is an exploratory reference distribution, not a test of the causal origin of tastes.''')
code('''year_stats = train.groupby(train.Date.dt.year).Rating.agg(['size','mean'])
display(year_stats); year_stats.to_csv(REPORTS/'train_year_summary.csv')
eligible_users = train.UserID.value_counts().loc[lambda x:x>=5].index
taste = train_joined[train_joined.UserID.isin(eligible_users)][['UserID','Type']].copy()
def concentration(frame):
    counts = frame.groupby(['UserID','Type']).size()
    return (counts.groupby(level=0).max()/counts.groupby(level=0).sum()).mean()
observed = concentration(taste)
rng = np.random.default_rng(SEED)
null = []
for _ in range(100):
    perm = taste.copy(); perm['Type'] = rng.permutation(perm.Type.to_numpy())
    null.append(concentration(perm))
print(f'Users with >=5 train ratings: {len(eligible_users):,}')
print(f'Mean dominant-type share: observed {observed:.3f}; shuffled {np.mean(null):.3f}')
(REPORTS/'taste_concentration.json').write_text(json.dumps({'eligible_users':len(eligible_users),
    'observed':observed,'permutation_mean':float(np.mean(null)),
    'permutation_2_5':float(np.quantile(null,.025)),
    'permutation_97_5':float(np.quantile(null,.975))},indent=2)+'\\n')
fig,ax = plt.subplots(1,2,figsize=(11,4),layout='constrained')
year_stats['mean'].plot(ax=ax[0],marker='o',color='#713e50')
ax[0].set(xlabel='Rating year',ylabel='Mean training rating (stars)',title='Observed temporal drift')
ax[1].hist(null,bins=15,color='gray'); ax[1].axvline(observed,color='#713e50',label='Observed')
ax[1].set(xlabel='Mean dominant-type share per user',ylabel='Permutation runs',title='Historical type concentration')
ax[1].legend(); fig.suptitle('Chronological training partition only')
fig.savefig(FIGURES/'time_taste.png',bbox_inches='tight'); plt.show()''')
md('''## Task 2 — features, preprocessing and leakage boundaries

- **IDs:** categorical `UserID`, `WineID`. Ridge on one-hot IDs is a regularized additive user/item bias reference, not matrix factorization.
- **Origin/content:** one-hot type, country, region, winery and elaborate; grape lists use fractional multi-hot values. ABV is centered/scaled using unique training wines, median-imputed if needed, with a missing indicator. Fit all vocabularies on train; unknown values contribute zero.
- **Time/context:** optional month sin/cos, scaled rating year, age = rating year minus rated Vintage; invalid/future ages become missing. `N.V.` stays missing, not year zero. These require the rating date and vintage to be known for the query.
- **Preferences:** user×type interactions test whether conditional preferences add signal. Users and categories with no train support back off to other terms.
- **Excluded:** price (absent), user location (absent), label/name/website (identifiers or difficult to justify), entire wine `Vintages` list (includes future availability), Body/Acidity in primary probes (partly derived from user comments), Harmonize (provenance uncertain).
- **Aggregates:** team models may use training-only means/counts at the evaluation cutoff. For training rows used by a downstream regressor, target aggregates require out-of-fold or strictly prior computation; never use that row's rating in its own feature. No target-aggregate features are passed to these ridge probes.

**Availability limitation:** the catalogue was assembled in 2022, after the 2012–2021 ratings. This is a retrospective experiment assuming stable item metadata; an event-time split cannot prove historical availability. ABV/grapes can vary by vintage. Cold-item results assume metadata for an unseen wine is known.''')
md('''## Task 2 — controlled feature experiments (preliminary Task 3 evidence)

The same sparse ridge objective, three regularization values (1, 10, 100), fixed partitions, clipping and metrics apply to all representations. Train/validation RMSE and runtime are saved. Probe comparisons isolate *predictive utility conditional on this model*, not causal feature importance. Metadata can be redundant with WineID for warm items; lack of improvement does not prove metadata is useless for MF or cold items.

Primary comparison: validation-selected metadata probe versus validation-tuned IDs reference, evaluated on the chronological test. Random and cold-item protocols are sensitivity analyses. Test labels never select alpha or representation. Exploratory validation sweeps are not significance tests; the reported paired confidence interval belongs to the selected test comparison.''')
code('''results = {}
results['chronological'] = run_protocol(wines,splits['chronological'],'chronological',REPORTS)
display(results['chronological']['grid'].sort_values('rmse').head(12))''')
code('''results['random'] = run_protocol(wines,splits['random'],'random',REPORTS)
display(results['random']['scores'])''')
code('''# Fewer purposeful representations for the disjoint-wine stress test.
cold_specs = {k:SPECS[k] for k in ['ids','content','ids_metadata','ids_metadata_time','user_taste']}
results['cold_item'] = run_protocol(wines,splits['cold_item'],'cold_item',REPORTS,specs=cold_specs)
display(results['cold_item']['scores'])''')
md('''## Task 2 — validity, significance and failure slices

Use a **paired user-cluster bootstrap**, 2,000 resamples with replacement, resampling entire users and comparing both predictions on identical rows. Negative ΔRMSE favors the metadata probe. The interval measures test-user sampling uncertainty conditional on a fixed split, fitted models and selected hyperparameters; it does not capture retraining or temporal distribution uncertainty. Users can share wines, so item dependence remains a limitation.

Check warm/new-user/new-item/new-both regimes and training activity/popularity bands. Micro-RMSE weights events; macro-user RMSE weights users equally. Small slices must show support. Report failure as a finding rather than tune on test. If models are changed after viewing this test, use a fresh final holdout or explicitly label further results exploratory.''')
code('''scores = pd.concat([r['scores'] for r in results.values()],ignore_index=True)
intervals = pd.DataFrame([r['bootstrap'] for r in results.values()])
display(scores.round(4)); display(intervals.round(4))
scores.to_csv(REPORTS/'all_test_metrics.csv',index=False)
intervals.to_csv(REPORTS/'all_uncertainty.csv',index=False)
slices = pd.concat([r['slices'].assign(protocol=p) for p,r in results.items()],ignore_index=True)
display(slices.query("protocol == 'chronological' and model in ['ids','selected_metadata']"))
macro = []
for p in protocols:
    predictions = pd.read_csv(REPORTS/f'{p}_test_predictions.csv')
    for name in ['global','user_mean','item_mean','ids','selected_metadata']:
        mse = (predictions.Rating-predictions[name])**2
        macro.append(dict(protocol=p,model=name,
            macro_user_rmse=float(np.sqrt(mse.groupby(predictions.UserID).mean()).mean())))
macro = pd.DataFrame(macro); display(macro.round(4))
macro.to_csv(REPORTS/'macro_user_metrics.csv',index=False)''')
code('''fig,ax = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
scores.pivot(index='model',columns='protocol',values='rmse').plot.barh(ax=ax[0])
ax[0].set(xlabel='Test RMSE (stars; lower is better)',ylabel='Model',title='Separate evaluation questions')
ax[0].legend(title='Partition protocol')
g = results['chronological']['grid']
for name in ['ids','content','ids_metadata','user_taste']:
    s = g[g.representation.eq(name)].sort_values('alpha')
    ax[1].plot(s.alpha,s.rmse,marker='o',label=name)
ax[1].set(xscale='log',xlabel='Ridge alpha (log scale)',ylabel='Validation RMSE (stars)',
          title='Chronological regularization sensitivity')
ax[1].legend(); fig.suptitle('Official Slim edition • fixed seed 258 • alpha selected on validation')
fig.savefig(FIGURES/'feature_results.png',bbox_inches='tight'); plt.show()''')
md('''## Task 5 — short, measured findings from Person A's probes

The following findings are generated from this execution. These are feature/evaluation diagnostics; C/D still need to add final KNN, MF, hybrid ablations and model interpretation.

**Follow-up diagnostic disclosure:** after seeing that the primary selected probe included time, we added an IDs+time/age-only **validation** probe. This separates temporal context from wine-metadata associations. It does not access test, change the original selection, or supply independent confirmatory evidence. Avoid attributing a combined metadata+time test gain entirely to static wine attributes.''')
code('''time_check = time_diagnostic(wines,splits['chronological'])
time_check.to_csv(REPORTS/'chronological_time_diagnostic.csv',index=False)
display(time_check)
best_val = results['chronological']['grid'].sort_values('rmse').drop_duplicates('representation')
display(best_val[['representation','alpha','train_rmse','rmse']].sort_values('rmse'))
print('IDs+time/age validation best:',round(time_check.rmse.min(),6))
print('IDs+metadata+time/age validation best:',round(best_val.set_index('representation').loc['ids_metadata_time','rmse'],6))''')
code('''primary = results['chronological']
ci = primary['bootstrap']
base = primary['scores'].set_index('model')
word = 'supports a reduction' if ci['ci_high'] < 0 else ('supports an increase' if ci['ci_low'] > 0 else 'does not resolve a difference')
facts = [
    f"Slim contains **{len(ratings):,} ratings, {ratings.UserID.nunique():,} users and {len(wines):,} wines**; sparsity is **{audit(wines,ratings)['sparsity']:.2%}**.",
    'Price sensitivity and consumer geography cannot be tested with the supplied schema.',
    f"On future ratings, IDs RMSE is **{base.loc['ids','rmse']:.4f}**; the selected **{primary['selected']}** probe is **{base.loc['selected_metadata','rmse']:.4f}**. This is a combined metadata/context probe, not isolated evidence for static attributes.",
    f"Paired ΔRMSE is **{ci['delta_rmse']:+.4f}**, 95% user-bootstrap CI **[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]**; this {word} conditional on this run.",
    f"Validation RMSE: IDs **{best_val.set_index('representation').loc['ids','rmse']:.4f}**, IDs+static metadata **{best_val.set_index('representation').loc['ids_metadata','rmse']:.4f}**, follow-up IDs+time/age **{time_check.rmse.min():.4f}**. Check the time ablation before claiming metadata drives improvement.",
    'Use the disjoint-wine result to discuss missing item histories; it is not evidence about real product launches.',
    'The plan covers the rubric, but must freeze shared splits, add uncertainty, confirm the exact deadline and reserve time for integration/peer grading.'
]
display(Markdown('\\n'.join('- '+f for f in facts)))
(REPORTS/'findings.md').write_text('# Person A: measured findings\\n\\n'+'\\n\\n'.join('- '+f for f in facts)+'\\n')''')
md('''## Handoff and feasible milestones

**Internal schedule relative to confirmed submission deadline D:**

- **D−10 to D−8 (A + B):** confirm deadline, obtain Slim, freeze input hashes/splits/feature contract, share baseline predictions and literature briefing.
- **D−7 to D−5 (C/D):** regularized biases + MF; content/hybrid; train/validation curves and runtime; use exact A partitions. B compares the matching literature protocols.
- **D−4:** freeze model choices; evaluate final test once, bootstrap paired comparisons and inspect cold/rare groups. If using the already-inspected A test during further development, treat it as exploratory and reserve another final holdout.
- **D−3 to D−2:** integrate the five notebook sections, reconcile metric definitions and uncertainty, write limitations and parameter interpretations.
- **D−1:** execute notebook from a clean download, rehearse to 20 minutes, verify submission assets and the separate peer-grading deadline.

If fewer days remain, compress dates but preserve this order. **Must-have:** one primary rating task, Slim, baselines + MF + one metadata/hybrid comparison, all five sections. **Stretch:** Full scalability, Top-K, genuine temporal cold starts. Do not promise every C/D experiment before the core comparison is complete.

Suggested talk: A 6 minutes (3+3 for Tasks 1/2), B 3 for Task 4, C 4 and D 4 for Tasks 3/5, shared synthesis/limitations 3. Equal section marks do not imply equal speaker minutes; this protects A's two sections.

### Optional ranking protocol (not executed)
Keep ranking secondary to rating prediction. Predeclare relevant = held-out rating ≥4. For a strict warm-catalogue experiment use wines with a training interaction, exclude each user's training-rated wines, and state that availability dates are unknown. Evaluate Recall@10/NDCG@10 over the entire eligible catalogue, not just a changing random negative sample; exclude users with no eligible relevant test item and report their number. Unobserved wines are unlabeled, not verified dislikes. Compare models on identical users/candidates; report catalogue coverage and diversity separately. Actual item availability at historical query dates is not established by this dataset.

### Team interface
See `docs/data_contract.md`. C/D return `RatingID,prediction` for the exact test RatingIDs; no row-order assumptions. Every model reports training edition, protocol, validation selection, clipping, seed, runtime and fallback behavior. B must compare published metrics only under matching protocols.

### References

- Supplied `assignment2_fa26.pdf`, especially PDF pp.2–3, 7–13, 14–20, 28–30.
- [Official X-Wines repository and version counts](https://github.com/rogerioxavier/X-Wines).
- [Official dataset distribution](https://github.com/rogerioxavier/X-Wines/blob/main/Dataset/README.MD).
- de Azambuja, Morais & Filipe (2023), [X-Wines: A Wine Dataset for Recommender Systems and Machine Learning](https://doi.org/10.3390/bdcc7010020). Use for data provenance/limitations; this notebook is not a replication of its published model scores.

This notebook does not claim that an 8–12-paper review, Full-edition experiment, Top-K task, price model or every restaurant-example question is mandatory.''')

nb = nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.11'}})
path = ROOT/'notebooks/person_a_tasks_1_2.ipynb'
path.parent.mkdir(exist_ok=True)
nbf.write(nb,path)
print(f'Built {path}: {len(cells)} cells')
