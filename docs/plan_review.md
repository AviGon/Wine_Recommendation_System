# Plan review against the supplied assignment

**Verdict: the division of work fits the five sections. The plan is ambitious enough; its main gaps are evaluation rigor, data availability and a dated integration schedule.** Tasks 1 and 2 each carry roughly the same marks as Tasks 3, 4 and 5, so Person A's work is a substantial contribution.

## What the assignment actually asks

- PDF p.2: due **early week 10**, submit through Gradescope. **No exact date, time or timezone appears.** Confirm the current Gradescope/course announcement; do not derive an official deadline from the filename or this review.
- PDF pp.3, 8–13: identify/explore data; define prediction/features and validity/significance; compare appropriate models/representations/baselines; discuss relevant research; interpret results and predictive parameters.
- PDF p.7: choose a dataset large enough for course models. Slim's 150,000 ratings are a defensible main experiment; the 1,000-rating Test edition is useful for debugging. The PDF specifies no minimum row count and does not require Full.
- PDF p.28: roughly five marks per section, groups up to four, a 20-minute presentation with the notebook available to viewers. The slide has a handwritten correction to “25 marks”; do not rely on the printed “25%” wording for course weighting.
- PDF p.30: peer grading is also described, roughly 4–5 assignments to grade and worth four marks. Check its separate instructions/deadline.

The restaurant example (pp.14–20) illustrates adaptation, not a checklist that forces price, coordinates or business survival into every dataset. An 8–12-paper review, every listed model, Top-K and Full-scale runs are team proposals, not stated requirements.

## Tasks 1–5 and responsibilities

1. **A / Task 1:** keep A1–A9, but mark A7 price **unavailable**. Separate wine-catalogue counts from rating exposure; examine activity tails, origins, dates, missingness, support and user effects. No consumer geography claim.
2. **A / Task 2:** freeze hashes, RatingID partitions, preprocessing, label availability, metrics and unknown-ID behavior before teammates report scores. Add paired uncertainty, a future-prediction primary split, and cold/rare coverage. This notebook implements those foundations.
3. **C/D / Task 3:** retain a controlled progression: global/user/item averages → regularized biases → KNN/MF → content/hybrid. Include validation tuning, training curves/runtime, noise/missingness and explanations of appropriate/inappropriate course alternatives. A's sparse ridge probes are an initial contract, not a replacement for final models.
4. **B / Task 4:** start early. Explain data origin/original use, compare similar datasets and methods, connect borrowed features/objectives to this task, and revisit the review after results exist. Avoid treating unmatched published scores as a benchmark to beat.
5. **C/D + team / Task 5:** analyze failures as well as gains, uncertainty, parameters, feature representations, active/new users and common/rare/unseen wines. Finish shared conclusions rather than four disconnected reports.

## Fix the research question

Use: **“Can historical user ratings predict later wine ratings, and does available wine metadata improve prediction beyond user/wine identities?”** Remove price until an actual justified source is obtained. Country and region describe production origin; no user residence, distance, sales or wine failure labels exist.

Metadata is a retrospective catalogue assembled after the ratings. Body/Acidity can incorporate review-derived descriptions; exclude them from primary features. Entire `Vintages` lists can reveal later years. Rated `Vintage` and rating time may be query context only if declared available. ABV and grapes may differ by vintage. These limitations belong in the presentation even with chronological splitting.

## Practical schedule

Let **D be the confirmed submission deadline**, not an inferred calendar date.

- **D−10 to D−8:** A freezes Slim input hashes, splits, preprocessing and baseline output format. B delivers dataset/literature briefing.
- **D−7 to D−5:** C/D implement their core models and ablations on the shared partitions; tune only on validation.
- **D−4:** freeze choices; one final test evaluation, paired uncertainty and failure slices. A's already-viewed holdout is exploratory if it influences later development: reserve a fresh final holdout in that case.
- **D−3 to D−2:** merge all five sections, reconcile claims/metrics and generate the complete notebook/presentation.
- **D−1:** clean rerun, 20-minute rehearsal, check Gradescope assets and peer-review instructions.

If time is short, drop Full, elaborate ranking and extra model families before dropping the five required sections. Suggested talk allocation: A six minutes, B three, C four, D four, shared interpretation/limitations three.

## Evidence and deliverables

Open `notebooks/person_a_tasks_1_2.ipynb` for executed Task 1/2 experiments, the restaurant-to-wine mapping and short findings. `reports/findings.md` contains measured headline results. `docs/data_contract.md` defines the C/D handoff. Remaining team deliverables are explicit; this review does not claim the complete final project is finished.

Sources: supplied `assignment2_fa26.pdf` (page references above); [official X-Wines repository](https://github.com/rogerioxavier/X-Wines); [official download information](https://github.com/rogerioxavier/X-Wines/blob/main/Dataset/README.MD); de Azambuja, Morais & Filipe (2023), [X-Wines](https://doi.org/10.3390/bdcc7010020).
