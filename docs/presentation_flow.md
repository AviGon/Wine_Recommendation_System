# Presentation flow

Use one shared deck with about ten slides and the cumulative notebook as the single walkthrough surface. All four speakers contribute slides. The data speaker shows saved tables/figures; the two model speakers give short implementation walkthroughs. The research speaker mainly uses slides and returns for the final synthesis.

## Research framing and published context: 0:00–4:00

Prepare two slides: **Prediction question and comparison design** and **Related research**. Define the input/output and why future rating prediction is the main question. Give the metric and reference-model plan briefly. Compare selected prior approaches with this study; avoid reading a bibliography. Hand over with the hypotheses that the data and models will examine.

Use the notebook anchors **Prediction objective and study design** and **Related research and model choices** as the evidence behind the slides. Return at 19:00 for the closing minute, giving this speaker five minutes overall.

## Data and preprocessing: 4:00–9:00

Prepare two slides: **Sample and data quality** and **Features and shared partitions**. Spend roughly two minutes on slides and three minutes on the notebook. Show the rating/activity figure, catalogue-versus-review composition, and chronological split coverage. Explain how the observed sparsity and drift motivate regularization, fallbacks and rating context. Use the short encoder excerpt if needed; do not scroll through raw CSV rows or installation cells.

Notebook anchors: **Dataset, collection and preprocessing**, then **Models, objectives and implementation** for the encoder/feature conventions. Finish with a clear statement of what is available before prediction and what training-only fitting means.

## Collaborative modeling: 9:00–14:00

Prepare two slides: **Biases and neighbours** and **Latent factors and regularization**. Use roughly two minutes of slides, ninety seconds of code and ninety seconds of saved results. Show the bias update or ALS solve, rather than an entire class. Explain the loss, regularization, unknown-ID fallback and why a lower training error does not establish better prediction.

Use the cumulative notebook's model excerpts and the CF side of the confidence-interval figure. The outcome is a small neighbour correction and no clear MF improvement over the tuned bias reference in this experiment. Do not generalize that finding to all users or recommender systems.

## Metadata, context and generalization: 14:00–19:00

Prepare three slides: **Static attributes versus rating context**, **Linear and interaction models**, and **Generalization and recommendation lists**. Explain the FM prediction calculation in about one minute. Use the separate Time/Age validation ablation, the common chronological comparison and the cold-item failure. Keep ranking/diversity to one short secondary finding; it answers a different question from rating prediction.

Notebook anchors: **Static attributes, calendar time and vintage age**, **Chronological rating prediction**, **Unseen wines and activity groups**, and **Ranking and recommendation diversity**. Explicitly identify validation results versus held-out results. Do not call the FM superior to its matched linear reference when their point estimates are nearly equal.

## Shared synthesis: 19:00–20:00

Use one joint closing slide, presented by the opening speaker. State the strongest supported conclusions and limitations: strong additive references, larger benefit from rating context than static attributes on warm wines, uncertain generalization to new wines, and the gap between rating accuracy and useful lists. Relate the findings back to the original question and the cited research.

## Recording workflow

Keep the ten slides in a single deck with one visual style. Open the cumulative notebook or its HTML export beside it. Use saved outputs; do not retrain, download dependencies or wait for long cells during recording. The visible notebook headings should match the slide topics. Give each speaker a precise starting anchor and ending sentence, and rehearse the screen transitions.

A useful distinction prevents repetition: the introduction states the prediction goal and promised comparison; exploration establishes what the observed data supports; modeling explains how predictions are produced; evaluation tests those promises; related research supplies outside evidence and explains agreement or disagreement with prior results.

The notebook and video are complementary. Export the complete, already-rendered notebook as `workbook.html`; record about twenty minutes and keep the video link separately as `video_url.txt` once the real recording exists. Slides support the recording, while the HTML must remain understandable without executing code.
