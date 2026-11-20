# Person A: measured findings

- Slim contains **150,000 ratings, 10,561 users and 1,007 wines**; sparsity is **98.59%**.

- Price sensitivity and consumer geography cannot be tested with the supplied schema.

- On future ratings, IDs RMSE is **0.4493**; the selected **ids_metadata_time** probe is **0.4332**. This is a combined metadata/context probe, not isolated evidence for static attributes.

- Paired ΔRMSE is **-0.0161**, 95% user-bootstrap CI **[-0.0185, -0.0136]**; this supports a reduction conditional on this run.

- Validation RMSE: IDs **0.4544**, IDs+static metadata **0.4539**, follow-up IDs+time/age **0.4456**. Check the time ablation before claiming metadata drives improvement.

- Use the disjoint-wine result to discuss missing item histories; it is not evidence about real product launches.

- The plan covers the rubric, but must freeze shared splits, add uncertainty, confirm the exact deadline and reserve time for integration/peer grading.
