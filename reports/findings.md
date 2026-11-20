# Findings

- Slim contains **150,000 ratings, 10,561 users and 1,007 wines**; sparsity is **98.59%**.

- Prices and consumer locations are unavailable in X-Wines.

- On future ratings, IDs RMSE is **0.4493**; the selected **ids_metadata_time** probe is **0.4332**. This is a combined metadata/context probe, not isolated evidence for static attributes.

- Paired ΔRMSE is **-0.0161**, 95% user-bootstrap CI **[-0.0185, -0.0136]**; this supports a reduction conditional on this run.

- Validation RMSE: IDs **0.4544**, IDs+static metadata **0.4539**, follow-up IDs+time/age **0.4456**. Check the time ablation before claiming metadata drives improvement.

- The disjoint-wine experiment measures prediction without item histories; it does not reconstruct product launches.
