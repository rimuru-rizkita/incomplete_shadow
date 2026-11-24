# Results

CSV files behind the tables of the letter and its supplementary material. They are produced by
`analyze_grsl.py`, `extra_stats.py`, and `manuscript/make_tables.py` from the per-scene confusion matrices
of each run. Shadow IoU, precision, recall, F1, and mIoU are in percent.

| File | Content |
|---|---|
| `per_seed.csv` | every run (configuration x seed) under each evaluation view |
| `table_by_view.csv` | mean and standard deviation over seeds per configuration and view |
| `pairwise_comparisons.csv` | all paired comparisons A - B: t-test over seeds (`p`, Holm-adjusted `p_holm`, seed CI) and 95% scene-bootstrap interval (`boot_low`, `boot_high`) |
| `rank_agreement.csv` | Kendall tau between the rankings of the five class-imbalance strategies under two views, with scene-bootstrap probabilities |
| `equivalence_controls.csv` | 90% scene-bootstrap intervals of the CBAM and hybrid-loss effects and their comparison with the sampler effect |
| `permutation_tests.csv` | scene-level paired sign-flip permutation tests (10,000 permutations, Holm-adjusted) |
| `prior_correction.csv` | post-hoc prior correction of oversampled U-Nets (base, fixed log-prior, validation-tuned bias) |
| `split_groups.csv` | train / val / test scenes per biome (L8 Biome) or latitude zone (L7 Irish) |

Views: `all` (all test scenes), `annotated` (test scenes with shadow labels), `geomaware` (Clear pixels inside
the shadow swath of unannotated scenes ignored), `sparcs` (external SPARCS test set, Landsat 8 models only).
