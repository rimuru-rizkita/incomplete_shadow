# Incomplete Shadow Annotations in Landsat Cloud-Shadow Benchmarks

Code for measuring how incomplete cloud-shadow labels in **L8 Biome** and **L7 Irish** affect
the ranking of segmentation methods, using **SPARCS** (complete shadow labels) as an external
test set, and for training with **geometry-aware partial labels** derived from the sun-projected
shadow swath of each cloud.

## Setup

```bash
# install a CUDA build of PyTorch first (https://pytorch.org), then:
pip install -r requirements.txt
```

Exact versions used in the paper are listed in `requirements-lock.txt`
(PyTorch 2.5.1, TorchGeo 0.8.1, segmentation_models_pytorch 0.6.0).

## Data

| Dataset | Location | Source |
|---|---|---|
| L8 Biome | `dataset/l8biome/` | TorchGeo `L8Biome` (download=True) |
| L7 Irish | `dataset/l7irish/` | TorchGeo `L7Irish` (download=True) |
| SPARCS | `dataset/sparcs/` | `sparcs_data_L8.zip` from https://emapr.ceoas.oregonstate.edu/sparcs/ |

Scenes are read in their native UTM grid at 30 m. The location-based splits used in the paper
are in `data/{l8biome,l7irish}_split.json`.

## Usage

All commands run from the repository root.

```bash
# 1. splits, patch index, normalisation statistics
python scripts/phase0_check_setup.py --dataset l7irish l8biome
python scripts/phase0_scan_train.py  --dataset l7irish l8biome --extra_splits val test

# 2. partial-label masks (geom = shadow swath, all = every Clear pixel)
python scripts/make_partial_masks.py --dataset l8biome l7irish --mode geom all
python scripts/prepare_sparcs.py

# 3. training (finished runs are skipped; use --dry_run to list runs)
python run_experiments.py --phase 7 8 9 10 11 12

# 4. evaluation on SPARCS and with the geometry-aware metric
python scripts/eval_sparcs.py
python scripts/geom_filter.py --dataset l8biome
python scripts/geom_filter.py --dataset l7irish

# 5. statistics (paired t-tests + Holm, scene bootstrap)
python analyze_grsl.py
```

| Phase | Experiment | Seeds |
|---|---|---|
| 7 | U-Net: random, non-empty, oversampling × partial labels {none, all, geom} | 5 |
| 8 | ResNet-18 (SSL4EO-L) and DeepLabV3+ (ImageNet) × partial {none, geom} | 3 |
| 9 | Oversampling probability *p* ∈ {0.2, 0.4, 0.8} | 3 |
| 10 | Factorial sampler × CBAM × loss; class-balanced loss, OHEM, class-aware sampling | 3 |
| 11 | Class-weighted OHEM | 3 |
| 12 | SegFormer MiT-B2 (ImageNet) × sampler × partial {none, geom} | 3 |

Results are written to `runs/` and `analysis/grsl/`.

## Repository layout

```
src/        data loading, samplers, losses, models, training, metrics
scripts/    data preparation, partial labels, SPARCS and geometry-aware evaluation
data/       location-based splits and partial-label statistics
run_experiments.py   experiment runner (phases defined in src/phases.py)
analyze_grsl.py      statistical analysis
```

See `RUNBOOK.md` for detailed notes on each step.

## Citation

The paper is under review; citation details will be added after publication.
