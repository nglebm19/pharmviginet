# MVP.md — Benchmark v0

What "v0" means, where the project stands and what comes next.
Every status line below was checked against files or commits; re-check before
changing it. Last verified 2026-09-27 against commit `b495b6c` plus the Task A change (FB-D11).

## Definition of v0

v0 is ready to make public when all of these hold:

1. The full pipeline runs from scripts, with no notebook step.
2. The data covers 2004Q1–2026Q1, or the gap is fixed and documented.
3. Task A has SIDER labels and classical baselines scored with per-fold `auc_strat`
   on ingredient-disjoint folds (FB-D11).
4. At least one learned baseline (LightGBM) is on the leaderboard.
5. README, labels and results agree with the code.

## Status

- [x] Collect 89 quarters, 2004Q1–2026Q1 (`collect_faers.py`)
- [x] Audit files and schemas (`audit_faers.py` → `data/logs/audit_report.json`)
- [x] Clean, dedup, join → `master.parquet`, 52.7M rows — **but 2012 onward only** (blocker 2)
- [~] Time split → `ml/{train,val,test}.parquet` — exists, built in a notebook (blocker 1)
- [x] Drug names → RxNorm ingredients: 76.4% of 33,802 names mapped
- [x] SIDER pairs: 134,844 (ingredient, PT) pairs, 1,264 ingredients
- [x] SIDER labels: 713,399 pairs, 29.2% positive
- [x] Task A dataset: 560,334 pairs in 5 ingredient-disjoint folds (`data/task_a.py`)
- [x] Classical baselines and drug-level control on Task A folds (below)
- [x] Tests: 19 pass (`pytest tests`)
- [ ] LightGBM baseline
- [ ] Task B, task C
- [ ] Public release (repo, dataset card, leaderboard), write-up. The GitHub repo
      `nglebm19/pharmviginet` stays private until v0 (FB-D10).

## Current leaderboard (task A)

Task A design: `DECISIONS.md` FB-D11. SIDER labels; eligible pairs have ≥ 3 distinct
reports up to 2022; every feature and score uses reports up to 2022 only; 5
ingredient-disjoint folds (seed 0). 560,334 pairs, 30.2% positive, 1,091 ingredients.
Source: `data/logs/task_a_results.json`.

| Method | AUC_strat mean ± sd (5 folds) | Paired diff vs IC, mean ± sd |
|---|---|---|
| EB05 | 0.609 ± 0.014 | +0.003 ± 0.006 |
| IC025 | 0.609 ± 0.013 | +0.003 ± 0.007 |
| EBGM (MGPS) | 0.606 ± 0.019 | +0.000 ± 0.001 |
| IC (BCPNN) | 0.606 ± 0.020 | reference |
| ROR | 0.602 ± 0.020 | −0.004 ± 0.001 |
| PRR | 0.602 ± 0.020 | −0.004 ± 0.001 |
| Drug-level-only control | 0.543 ± 0.035 | −0.063 ± 0.018 |

Fold quality (checked before scoring): 107–116K pairs, 27.8–32.2% positive,
164–217 ingredients and 895–980 usable PT strata per fold; the largest ingredient
holds ≤ 6.7% of a fold's pairs; 26,678 combination pairs (4.5%) whose ingredients span
folds are dropped.

Reading the table:
- Classical methods are within 0.007 of each other, well inside the fold-to-fold sd
  (~0.02). Only ROR/PRR vs IC is a consistent paired difference (−0.003 to
  −0.005, negative in every fold).
- The drug-level control, which sees only drug features (report volume, PTs reported,
  first year, ingredient count), scores well below every classical method. Those
  drug-level features alone do not reproduce the classical ranking.
- Bar for learned models: beat IC's paired per-fold `auc_strat` (≈ 0.61) and the
  drug-level control.
- EBGM ranks well, but its prior is near-degenerate (see ARCHITECTURE.md); don't
  report absolute EBGM values until the prior is fixed.
- Pooled AUC and AUPRC are in the JSON for reference; F1 is not reported.

### Legacy time-split reference (not Task A)

The earlier table (test split 2024–2026Q1, 309,522 pairs, IC/EBGM 0.615) is kept in
`data/logs/baseline_results_sider.json` (`baseline --task timesplit --labels sider`).
It is not comparable with Task A: its eval set and its SIDER eligibility used post-2022
reports, `ROR_all` and within-split `PRR` used eval-period data, and the notebook-built
`ror_train` column cannot be reproduced from `train.parquet`.

## Blockers for v0

1. **Split is not reproducible.** `clean_faers.py --stage join` writes a plain time
   split, but the current `data/processed/ml/*.parquet` came from
   `notebooks/PharmVigiNet_train.ipynb`: val/test carry `ror_train`, which no script
   produces. `scripts/run_pipeline.sh` only runs model training, not the pipeline.
2. **2004–2011 data is missing.** `demo.parquet` and `reac.parquet` start in 2004, but
   `cases_deduped.parquet` and `drug_ps.parquet` start in 2012, so legacy AERS rows
   are lost at the dedup stage. `master.parquet` and `train.parquet` therefore start
   in 2012 (2012 has only 674K rows). `master.parquet` still carries uppercase legacy
   columns (`ISR`, `CASE`), which suggests the lowercase-only `RENAME_MAP`
   (`isr` → `primaryid`) misses legacy headers. Not yet confirmed.
3. **Old ROR label does not match `ror`.** The `label` column was supposedly derived
   from `ror`, yet `ROR_all` scores only AUC 0.787 against it on test (`ROR_train`
   0.835; `data/logs/baseline_results.json`). It should be near 1.0. Probably computed
   from different data; compare `compute_ror` in `clean_faers.py` with the notebook.
4. **`n_reports` leaks the eval period.** `build_labels.faers_pairs` applies the
   ≥ 3-reports filter using counts from all years, including test. Task A does not
   use this column; it applies its own filter on reports up to 2022.
5. **Task A must be rerun after blocker 2.** "Reports up to 2022" currently means
   2012–2022.

## Queue (agreed order)

1. **Reproducible pipeline.** No plan approved yet; start in PLAN mode. Move the split into a script (e.g.
   `pharmviginet/data/build_dataset.py`); make one command run
   collect → audit → clean → split → labels → baselines. Resolve blockers 2–4 here.
2. **LightGBM baseline** on per-pair features: time trends, age/sex mix, outcomes,
   reporter type, indications and the classical scores. Evaluate on the Task A
   ingredient-disjoint folds (FB-D11, resolved); beat IC per fold and the drug-level control.
3. **Fix the EBGM prior fit.** Try fitting without zero truncation, or floor α.
4. **Task B:** FDA SrLC label-change dates. SrLC has no bulk export, so this needs a
   scraper. Most novel part of the benchmark.
5. **Task C:** report-sparse drugs (few or no reports up to the cutoff). Where ChemBERTa may help.

Deprioritized: PubMedBERT text model and fusion (no narratives in public FAERS, FB-D9).
