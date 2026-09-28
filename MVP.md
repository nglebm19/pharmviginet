# MVP.md — Benchmark v0

What "v0" means, where the project stands and what comes next.
Every status line below was checked against files or commits; re-check before
changing it. Last verified 2026-09-27 after the legacy AERS rebuild (FB-D12), on top of commit `95154af`.

## Definition of v0

v0 is ready to make public when all of these hold:

1. The full pipeline runs from scripts, with no notebook step.
2. The data covers 2004Q1–2026Q1. (Done: FB-D12.)
3. Task A has SIDER labels and classical baselines scored with per-fold `auc_strat`
   on ingredient-disjoint folds (FB-D11).
4. At least one learned baseline (LightGBM) is on the leaderboard.
5. README, labels and results agree with the code.

## Status

- [x] Collect 89 quarters, 2004Q1–2026Q1 (`collect_faers.py`)
- [x] Audit files and schemas (`audit_faers.py` → `data/logs/audit_report.json`)
- [x] Clean, dedup, join → `master.parquet`, 62,732,870 rows, 2004–2026
      (52,728,487 FAERS + 10,004,383 legacy AERS rows; `cases_deduped` 20,328,567 cases)
- [ ] Time split → `ml/{train,val,test}.parquet` — **missing** (blocker 1). The join
      stage was killed while writing it; the old notebook-built split is kept in
      `data/processed/ml_notebook_legacy/` (2012+ data only, stale)
- [x] Drug names → RxNorm ingredients: 74.5% of 43,099 names mapped
- [x] SIDER pairs: 134,844 (ingredient, PT) pairs, 1,264 ingredients
- [x] SIDER labels: 860,422 pairs, 28.2% positive, 11,732 drug names
- [x] Task A dataset: 715,034 pairs in 5 ingredient-disjoint folds (`data/task_a.py`)
- [x] Classical baselines and drug-level control on Task A folds (below)
- [x] Tests: 27 pass (`pytest tests`)
- [ ] LightGBM baseline
- [ ] Task B, task C
- [ ] Public release (repo, dataset card, leaderboard), write-up. The GitHub repo
      `nglebm19/pharmviginet` stays private until v0 (FB-D10).

## Current leaderboard (task A)

Task A design: `DECISIONS.md` FB-D11. SIDER labels; eligible pairs have ≥ 3 distinct
reports up to 2022; every feature and score uses reports up to 2022 only; 5
ingredient-disjoint folds (seed 0). Reports up to 2022 now cover 2004–2022 (FB-D12).
715,034 pairs, 28.9% positive, 1,138 ingredients. Source: `data/logs/task_a_results.json`.

| Method | AUC_strat mean ± sd (5 folds) | Paired diff vs IC, mean ± sd |
|---|---|---|
| EB05 | 0.620 ± 0.005 | +0.006 ± 0.007 |
| IC025 | 0.619 ± 0.004 | +0.005 ± 0.008 |
| EBGM (MGPS) | 0.615 ± 0.011 | +0.001 ± 0.001 |
| IC (BCPNN) | 0.614 ± 0.012 | reference |
| ROR | 0.610 ± 0.012 | −0.004 ± 0.001 |
| PRR | 0.610 ± 0.012 | −0.004 ± 0.001 |
| Drug-level-only control | 0.554 ± 0.021 | −0.059 ± 0.011 |

Fold quality (checked before scoring): 140–145K pairs, 27.1–32.0% positive,
177–250 ingredients and 1,029–1,140 usable PT strata per fold; the largest ingredient
holds ≤ 5.6% of a fold's pairs; 31,328 combination pairs (4.2%) whose ingredients span
folds are dropped.

Before the legacy rebuild (2012–2022 data only, 560,334 pairs) the same table read
IC 0.606 ± 0.020, EB05 0.609 ± 0.014, drug control 0.543 ± 0.035; the ranking is unchanged.

Reading the table:
- Classical methods are within 0.010 of each other, close to the fold-to-fold sd
  (~0.005–0.012). Only ROR/PRR vs IC is a consistent paired difference (−0.002 to
  −0.004, negative in every fold). EB05/IC025 beat IC in 4 of 5 folds.
- The drug-level control, which sees only drug features (report volume, PTs reported,
  first year, ingredient count), scores well below every classical method. Those
  drug-level features alone do not reproduce the classical ranking.
- Bar for learned models: beat IC's paired per-fold `auc_strat` (≈ 0.61–0.62) and the
  drug-level control.
- EBGM ranks well, but its prior is near-degenerate (p ≈ 0.95, α₁ ≈ 2.4e-4); don't
  report absolute EBGM values until the prior is fixed.
- Pooled AUC and AUPRC are in the JSON for reference; F1 is not reported.

### Legacy time-split reference (not Task A)

The earlier table (test split 2024–2026Q1, 309,522 pairs, IC/EBGM 0.615) is kept in
`data/logs/baseline_results_sider.json` (`baseline --task timesplit --labels sider`).
It is not comparable with Task A: its eval set and its SIDER eligibility used post-2022
reports, `ROR_all` and within-split `PRR` used eval-period data, and the notebook-built
`ror_train` column cannot be reproduced from `train.parquet`. It was computed on the
2012+ data and can no longer be rerun (it needs `ror_train`); the JSON is historical.

## Blockers for v0

1. **No reproducible train/val/test split.** `data/processed/ml/` is currently empty.
   The in-memory split writer in `clean_faers.py --stage join` is not safe at the
   rebuilt size: on 2026-09-27 the process was killed (16 GB machine, 5.9 GB peak RSS)
   after `master.parquet` was fully written but before any split file was written.
   A fix should write each split straight from `master.parquet` with pyarrow year
   filters (or another memory-safe approach). The earlier split came from
   `notebooks/PharmVigiNet_train.ipynb` and is kept, stale, in
   `data/processed/ml_notebook_legacy/`. Task A does not need `ml/`; the paused
   text/mol models and `baseline --task timesplit` do. `scripts/run_pipeline.sh`
   only runs model training, not the pipeline.
2. ~~2004–2011 data is missing.~~ **Resolved (FB-D12).** Legacy rows were lost at
   parsing (trailing `$` shifted every column) and dropped at dedup. Now 2004–2026.
3. **Old ROR label does not match `ror`.** The `label` column was supposedly derived
   from `ror`, yet `ROR_all` scores only AUC 0.787 against it on test (`ROR_train`
   0.835; `data/logs/baseline_results.json`). It should be near 1.0. Probably computed
   from different data; compare `compute_ror` in `clean_faers.py` with the notebook.
4. **`n_reports` leaks the eval period.** `build_labels.faers_pairs` applies the
   ≥ 3-reports filter using counts from all years, including test. Task A does not
   use this column; it applies its own filter on reports up to 2022.

## Queue (agreed order)

1. **Reproducible pipeline.** No plan approved yet; start in PLAN mode. Move the split into a script (e.g.
   `pharmviginet/data/build_dataset.py`); make one command run
   collect → audit → clean → split → labels → baselines, with a memory-safe split
   writer (blocker 1). Resolve blockers 3–4 here.
2. **LightGBM baseline** on per-pair features: time trends, age/sex mix, outcomes,
   reporter type, indications and the classical scores. Evaluate on the Task A
   ingredient-disjoint folds (FB-D11, resolved); beat IC per fold and the drug-level control.
3. **Fix the EBGM prior fit.** Try fitting without zero truncation, or floor α.
4. **Task B:** FDA SrLC label-change dates. SrLC has no bulk export, so this needs a
   scraper. Most novel part of the benchmark.
5. **Task C:** report-sparse drugs (few or no reports up to the cutoff). Where ChemBERTa may help.

Deprioritized: PubMedBERT text model and fusion (no narratives in public FAERS, FB-D9).
