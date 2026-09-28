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
- [x] Time split → `ml/{train,val,test}.parquet` from `clean_faers.py --stage split`
      (FB-D13): 46,589,011 / 4,237,245 / 11,906,614 rows (2004–2022 / 2023 / 2024–2026),
      `ml/split_manifest.json` written last. The old notebook split is kept, stale, in
      `data/processed/ml_notebook_legacy/`
- [x] Drug names → RxNorm ingredients: 74.5% of 43,099 names mapped
- [x] SIDER pairs: 134,844 (ingredient, PT) pairs, 1,264 ingredients
- [x] SIDER labels: 860,422 pairs, 28.2% positive, 11,732 drug names
- [x] Task A dataset: 715,034 pairs in 5 ingredient-disjoint folds (`data/task_a.py`)
- [x] Classical baselines and drug-level control on Task A folds (below)
- [x] Tests: 35 pass (`pytest tests`)
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

`baseline --task timesplit --labels sider` writes `data/logs/baseline_results_sider.json`.
It now runs on the script-built split (2004–2026 data, 327,514 test pairs) and is kept
only as a downstream compatibility check; its numbers are not analysed. The earlier
notebook-based table (2012+ data, 309,522 test pairs, IC/EBGM 0.615) is preserved in
`data/logs/baseline_results_sider_notebook_2012.json`.
It is not comparable with Task A: its eval set and its SIDER eligibility used post-2022
reports, `ROR_all` and within-split `PRR` used eval-period data, and the notebook-built
`ror_train` column could not be reproduced from `train.parquet`. `ROR_train` is now
computed in code from train-split counts (FB-D13).

## Blockers for v0

1. **Pipeline is not one command yet.** The split is now scripted and memory-bounded
   (FB-D13; resolved 2026-09-27). The earlier in-memory writer was killed at 5.9 GB peak
   RSS; the streamed writer peaks at ~2.32 GB (operational limit 3 GB, machine 16 GB).
   Still open: `scripts/run_pipeline.sh` only runs model training, so collect → audit →
   clean → split → labels → Task A is not yet a single command.
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

1. **LightGBM baseline (next)** on per-pair features: time trends, age/sex mix, outcomes,
   reporter type, indications and the classical scores. Evaluate on the Task A
   ingredient-disjoint folds (FB-D11); beat IC per fold and the drug-level control.
   The split writer is done (FB-D13); a one-command `run_pipeline.sh` and blockers 3–4
   remain open but are not on its path.
2. **Fix the EBGM prior fit.** Try fitting without zero truncation, or floor α.
3. **Task B:** FDA SrLC label-change dates. SrLC has no bulk export, so this needs a
   scraper. Most novel part of the benchmark.
4. **Task C:** report-sparse drugs (few or no reports up to the cutoff). Where ChemBERTa may help.

Deprioritized: PubMedBERT text model and fusion (no narratives in public FAERS, FB-D9).
