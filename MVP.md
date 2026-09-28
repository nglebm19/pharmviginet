# MVP.md — Benchmark v0

What "v0" means, where the project stands and what comes next.
Every status line below was checked against files or commits; re-check before
changing it. Last verified 2026-09-27 against commit `483ec3a`.

## Definition of v0

v0 is ready to make public when all of these hold:

1. The full pipeline runs from scripts, with no notebook step.
2. The data covers 2004Q1–2026Q1, or the gap is fixed and documented.
3. Task A has SIDER labels and classical baselines scored with `auc_strat`.
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
- [x] Classical baselines on SIDER labels (below)
- [x] Tests: 12 pass (`pytest tests`)
- [ ] LightGBM baseline
- [ ] Task B, task C
- [ ] Public release (repo, dataset card, leaderboard), write-up. The GitHub repo
      `nglebm19/pharmviginet` stays private until v0 (FB-D10).

## Current leaderboard (task A)

Test split (2024–2026Q1), SIDER labels, one score per unique (drug, PT) pair:
309,522 pairs, 29.0% positive, 1,574 PT strata. Source: `data/logs/baseline_results_sider.json`.
`_train` methods use train-period counts only; unseen pairs get neutral scores.

| Model | AUC pooled | **AUC_strat** |
|---|---|---|
| IC (BCPNN) | 0.52 | **0.615** |
| EBGM (MGPS) | 0.53 | **0.615** |
| PRR_train | 0.52 | 0.612 |
| ROR_train | 0.53 | 0.608 |
| ROR_all (uses eval-period data) | 0.48 | 0.605 |
| EB05 | 0.55 | 0.605 |
| IC025 | 0.55 | 0.602 |
| PRR (within eval split) | 0.46 | 0.583 |

All classical methods land at 0.60–0.62. This is the bar for learned models.
EBGM ranks well, but its prior is near-degenerate (see ARCHITECTURE.md); don't report
absolute EBGM values until the prior is fixed. The F1 column in the JSON uses a fixed
threshold on log scores and is meaningless; ignore it.

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
   ≥ 3-reports filter using counts from all years, including test.

## Queue (agreed order)

1. **Reproducible pipeline.** No plan approved yet; start in PLAN mode. Move the split into a script (e.g.
   `pharmviginet/data/build_dataset.py`); make one command run
   collect → audit → clean → split → labels → baselines. Resolve blockers 2–4 here.
2. **LightGBM baseline** on per-pair features: time trends, age/sex mix, outcomes,
   reporter type, indications and the classical scores. Target: beat AUC_strat 0.615.
   Decide the leakage-safe split first (FB-D11 in `DECISIONS.md`).
3. **Fix the EBGM prior fit.** Try fitting without zero truncation, or floor α.
4. **Task B:** FDA SrLC label-change dates. SrLC has no bulk export, so this needs a
   scraper. Most novel part of the benchmark.
5. **Task C:** cold-start split for drugs with few reports. Where ChemBERTa may help.

Deprioritized: PubMedBERT text model and fusion (no narratives in public FAERS, FB-D9).
