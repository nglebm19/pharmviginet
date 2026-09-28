# ARCHITECTURE.md — FAERS-Bench

How the pipeline and code are put together. Why things are this way: `DECISIONS.md`.
Status and blockers: `MVP.md`.

## Data flow

```
FDA quarterly zips ─ collect_faers.py ─▶ data/raw/ ─▶ data/extracted/<YYYYQn>/
  ─ audit_faers.py ─▶ data/logs/audit_report.json, schema_map.json
  ─ clean_faers.py
      clean   ─▶ data/processed/{demo,drug,reac,outc,rpsr,ther,indi}.parquet
      dedup   ─▶ cases_deduped.parquet (one row per caseid, FB-D12), drug_ps.parquet (role_cod == PS)
      smiles  ─▶ drug_smiles_map.parquet (PubChem)
      join    ─▶ master.parquet (one row per case × PS drug × PT) + time split (currently fails, see gaps)
  ─ notebook  ─▶ data/processed/ml_notebook_legacy/ (old 2012+ split with ror_train; stale)

master.parquet ─ labels/drug_norm.py ─▶ data/external/rxnorm_map.parquet
SIDER download ─ labels/sider.py     ─▶ data/external/sider_pairs.parquet
both           ─ labels/build_labels.py ─▶ data/processed/labels_sider.parquet
ml/*.parquet + labels ─ models/baseline.py ─▶ data/logs/baseline_results_sider.json   (legacy time split)

master.parquet (year ≤ 2022) + labels ─ data/task_a.py ─▶ data/processed/task_a_pairs.parquet, task_a_folds.parquet
task_a_pairs ─ models/baseline.py --task a ─▶ data/logs/task_a_results.json
```

## Stages

| Stage | Entry point | Output | Status |
|---|---|---|---|
| 0 Collect | `collect_faers.py --all` | 89 quarters, 2004Q1–2026Q1 | Done |
| 1 Audit | `audit_faers.py` | `data/logs/audit_report.json` | Done |
| 2–5 Clean, dedup, SMILES, join | `clean_faers.py --all` | `master.parquet`, 62.7M rows, 2004–2026 | Done (SMILES not rerun for legacy names) |
| 6 Split | `clean_faers.py --stage join` | `ml/{train,val,test}.parquet` | **Missing**: writer killed for memory |
| 7 Drug normalization | `python -m pharmviginet.labels.drug_norm` | `rxnorm_map.parquet` | Done |
| 8 SIDER pairs | `python -m pharmviginet.labels.sider` | `sider_pairs.parquet` | Done |
| 9 Labels | `python -m pharmviginet.labels.build_labels` | `labels_sider.parquet` | Done |
| 10 Baselines, legacy time split | `python -m pharmviginet.models.baseline --labels sider` | `baseline_results_sider.json` | Done; not Task A |
| 11 Task A pairs + folds | `python -m pharmviginet.data.task_a` | `task_a_pairs.parquet`, `task_a_folds.parquet` | Done |
| 12 Task A scoring | `python -m pharmviginet.models.baseline --task a` | `task_a_results.json` | Done |
| — Text / mol models | `python -m pharmviginet.train.train` | `model_checkpoints/*.pt` | Paused, still on ROR labels (text: 1 epoch on 10K rows, AUC 0.64) |

`scripts/run_pipeline.sh` runs only the text/mol training step.

## Module map

```
collect_faers.py, audit_faers.py, clean_faers.py   # stages 0–5 (clean_faers.py hardcodes its own paths)
pharmviginet/
├── config.py                 # paths and model hyperparameters
├── data/faers.py             # PyTorch dataset: text_input, smiles, label
├── data/smiles.py            # PyTorch dataset: smiles, label
├── data/clean.py             # empty
├── data/task_a.py            # Task A: counts up to 2022, eligibility, ingredient-disjoint folds, fold quality, scores
├── labels/drug_norm.py       # RxNav client (≤ 15 req/s, retries), resumable cache
├── labels/sider.py           # SIDER download, PT rows, drug → ingredient (RxNav, PubChem fallback)
├── labels/build_labels.py    # SIDER labels for FAERS pairs with ≥ 3 reports
├── models/disproportionality.py  # pair counts, PRR, BCPNN, MGPS
├── models/baseline.py        # classical methods: legacy time split, or --task a
├── models/drug_control.py    # Task A drug-level-only control (out-of-fold, sklearn)
├── models/text.py, mol.py    # PubMedBERT / ChemBERTa fine-tuning
├── models/fusion.py          # empty
├── train/train.py, evaluate.py   # runs text then mol; evaluates on 100K-row samples
├── utils/metrics.py          # AUC, AUPRC, F1, stratified_auc, fold_metrics
└── utils/logging.py          # empty
tests/                        # disproportionality, labels, metrics, task_a
```

## Labels (task A)

For each FAERS (drugname, PT) pair with ≥ 3 reports:

- `drug_norm` maps the FAERS name to RxNorm ingredients (`approximateTerm` → `related?tty=IN`).
- `sider` maps SIDER drugs the same way, falling back to the PubChem title for truncated
  names, and keeps single-ingredient drugs and `meddra_type == "PT"` rows.
- `build_labels` sets **1** if SIDER lists (ingredient, PT) for any ingredient, **0** if
  every ingredient and the PT are in SIDER but the pair is not listed, and drops the rest.
  PTs are matched lowercase.

## Task A folds

`data/task_a.py`, all counts from reports with `year <= FEATURE_CUTOFF_YEAR` (2022):

1. Pair counts and scores up to the cutoff; MGPS prior fitted on all 5.5M pairs.
2. Eligible = ≥ 3 distinct reports up to the cutoff, inner-joined with `labels_sider`
   (`drugname, pt, ingredients, label` only).
3. Ingredients are shuffled (seed 0) and each goes to the fold with the fewest pairs
   so far. A pair takes its ingredients' fold; pairs whose ingredients span folds are dropped.
4. Fold quality is checked before scoring; the run stops if the largest/smallest fold
   ratio > 1.5, a fold's positive rate is > 5 pp from the overall rate, a fold has
   < 500 usable PT strata, or one ingredient holds > 10% of a fold's pairs.
5. Drug-level features for the control: `log_n_drug`, `n_pt`, `first_year`, `n_ingredients`.

## Scoring

Per pair, from distinct reports: `a` (drug and event), `n_drug`, `n_reac`, `N`,
`E = n_drug · n_reac / N`.

- **ROR:** `a·d / ((b+0.5)(c+0.5))`. Task A uses `disproportionality.ror` on counts up
  to 2022. Legacy columns: `ror` (`clean_faers.compute_ror`, all of `master.parquet`,
  eval years included) and `ror_train` (notebook; cannot be reproduced from `train.parquet`).
- **PRR:** `(a / n_drug) / ((n_reac − a) / (N − n_drug))`.
- **BCPNN:** `IC = log2((a+0.5)/(E+0.5))`; IC025 by Norén's closed-form approximation.
- **MGPS:** a two-gamma mixture prior fitted by maximum marginal likelihood on a
  zero-truncated negative binomial, with pairs grouped on (a, E rounded to 3 significant
  digits). EBGM is `exp(E[log λ | a])`; EB05 comes from bisection on the mixture CDF.
  **Known issue:** the fit is near-degenerate (5.5M pairs up to 2022: p ≈ 0.95, α₁ ≈ 2.4e-4).
- Legacy `_train` variants use train-period counts; unseen pairs get IC 0, EBGM 1, PRR 1.
  Task A pairs all have ≥ 3 reports up to the cutoff, so no neutral fill is needed;
  PRR is undefined when `n_reac == a` (0 pairs in the current build) and is then set to
  the largest finite PRR.

## Metric

`utils/metrics.stratified_auc`: ROC AUC within each PT, skipping PTs with < 20 pairs
or one class, averaged weighted by PT size. Reported as `auc_strat` with `n_strata`.

`utils/metrics.fold_metrics`: Task A scores every method on the same rows of each fold,
then reports `auc_strat` mean and sample sd over folds, plus the per-fold paired
difference from a reference method (IC).

## Known data gaps

- No script-built `ml/` split: the in-memory split writer in the join stage is killed at
  the rebuilt size (`MVP.md`, blocker 1).
- `drug_smiles_map` was not rebuilt after FB-D12, so legacy-only drug names get `[UNK-MOL]`.
- Non-key schema differences between eras remain (e.g. legacy `gndr_cod` vs modern `sex`).
- The `label` column in `master.parquet` is the old ROR rule and does not match
  `ror` (`MVP.md`, blocker 3). Text/mol datasets still train on it.
- `text_input` is a template, not a narrative.
- ~15% of drugs have no SMILES (`[UNK-MOL]`).

## Planned components (not designed yet)

- Split script replacing the notebook; one-command pipeline.
- LightGBM on per-pair features (time trends, demographics, outcomes, reporter type,
  indications, classical scores).
- Task B: SrLC label-change scraper and a lead-time metric.
- Task C: report-sparse drugs (few or no reports up to the cutoff).
