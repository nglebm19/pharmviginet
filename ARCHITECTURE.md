# ARCHITECTURE.md — FAERS-Bench

How the pipeline and code are put together. Why things are this way: `DECISIONS.md`.
Status and blockers: `MVP.md`.

## Data flow

```
FDA quarterly zips ─ collect_faers.py ─▶ data/raw/ ─▶ data/extracted/<YYYYQn>/
  ─ audit_faers.py ─▶ data/logs/audit_report.json, schema_map.json
  ─ clean_faers.py
      clean   ─▶ data/processed/{demo,drug,reac,outc,rpsr,ther,indi}.parquet
      dedup   ─▶ cases_deduped.parquet (latest caseversion per caseid), drug_ps.parquet (role_cod == PS)
      smiles  ─▶ drug_smiles_map.parquet (PubChem)
      join    ─▶ master.parquet (one row per case × PS drug × PT) + plain time split
  ─ notebook  ─▶ data/processed/ml/{train,val,test}.parquet (adds ror_train; no script yet)

master.parquet ─ labels/drug_norm.py ─▶ data/external/rxnorm_map.parquet
SIDER download ─ labels/sider.py     ─▶ data/external/sider_pairs.parquet
both           ─ labels/build_labels.py ─▶ data/processed/labels_sider.parquet
ml/*.parquet + labels ─ models/baseline.py ─▶ data/logs/baseline_results_sider.json
```

## Stages

| Stage | Entry point | Output | Status |
|---|---|---|---|
| 0 Collect | `collect_faers.py --all` | 89 quarters, 2004Q1–2026Q1 | Done |
| 1 Audit | `audit_faers.py` | `data/logs/audit_report.json` | Done |
| 2–5 Clean, dedup, SMILES, join | `clean_faers.py --all` | `master.parquet`, 52.7M rows | Done, 2012+ only |
| 6 Split | notebook | `ml/{train,val,test}.parquet` | Not scripted |
| 7 Drug normalization | `python -m pharmviginet.labels.drug_norm` | `rxnorm_map.parquet` | Done |
| 8 SIDER pairs | `python -m pharmviginet.labels.sider` | `sider_pairs.parquet` | Done |
| 9 Labels | `python -m pharmviginet.labels.build_labels` | `labels_sider.parquet` | Done |
| 10 Baselines | `python -m pharmviginet.models.baseline --labels sider` | `baseline_results_sider.json` | Done |
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
├── labels/drug_norm.py       # RxNav client (≤ 15 req/s, retries), resumable cache
├── labels/sider.py           # SIDER download, PT rows, drug → ingredient (RxNav, PubChem fallback)
├── labels/build_labels.py    # SIDER labels for FAERS pairs with ≥ 3 reports
├── models/disproportionality.py  # pair counts, PRR, BCPNN, MGPS
├── models/baseline.py        # scores and evaluates all classical methods
├── models/text.py, mol.py    # PubMedBERT / ChemBERTa fine-tuning
├── models/fusion.py          # empty
├── train/train.py, evaluate.py   # runs text then mol; evaluates on 100K-row samples
├── utils/metrics.py          # AUC, AUPRC, F1, stratified_auc
└── utils/logging.py          # empty
tests/                        # disproportionality, labels, metrics
```

## Labels (task A)

For each FAERS (drugname, PT) pair with ≥ 3 reports:

- `drug_norm` maps the FAERS name to RxNorm ingredients (`approximateTerm` → `related?tty=IN`).
- `sider` maps SIDER drugs the same way, falling back to the PubChem title for truncated
  names, and keeps single-ingredient drugs and `meddra_type == "PT"` rows.
- `build_labels` sets **1** if SIDER lists (ingredient, PT) for any ingredient, **0** if
  every ingredient and the PT are in SIDER but the pair is not listed, and drops the rest.
  PTs are matched lowercase.

## Scoring

Per pair, from distinct reports: `a` (drug and event), `n_drug`, `n_reac`, `N`,
`E = n_drug · n_reac / N`.

- **ROR:** `a·d / ((b+0.5)(c+0.5))`, computed in `clean_faers.compute_ror` over all of
  `master.parquet` (`ror`), and in the notebook over train only (`ror_train`).
- **PRR:** `(a / n_drug) / ((n_reac − a) / (N − n_drug))`.
- **BCPNN:** `IC = log2((a+0.5)/(E+0.5))`; IC025 by Norén's closed-form approximation.
- **MGPS:** a two-gamma mixture prior fitted by maximum marginal likelihood on a
  zero-truncated negative binomial, with pairs grouped on (a, E rounded to 3 significant
  digits). EBGM is `exp(E[log λ | a])`; EB05 comes from bisection on the mixture CDF.
  **Known issue:** on 4.3M train pairs the fit is near-degenerate (p ≈ 0.99, α ≈ 3e-5).
- `_train` variants use train-period counts; unseen pairs get IC 0, EBGM 1, PRR 1.

## Metric

`utils/metrics.stratified_auc`: ROC AUC within each PT, skipping PTs with < 20 pairs
or one class, averaged weighted by PT size. Reported as `auc_strat` with `n_strata`.

## Known data gaps

- Legacy AERS rows (2004–2011) are lost at the dedup stage (`MVP.md`, blocker 2).
- `master.parquet` keeps uppercase legacy columns next to the modern ones, unmerged.
- The `label` column in `master.parquet` is the old ROR rule and does not match
  `ror` (`MVP.md`, blocker 3). Text/mol datasets still train on it.
- `text_input` is a template, not a narrative.
- ~15% of drugs have no SMILES (`[UNK-MOL]`).

## Planned components (not designed yet)

- Split script replacing the notebook; one-command pipeline.
- LightGBM on per-pair features (time trends, demographics, outcomes, reporter type,
  indications, classical scores).
- Task B: SrLC label-change scraper and a lead-time metric.
- Task C: drug-disjoint cold-start split.
