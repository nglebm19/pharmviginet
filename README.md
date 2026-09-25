# PharmVigiNet — FAERS-Bench

An open, non-commercial benchmark for adverse drug event signal detection on FDA FAERS.

> **Status: work in progress.** The data pipeline, independent SIDER labels and
> classical disproportionality baselines are done. ML baselines, the label-change
> task, the cold-start split and the public leaderboard are planned.

## Why

- Classical signal scores (ROR, PRR) are often evaluated against labels derived
  from the same scores — a circular setup that inflates results.
- There is no shared, clean, deduplicated, time-split FAERS dataset, so methods
  are hard to compare fairly.
- FAERS-Bench provides that dataset, labels that are independent of the scores
  (SIDER 4.1), and an event-stratified metric.

## What's in the benchmark

- **Data:** FAERS / legacy AERS quarterly files, 2004Q1–2026Q1 (89 quarters).
  Primary-suspect drugs only (`role_cod == "PS"`), deduplicated to the latest
  `caseversion` per `caseid`. 52.7M rows in `master.parquet`.
- **Time split:** train ≤ 2022, val = 2023, test ≥ 2024 (2026Q1 is treated as
  part of the holdout). No random splits — they leak future reports.
- **Unit of evaluation:** one (drug, MedDRA PT) pair with ≥ 3 reports.

## Labels (task A)

FAERS drug names are free text, so both FAERS and SIDER drugs are mapped to
RxNorm ingredients via the NLM RxNav API (76.4% of 33.8K FAERS drug names mapped).

| Label | Rule |
|---|---|
| positive | SIDER 4.1 lists (ingredient, PT) for any of the drug's ingredients |
| negative | every ingredient and the PT are known to SIDER, but the pair is not listed |
| dropped | anything else (drug unmapped / not in SIDER, or PT unknown to SIDER) |

Result: 713K labeled pairs, 29% positive. Negatives rely on a closed-world
assumption and are therefore somewhat noisy.

The older rule `ROR ≥ 2.0 AND lower_CI > 1.0` is circular (ROR scores against
its own formula, AUC 0.94) and is kept only for comparison.

## Metric

Primary metric: **`auc_strat`** — ROC AUC computed within each PT, then a
weighted mean across PTs (`pharmviginet/utils/metrics.py`).

Pooled AUC is misleading here: SIDER positives are dominated by common,
non-specific events (nausea, headache) that have low disproportionality for
every drug, which pushes pooled AUC below chance. Comparing drugs within the
same event is the meaningful question and matches published ROR-vs-SIDER results.

## Baselines

SIDER labels, test split, one score per unique (drug, PT) pair.
`_train` methods use train-period counts only; unseen pairs get neutral scores.

| Model | AUC pooled | **AUC_strat** |
|---|---|---|
| IC (BCPNN) | 0.52 | **0.615** |
| EBGM (MGPS) | 0.53 | **0.615** |
| PRR_train | 0.52 | 0.612 |
| ROR_train | 0.53 | 0.608 |
| EB05 | 0.55 | 0.605 |
| IC025 | 0.55 | 0.602 |
| ROR_all (uses eval-period data) | 0.48 | 0.606 |
| PRR (within eval split) | 0.46 | 0.583 |

All classical methods land at 0.60–0.62 — the bar for learned models.
Note: the MGPS prior fit is near-degenerate; EBGM still ranks well, but its
absolute values should be checked before being reported.

## Reproduce

Needs Python ≥ 3.10 and roughly 21 GB of disk (~3 GB zips + ~18 GB extracted).

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Download + extract all FAERS/AERS quarters
python collect_faers.py --all

# 3. Audit files and schemas → data/logs/audit_report.json
python audit_faers.py

# 4. Clean, deduplicate, normalize, join → data/processed/master.parquet
python clean_faers.py --all

# 5. Time split → data/processed/ml/{train,val,test}.parquet
#    TODO: currently done in a notebook; a script is planned.
#    Rule: train = year <= 2022, val = year == 2023, test = year >= 2024

# 6. Map FAERS drug names to RxNorm ingredients → data/external/rxnorm_map.parquet
python -m pharmviginet.labels.drug_norm

# 7. Download SIDER 4.1 and build known pairs → data/external/sider_pairs.parquet
python -m pharmviginet.labels.sider

# 8. Build labels → data/processed/labels_sider.parquet
python -m pharmviginet.labels.build_labels

# 9. Score baselines → data/logs/baseline_results_sider.json
python -m pharmviginet.models.baseline --labels sider

# 10. Tests
pytest tests/
```

External APIs are rate-limited in code: RxNav ≤ 15 req/s, PubChem ≤ 5 req/s.
RxNav lookups are cached and resumable.

## Repo layout

```
collect_faers.py        # stage 0 — download + extract quarters
audit_faers.py          # stage 1 — file / schema audit
clean_faers.py          # stages 2–5 — clean, dedup, normalize, join
pharmviginet/
├── config.py           # paths and hyperparameters
├── data/               # FAERS loading, cleaning, SMILES lookup
├── labels/             # RxNorm normalization, SIDER pairs, label building
├── models/             # baselines (ROR/PRR/IC/EBGM), text, mol, fusion
├── train/              # training + evaluation loops
└── utils/              # logging, metrics (stratified AUC)
scripts/                # server setup, training runner
tests/
data/                   # gitignored — raw, extracted, processed, external, logs
```

## Roadmap

- [x] Collect all quarters (2004Q1–2026Q1)
- [x] Audit + clean + deduplicate → `master.parquet`
- [x] Independent labels from SIDER 4.1 (task A)
- [x] Classical baselines: ROR, PRR, IC/IC025, EBGM/EB05
- [ ] Split script (replace notebook)
- [ ] FDA label-change dates (SrLC) — early-detection task (task B)
- [ ] Cold-start split — unseen drugs (task C)
- [ ] ML baselines: LightGBM, ChemBERTa, fusion
- [ ] Public release: Hugging Face dataset + leaderboard
- [ ] Write-up / preprint

Public FAERS quarterly files contain no case narratives, so text models only
see templated input (`"<DRUG> caused <PT>"`).

## Licensing and data

- **Code:** MIT — see [LICENSE](LICENSE). Covers code only.
- **FAERS:** U.S. public domain.
- **SIDER 4.1:** CC BY-NC-SA and contains MedDRA terms. It is **not**
  redistributed here; `pharmviginet.labels.sider` downloads it locally, and you
  are responsible for accepting its terms.
- **MedDRA®** is a registered trademark of ICH.

This is a research benchmark. It is not medical advice and not for regulatory use.

Author: Dinh Nguyen Le.

## References

- Rothman KJ, Lanes S, Sacks ST. The reporting odds ratio and its advantages over the proportional reporting ratio. *Pharmacoepidemiol Drug Saf* 2004.
- Bate A et al. A Bayesian neural network method for adverse drug reaction signal generation (BCPNN). *Eur J Clin Pharmacol* 1998.
- DuMouchel W. Bayesian data mining in large frequency tables (MGPS). *Am Stat* 1999.
- Kuhn M, Letunic I, Jensen LJ, Bork P. The SIDER database of drugs and side effects. *Nucleic Acids Res* 2016.
- Gu Y et al. Domain-specific language model pretraining for biomedical NLP (PubMedBERT). arXiv:2007.15779.
- Chithrananda S et al. ChemBERTa. arXiv:2010.09885.
- openFDA drug adverse event API: https://open.fda.gov/apis/drug/event/

## Citation

Preprint forthcoming.
