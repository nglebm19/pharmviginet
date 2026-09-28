# PharmVigiNet — FAERS-Bench

An open, non-commercial benchmark for adverse drug event signal detection on FDA FAERS.

> **Status: work in progress.** The data pipeline, independent SIDER labels and
> classical disproportionality baselines are done. ML baselines, the label-change
> task, the report-sparse drug task and the public leaderboard are planned.

## Why

- Classical signal scores (ROR, PRR) are often evaluated against labels derived
  from the same scores — a circular setup that inflates results.
- There is no shared, clean, deduplicated, time-split FAERS dataset, so methods
  are hard to compare fairly.
- FAERS-Bench provides that dataset, labels that are independent of the scores
  (SIDER 4.1), and an event-stratified metric.

## What's in the benchmark

- **Data:** FAERS / legacy AERS quarterly files, 2004Q1–2026Q1 (89 quarters
  collected). Primary-suspect drugs only (`role_cod == "PS"`), deduplicated to the
  latest version per `caseid`. 62.7M rows in `master.parquet`, 2004–2026.
- **Task A split:** features from reports up to 2022 only; 5 folds grouped by RxNorm
  ingredient (label-held-out ingredients). Evaluation over time is Task B.
- **Time split** (legacy reference and Task B): train ≤ 2022, val = 2023,
  test ≥ 2024 (2026Q1 is treated as part of the holdout). No random splits.
- **Unit of evaluation:** one (drug, MedDRA PT) pair with ≥ 3 reports up to 2022.

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

The older rule `ROR ≥ 2.0 AND lower_CI > 1.0` is circular (ROR is scored against
labels derived from ROR) and is kept only for comparison.

## Metric

Primary metric: **`auc_strat`** — ROC AUC computed within each PT, then a
weighted mean across PTs (`pharmviginet/utils/metrics.py`). For Task A it is
computed per fold and reported as mean ± sd over the 5 folds.

Pooled AUC is misleading here: SIDER positives are dominated by common,
non-specific events (nausea, headache) that have low disproportionality for
every drug, which pushes pooled AUC below chance. Comparing drugs within the
same event is the meaningful question and matches published ROR-vs-SIDER results.

## Baselines (task A)

SIDER labels; pairs with ≥ 3 reports up to 2022; all scores computed from reports up
to 2022 only; 5 ingredient-disjoint folds, so a learned model never trains on the
SIDER labels of the ingredients it is scored on. 715,034 pairs.

| Method | AUC_strat, mean ± sd over folds |
|---|---|
| EB05 | 0.620 ± 0.005 |
| IC (BCPNN) | 0.614 ± 0.012 |
| ROR | 0.602 ± 0.020 |
| Drug-level-only control | 0.554 ± 0.021 |

All classical methods land at about 0.61–0.62. Full table, paired differences and
fold checks: [MVP.md](MVP.md). Design: FB-D11 in [DECISIONS.md](DECISIONS.md).

## Known limitations

- **No train/val/test split files:** the split writer runs out of memory at the
  current data size and is being replaced. Task A does not need them.
- **Label noise:** SIDER negatives rely on a closed-world assumption.
- **EBGM prior:** the MGPS prior fit is near-degenerate; EBGM still ranks well,
  but its absolute values should not be reported until the fit is fixed.
- **No narratives:** public FAERS has no case text, so text models only see
  templated input.

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
#    TODO: currently done in a notebook; a script is planned (see MVP.md).
#    Rule: train = year <= 2022, val = year == 2023, test = year >= 2024

# 6. Map FAERS drug names to RxNorm ingredients → data/external/rxnorm_map.parquet
python -m pharmviginet.labels.drug_norm

# 7. Download SIDER 4.1 and build known pairs → data/external/sider_pairs.parquet
python -m pharmviginet.labels.sider

# 8. Build labels → data/processed/labels_sider.parquet
python -m pharmviginet.labels.build_labels

# 9. Task A pairs + ingredient-disjoint folds → data/processed/task_a_*.parquet
python -m pharmviginet.data.task_a

# 10. Score Task A baselines → data/logs/task_a_results.json
python -m pharmviginet.models.baseline --task a

# (legacy time-split reference → data/logs/baseline_results_sider.json)
python -m pharmviginet.models.baseline --labels sider

# 11. Tests
pytest tests/
```

External APIs are rate-limited in code: RxNav ≤ 15 req/s, PubChem ≤ 5 req/s.
RxNav lookups are cached and resumable.

## Project docs

- [PRODUCT.md](PRODUCT.md) — purpose, users, benchmark tasks, scope, non-goals
- [MVP.md](MVP.md) — v0 definition, status, full leaderboard, blockers, roadmap
- [ARCHITECTURE.md](ARCHITECTURE.md) — data flow, stages, module map, scoring
- [DECISIONS.md](DECISIONS.md) — design decisions and open questions

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
