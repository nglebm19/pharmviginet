# DECISIONS.md — FAERS-Bench

Project-level decisions. Workspace-level decisions live in `../DECISIONS.md`
(D1, D2, …); project entries use the `FB-D` prefix to avoid clashes.
Don't relitigate an entry without an explicit reason. Add new entries at the bottom.

## FB-D1 — Non-commercial open benchmark (2026-09-24)
FAERS-Bench is an open research benchmark and portfolio project, not a product.
Code is MIT. **Why:** the value is a fair, reproducible comparison of methods;
SIDER's NC license also rules out commercial use of the labels. (`7937704`, `621f12f`)

## FB-D2 — SIDER 4.1 labels replace the ROR-derived label (2026-09-24)
Positive if SIDER lists (ingredient, PT); negative if every ingredient and the PT
are known to SIDER but the pair is unlisted; otherwise dropped.
**Why:** the old rule `ROR ≥ 2.0 AND lower_CI > 1.0` is circular — ROR baselines
were scored against their own formula. The old label is kept only for comparison. (`0f22503`)

## FB-D3 — Closed-world negatives accepted as documented noise (2026-09-24)
Unlisted pairs count as negatives even though some are true ADEs
(e.g. OxyContin / overdose = 0) and some high-ROR pairs are reporting artifacts
(e.g. lawsuit-driven Zantac cancer reports, labeled 0).
**Why:** no complete negative set exists; the noise is disclosed in the write-up.

## FB-D4 — Drugs normalized to RxNorm ingredients via RxNav (2026-09-24)
FAERS and SIDER names both map through `approximateTerm` → `related?tty=IN`.
Only single-ingredient SIDER drugs are used; a FAERS product matches if any of its
ingredients does. **Why:** FAERS drug names are free text; SIDER is keyed by compound.
Single-ingredient SIDER drugs keep each pair's ingredient unambiguous.

## FB-D5 — Time split: train ≤ 2022, val 2023, test ≥ 2024 (2026-09-24)
2026Q1 is part of the test holdout. No random splits.
**Why:** random splits leak future reports into training.

## FB-D6 — Score one row per unique (drug, PT) pair (2026-09-25)
**Why:** per-report scoring weights common pairs by report volume and repeats the
same score thousands of times. (`3192265`)

## FB-D7 — Primary metric is event-stratified AUC (`auc_strat`) (2026-09-25)
AUC within each PT (≥ 20 pairs, both classes), weighted mean by PT size.
**Why:** SIDER positives are dominated by common, non-specific events (nausea,
headache) with low disproportionality for every drug, which pushes pooled AUC
below chance. Comparing drugs within one event matches published ROR-vs-SIDER work. (`3192265`)

## FB-D8 — Classical `_train` baselines use train-period counts only (2026-09-25)
Pairs unseen in train get neutral scores (IC 0, EBGM 1, PRR 1).
**Why:** a fair baseline cannot use eval-period data. `ROR_all` and within-split
PRR are kept only as references. (`539ad14`)

## FB-D9 — Text model deprioritized (2026-09-25)
**Why:** public FAERS quarterly files have no narrative text; `text_input` is a
template (`"<DRUG> caused <PT>"`). A language model would mostly recall literature
knowledge rather than learn from reports.

## FB-D10 — Repo stays private until v0; SIDER never redistributed (2026-09-25)
Nothing under `data/` is committed. Users rebuild labels locally with the scripts.
**Why:** v0 criteria in `MVP.md` are not met; SIDER's CC BY-NC-SA / MedDRA terms.

## FB-D11 — Task A uses ingredient-disjoint folds (resolved 2026-09-27)
**Decision.** Task A = SIDER labels + features from reports up to 2022 only +
5 ingredient-disjoint folds grouped by RxNorm ingredient.
- Eligibility: ≥ 3 distinct reports up to 2022, computed in `data/task_a.py`.
  From `labels_sider` only `drugname`, `pt`, `ingredients` and `label` are used.
- Each ingredient belongs to one fold. For fold k, its ingredients are
  **label-held-out ingredients**: their FAERS features are available, their SIDER
  labels are not used for training. PTs appear in every fold, as they should.
  Combination products whose ingredients span folds are dropped (4.5% of pairs).
- Primary metric: per-fold `auc_strat`, reported as mean ± sd, plus the paired
  per-fold difference from IC. All methods are scored on the same fold partition.
- Required rows: classical scores (computed from reports up to 2022, no labels) and
  a drug-level-only control (`models/drug_control.py`).
- Evaluation over time belongs to Task B (dated FDA label changes); report-sparse
  drugs belong to Task C. Task A has no ≥ 2024 filter on its eval set.

**Why.** Measured on the old time split (309,522 test pairs):
- 94.5% of test (ingredients, PT) labels were already present among train-period
  labeled pairs; a label lookup scored `auc_strat` 0.999.
- The drug's mean train label alone (leaving out the pair) scored 0.747, above every
  classical method, with no pair-level signal.
- Requiring eval pairs to appear in ≥ 2024 data kept 199K of 587K eligible pairs and
  986 of 1,236 drug sets, raised the positive rate 31.2% → 33.9% and the median count
  6 → 10. Within each train-volume band, survival did not depend on the label. It added
  no temporal information (SIDER labels are static and mostly pre-2015) and only cut power.

**Removed from the Task A leaderboard** (checked against code and data):
- `ROR_all`: `clean_faers.compute_ror` over all of `master.parquet`, 2023–2026Q1 included
  (reproduced exactly).
- Within-split `PRR`: `baseline.load_split` computes it on the eval split itself.
- `ror_train`: built in the notebook; missing exactly where a pair is absent from
  train, but only 2.8% of values match ROR recomputed from `train.parquet`. Replaced
  by `disproportionality.ror` computed in code from reports up to 2022.

**Open limits.** "Up to 2022" is currently 2012–2022 (legacy rows missing, MVP blocker 2).
Classical scores still benefit from reporting that follows labeling, equally for all
methods. Revisit this decision if a drug-level-only model reaches the classical level,
if gains concentrate on drugs with same-class neighbours in training (consider
ATC-grouped folds), or if fold-to-fold sd hides method differences.
