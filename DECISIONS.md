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

## FB-D11 — OPEN: leakage-safe split for learned models
SIDER labels are static, so a model trained on train-period pairs sees most test
pairs, with their labels, during training. The time split alone does not prevent
this. Options: drug-disjoint (overlaps task C), pair-disjoint, or train only on pairs
first seen before a cutoff and test on pairs first seen after it.
Must be decided before the LightGBM baseline.
