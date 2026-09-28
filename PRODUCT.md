# PRODUCT.md — FAERS-Bench

What this benchmark is for, who it serves and what it deliberately leaves out.
Status and next work: `MVP.md`. How the code works: `ARCHITECTURE.md`.

## Purpose

FAERS-Bench is an open, non-commercial research benchmark for adverse drug
event (ADE) signal detection on the FDA Adverse Event Reporting System (FAERS).

It addresses three problems in published signal-detection work:

1. **Circular evaluation.** Disproportionality scores (ROR, PRR) are often
   scored against labels derived from the same scores, which inflates results.
2. **No shared dataset.** FAERS is raw, duplicated and split across two schemas,
   so every study rebuilds it differently and results are not comparable.
3. **Misleading metrics.** Pooled AUC rewards predicting which events are common,
   not which drugs cause them.

The benchmark provides a deduplicated, time-split FAERS dataset, labels that are
independent of the scores being evaluated, an event-stratified metric and
reproducible classical baselines.

## Intended users

- Pharmacovigilance and drug-safety researchers comparing signal-detection methods.
- ML researchers who want a leakage-aware benchmark on real post-marketing data.
- Reviewers of the author's work: this is also a portfolio and write-up project
  (solo builder, correctness over speed).

## Benchmark tasks

| Task | Question | Labels | Status |
|---|---|---|---|
| A — Known ADE ranking | Within one event (MedDRA PT), which drugs are known to cause it, for ingredients whose labels were held out? | SIDER 4.1, ingredient-disjoint folds | Built |
| B — Early detection | How early does a method flag a pair before the FDA adds it to the drug label? | FDA SrLC label-change dates | Planned |
| C — Report-sparse drugs | Can a method score drugs with few or no reports up to the cutoff? | SIDER 4.1 | Planned |

Unit of evaluation: one (drug, PT) pair. Primary metric: `auc_strat`, the ROC AUC
computed within each PT and averaged weighted by PT size.

## Scope

- FAERS and legacy AERS public quarterly files, 2004Q1–2026Q1 (target coverage;
  the current build only has 2012 onward, see `MVP.md`).
- Primary-suspect drugs only, latest case version only.
- Classical baselines (ROR, PRR, BCPNN/IC, MGPS/EBGM) and learned baselines
  (LightGBM on per-pair features first; molecular models for task C).
- Code that rebuilds every artifact locally, including labels.

## Non-goals

- Not a commercial product, service or regulatory tool. Outputs are not medical advice.
- No causal claims about any drug; the benchmark ranks reporting signals.
- No modeling of case narratives: public FAERS files contain none.
- No redistribution of SIDER or MedDRA content.

## Licensing

- **Code:** MIT (`LICENSE`). Covers code only, not data or third-party labels.
- **FAERS:** U.S. public domain.
- **SIDER 4.1:** CC BY-NC-SA and contains MedDRA terms. Never commit or redistribute
  SIDER files or labels derived from them; ship scripts that rebuild labels locally.
- **MedDRA®** is a registered trademark of ICH.
