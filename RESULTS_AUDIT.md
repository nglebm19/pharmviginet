# Results audit — 2026-10-28

Canonical evaluation is FB-D11/FB-D12: SIDER 4.1, features from 2004–2022,
≥ 3 distinct reports, 5 ingredient-disjoint folds, seed 0. Within each fold,
`utils.metrics.stratified_auc` uses PT strata with ≥ 20 pairs and both classes,
weighted by stratum pair count. `fold_metrics` reports the equal-weight fold mean
and sample SD (`ddof=1`), including paired differences against IC.

The machine-readable [manifest](results_manifest.json) records exact values,
per-fold metrics, verification scope, environment, UTC modification times,
SHA-256 hashes and relative paths. It contains aggregate results, not SIDER labels.
The checkout audited was `60b1a28`; **this is not an assertion that this commit
produced the original artifacts**. Those runs did not save commit IDs, source
hashes or dependency versions. Current source hashes identify the replay code.

## Claims and generating code

| Claim | Before | Verified value / disposition | Artifact and generating code |
|---|---|---|---|
| RxNorm name coverage | README 76.4% / 33.8K; MVP 74.5% / 43,099 | 32,090 / 43,099 = 74.4565%; display 74.5% | `data/external/rxnorm_map.parquet`; `labels/drug_norm.py:faers_drugnames`, `normalize_names`, `main` |
| All-years SIDER labels | README 713K / 29%; MVP 860,422 / 28.2% | 860,422; 242,296 positive; 11,732 drug names | `data/processed/labels_sider.parquet`; `labels/build_labels.py:faers_pairs`, `label_pairs` |
| Task A evaluation pairs | 715,034 | 715,034; positive fraction 0.2888394118321646 | `data/processed/task_a_pairs.parquet`; `data/task_a.py:eligible_pairs`, `assign_folds`, `pair_folds` |
| Task A ingredients | 1,138 described alongside evaluated pairs | 1,138 assigned folds; 1,032 retained in evaluated pairs | `task_a_folds.parquet`, `task_a_pairs.parquet`; `assign_folds`, `pair_folds` |
| SIDER reference pairs | 134,844 / 1,264 ingredients | Unchanged | `data/external/sider_pairs.parquet`; `labels/sider.py:main` (PT-only, single-ingredient SIDER drugs) |
| ROR AUC_strat | README 0.602 ± 0.020; MVP 0.610 ± 0.012 | 0.6102111609794475 ± 0.012335699606735185 | `data/logs/task_a_results.json`; `models/disproportionality.py:ror`, `models/baseline.py:evaluate_task_a`, `utils/metrics.py:fold_metrics` |
| PRR AUC_strat | 0.610 ± 0.012 | Unchanged | Same result JSON; `disproportionality.prr` |
| IC / IC025 AUC_strat | 0.614 ± 0.012 / 0.619 ± 0.004 | Unchanged | Same result JSON; `disproportionality.bcpnn` |
| EBGM / EB05 AUC_strat | 0.615 ± 0.011 / 0.620 ± 0.005 | Unchanged | Same result JSON; `disproportionality.ebgm`, `task_a_pairs.prior.json` |
| Drug-only control AUC_strat | 0.554 ± 0.021 | Unchanged | Same result JSON; `models/drug_control.py:oof_scores` |
| ROR / PRR paired difference vs IC | −0.004 ± 0.001 | −0.003 ± 0.001 (each, rounded independently from full precision) | Same result JSON, `summary.*.diff_vs_ref_mean` |
| Master size and eras | 62,732,870; 52,728,487 modern + 10,004,383 legacy | Unchanged | `master.parquet`; `clean_faers.py:stage_join`; bounded scan of `year`, `era`, `quarter`, `role_cod` |
| Deduplicated cases | 20,328,567 | Row count confirmed; uniqueness/dedup selection not independently rerun | `cases_deduped.parquet`; `clean_faers.py:dedup_demo`, `stage_dedup` |
| Train / val / test rows | 46,589,011 / 4,237,245 / 11,906,614 | Unchanged; split metadata checked and master year totals matched | `ml/*.parquet`, `ml/split_manifest.json`; `clean_faers.py:write_splits`, `check_split` |
| Quarters | 89, 2004Q1–2026Q1 | Historical collection/audit logs report 89; processed coverage checked | `download_log.json`, `audit_report.json`, master quarter scan; `collect_faers.py`, `audit_faers.py` |

Module paths in this table are relative to `pharmviginet/` unless they name a
root-level script. Exact leaderboard values and fold-quality aggregates are in
the manifest; the rounded tables in README and MVP are checked by tests.

## Why the numbers differed

**RxNorm: changed dataset, stale README.** `drug_norm.log` explicitly reports
33,802 names with ≥ 10 master rows and 76.4% mapped. It even retains the earlier
repository path. Commit `95154af` documents those values before the legacy rebuild.
Commit `1fd021a` updates MVP to 43,099 / 74.5% but leaves README unchanged. The
mapping denominator is joined master rows, not distinct reports, all unique FAERS
names, or Task A pairs. The audit checks that the current denominator equals all
names with ≥ 10 rows and that mapping ingredients agree with the current cache.
Restricting the current master to modern FAERS recovers exactly 33,802 candidate
names, independently corroborating the historical denominator.
The cache is keyed by cleaned name (40,590 entries), whereas coverage is measured
over original names (43,099). These denominators must not be interchanged.
No evidence supports changing the ≥ 10-row filter. The old mapping snapshot and
RxNav release were not preserved, so the historical numerator and any individual
mapping changes cannot be independently reconstructed from the old log alone.

**ROR: historical run left in a current table.** `git diff 95154af 1fd021a -- README.md`
shows that the FB-D12 update changed the pair count, EB05, IC and control rows,
but skipped ROR. The old 0.602 ± 0.020 belonged to 560,334 Task A pairs with
2012–2022 features. Current 0.610 ± 0.012 is supported by local artifacts and
replayed evaluation. This is a data-recovery change, not evidence of a model
improvement, a changed statistic, or a reason to select the larger score.

**713K versus 715,034: different populations as well as different runs.**
713K abbreviates the historical 713,399 **all-years SIDER label pool**, recorded
in `95154af:MVP.md` and FB-D12. The corresponding old Task A set had 560,334 pairs.
After recovering legacy AERS, the label pool is 860,422 pairs. Task A applies
its own cutoff counts (746,362 eligible pairs) and drops 31,328 combinations
spanning folds, giving 715,034, with 206,530 positives. Although 1,138 ingredients
are assigned folds, only 1,032 occur in the final evaluated pairs; the remaining
ingredients have no retained pairs after combination filtering.
All-years `labels_sider.n_reports` is never used for Task A eligibility.

## Provenance and historical artifacts

- Current master mtime: `2026-09-28T04:18:50.881999+00:00`; mapping:
  `04:55:14.025530+00:00`; labels: `04:55:41.631211+00:00`; Task A pairs:
  `04:56:49.807503+00:00`; results: `04:57:55.483025+00:00` on the same UTC date.
  This sequence is consistent with the FB-D12 rebuild on September 27 Pacific
  time. It does not prove the original source revision. Reconstructed content
  and replay checks provide stronger evidence than timestamps.
- `SIDER_PAIRS` mtime is September 25 UTC and predates FB-D12, as expected for
  an independent label reference. Stored labels identify `source=sider4.1`.
  Local SIDER inputs and both RxNav caches are fingerprinted. Neither upstream
  source release hashes nor per-lookup timestamps were recorded at acquisition.
- The split manifest records `created=2026-09-27T22:56:15` without a timezone.
  It records source path, schema, rows and yearly coverage, but no source hash.
  This audit adds hashes; it does not retroactively authenticate its creation.
- `audit_report.json` is dated May 29 and `validation_report.json` May 27.
  The latter contains legacy missing-file warnings and is stale for the rebuilt
  pipeline. Neither certifies September cleaning. Raw/extracted files were not read.
- `baseline_results_sider.json` is the September 28 UTC legacy time-split output:
  327,514 test pairs, IC AUC_strat 0.6275958586352781. This is a stored reference,
  not a replayed Task A score. `baseline_results_sider_notebook_2012.json` is the
  September 25 historical notebook output: 309,522 test pairs, IC
  0.615449278171789 and EBGM 0.6150186110202545. Keep them separate from Task A.
- `baseline_results.json` predates FB-D12; its test pooled ROR_all AUC 0.7871971129566836
  and notebook ROR_train AUC 0.8349333249486245 do not establish a bug in current
  Task A. The circular label also applies a confidence-bound threshold; ROR alone
  need not yield perfect AUC. No evaluation code was changed during this audit.
- `text_results.json` is a May 31 one-epoch, 10,000-row legacy output, without
  Task A configuration. It does not establish a trained Task A text baseline.
- Historical pre-FB-D12 Task A scores remain documented in `95154af`; their
  original pair/result files are not available as separate local snapshots.
  Timing and memory claims in MVP/FB-D13 remain historical measurements, not
  measurements from this audit. Private/public repository status was not checked remotely.

## Reproduce the verification

From the repository root with the existing Python environment and local data:

```bash
# Read-only artifact checks, metric replay (including control), coverage and hashes.
OMP_NUM_THREADS=2 .venv/bin/python -m scripts.audit_results --output /tmp/results.json

# Also reconstruct cutoff counts, eligibility, folds and scores from master.
# Uses the saved MGPS prior; does not refit it, call APIs or overwrite data artifacts.
OMP_NUM_THREADS=2 .venv/bin/python -m scripts.audit_results --counts --output /tmp/results-full.json

OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests
git diff --check
```

To regenerate Task A artifacts intentionally from the existing master and labels:

```bash
.venv/bin/python -m pharmviginet.data.task_a
.venv/bin/python -m pharmviginet.models.baseline --task a
```

These last commands overwrite Task A outputs and refit the prior; they were not
used for this audit. Upstream rebuild commands are in README. Preserve the
fingerprinted inputs and caches before rebuilding; live RxNav responses can change.

Executed during this audit: the full `--counts` verification above passed,
including replay of all seven methods' per-fold metrics, deterministic fold
reconstruction, stored-prior score reconstruction, mapping/cache checks and
bounded master coverage scans. The original suite passed 35 tests; with three
new consistency tests, `pytest -q tests` passed **38 tests**. `git diff --check`
passed. No downloads, API calls, full pipeline reruns, commits or pushes occurred.

## Addendum — 2026-11-10 re-check and leakage claim

Re-run on the same checkout (`60b1a28`, docs uncommitted): `pytest -q tests` passed
38 tests; `python -m scripts.audit_results` (without `--counts`) passed, its replayed
results equal the manifest exactly, and all artifact SHA-256 hashes match the manifest.

**Time-split leakage (FB-D11).** The historical figures — 94.5% of test
(ingredients, PT) labels already present in the train period, label lookup
`auc_strat` 0.999, on 309,522 pairs from the 2012+ notebook split — are
**unverified**: no script or log was kept and their inputs were overwritten.
`scripts/leakage_check.py` measures the same quantity on the current script-built
split (read-only, ~17 s, ~2.2 GB peak RSS):

| Quantity | Value |
|---|---|
| Test pairs (same rule as `baseline.load_split`, SIDER labels) | 327,514 |
| Test keys already labeled in train period | 0.9595925670353023 |
| Label-lookup `auc_strat` (unseen keys scored at train positive rate) | 0.9994607647569387 (1,626 strata) |
| Same, shared keys only | 1.0 (1,569 strata) |

The test-pair count equals the stored `baseline_results_sider.json` run. A SIDER
label is a static function of (ingredients, PT), so a time split mostly re-tests
labels already seen in training; that is why Task A holds out ingredients instead.
The FB-D11 "drug mean train label = 0.747" figure is **unverified**: like the
94.5% / 0.999 figures, it has no saved script or log and was not re-measured.

Pooled AUC on Task A is 0.487–0.549 per fold for the classical scores and
0.635–0.682 for the drug-level-only control (`task_a_results.json`), so README
now says "near chance" rather than "below chance".

`tests/test_leakage_check.py` covers the check on synthetic data (key on ingredients
rather than drug name, conflicting-label guard, null-`ror`/duplicate filtering), so
the suite now has 41 tests. `scripts/leakage_check.py` was added to the audit's source
list. In `results_manifest.json`, only the `source_fingerprints` entries for
`scripts/audit_results.py` and `scripts/leakage_check.py` were updated (2026-11-11 UTC
mtimes); `audited_at_utc`, verification flags, results and artifact hashes are those
of the 2026-10-28 audit and were not regenerated.

## Verification limits and application wording

The audit replays every current Task A metric, reapplies SIDER rules to every
stored label, checks fold disjointness/quality, reconstructs cutoff counts and
folds, and recomputes scores using the saved prior. It does not refit MGPS,
rebuild the all-years label candidate pool, replay legacy evaluations, or verify
raw-to-master joins and deduplication end to end. Those checks require separately
preserved inputs and a deliberate upstream rerun; a rerun cannot recover missing
historical provenance. The existing near-degenerate MGPS prior remains a limitation.

For the application, describe a research benchmark with verified classical
baselines and a drug-only control. Task A is ingredient-held-out label evaluation
with pre-2023 reporting features, not prospective discovery: SIDER is static,
and reports may follow labeling. Do not describe the label pool as the evaluation
set, ROR's data-rebuild change as algorithmic progress, or the project as a
completed learned-model benchmark. LightGBM, Tasks B/C and a public leaderboard
remain unfinished. Closed-world negative labels and normalization errors remain
limitations. Retain the project's non-commercial/SIDER restrictions.
