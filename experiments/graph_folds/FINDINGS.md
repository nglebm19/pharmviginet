# Experiment — graph-based ingredient-disjoint folds (Method F)

Branch `experiment/graph-ingredient-splitting`, 2026-11-23. **Not integrated:** the
production pipeline (`pharmviginet/data/task_a.py`), Task A artifacts and the published
leaderboard are unchanged (SHA-256 of `task_a_results.json`, `task_a_pairs.parquet`,
`task_a_folds.parquet`, the prior JSON and `labels_sider.parquet` checked before and after).

## Method F
`components.py`: ingredients are nodes, co-occurrence in one FAERS product's ingredient
set is an edge; Union-Find gives connected components; whole components go to the fold
with the fewest pairs, largest component first (LPT). Ties: component id (deterministic),
or random order with `seed=` (sensitivity check). Inputs are ingredient strings only —
labels cannot affect the partition (tested).

## Graph (746,362 eligible Task A pairs, before fold filtering)
| | |
|---|---|
| Ingredient sets | 1,297 (281 combinations, 40,917 pairs) |
| Ingredients / edges / components | 1,138 / 625 / 855 (793 singletons) |
| Largest component | 69 ingredients, 69,888 pairs = 9.4% (fold target 20%) |
| Next largest (pairs) | 43,445; 18,342; 16,308; 9,112 |

No giant component, so whole components pack into 5 folds without cutting any edge.
Production greedy assigns ingredients individually in random order and drops the
31,328 combination pairs (4.2%) whose ingredients land in different folds.

## Validation
- **Wrapper replay:** production evaluation re-run through `evaluate.py` equals
  `data/logs/task_a_results.json` (all methods, rtol 1e-9).
- **Scores:** recomputed scores for the 715,034 production pairs match stored ones
  (EBGM/EB05 within 3e-8 relative — the saved prior JSON is rounded; same tolerance as
  `scripts/audit_results --counts`); stored scores are then used for shared pairs.
- **Leakage:** in every scenario, 0 ingredients and 0 drug names in more than one fold,
  0 pairs spanning folds.
- **Fold quality** (`task_a.fold_quality`, production limits; labels used only to check):

| Partition | Pairs | Max/min fold | Positive rate | Min strata | Max top-1 share |
|---|---|---|---|---|---|
| Production greedy | 715,034 | 1.036 | 27.1–32.0% | 1,029 | 5.6% |
| F, production pairs | 715,034 | 1.00001 | 28.0–31.4% | 1,051 | 5.6% |
| F, all eligible pairs | 746,362 | 1.00001 | 28.1–31.4% | 1,033 | 5.4% |
| F seeded ×5, all pairs | 746,362 | 1.00001 | all pass | — | — |

## Results — `auc_strat` mean ± sd over 5 folds (paired diff vs IC)
| Method | Production (greedy) | F, same 715,034 pairs | F, all 746,362 pairs | F seeded ×5, all pairs (range of means) |
|---|---|---|---|---|
| EB05 | 0.6196 ± 0.0052 (+0.0059) | 0.6189 ± 0.0105 (+0.0061) | 0.6172 ± 0.0114 (+0.0059) | 0.6166–0.6175 |
| IC025 | 0.6190 ± 0.0044 (+0.0054) | 0.6186 ± 0.0095 (+0.0057) | 0.6167 ± 0.0116 (+0.0054) | 0.6162–0.6170 |
| EBGM | 0.6148 ± 0.0107 (+0.0011) | 0.6138 ± 0.0125 (+0.0009) | 0.6124 ± 0.0105 (+0.0011) | 0.6117–0.6129 |
| IC | 0.6137 ± 0.0116 | 0.6128 ± 0.0125 | 0.6113 ± 0.0105 | 0.6106–0.6118 |
| ROR | 0.6102 ± 0.0123 (−0.0035) | 0.6092 ± 0.0130 (−0.0036) | 0.6080 ± 0.0101 (−0.0033) | 0.6073–0.6086 |
| PRR | 0.6102 ± 0.0123 (−0.0035) | 0.6092 ± 0.0130 (−0.0036) | 0.6080 ± 0.0101 (−0.0033) | 0.6073–0.6085 |
| Drug-level control | 0.5544 ± 0.0210 (−0.0593) | 0.5515 ± 0.0187 (−0.0613) | 0.5601 ± 0.0082 (−0.0512) | 0.5581–0.5625 |

Subsets, F on all pairs (same fitted scores, metric restricted to the subset):

| Method | Single-ingredient (705,445) | Combination (40,917; 29–85 strata/fold) | Recovered (31,328; 23–75 strata/fold) |
|---|---|---|---|
| EB05 | 0.6184 ± 0.0111 | 0.6102 ± 0.0349 | 0.6132 ± 0.0504 |
| IC | 0.6115 ± 0.0114 | 0.6127 ± 0.0218 | 0.6206 ± 0.0290 |
| ROR | 0.6080 ± 0.0111 | 0.6132 ± 0.0206 | 0.6225 ± 0.0262 |
| Drug-level control | 0.5492 ± 0.0097 | 0.5918 ± 0.0830 | 0.6118 ± 0.0610 |

Single-ingredient pairs under production greedy: EB05 0.6191, IC 0.6128, ROR 0.6093,
control 0.5499 — directly comparable with the F single-ingredient column (same 705,445
pairs, different folds).

**The combination columns have no production counterpart.** Of the 40,917 combination
pairs, production keeps only the 9,589 whose ingredients fall in one fold and drops
the other 31,328 (the "recovered" subset, so recovered ⊂ combination). Those 9,589
pairs have 0–1 usable PT strata per fold (≥ 20 pairs and both classes per PT), so
`auc_strat` is undefined for them under either partition. The combination and
recovered figures are therefore F-only estimates, not a production-vs-F comparison.
They are also weak: 23–85 strata per fold against > 1,000 for single-ingredient
pairs, fold sd 0.02–0.08, so the ROR/IC vs EB05 ordering on combinations is within
noise. Scores come from the all-pairs fit (the drug-level control is trained on all
training-fold pairs, not on combinations alone); only the metric is restricted.

## Reading
- Changing only the partition (same pairs) lowers every classical mean by ≤ 0.001 and
  the control by 0.003; paired differences vs IC are unchanged. The EB05/IC025 fold sd
  roughly doubles (≈0.005 → ≈0.010), so method gaps of ~0.006 are within one fold sd.
- Adding the 31,328 recovered combinations lowers classical means by a further
  0.001–0.002 and **raises the control by ~0.009** (0.5515 → 0.5601). On the recovered
  subset the control (0.612) nearly reaches IC (0.621) and ROR (0.622), with large sd
  and few strata.
- Method ranking (EB05 ≥ IC025 > EBGM > IC > ROR = PRR ≫ control) holds in every
  whole-set scenario and seeded variant, and on single-ingredient pairs. It does not
  hold on the combination/recovered subsets, where ROR/IC ≥ EB05.

## Risks
1. **Headline numbers change** (classical −0.002, control +0.006 on all pairs); the
   audited leaderboard and `results_manifest.json` would need a rerun and a FB-D11
   amendment. Not comparable to the published table.
2. **Combination labels:** positive if *any* ingredient lists the PT. On combinations the
   drug-level control comes close to the classical methods — FB-D11's revisit trigger
   ("a drug-level-only model reaches the classical level") is near on this subset.
   Evidence is weak (29–85 strata/fold, sd up to 0.08) but it argues for reporting
   single-ingredient and combination results separately.
3. **Partition sensitivity:** one deterministic partition; seeded variants move means by
   ≤ 0.0007 (classical) and ≤ 0.0024 (control) from the deterministic run.
4. **Component coupling:** the largest component's 69 ingredients share one fold
   (membership not inspected); RxNav mapping errors could join unrelated components.
5. **Cost:** `evaluate.py` ~10.5 min, 4.0 GB max RSS (10.3 GB peak footprint on macOS);
   the 5 seeded variants dominate.

## Recommendation: conditional GO
Method F is correct, label-free, leakage-free, deterministic and passes every fold-quality
limit while keeping all 746,362 pairs with better balance than production. Integrate it
only as a deliberate leaderboard version change, not as a silent fix:
1. Replace `task_a.assign_folds` + drop step with component LPT behind an FB-D11 amendment.
2. Rerun Task A, regenerate `results_manifest.json`, re-audit.
3. Report all-pairs `auc_strat` as the headline plus single-ingredient and combination
   subsets; flag the combination control result.
4. Keep the current greedy table as the historical v0-pre-F result.

## Reproduce
```bash
python -m pytest -q tests/test_graph_folds.py
python -m experiments.graph_folds.evaluate --output experiments/graph_folds/results.json
```
`results.json` holds aggregate metrics, counts and fold-quality summaries only — no pair
rows, drug names or labels.
