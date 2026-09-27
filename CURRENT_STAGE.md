# Current Stage — Handoff for Next Claude Session

_Last updated: 2026-09-25. Latest commit: `ee8fe92`._

Read this file first, then `CLAUDE.md`. `CLAUDE.md` holds the project facts (RIPER-5 protocol, data facts,
pipeline stage table, baseline leaderboard, milestones). This file covers only what `CLAUDE.md` does not:
the latest decisions, open issues, and the agreed next step.

---

## Where the project stands

- **Goal**: FAERS-Bench, an open-source, non-commercial benchmark for adverse drug event signal detection.
  It is a portfolio project with no revenue goal. The code is MIT licensed.
- **Repo**: private GitHub repo `nglebm19/pharmviginet`, branch `main`, in sync with origin.
  It stays private until benchmark v0 is ready.
- **Task A is complete end to end**: SIDER labels, 8 classical baselines, and event-stratified AUC.
  See the tables under "Baseline to Beat" in `CLAUDE.md`. Best baselines: IC and EBGM, AUC_strat 0.615.
- `README.md` was written in a separate session (`91ca5ec`). Check that it matches the current numbers.
- The parent folder was renamed from `theblackswan` to `Alethenity`. The tool config paths were updated (`ee8fe92`).

## History, in order (see `git log` for details)

1. Initialized git and set up `.gitignore` (`/data/` and `/model_checkpoints/` are excluded).
2. Reframed the goal and added the MIT license.
3. Built SIDER labels. The code is in `pharmviginet/labels/`. The circular ROR label is kept only for comparison.
4. Switched scoring to one score per unique (drug, pt) pair. Added `stratified_auc` and fixed the PRR formula.
5. Added BCPNN (IC, IC025) and MGPS (EBGM, EB05) in `pharmviginet/models/disproportionality.py`.

---

## Agreed next step

**Step 1: make the pipeline reproducible.** The user has not yet approved a PLAN for it.
Start in PLAN mode, per the RIPER-5 protocol in `CLAUDE.md`.

- The train/val/test split (`data/processed/ml/*.parquet`) was built in `notebooks/PharmVigiNet_train.ipynb`
  and has no script. Move that logic into a script such as `pharmviginet/data/build_dataset.py`.
- Make `scripts/run_pipeline.sh` run the whole pipeline with one command:
  collect → audit → clean → split → labels (`drug_norm`, `sider`, `build_labels`) → baselines.
- Investigate one open problem during this step. On the old ROR labels, `ROR_all` scores only AUC 0.79,
  while `ROR_train` scores 0.83. The label was supposedly derived from `ror`, so it should score near 1.0.
  The `label` column was probably computed from different data than the `ror` column.
  Compare `compute_ror` in `clean_faers.py` with the notebook.

## Queue after step 1 (the user agreed to this order)

2. **LightGBM baseline** on per-pair features: time trends, age/sex mix, outcomes, reporter type,
   indications, and the classical scores. The target is to beat AUC_strat 0.615.
3. **Fix the EBGM prior fit.** The current prior is near-degenerate (p=0.99, α≈3e-5).
   Try fitting without zero truncation, or put a floor on α.
4. **Task B**: FDA SrLC label-change dates for early detection. The SrLC site has no bulk export,
   so this needs a scraper. It is the most novel part of the benchmark.
5. **Task C**: a cold-start split for new drugs with few reports. This is where ChemBERTa may help.
6. **Deprioritized**: the PubMedBERT text model. Public FAERS has no narrative text.

---

## Known issues and gotchas

- **Label noise**: pairs that SIDER does not list are treated as negatives, which is wrong for some pairs
  (for example OxyContin / Overdose = 0). Reports driven by lawsuits (Zantac cancers) have high ROR
  but are labeled negative. Document both in the write-up.
- **Drug mapping**: 76.4% of the 33.8K FAERS names map to RxNorm. The unmapped names are mostly
  consumer products. RxNav fuzzy matching sometimes picks the wrong ingredient
  (for example SIDER's `fenofibric` → fenofibrate).
- **F1 in the metrics output is meaningless**: it uses a fixed threshold on log scores. Ignore it
  until a proper threshold is chosen.
- **`git push` can hang**: the macOS keychain credential helper can block on a hidden prompt.
  Workaround:
  `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`
  Or have the user run `gh auth setup-git` once.
- **Rebuilding `.venv`**: it was rebuilt with only what the labels and baselines need:
  `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python pandas pyarrow requests pytest tqdm scikit-learn scipy`.
  torch and transformers are **not** installed. Add them only when a model needs them.
- **Memory**: loading `train.parquet` for pair counts peaks at about 3.8 GB of RAM. A full baseline run takes about 95 s.
- **Licensing**: never commit anything under `data/external/`, which holds the SIDER download,
  the RxNav caches and `sider_pairs`.

## Useful commands

```bash
.venv/bin/python -m pytest -q tests                                  # 12 tests, all pass
.venv/bin/python -m pharmviginet.models.baseline --labels sider      # leaderboard → data/logs/baseline_results_sider.json
.venv/bin/python -m pharmviginet.labels.build_labels                 # rebuild labels_sider.parquet
.venv/bin/python -m pharmviginet.labels.drug_norm                    # RxNav mapping; resumable, takes about 2.5 h cold
```

---

## Suggested skills

- `grill-me` or `grill-with-docs`: stress-test the step 1 plan before executing it.
- `tdd`: for `build_dataset.py` and the LightGBM feature code. Existing tests are in `tests/`.
- `diagnose`: for the ROR-label mismatch investigation.
- `explore-codebase` / `review-changes`: the code-review-graph MCP is configured for this repo.
- `to-issues`: to turn the queue above into GitHub issues, if the user wants to track them.
- `superpowers:verification-before-completion`: before claiming any leaderboard numbers.
