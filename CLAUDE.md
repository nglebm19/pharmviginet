# CLAUDE.md — PharmVigiNet (FAERS-Bench)

Agent operating manual. Project facts live in the files below; don't repeat them here.

## Docs map
- `PRODUCT.md` — purpose, users, benchmark tasks, scope, non-goals, licensing
- `MVP.md` — v0 definition, verified status, leaderboard, blockers, next-work queue
- `DECISIONS.md` — project decisions (`FB-D1`, …) and open questions
- `ARCHITECTURE.md` — data flow, stages, module map, labels, scoring, metric
- `README.md` — public overview and reproduce steps

Verify any status or number against files, `data/logs/*.json` or commits before
writing it into a doc. The docs have drifted before.

---

## RIPER-5 Protocol

Operate in exactly one mode at a time. Declare mode at the start of every response: **[MODE: X]**. Never switch modes without explicit user request. Default mode on session start: **RESEARCH**.

### MODE 1 — RESEARCH
- Read files, explore code, ask clarifying questions
- Output: findings and observations only
- ❌ No suggestions, no implementations, no opinions

### MODE 2 — INNOVATE
- Brainstorm approaches, discuss trade-offs
- Output: possibilities only, no concrete plans
- ❌ No code, no step-by-step plans

### MODE 3 — PLAN
- Write exhaustive, unambiguous implementation plan
- Every file edit, every function, every decision specified
- End with: **"PLAN COMPLETE. Confirm to EXECUTE?"**
- ❌ No implementation yet

### MODE 4 — EXECUTE
- Implement the approved plan exactly — zero deviation
- If unexpected obstacle: **STOP**, explain, return to PLAN
- ❌ No improvisation, no scope creep, no "while I'm here" fixes

### MODE 5 — REVIEW
- Compare implementation vs approved plan line by line
- Output: **PASS** or **FAIL** with specific deviations listed
- ❌ No new changes during review

---

## Do Not Read — Skip These (save tokens)
```
data/raw/          # 160 zips, ~3GB — never read
data/extracted/    # quarter folders, ~18GB — never read
__pycache__/       # bytecode — never read
.venv/             # packages — never read
notebooks/         # skip unless explicitly asked
```
Read only when needed: `data/logs/`, `data/processed/`, `pharmviginet/`

---

## Critical Data Facts

### Two eras — different schemas
- **Legacy AERS** (2004Q1–2012Q3): key = `isr`, case = `case`
- **Modern FAERS** (2012Q4–present): key = `primaryid`, case = `caseid`
- Era detection: `year < 2012` OR `(year == 2012 AND quarter <= 3)`

### Always read files like this — never change these params
```python
pd.read_csv(path, sep="$", encoding="latin1", low_memory=False, index_col=False)
```
`index_col=False` is required: legacy AERS data rows end with a trailing `$` that the
header lacks, and without it pandas turns ISR into the index and shifts every column
left (this silently dropped all 2004–2012Q3 data until FB-D12). Then lowercase the
headers (legacy files are uppercase: `ISR`, `CASE`, `PT`). Use `clean_faers.load_table_file`.

### 7 files per quarter
- `DEMO` — one row per case, primary join key
- `DRUG` — one row per drug per case; filter `role_cod == "PS"` only
- `REAC` — MedDRA reactions (event side of each pair)
- `OUTC` — patient outcomes
- `RPSR` — report source
- `THER` — therapy dates
- `INDI` — drug indication

### No narratives in public FAERS
- Public quarterly files have **no NARR / free-text narrative file**.
- `text_input` in `master.parquet` is a template (`"<DRUG> caused <PT>"`), not real text.
- Don't plan around NARR.

### Labels
- Benchmark label = SIDER 4.1 (`FB-D2`). Rules: `ARCHITECTURE.md`.
- The `label` column in `master.parquet` / `ml/*.parquet` is the **old circular ROR
  rule**, kept only for comparison. Don't train or score against it.
- The `ror_train` column in `ml/{val,test}.parquet` was built in the notebook and cannot
  be reproduced from `train.parquet`. Don't use it; Task A computes ROR in code.
- Task A eligibility comes from reports up to 2022 only; never use `labels_sider.n_reports`
  (all years) for Task A.

---

## Hard Constraints
```python
# ❌ random split — leaks future data
train, test = train_test_split(df)

# ✅ time split
train = df[df.year <= 2022]
val   = df[df.year == 2023]
test  = df[df.year >= 2024]

# ❌ all drugs
df_drug = load_drug_table()

# ✅ primary suspect only
df_drug = df_drug[df_drug.role_cod == "PS"]

# ❌ no minimum
ror = compute_ror(df)

# ✅ minimum 3 reports
ror = compute_ror(df)[lambda x: x.n_reports >= 3]

# ❌ dedup skip — inflates ROR scores
# ✅ always dedup first
df = df.sort_values("caseversion").groupby("caseid").last()
```

### Never do
- Read raw `.txt` directly into model training — always via processed parquet
- Skip deduplication — ~20% of FAERS rows are duplicates
- Commit anything under `data/` (raw, extracted, processed, external, logs)
- Commit or redistribute SIDER files or labels derived from them
- Report pooled AUC as the headline metric — use `auc_strat` (`FB-D7`)
- Evaluate supervised models for Task A without ingredient-disjoint folds (`FB-D11`)

---

## Gotchas & Tribal Knowledge
- **Trailing `$`** — if a header also ends with `$`, pandas adds an empty col: drop `Unnamed*` columns. If only the data rows end with `$` (legacy AERS), see `index_col=False` above
- **Canonical forms** — `pt`/`indi_pt` uppercase (FAERS side); SIDER side lowercase; `build_labels` compares both as strip + lower. `primaryid`/`caseid` are `Int64` in every table
- **Rate limits** — PubChem ≤ 5 req/s (`time.sleep(0.2)`); RxNav ≤ 15 req/s (enforced in `RxNormClient`)
- **SMILES missing ~15%** — biologics/vaccines have no SMILES, use learned `[UNK-MOL]` embedding
- **Class balance** — 59% positive under old ROR labels, 29% under SIDER; `POS_WEIGHT=15` in `config.py` fits neither
- **role_cod values** — PS=primary suspect, SS=secondary, C=concomitant, I=interacting
- **caseversion** — keep only `max(caseversion)` per `caseid`, reports get updated over time
- **2026Q1 exists** — collector grabbed an early 2026 quarter, treat as part of the test holdout
- **F1 in metrics output is meaningless** — fixed threshold on log scores; ignore it
- **Drug mapping errors** — RxNav fuzzy matching sometimes picks the wrong ingredient
  (e.g. SIDER `fenofibric` → fenofibrate); unmapped FAERS names are mostly consumer products
- **Memory** — machine has 16 GB. `clean_faers.py --stage join` peaks at ~5.9 GB and is killed
  while writing the in-memory split (MVP blocker 1). Read big parquet with column lists and year filters
- **`.venv` is minimal** — torch and transformers are **not** installed; add them only when a model needs them. Rebuild:
  `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python pandas pyarrow requests pytest tqdm scikit-learn scipy`
- **`git push` can hang** — the macOS keychain helper can block on a hidden prompt. Workaround:
  `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`,
  or have the user run `gh auth setup-git` once

---

## Useful commands
```bash
.venv/bin/python -m pytest -q tests                                  # 27 tests
.venv/bin/python -m pharmviginet.data.task_a                         # Task A pairs + folds, ~50 s; stops on bad fold quality
.venv/bin/python -m pharmviginet.models.baseline --task a            # Task A leaderboard → data/logs/task_a_results.json, ~50 s
.venv/bin/python -m pharmviginet.models.baseline --labels sider      # legacy time-split reference → data/logs/baseline_results_sider.json
.venv/bin/python -m pharmviginet.labels.build_labels                 # rebuild labels_sider.parquet
.venv/bin/python -m pharmviginet.labels.drug_norm                    # RxNav mapping; resumable, ~2.5 h cold
```

## Suggested skills
- `grill-me` / `grill-with-docs` — stress-test a plan before EXECUTE
- `tdd` — split script and LightGBM feature code (tests in `tests/`)
- `diagnose` — ROR-label mismatch, missing 2004–2011 rows
- `superpowers:verification-before-completion` — before claiming any leaderboard number
- `explore-codebase` / `review-changes` — code-review-graph MCP is configured for this repo
- `to-issues` — turn the `MVP.md` queue into GitHub issues, if wanted

---

## Pointers
- Data dictionary: `data/extracted/2024Q4/ASCII/README.pdf`
- ROR/PRR: Rothman et al. 2004
- PubMedBERT: arxiv.org/abs/2007.15779
- ChemBERTa: arxiv.org/abs/2010.09885
- OpenFDA API: open.fda.gov/apis/drug/event/
