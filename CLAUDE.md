# CLAUDE.md — PharmVigiNet

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
data/extracted/    # 91 quarter folders, ~18GB — never read
__pycache__/       # bytecode — never read
.venv/             # packages — never read
notebooks/         # skip unless explicitly asked
```
Read only when needed: `data/logs/`, `data/processed/`, `pharmviginet/`

---


## Motivational Intent
Open-source, non-commercial benchmark for adverse drug event signal detection
on FDA FAERS data ("FAERS-Bench"). Portfolio project — no revenue goal.
Provides a clean, deduplicated, time-split FAERS dataset, independent labels
(SIDER, later FDA label-change dates), and reproducible baselines
(ROR, PRR, BCPNN/IC, EBGM, LightGBM, ChemBERTa) so methods can be compared fairly.
Goal: public GitHub repo + leaderboard, write-up/preprint, strong portfolio piece.
Solo builder. Correctness over speed.

### Licensing constraints (non-commercial)
- Code: MIT (see LICENSE). Covers code only, not data or third-party labels.
- SIDER is CC BY-NC-SA and contains MedDRA terms — never commit or redistribute
  raw SIDER files; ship scripts that rebuild labels locally instead.
- FAERS data is public domain.

---

## Actual Folder Structure
```
pharmviginet/
├── CLAUDE.md / AGENTS.md
├── collect_faers.py   # stage 0 ✅
├── audit_faers.py     # stage 1
├── stage1.txt         # stage 1 notes
├── pyproject.toml / requirements.txt / README.md / LICENSE
├── data/
│   ├── raw/           # SKIP — 160 zips
│   ├── extracted/     # SKIP — 91 quarters 2004Q1–2026Q1
│   ├── processed/     # target — parquet outputs
│   ├── external/      # SIDER download, RxNav caches, rxnorm_map, sider_pairs (never commit)
│   └── logs/          # READ — audit_report.json, schema_map.json
├── pharmviginet/      # READ — source package
│   ├── config.py
│   ├── data/{faers,clean,smiles}.py
│   ├── labels/{drug_norm,sider,build_labels}.py   # independent labels
│   ├── models/{baseline,text,mol,fusion}.py
│   ├── train/{train,evaluate}.py
│   └── utils/{logging,metrics}.py
├── scripts/ / notebooks/ / tests/
```
**Data flow:** raw zips → extracted txt → processed parquet → master.parquet → ml/train|val|test.parquet → model

---

## Pipeline Stages
| Stage | Script | Status | Output |
|---|---|---|---|
| 0 — Collect | collect_faers.py | ✅ Done | 89 quarters 2004Q1–2026Q1 |
| 1 — Audit | audit_faers.py | ✅ Done | data/logs/audit_report.json |
| 2–5 — Clean/Dedup/Normalize/Join | clean_faers.py | ✅ Done | master.parquet (52.7M rows) |
| 6 — ML split | (notebook) | ✅ Done | ml/{train,val,test}.parquet |
| 7 — Baseline (ROR labels) | models/baseline.py | ✅ Done — circular, see below | baseline_results.json |
| 8 — Drug normalization | labels/drug_norm.py | ✅ 76.4% of 33.8K names mapped | external/rxnorm_map.parquet |
| 9 — SIDER pairs | labels/sider.py | ✅ Done | external/sider_pairs.parquet |
| 10 — SIDER labels | labels/build_labels.py | ✅ 713K pairs, 29% positive | processed/labels_sider.parquet |
| 11 — Baseline (SIDER labels) | models/baseline.py --labels sider | ✅ | baseline_results_sider.json |
| — Text (PubMedBERT) | models/text.py | ⏸ 1 epoch on 10K sample, AUC 0.64 | text_best.pt |
| — Mol / Fusion | models/mol.py, fusion.py | ⏸ | fusion.py empty |

---

## Critical Data Facts

### Two eras — different schemas
- **Legacy AERS** (2004Q1–2012Q3): key = `isr`, case = `case`
- **Modern FAERS** (2012Q4–present): key = `primaryid`, case = `caseid`
- Era detection: `year < 2012` OR `(year == 2012 AND quarter <= 3)`

### Always read files like this — never change these params
```python
pd.read_csv(path, sep="$", encoding="latin1", low_memory=False)
```

### 7 files per quarter
- `DEMO` — one row per case, primary join key
- `DRUG` — one row per drug per case; filter `role_cod == "PS"` only
- `REAC` — MedDRA reactions (event side of each pair)
- `OUTC` — patient outcomes
- `RPSR` — report source
- `THER` — therapy dates
- `INDI` — drug indication

### No narratives in public FAERS
- Public quarterly files have **no NARR / free-text narrative file** (audit: NARR ≥10% never reached).
- `text_input` in master.parquet is a template (`"<DRUG> caused <PT>"`), not real text.
- Text models therefore have no real narrative input — don't plan around NARR.

### Labels
- Old label `ROR ≥ 2.0 AND lower_CI > 1.0` is **circular** — ROR baselines score against their own formula (AUC 0.94). Kept only for comparison.
- Benchmark label = SIDER 4.1 (see `pharmviginet/labels/`): positive if SIDER lists (ingredient, pt);
  negative if drug and pt both known to SIDER but pair unlisted (closed-world — noisy); else unlabeled.
- Drug names normalized to RxNorm ingredients via RxNav (`approximateTerm` → `related?tty=IN`), max 15 req/s.

---

## Model Architecture
```
SMILES → ChemBERTa → [mol_emb: 384]  ─┐
                                        ├→ concat [1152] → MLP → signal_score
NARR  → PubMedBERT → [text_emb: 768] ─┘
```
- Text: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Mol: `seyonec/ChemBERTa-zinc-base-v2`
- Optimizer: AdamW lr=2e-5, weight_decay=0.01
- Loss: BCE (current code pos_weight=15 — wrong for the 59% positive ROR labels; revisit per label set)
- Label: see Labels above — not the ROR rule

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
- Commit `data/raw/` or `data/extracted/` to git

---

## Gotchas & Tribal Knowledge
- **Phantom column** — trailing `$` creates empty col: `df = df.loc[:, ~df.columns.str.startswith('Unnamed')]`
- **PubChem rate limit** — 5 req/sec max: `time.sleep(0.2)` between calls
- **SMILES missing ~15%** — biologics/vaccines have no SMILES, use learned `[UNK-MOL]` embedding
- **Class balance** — 59% positive under old ROR labels; recheck under SIDER labels before choosing loss weights
- **role_cod values** — PS=primary suspect, SS=secondary, C=concomitant, I=interacting
- **caseversion** — keep only `max(caseversion)` per `caseid`, reports get updated over time
- **2026Q1 exists** — collector grabbed an early 2026 quarter, treat as 2025 holdout

---

## Baseline to Beat
| Model | AUC |
|---|---|
| ROR on SIDER labels, pooled (test, row-level) | 0.48 |
| ROR on SIDER labels, pooled (test, unique pairs) | 0.48 |
| ROR on SIDER labels, **within-PT** (weighted mean) | **0.61** |

Pooled AUC is below chance because SIDER positives are mostly common, non-specific
events (nausea, headache) that have low ROR for every drug. Compare drugs within the
same event (within-PT AUC) — this matches published ROR-vs-SIDER results.
Benchmark metric must be event-stratified, not pooled.
| ROR_all on ROR labels (circular, ignore) | 0.94 test |

---

## Pointers
- Data dictionary: `data/extracted/2024Q4/ASCII/README.pdf`
- ROR/PRR: Rothman et al. 2004
- PubMedBERT: arxiv.org/abs/2007.15779
- ChemBERTa: arxiv.org/abs/2010.09885
- OpenFDA API: open.fda.gov/apis/drug/event/

---

## Milestones
| Milestone | Done? |
|---|---|
| All quarters collected | ✅ |
| Audit clean | ✅ |
| master.parquet built | ✅ |
| Git repo + backup | ✅ |
| Independent labels (SIDER) — task A | ✅ |
| Statistical baselines rescored on new labels | ✅ (pooled metric flawed — see Baseline) |
| Label-change dates (SrLC) — task B early detection | ⬜ |
| Cold-start split — task C | ⬜ |
| Benchmark v0 public (repo + HF dataset + leaderboard) | ⬜ |
| ML / ChemBERTa baselines | ⬜ |
| Write-up / preprint | ⬜ |