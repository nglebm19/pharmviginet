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
- Code: open-source license (TBD, e.g. MIT).
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
│   ├── external/      # DrugBank, SIDER
│   └── logs/          # READ — audit_report.json, schema_map.json
├── pharmviginet/      # READ — source package
│   ├── config.py
│   ├── data/{faers,clean,smiles}.py
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
| 0 — Collect | collect_faers.py | ✅ Done | 91 quarters incl 2026Q1 |
| 1 — Audit | audit_faers.py | 🔄 Next | data/logs/audit_report.json |
| 2 — Clean | clean_faers.py | ⏳ | data/processed/*.parquet |
| 3 — Dedup | clean_faers.py | ⏳ | cases_deduped.parquet |
| 4 — Normalize | clean_faers.py | ⏳ | drug_smiles_map.parquet |
| 5 — Join | clean_faers.py | ⏳ | master.parquet |
| 6 — ML split | build_dataset.py | ⏳ | ml/{train,val,test}.parquet |
| 7 — Baseline | models/baseline.py | ⏳ | ROR/PRR scores |
| 8 — Text | models/text.py | ⏳ | PubMedBERT fine-tuned |
| 9 — Mol | models/mol.py | ⏳ | ChemBERTa fine-tuned |
| 10 — Fusion | models/fusion.py | ⏳ | PharmVigiNet v1 |

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
- `REAC` — MedDRA reactions = your labels
- `NARR` — free-text = NLP input (sparse before 2015, missing in old quarters = normal)
- `OUTC` — patient outcomes
- `RPSR` — report source
- `INDI` — drug indication

### NARR coverage by era
- Before 2013: <5% — skip for NLP
- 2013–2017: 10–30% — use with caution
- 2018–present: 50–65% — primary NLP range

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
- Loss: BCE with class weighting pos:neg ≈ 1:15
- Label: `ROR ≥ 2.0 AND lower_CI > 1.0`

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
- **Class imbalance** — true signals are 5–8% of pairs, always use weighted loss
- **role_cod values** — PS=primary suspect, SS=secondary, C=concomitant, I=interacting
- **caseversion** — keep only `max(caseversion)` per `caseid`, reports get updated over time
- **2026Q1 exists** — collector grabbed an early 2026 quarter, treat as 2025 holdout

---

## Baseline to Beat
| Model | AUC |
|---|---|
| ROR classical | ~0.71 |
| PRR classical | ~0.69 |
| Text-only PubMedBERT | TBD |
| **PharmVigiNet multi-modal** | **target: text+8pts** |

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
| Independent labels (SIDER) — task A | ⬜ |
| Statistical baselines rescored on new labels | ⬜ |
| Label-change dates (SrLC) — task B early detection | ⬜ |
| Cold-start split — task C | ⬜ |
| Benchmark v0 public (repo + HF dataset + leaderboard) | ⬜ |
| ML / ChemBERTa baselines | ⬜ |
| Write-up / preprint | ⬜ |