from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA      = ROOT / "data"
PROCESSED = DATA / "processed"
ML_DIR    = PROCESSED / "ml"
LOGS      = DATA / "logs"
CKPT_DIR  = ROOT / "model_checkpoints"

MASTER_PARQUET = PROCESSED / "master.parquet"
TRAIN_PARQUET  = ML_DIR / "train.parquet"
VAL_PARQUET    = ML_DIR / "val.parquet"
TEST_PARQUET   = ML_DIR / "test.parquet"
SMILES_MAP     = PROCESSED / "drug_smiles_map.parquet"

PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
CHEMBERT   = "seyonec/ChemBERTa-zinc-base-v2"

LR           = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE   = 32
MAX_EPOCHS   = 10
POS_WEIGHT   = 15.0
TEXT_MAX_LEN = 128
MOL_MAX_LEN  = 128

EXTERNAL       = DATA / "external"
RXNORM_MAP     = EXTERNAL / "rxnorm_map.parquet"
SIDER_DIR      = EXTERNAL / "sider"
SIDER_PAIRS    = EXTERNAL / "sider_pairs.parquet"
LABELS_SIDER   = PROCESSED / "labels_sider.parquet"

# Task A — ingredient-disjoint folds, features from reports up to the cutoff year
FEATURE_CUTOFF_YEAR = 2022
TASK_A_MIN_REPORTS  = 3
TASK_A_N_FOLDS      = 5
TASK_A_SEED         = 0
TASK_A_PAIRS        = PROCESSED / "task_a_pairs.parquet"
TASK_A_FOLDS        = PROCESSED / "task_a_folds.parquet"
TASK_A_RESULTS      = LOGS / "task_a_results.json"

# Train/val/test split of master.parquet (clean_faers.py --stage split)
TRAIN_MAX_YEAR = FEATURE_CUTOFF_YEAR
VAL_YEARS      = (2023, 2023)
TEST_MIN_YEAR  = 2024
SPLIT_MANIFEST = ML_DIR / "split_manifest.json"
