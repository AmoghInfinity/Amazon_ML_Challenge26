"""
Configuration for Business Entity Resolution Pipeline
"""
import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Data Files ===
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

# === Output Files ===
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# === Retrieval Config ===
RETRIEVAL = {
    # Lexical retrieval
    "lexical_top_k": 100,        # top-k from each lexical channel
    "bm25_top_k": 100,
    
    # Embedding retrieval
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_top_k": 100,
    "embedding_batch_size": 512,
    
    # Fusion
    "fusion_top_k": 50,          # after fusing all retrievers
    
    # Adaptive-K
    "adaptive_k_min": 0,
    "adaptive_k_max": 30,
    "score_gap_threshold": 0.15,  # tuned on validation
}

# === Feature Extraction ===
FEATURES = {
    "name_ngram_range": (3, 4),   # char n-gram range (3-4 avoids nnz int32 overflow)
    "address_ngram_range": (3, 4),
    "tfidf_max_features": 25000,
}

# === Model Config ===
MODEL = {
    "lgbm_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "verbose": -1,
        "n_jobs": -1,
        "is_unbalance": True,
    },
    "early_stopping_rounds": 50,
}

# === Decision Layer ===
DECISION = {
    "match_threshold": 0.5,        # base threshold, calibrated on validation
    "singleton_threshold": 0.3,    # below this max score → declare singleton
    "score_margin_weight": 0.3,    # weight for score margin in singleton detection
    "f_beta": 0.5,                 # F-beta parameter
}

# === Validation ===
VALIDATION = {
    "val_fraction": 0.2,           # fraction of S1 entities for validation
    "stratify_by": ["country", "match_count_bin"],
    "random_seed": 42,
}
