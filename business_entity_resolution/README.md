# Business Entity Resolution — Team Odyssey

## Overview

ML solution for resolving business entity identities across 3 independent, noisy data sources. Given business records with `business_name`, `business_address`, and `country`, determines which records across sources refer to the same real-world business.

## Architecture

```
S1 / S2 / S3 records
        │
        ▼
Universal normalization (names + addresses)
        │
   ┌────┼────────────┐
   ▼    ▼             ▼
TF-IDF   Token      Embedding
(char   blocking    (multilingual
n-gram)             MiniLM)
   │       │             │
   └───────┼─────────────┘
           ▼
   Reciprocal Rank Fusion + Adaptive-K cutoff
           │  → candidate_pairs.tsv
           ▼
   Pairwise feature extraction (name/address/semantic/retrieval)
           │
           ▼
   LightGBM matcher (hard-negative trained)
           │
           ▼
   Entity-level decision layer
   (singleton detection, macro-F0.5-calibrated thresholds, one-to-one)
           │  → matching_results.tsv
           ▼
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (train + inference)
```bash
python src/pipeline.py --mode full
```

### 3. Run training only
```bash
python src/pipeline.py --mode train
```

### 4. Run inference only (requires saved model)
```bash
python src/pipeline.py --mode inference
```

### 5. Fast mode (no embeddings, lexical-only)
```bash
python src/pipeline.py --mode full --no-embeddings
```

### 6. Validate output
```bash
python ../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

## Project Structure

```
business_entity_resolution/
├── src/
│   ├── config.py                  # Central configuration
│   ├── pipeline.py                # Main pipeline orchestrator
│   ├── phase0_analysis.py         # Phase 0 data analysis
│   ├── data/
│   │   ├── loader.py              # Data I/O, validation splits
│   │   └── __init__.py
│   ├── normalization/
│   │   ├── names.py               # Business name normalization
│   │   ├── addresses.py           # Address normalization
│   │   ├── pipeline.py            # Normalization pipeline
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── lexical.py             # TF-IDF char n-gram retrieval
│   │   ├── embeddings.py          # Sentence-transformer + FAISS
│   │   ├── fusion.py              # RRF fusion + adaptive-K
│   │   └── __init__.py
│   ├── features/
│   │   ├── pairwise.py            # Pairwise feature extraction
│   │   └── __init__.py
│   ├── models/
│   │   ├── pair_matcher.py        # LightGBM matcher
│   │   ├── hard_negatives.py      # Hard negative mining
│   │   └── __init__.py
│   ├── decision/
│   │   ├── entity_resolution.py   # Decision layer + calibration
│   │   └── __init__.py
│   └── evaluation/
│       ├── metrics.py             # F0.5, blocking metrics
│       └── __init__.py
├── output/
│   ├── matching_results.tsv       # Final matches
│   └── candidate_pairs.tsv        # Blocking candidates
├── models/                        # Saved model artifacts
├── requirements.txt
└── README.md
```

## Key Design Decisions

1. **Adaptive-K cutoff** instead of fixed top-N — preserves recall for multi-match entities
2. **Country as weak signal only** — `country_match` flag, no one-hot encoding, no hardcoded country logic
3. **One-to-one constraint** — validated empirically from ground truth (each S2/S3 maps to ≤1 S1)
4. **Hard negative mining** — retrieval-surfaced non-matches are the hardest, most informative negatives
5. **Macro-F0.5 calibration** — thresholds tuned on the actual evaluation metric, not pairwise accuracy
6. **Singleton detection** — dedicated mechanism since singletons are 5.6% of entities but scored at full weight
