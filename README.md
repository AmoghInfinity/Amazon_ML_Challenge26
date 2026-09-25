# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains our solution for the **Amazon ML Challenge 2026: Business Entity Resolution**. The task is to resolve business identities across three independent, noisy data sources using a machine learning pipeline.

## 🚀 Quick Start

### 1. Set Up the Environment
We recommend using a virtual environment to manage dependencies:

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r business_entity_resolution/requirements.txt
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r business_entity_resolution/requirements.txt
```

### 2. Dataset Structure
Make sure the raw data is organized in the `dataset/` directory at the root of the project:
```
Team_odyssey_Amazon/
├── dataset/
│   ├── train/        # train_source1.tsv, train_source2.tsv, etc.
│   └── test/         # test_source1.tsv, test_source2.tsv, etc.
```

### 3. Run the Pipeline
Once the environment is set up and data is in place, you can run the full pipeline (training + inference) from the root directory:

```bash
cd business_entity_resolution
python src/pipeline.py --mode full
```

*(Note: The full pipeline with embeddings is compute-intensive. For a faster iteration without embedding retrieval, you can add the `--no-embeddings` flag.)*

---

## 🏗️ Architecture Overview

Our solution utilizes a **Three-Pillar Design**:
1. **Multi-channel Retrieval:** Combines TF-IDF lexical retrieval (character n-grams) with Semantic Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) via FAISS.
2. **Adaptive-K Fusion:** Uses Reciprocal Rank Fusion (RRF) with a score-gap-based cutoff to ensure high recall without a fixed candidate limit.
3. **Calibrated Matcher:** A LightGBM pairwise classifier trained with hard-negative mining, followed by a decision layer calibrated specifically to maximize the target **Macro-F0.5 score** (including dedicated singleton detection).

## 📂 Repository Structure

- `business_entity_resolution/src/` — Core pipeline code (retrieval, normalization, features, matching).
- `business_entity_resolution/output/` — Where the final predictions (`matching_results.tsv`, `candidate_pairs.tsv`) are saved.
- `business_entity_resolution/requirements.txt` — Exact pinned dependencies needed to run the project.
- `dataset/` — Directory for the raw training and test data (ignored by git).

For more detailed technical notes on our methodology and design decisions, please refer to the inner [`business_entity_resolution/README.md`](business_entity_resolution/README.md).
