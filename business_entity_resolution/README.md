# Business Entity Resolution — Team Odyssey

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026: Business Entity Resolution. 

The pipeline uses a hybrid blocking approach (Lexical TF-IDF + Semantic FAISS) merged with Reciprocal Rank Fusion, feeding into a LightGBM pairwise precision model calibrated explicitly for the Macro-F0.5 metric.

## 📂 Project Structure

```text
business_entity_resolution/
├── src/
│   ├── config.py                 # Global configurations & hyperparameters
│   ├── pipeline.py               # Main execution entrypoint
│   ├── data/                     # Data loading and saving utilities
│   ├── normalization/            # Text standardization (names, addresses)
│   ├── retrieval/                # Lexical and Embedding retrievers (Blocking)
│   ├── features/                 # Jaro-Winkler, Levenshtein, N-gram extractors
│   ├── decision/                 # Singleton detection and F0.5 calibrator
│   ├── evaluation/               # F0.5 metrics and France-simulation stress tests
│   └── models/
│       ├── pair_matcher.py       # LightGBM pairwise model wrap
│       ├── hard_negatives.py     # Explicit name/address collision mining
│       ├── fine_tune_encoder.py  # Phase 2: Dual-Encoder training script
│       └── fine_tune_cross_encoder.py # Phase 2: Cross-Encoder training script
├── output/                       # Output TSVs generated here
├── requirements.txt              # Pipeline dependencies (CPU/GPU)
└── Documentation_template.md     # Official methodology write-up
```

## 🚀 How to Reproduce

### 1. Environment Setup
Create a Python 3.11 virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Data Path
Ensure that the raw challenge dataset is located at:
`C:\Users\LENOVO\Team_odyssey_Amazon\dataset`
*(If your data is elsewhere, simply modify `DATA_DIR` in `src/config.py`).*

### 3. Run the ML Pipeline (Baseline)
The main pipeline executes both training and inference. To run the baseline pipeline across the full dataset:
```bash
python src/pipeline.py --mode full
```
This automatically:
1. Performs lexical & dense semantic retrieval.
2. Generates hard negative pairs.
3. Extracts string similarity features.
4. Trains the LightGBM model.
5. Calibrates match/singleton thresholds against the Macro-F0.5 metric.
6. Runs inference on the `test` split and generates `candidate_pairs.tsv` and `matching_results.tsv` in the `output/` directory.

### 4. Phase 2 Fine-Tuning (Optional Overnight Run)
For maximum precision, you can optionally fine-tune the deep learning models prior to running the main pipeline.
1. Run the dual-encoder training: `python src/models/fine_tune_encoder.py --epochs 3`
2. Run the cross-encoder training: `python src/models/fine_tune_cross_encoder.py --epochs 2`
3. Update `config.py` to point to the newly saved models, then run `python src/pipeline.py --mode full`.

### 5. Validate the Submission Format
You can verify the pipeline output complies with challenge rules using the official validator:
```bash
python ../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../dataset/test
```
