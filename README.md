# Amazon ML Challenge 2026 — Team Odyssey

This repository contains our complete end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**. 

We tackle the challenge of resolving noisy business identities across three independent sources by utilizing a highly scalable **Adaptive-K Retrieval** strategy and a precision-focused **LightGBM Meta-Model**. Our architecture explicitly optimizes the competition's target **Macro-F0.5** score and is hardened against distribution shifts (like the unseen France dataset) via synthetic stress-testing and deep semantic embeddings.

---

## 🚀 Quick Start

### 1. Set Up the Environment
Create a virtual environment and install the pinned dependencies:

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r business_entity_resolution/requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r business_entity_resolution/requirements.txt
```

### 2. Dataset Structure
Make sure the raw training and test data folders are located at the root of the project exactly like this:
```text
Team_odyssey_Amazon/
├── dataset/
│   ├── train/        # train_source1.tsv, train_source2.tsv, etc.
│   └── test/         # test_source1.tsv, test_source2.tsv, etc.
```

### 3. Run the Core Pipeline (Phase 1 Baseline)
Navigate into the core pipeline directory and run the fully automated training and inference flow.
```bash
cd business_entity_resolution
python src/pipeline.py --mode full
```
*(If you want to test the execution speed on a smaller batch first, append `--sample-size 5000` to the command).*

---

## 🔬 Phase 2: Neural Network Fine-Tuning
For teams looking to maximize their precision on the private leaderboard, this repository also includes scripts to fine-tune the core deep-learning models on generated hard-negatives. 

These are standalone scripts designed to be run overnight on a GPU:
1. **Fine-Tune Dual-Encoder:** `python src/models/fine_tune_encoder.py` (Aligns embeddings to explicitly separate same-name/different-address collisions).
2. **Fine-Tune Cross-Encoder:** `python src/models/fine_tune_cross_encoder.py` (Trains an ultra-precise `stsb-distilroberta-base` reranker).

*After fine-tuning, you simply update `business_entity_resolution/src/config.py` to point to the saved models, and the main `pipeline.py` will automatically utilize them!*

---

## 📂 Repository Layout

- `business_entity_resolution/src/` — Core ML pipeline logic (retrieval, features, LightGBM matcher, decision logic).
- `business_entity_resolution/output/` — Where `matching_results.tsv` and `candidate_pairs.tsv` are saved for submission.
- `business_entity_resolution/Documentation_template.md` — The completed methodology document outlining our F0.5 optimization strategy.
- `dataset/` — Directory for the raw challenge data.

For granular instructions on how to use the evaluation validation scripts (`validate_submission.py`), see the inner [`business_entity_resolution/README.md`](business_entity_resolution/README.md).
