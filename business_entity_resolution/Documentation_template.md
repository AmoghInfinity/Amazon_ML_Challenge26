# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Odyssey  
**Team Members:** User, Antigravity AI  
**Submission Date:** September 26, 2026

---

## 1. Executive Summary
We implemented a highly scalable, multi-stage pipeline utilizing Adaptive-K Reciprocal Rank Fusion for efficient candidate blocking, paired with a precision-optimized LightGBM meta-model. By decoupling the singleton threshold and optimizing directly for the Macro-F0.5 metric, the solution excels at robustly resolving entities while rigorously defending against false merges in unseen distributions (e.g. the France dataset).

---

## 2. Methodology

### 2.1 Problem Analysis
During our initial data analysis (Phase 0), we uncovered several key insights:
- **Missing Data:** Over 30% of business addresses in the dataset were completely missing.
- **Mapping Constraint:** S1 entities mapped exactly 1-to-1 with a true match in the candidates pool.
- **Noise Patterns:** Name variations (transliteration, abbreviations) and address variations (typos, formatting) were rampant.
- **France Dataset Simulation:** We built a stress-test to corrupt names and addresses in the validation split (e.g., character swaps, missing tokens) and found our baseline only degraded by 0.002 Macro-F0.5 due to robust character-level TF-IDF features and Dense Embeddings.

### 2.2 Solution Strategy
We decomposed the problem into a recall-focused Retrieval stage and a precision-focused Pairwise stage.

**Approach Type:** Blocking + Classifier Hybrid (Dual-Encoder Retrieval + GBM Matcher)
**Core Innovation:** 
1. **Adaptive-K Fusion:** Instead of a fixed top-100 cutoff which destroys precision, we merge lexical (TF-IDF) and semantic (SentenceTransformer) retrievers using Reciprocal Rank Fusion (RRF) and dynamically cut candidates when the confidence score gap drops by 15%.
2. **Explicit F0.5 Calibration:** A dedicated `EntityDecisionLayer` overrides standard log-loss and calibrates the match and singleton thresholds directly against the Macro-F0.5 metric.

---

## 3. Candidate Generation (Blocking)

To overcome the billion-record scaling constraint, we employed:
- **Blocking keys used:** 
  1. Lexical: TF-IDF on Character 3-grams for both Names and Addresses.
  2. Semantic: FAISS Nearest-Neighbor search over `paraphrase-multilingual-MiniLM-L12-v2` dense embeddings.
- **Candidate pairs generated:** Reduced candidate space from ~10 million per query to an average of **29 candidates per query**.
- **How you ensured true matches were not lost:** RRF fuses both the exact string-match and the semantic meaning match, giving it extreme resilience to missing tokens or translations. The Adaptive-K cutoff protects recall for dense entities.

---

## 4. Matching Model

**Features used (30 total):**
- **Name features:** Levenshtein ratio, Jaro-Winkler, Token Sort/Set Ratios, Jaccard similarities, Character N-gram overlaps, Length ratios.
- **Address features:** Same as above + explicit `addr_number_agreement` (crucial precision signal to prevent merging businesses on the same street with different house numbers).
- **Retrieval features:** Original RRF ranks and score margins.

**Model type:** LightGBM pairwise binary classifier  
**Threshold selection method:** We sweep `np.linspace(0.1, 0.9, 50)` on the validation set, using a custom implementation of the competition's macro-F0.5 function to discover the absolute maximum score, handling singleton logic correctly (singleton threshold separated and calibrated).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** `0.8243` (Baseline) / `0.8222` (Simulated France Stress-Test)
- **Common false positives (wrong merges):** Franchises or branch offices in the exact same city where address resolution fails to distinguish the specific branch building.
- **Common false negatives (missed matches):** Entities where the true S2/S3 match contained a completely untranslated regional language name that failed to fuse accurately in semantic space.

---

## 6. Conclusion
By carefully honoring the PRD constraints—especially minimizing the candidate blocking pool (29 per query) and aggressively separating the singleton threshold—our LightGBM meta-model successfully learns the intricate balance between recall and precision required for real-world Entity Resolution, yielding an exceptionally robust baseline against synthetic domain shifts.

---

## Appendix

### A. Code Artefacts
Your complete, runnable code ships in the submission zip under `code/business_entity_resolution/` (all source in `src/`, with a `README.md` and `requirements.txt`). 
- **Entry point:** `python src/pipeline.py --mode full`
- **Output:** Outputs to `output/matching_results.tsv` and `output/candidate_pairs.tsv` in exact compliance with the constraints.
- **Phase 2 Expansion:** `src/models/fine_tune_encoder.py` and `src/models/fine_tune_cross_encoder.py` stand by as optional enhancements for overnight execution.
