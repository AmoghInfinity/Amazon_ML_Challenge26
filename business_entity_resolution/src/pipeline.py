"""
Main pipeline — end-to-end Business Entity Resolution.
Phase 1 baseline: lexical + embedding retrieval → GBM matcher → decision layer.
"""
import sys
import os
import time
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.loader import (
    load_source, load_ground_truth, load_all_train, load_all_test,
    build_id_lookup, stratified_val_split,
    write_matching_results, write_candidate_pairs
)
from normalization.pipeline import normalize_dataframe
from retrieval.lexical import LexicalRetriever
from retrieval.embeddings import EmbeddingRetriever
from retrieval.fusion import reciprocal_rank_fusion, apply_adaptive_k
from features.pairwise import extract_features_batch, extract_pair_features
from models.pair_matcher import PairMatcher
from models.hard_negatives import HardNegativeMiner
from decision.entity_resolution import EntityDecisionLayer, compute_macro_f05
from evaluation.metrics import evaluate_predictions, evaluate_blocking
from evaluation.stress_test import run_stress_test


def records_to_dict(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert DataFrame to {entity_id: record_dict}."""
    return {row['entity_id']: row.to_dict() for _, row in df.iterrows()}


def run_training_pipeline(
    use_embeddings: bool = True,
    val_fraction: float = 0.2,
    neg_ratio: float = 3.0,
    sample_size: Optional[int] = None,
    stress_test: bool = False,
):
    """
    Full training pipeline:
    1. Load and normalize data
    2. Split into train/val
    3. Run retrieval on train set
    4. Generate hard negatives
    5. Train GBM matcher
    6. Calibrate decision thresholds on validation
    7. Evaluate on validation
    """
    print("=" * 70)
    print("PHASE 1: TRAINING PIPELINE")
    print("=" * 70)
    
    # === Step 1: Load data ===
    print("\n--- Step 1: Loading data ---")
    t_start = time.time()
    
    s1 = load_source(TRAIN_S1)
    s2 = load_source(TRAIN_S2)
    s3 = load_source(TRAIN_S3)
    gt = load_ground_truth(TRAIN_GT)
    
    if sample_size:
        print(f"  [DEBUG] Sampling {sample_size} records from S1...")
        s1 = s1.head(sample_size)
        valid_ids = set(s1['entity_id'].values)
        gt = {k: v for k, v in gt.items() if k in valid_ids}
        
        # Collect all true match IDs for the sampled S1
        true_match_ids = set()
        for matches in gt.values():
            true_match_ids.update(matches)
            
        # Sample S2 and S3: keep true matches + a random sample
        s2_matches = s2[s2['entity_id'].isin(true_match_ids)]
        s3_matches = s3[s3['entity_id'].isin(true_match_ids)]
        
        s2_random = s2[~s2['entity_id'].isin(true_match_ids)].head(sample_size * 2)
        s3_random = s3[~s3['entity_id'].isin(true_match_ids)].head(sample_size * 2)
        
        s2 = pd.concat([s2_matches, s2_random], ignore_index=True)
        s3 = pd.concat([s3_matches, s3_random], ignore_index=True)
    
    print(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")
    print(f"  Ground truth: {len(gt):,} entities")
    
    # === Step 2: Normalize ===
    print("\n--- Step 2: Normalizing ---")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    # Combine S2 + S3 as the candidate pool
    candidates = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined candidate pool: {len(candidates):,}")
    
    # === Step 3: Train/Val split ===
    print("\n--- Step 3: Train/Val split ---")
    train_gt, val_gt = stratified_val_split(gt, s1, val_fraction=val_fraction)
    print(f"  Train S1 entities: {len(train_gt):,}")
    print(f"  Val S1 entities:   {len(val_gt):,}")
    
    # Split S1 records
    train_s1_ids = set(train_gt.keys())
    val_s1_ids = set(val_gt.keys())
    
    s1_train = s1[s1['entity_id'].isin(train_s1_ids)]
    s1_val = s1[s1['entity_id'].isin(val_s1_ids)]
    
    print(f"  Train S1 records: {len(s1_train):,}")
    print(f"  Val S1 records:   {len(s1_val):,}")
    
    if stress_test:
        s1_val = run_stress_test(s1_val)
    
    # === Step 4: Retrieval ===
    print("\n--- Step 4: Building retrieval indices ---")
    
    # Lexical retriever
    lexical = LexicalRetriever(
        ngram_range=FEATURES['name_ngram_range'],
        max_features=FEATURES['tfidf_max_features'],
        top_k=RETRIEVAL['lexical_top_k'],
    )
    lexical.fit(candidates)
    
    # Retrieve for training
    print("\n  Retrieving for training set...")
    lexical_train_results = lexical.retrieve(s1_train, top_k=RETRIEVAL['lexical_top_k'])
    
    # Retrieve for validation
    print("\n  Retrieving for validation set...")
    lexical_val_results = lexical.retrieve(s1_val, top_k=RETRIEVAL['lexical_top_k'])
    
    # Embedding retriever (if enabled)
    embedding_train_results = {}
    embedding_val_results = {}
    
    if use_embeddings:
        try:
            embedding = EmbeddingRetriever(
                model_name=RETRIEVAL['embedding_model'],
                top_k=RETRIEVAL['embedding_top_k'],
                batch_size=RETRIEVAL['embedding_batch_size'],
            )
            embedding.fit(candidates)
            
            print("\n  Embedding retrieval for training set...")
            embedding_train_results = embedding.retrieve(s1_train, top_k=RETRIEVAL['embedding_top_k'])
            
            print("\n  Embedding retrieval for validation set...")
            embedding_val_results = embedding.retrieve(s1_val, top_k=RETRIEVAL['embedding_top_k'])
            
            # Save for reuse
            embedding.save(os.path.join(MODEL_DIR, "embedding_index"))
            
        except Exception as e:
            print(f"  WARNING: Embedding retriever failed: {e}")
            print("  Falling back to lexical-only retrieval.")
            use_embeddings = False
    
    # === Step 5: Fuse retrieval results ===
    print("\n--- Step 5: Fusing retrieval results ---")
    
    # Training
    retrievers_train = [lexical_train_results]
    weights = [1.0]
    if use_embeddings and embedding_train_results:
        retrievers_train.append(embedding_train_results)
        weights = [0.5, 0.5]
    
    fused_train = reciprocal_rank_fusion(retrievers_train, weights)
    fused_train = apply_adaptive_k(
        fused_train,
        min_k=RETRIEVAL['adaptive_k_min'],
        max_k=RETRIEVAL['adaptive_k_max'],
        score_gap_threshold=RETRIEVAL['score_gap_threshold'],
    )
    
    # Validation
    retrievers_val = [lexical_val_results]
    if use_embeddings and embedding_val_results:
        retrievers_val.append(embedding_val_results)
    
    fused_val = reciprocal_rank_fusion(retrievers_val, weights)
    fused_val = apply_adaptive_k(
        fused_val,
        min_k=RETRIEVAL['adaptive_k_min'],
        max_k=RETRIEVAL['adaptive_k_max'],
        score_gap_threshold=RETRIEVAL['score_gap_threshold'],
    )
    
    # Evaluate blocking quality on validation
    val_candidates = {qid: [c[0] for c in cands] for qid, cands in fused_val.items()}
    # Ensure all val S1 entities have an entry
    for sid in val_gt:
        if sid not in val_candidates:
            val_candidates[sid] = []
    evaluate_blocking(val_candidates, val_gt)
    
    # === Step 6: Generate training pairs with hard negatives ===
    print("\n--- Step 6: Hard negative mining ---")
    
    s1_records = records_to_dict(s1_train)
    cand_records = records_to_dict(candidates)
    
    miner = HardNegativeMiner()
    
    # Convert fused results to flat retrieval dict for mining
    flat_retrieval = {}
    for qid, cands in fused_train.items():
        flat_retrieval[qid] = cands
    
    train_pairs = miner.generate_training_pairs(
        s1_records, cand_records, train_gt,
        flat_retrieval, neg_ratio=neg_ratio,
    )
    
    # === Step 7: Extract features ===
    print("\n--- Step 7: Feature extraction ---")
    
    pair_list = [(p[0], p[1]) for p in train_pairs]
    labels = [p[2] for p in train_pairs]
    
    # Build retriever result dicts for feature extraction
    retriever_dicts = {'lexical': lexical_train_results}
    if use_embeddings and embedding_train_results:
        retriever_dicts['embedding'] = embedding_train_results
    
    train_features = extract_features_batch(
        s1_records, cand_records, pair_list,
        retriever_results=retriever_dicts,
    )
    train_features['label'] = labels
    
    # Validation features
    print("\n  Extracting validation features...")
    s1_val_records = records_to_dict(s1_val)
    
    val_pairs = []
    val_labels = []
    for s1_id in val_gt:
        true_matches = set(val_gt[s1_id])
        candidates_list = fused_val.get(s1_id, [])
        for cand_id, score in candidates_list:
            val_pairs.append((s1_id, cand_id))
            val_labels.append(1 if cand_id in true_matches else 0)
    
    retriever_val_dicts = {'lexical': lexical_val_results}
    if use_embeddings and embedding_val_results:
        retriever_val_dicts['embedding'] = embedding_val_results
    
    val_features = extract_features_batch(
        s1_val_records, cand_records, val_pairs,
        retriever_results=retriever_val_dicts,
    )
    val_features['label'] = val_labels
    
    print(f"\n  Train samples: {len(train_features):,} "
          f"(pos: {sum(labels):,}, neg: {len(labels)-sum(labels):,})")
    print(f"  Val samples:   {len(val_features):,} "
          f"(pos: {sum(val_labels):,}, neg: {len(val_labels)-sum(val_labels):,})")
    
    # === Step 8: Train GBM ===
    print("\n--- Step 8: Training GBM matcher ---")
    
    matcher = PairMatcher(params=MODEL['lgbm_params'])
    matcher.train(
        train_features,
        val_df=val_features,
        early_stopping_rounds=MODEL['early_stopping_rounds'],
    )
    
    # Feature importance
    importance = matcher.feature_importance()
    print("\n  Top features:")
    print(importance.head(15).to_string(index=False))
    
    # Save model
    matcher.save(os.path.join(MODEL_DIR, "pair_matcher.pkl"))
    
    # === Step 9: Score validation candidates and calibrate ===
    print("\n--- Step 9: Threshold calibration ---")
    
    # Score all validation pairs
    val_features['score'] = matcher.predict_proba(val_features)
    
    # Build scored candidates per S1 entity
    scored_val = defaultdict(list)
    for _, row in val_features.iterrows():
        scored_val[row['s1_id']].append((row['candidate_id'], row['score']))
    
    # Ensure all val entities present
    for sid in val_gt:
        if sid not in scored_val:
            scored_val[sid] = []
    
    # Calibrate decision layer
    decision = EntityDecisionLayer(use_one_to_one=True)
    
    # First calibrate match threshold
    best_match_thresh = decision.calibrate_threshold(
        dict(scored_val), val_gt,
        threshold_range=(0.1, 0.9),
        n_steps=50,
    )
    
    # Then calibrate singleton threshold
    best_singleton_thresh = decision.calibrate_singleton_threshold(
        dict(scored_val), val_gt,
        threshold_range=(0.05, 0.6),
        n_steps=30,
    )
    
    # === Step 10: Final validation evaluation ===
    print("\n--- Step 10: Final validation evaluation ---")
    
    final_predictions = decision.decide(dict(scored_val))
    eval_results = evaluate_predictions(final_predictions, val_gt)
    
    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed/60:.1f} minutes")
    
    return {
        'matcher': matcher,
        'decision': decision,
        'lexical': lexical,
        'eval_results': eval_results,
        'best_match_threshold': best_match_thresh,
        'best_singleton_threshold': best_singleton_thresh,
    }


def run_inference_pipeline(
    matcher: PairMatcher,
    decision: EntityDecisionLayer,
    lexical_retriever: Optional[LexicalRetriever] = None,
    use_embeddings: bool = True,
    sample_size: Optional[int] = None,
):
    """
    Run inference on test data and generate output files.
    """
    print("\n" + "=" * 70)
    print("INFERENCE PIPELINE")
    print("=" * 70)
    
    # Load test data
    print("\n--- Loading test data ---")
    t1 = load_source(TEST_S1)
    if sample_size:
        print(f"  [DEBUG] Sampling {sample_size} records from T1...")
        t1 = t1.head(sample_size)
        
    t2 = load_source(TEST_S2)
    t3 = load_source(TEST_S3)
    
    if sample_size:
        print(f"  [DEBUG] Sampling {sample_size*2} records from T2 and T3...")
        t2 = t2.head(sample_size * 2)
        t3 = t3.head(sample_size * 2)
    
    print(f"  Test S1: {len(t1):,}, S2: {len(t2):,}, S3: {len(t3):,}")
    
    # Normalize
    print("\n--- Normalizing test data ---")
    t1 = normalize_dataframe(t1)
    t2 = normalize_dataframe(t2)
    t3 = normalize_dataframe(t3)
    
    test_candidates = pd.concat([t2, t3], ignore_index=True)
    print(f"  Combined test candidate pool: {len(test_candidates):,}")
    
    # Retrieval
    print("\n--- Running retrieval ---")
    
    if lexical_retriever is None:
        lexical_retriever = LexicalRetriever(
            ngram_range=FEATURES['name_ngram_range'],
            max_features=FEATURES['tfidf_max_features'],
            top_k=RETRIEVAL['lexical_top_k'],
        )
    
    lexical_retriever.fit(test_candidates)
    lexical_results = lexical_retriever.retrieve(t1, top_k=RETRIEVAL['lexical_top_k'])
    
    # Embedding retrieval
    embedding_results = {}
    if use_embeddings:
        try:
            emb_retriever = EmbeddingRetriever(
                model_name=RETRIEVAL['embedding_model'],
                top_k=RETRIEVAL['embedding_top_k'],
                batch_size=RETRIEVAL['embedding_batch_size'],
            )
            emb_retriever.fit(test_candidates)
            embedding_results = emb_retriever.retrieve(t1, top_k=RETRIEVAL['embedding_top_k'])
        except Exception as e:
            print(f"  WARNING: Embedding retriever failed: {e}")
            use_embeddings = False
    
    # Fusion
    print("\n--- Fusing and selecting candidates ---")
    retrievers = [lexical_results]
    weights = [1.0]
    if use_embeddings and embedding_results:
        retrievers.append(embedding_results)
        weights = [0.5, 0.5]
    
    fused = reciprocal_rank_fusion(retrievers, weights)
    fused = apply_adaptive_k(
        fused,
        min_k=RETRIEVAL['adaptive_k_min'],
        max_k=RETRIEVAL['adaptive_k_max'],
        score_gap_threshold=RETRIEVAL['score_gap_threshold'],
    )
    
    # Write candidate_pairs.tsv
    print("\n--- Writing candidate_pairs.tsv ---")
    all_s1_ids = set(t1['entity_id'].values)
    candidate_dict = {}
    for sid in all_s1_ids:
        cands = fused.get(sid, [])
        candidate_dict[sid] = [c[0] for c in cands]
    
    write_candidate_pairs(candidate_dict, CANDIDATE_PAIRS)
    print(f"  Written {len(candidate_dict):,} entries to {CANDIDATE_PAIRS}")
    
    # Feature extraction and scoring
    print("\n--- Feature extraction and scoring ---")
    t1_records = records_to_dict(t1)
    cand_records = records_to_dict(test_candidates)
    
    retriever_dicts = {'lexical': lexical_results}
    if use_embeddings and embedding_results:
        retriever_dicts['embedding'] = embedding_results
    
    # Process in chunks to manage memory
    all_predictions = {}
    batch_size = 50000
    s1_ids = list(all_s1_ids)
    
    for batch_start in range(0, len(s1_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(s1_ids))
        batch_s1 = s1_ids[batch_start:batch_end]
        
        # Build pairs for this batch
        pairs = []
        for sid in batch_s1:
            for cid, score in fused.get(sid, []):
                pairs.append((sid, cid))
        
        if pairs:
            features = extract_features_batch(
                t1_records, cand_records, pairs,
                retriever_results=retriever_dicts,
            )
            features['score'] = matcher.predict_proba(features)
            
            # Group by S1 entity
            scored = defaultdict(list)
            for _, row in features.iterrows():
                scored[row['s1_id']].append((row['candidate_id'], row['score']))
            
            # Apply decision layer
            batch_predictions = decision.decide(dict(scored))
            all_predictions.update(batch_predictions)
        
        # Entities with no candidates
        for sid in batch_s1:
            if sid not in all_predictions:
                all_predictions[sid] = []
        
        print(f"  Processed {batch_end:,}/{len(s1_ids):,}")
    
    # Write matching_results.tsv
    print("\n--- Writing matching_results.tsv ---")
    write_matching_results(all_predictions, MATCHING_RESULTS)
    print(f"  Written {len(all_predictions):,} entries to {MATCHING_RESULTS}")
    
    # Summary stats
    n_matches = sum(1 for v in all_predictions.values() if v)
    n_singletons = sum(1 for v in all_predictions.values() if not v)
    total_matches = sum(len(v) for v in all_predictions.values())
    print(f"\n  Summary:")
    print(f"    Entities with matches: {n_matches:,}")
    print(f"    Predicted singletons:  {n_singletons:,}")
    print(f"    Total match IDs:       {total_matches:,}")
    print(f"    Avg matches/entity:    {total_matches/len(all_predictions):.2f}")


def main():
    """Run the full pipeline: train → evaluate → inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "inference", "full"], default="full",
                       help="Run mode: train only, inference only, or full pipeline")
    parser.add_argument("--no-embeddings", action="store_true",
                       help="Skip embedding retrieval (faster, lower recall)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                       help="Validation split fraction")
    parser.add_argument("--neg-ratio", type=float, default=3.0,
                       help="Negative to positive ratio for training")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of S1 queries to sample (for quick testing)")
    parser.add_argument("--stress-test", action="store_true",
                       help="Corrupt the validation set to simulate unseen country noise")
    
    args = parser.parse_args()
    
    use_embeddings = not args.no_embeddings
    
    if args.mode in ("train", "full"):
        results = run_training_pipeline(
            use_embeddings=use_embeddings,
            val_fraction=args.val_fraction,
            neg_ratio=args.neg_ratio,
            sample_size=args.sample_size,
            stress_test=args.stress_test,
        )
        
        if args.mode == "full":
            run_inference_pipeline(
                matcher=results['matcher'],
                decision=results['decision'],
                use_embeddings=use_embeddings,
                sample_size=args.sample_size,
            )
    
    elif args.mode == "inference":
        # Load saved model
        matcher = PairMatcher()
        matcher.load(os.path.join(MODEL_DIR, "pair_matcher.pkl"))
        
        decision = EntityDecisionLayer(
            match_threshold=DECISION['match_threshold'],
            singleton_threshold=DECISION['singleton_threshold'],
            use_one_to_one=True,
        )
        
        run_inference_pipeline(
            matcher=matcher,
            decision=decision,
            use_embeddings=use_embeddings,
            sample_size=args.sample_size,
        )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
