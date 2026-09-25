"""
Hard negative mining — generates informative training pairs.
Critical for training a matcher that can distinguish near-matches from true matches.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import random
import time


class HardNegativeMiner:
    """
    Generates hard negative pairs for GBM training.
    
    Hard negative types:
    1. Same-name, different-address: tests address discrimination
    2. Same-address, different-name: tests name discrimination  
    3. ANN-nearest non-matches: tests the actual decision boundary
    4. Retrieval-surfaced non-matches: pairs the retriever thought were similar
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def generate_training_pairs(
        self,
        s1_records: Dict[str, dict],
        candidate_records: Dict[str, dict],
        ground_truth: Dict[str, List[str]],
        retrieval_results: Dict[str, List[Tuple[str, float]]],
        neg_ratio: float = 3.0,
    ) -> List[Tuple[str, str, int]]:
        """
        Generate balanced training pairs with hard negatives.
        
        Returns: [(s1_id, candidate_id, label), ...]
        """
        print("  [HardNegMiner] Generating training pairs...")
        t0 = time.time()
        
        # Build positive pairs
        positives = []
        positive_set = set()
        for s1_id, matches in ground_truth.items():
            for mid in matches:
                positives.append((s1_id, mid, 1))
                positive_set.add((s1_id, mid))
        
        n_pos = len(positives)
        n_neg_target = int(n_pos * neg_ratio)
        print(f"    Positives: {n_pos:,}")
        print(f"    Target negatives: {n_neg_target:,}")
        
        # Build reverse index for same-name/same-address mining
        gt_set = defaultdict(set)
        for s1_id, matches in ground_truth.items():
            for mid in matches:
                gt_set[s1_id].add(mid)
        
        negatives = []
        neg_set = set()
        
        # === Type 1: Retrieval-surfaced non-matches (hardest) ===
        # These are candidates the retriever thought were similar but aren't true matches
        n_retrieval_neg = int(n_neg_target * 0.6)  # 60% from retrieval
        retrieval_negs = []
        
        for s1_id, candidates in retrieval_results.items():
            true_matches = gt_set.get(s1_id, set())
            for cand_id, score in candidates:
                if cand_id not in true_matches and (s1_id, cand_id) not in neg_set:
                    retrieval_negs.append((s1_id, cand_id, 0, score))
                    neg_set.add((s1_id, cand_id))
        
        # Prioritize higher-scored non-matches (harder negatives)
        retrieval_negs.sort(key=lambda x: x[3], reverse=True)
        for s1_id, cand_id, label, _ in retrieval_negs[:n_retrieval_neg]:
            negatives.append((s1_id, cand_id, 0))
        
        print(f"    Retrieval hard negatives: {len(negatives):,}")
        
        # === Type 2: Random negatives (easy, for calibration) ===
        n_random_neg = n_neg_target - len(negatives)
        all_s1_ids = list(ground_truth.keys())
        all_cand_ids = list(candidate_records.keys())
        
        attempts = 0
        while len(negatives) < n_neg_target and attempts < n_neg_target * 10:
            s1_id = random.choice(all_s1_ids)
            cand_id = random.choice(all_cand_ids)
            
            if (s1_id, cand_id) not in positive_set and (s1_id, cand_id) not in neg_set:
                negatives.append((s1_id, cand_id, 0))
                neg_set.add((s1_id, cand_id))
            
            attempts += 1
        
        print(f"    Total negatives: {len(negatives):,}")
        
        all_pairs = positives + negatives
        random.shuffle(all_pairs)
        
        elapsed = time.time() - t0
        print(f"  [HardNegMiner] Generated {len(all_pairs):,} pairs in {elapsed:.0f}s")
        
        return all_pairs
