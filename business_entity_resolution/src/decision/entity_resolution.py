"""
Entity-level decision layer.
Converts pairwise match scores into final entity-level decisions.
Includes singleton detection and macro-F0.5-aware threshold calibration.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.optimize import minimize_scalar


def compute_entity_f05(predicted: List[str], truth: List[str]) -> float:
    """
    Compute F0.5 for a single entity.
    Handles singletons correctly: empty prediction on true singleton = 1.0.
    """
    pred_set = set(predicted)
    true_set = set(truth)
    
    # Both empty = singleton correctly identified
    if not pred_set and not true_set:
        return 1.0
    
    # Predicted matches but truth is singleton
    if pred_set and not true_set:
        return 0.0
    
    # Predicted singleton but has true matches
    if not pred_set and true_set:
        return 0.0
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        return 0.0
    
    beta = 0.5
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return f_beta


def compute_macro_f05(
    all_predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """Compute macro-averaged F0.5 across all S1 entities."""
    scores = []
    for s1_id in ground_truth:
        pred = all_predictions.get(s1_id, [])
        truth = ground_truth[s1_id]
        scores.append(compute_entity_f05(pred, truth))
    return np.mean(scores) if scores else 0.0


class EntityDecisionLayer:
    """
    Converts pairwise scores to entity-level match decisions.
    
    Key decisions:
    1. Threshold calibration (optimized for macro-F0.5)
    2. Singleton detection
    3. Optional one-to-one constraint (since data shows 1:1 is true)
    """
    
    def __init__(
        self,
        match_threshold: float = 0.5,
        singleton_threshold: float = 0.3,
        use_one_to_one: bool = True,
    ):
        self.match_threshold = match_threshold
        self.singleton_threshold = singleton_threshold
        self.use_one_to_one = use_one_to_one
    
    def decide(
        self,
        scored_candidates: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, List[str]]:
        """
        Make entity-level decisions from pairwise scores.
        
        Args:
            scored_candidates: {s1_id: [(candidate_id, score), ...]}
        
        Returns:
            {s1_id: [matched_ids]} — final match decisions
        """
        results = {}
        used_candidates = set()  # For one-to-one constraint
        
        # Sort S1 entities by max score descending (highest confidence first)
        # This helps one-to-one assignment give priority to strongest matches
        s1_order = sorted(
            scored_candidates.keys(),
            key=lambda s1: max([s for _, s in scored_candidates[s1]], default=0),
            reverse=True,
        )
        
        for s1_id in s1_order:
            candidates = scored_candidates[s1_id]
            
            if not candidates:
                results[s1_id] = []
                continue
            
            # Sort by score descending
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            # Singleton detection
            max_score = candidates[0][1]
            if max_score < self.singleton_threshold:
                results[s1_id] = []
                continue
            
            # Apply threshold
            matches = []
            for cand_id, score in candidates:
                if score >= self.match_threshold:
                    if self.use_one_to_one and cand_id in used_candidates:
                        continue  # Already assigned to another S1
                    matches.append(cand_id)
                    if self.use_one_to_one:
                        used_candidates.add(cand_id)
            
            results[s1_id] = matches
        
        return results
    
    def calibrate_threshold(
        self,
        scored_candidates: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, List[str]],
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        n_steps: int = 50,
    ) -> float:
        """
        Find the threshold that maximizes macro-F0.5 on validation data.
        """
        best_threshold = 0.5
        best_f05 = 0.0
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        
        for threshold in thresholds:
            self.match_threshold = threshold
            predictions = self.decide(scored_candidates)
            f05 = compute_macro_f05(predictions, ground_truth)
            
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = threshold
        
        self.match_threshold = best_threshold
        print(f"  [DecisionLayer] Calibrated threshold: {best_threshold:.3f} "
              f"(macro-F0.5 = {best_f05:.4f})")
        
        return best_threshold
    
    def calibrate_singleton_threshold(
        self,
        scored_candidates: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, List[str]],
        threshold_range: Tuple[float, float] = (0.05, 0.6),
        n_steps: int = 30,
    ) -> float:
        """
        Calibrate singleton threshold separately.
        """
        best_threshold = 0.3
        best_f05 = 0.0
        
        # Save current match threshold
        saved_match = self.match_threshold
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        
        for threshold in thresholds:
            self.singleton_threshold = threshold
            predictions = self.decide(scored_candidates)
            f05 = compute_macro_f05(predictions, ground_truth)
            
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = threshold
        
        self.singleton_threshold = best_threshold
        self.match_threshold = saved_match
        print(f"  [DecisionLayer] Calibrated singleton threshold: {best_threshold:.3f} "
              f"(macro-F0.5 = {best_f05:.4f})")
        
        return best_threshold
