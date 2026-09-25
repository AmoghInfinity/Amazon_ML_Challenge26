"""
Candidate fusion — combines results from multiple retrievers.
Implements Reciprocal Rank Fusion (RRF) and adaptive-K cutoff.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def reciprocal_rank_fusion(
    retrieval_results: List[Dict[str, List[Tuple[str, float]]]],
    retriever_weights: Optional[List[float]] = None,
    k: int = 60,  # RRF constant
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Fuse results from multiple retrievers using Reciprocal Rank Fusion.
    
    RRF score for candidate c from retriever i at rank r_i:
        score(c) = sum_i (weight_i / (k + r_i))
    
    Args:
        retrieval_results: list of {query_id: [(candidate_id, score), ...]}
        retriever_weights: weight per retriever (default: equal)
        k: RRF constant (higher = more weight to lower ranks)
    
    Returns:
        {query_id: [(candidate_id, fused_score), ...]} sorted by score desc
    """
    if retriever_weights is None:
        retriever_weights = [1.0] * len(retrieval_results)
    
    # Normalize weights
    total_w = sum(retriever_weights)
    retriever_weights = [w / total_w for w in retriever_weights]
    
    # Collect all query IDs
    all_query_ids = set()
    for results in retrieval_results:
        all_query_ids.update(results.keys())
    
    fused = {}
    
    for qid in all_query_ids:
        candidate_scores = defaultdict(float)
        candidate_sources = defaultdict(set)
        
        for i, results in enumerate(retrieval_results):
            if qid not in results:
                continue
            
            candidates = results[qid]
            weight = retriever_weights[i]
            
            for rank, (cid, score) in enumerate(candidates, start=1):
                candidate_scores[cid] += weight / (k + rank)
                candidate_sources[cid].add(i)
        
        # Sort by fused score
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        fused[qid] = [(cid, score) for cid, score in sorted_candidates]
    
    return fused


def adaptive_k_cutoff(
    candidates: List[Tuple[str, float]],
    min_k: int = 0,
    max_k: int = 30,
    score_gap_threshold: float = 0.15,
    min_score: float = 0.001,
) -> List[Tuple[str, float]]:
    """
    Adaptive-K candidate selection based on score gaps.
    
    Instead of fixed top-N, cut where the score drops significantly.
    This preserves recall for entities with many true matches while
    keeping the candidate set small for easy entities.
    
    Args:
        candidates: sorted list of (id, score) in descending order
        min_k: minimum candidates to always return
        max_k: maximum candidates to return
        score_gap_threshold: relative score drop to trigger cutoff
        min_score: absolute minimum score to include
    
    Returns:
        Truncated candidate list
    """
    if not candidates:
        return []
    
    # Always include at least min_k
    result = candidates[:min_k] if min_k > 0 else []
    
    if len(candidates) <= min_k:
        return candidates
    
    # Scan remaining candidates for score gap
    max_score = candidates[0][1]
    if max_score <= 0:
        return result
    
    for i in range(max(min_k, 1), min(len(candidates), max_k)):
        cid, score = candidates[i]
        
        # Skip if below absolute minimum
        if score < min_score:
            break
        
        # Check relative score gap from previous
        prev_score = candidates[i-1][1] if i > 0 else max_score
        if prev_score > 0:
            relative_gap = (prev_score - score) / prev_score
            if relative_gap > score_gap_threshold:
                # Also check if score is significantly below max
                if score < max_score * 0.3:
                    break
        
        result.append((cid, score))
    
    return result


def apply_adaptive_k(
    fused_results: Dict[str, List[Tuple[str, float]]],
    min_k: int = 0,
    max_k: int = 30,
    score_gap_threshold: float = 0.15,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Apply adaptive-K cutoff to all query results.
    """
    result = {}
    
    total_before = 0
    total_after = 0
    
    for qid, candidates in fused_results.items():
        total_before += len(candidates)
        trimmed = adaptive_k_cutoff(
            candidates,
            min_k=min_k,
            max_k=max_k,
            score_gap_threshold=score_gap_threshold,
        )
        result[qid] = trimmed
        total_after += len(trimmed)
    
    n = len(fused_results)
    if n > 0:
        print(f"  [Adaptive-K] Before: {total_before/n:.1f} avg candidates/query, "
              f"After: {total_after/n:.1f} avg candidates/query")
    
    return result
