"""
Evaluation metrics — entity-level and blocking quality metrics.
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def entity_f05(predicted: list, truth: list) -> dict:
    """
    Compute per-entity F0.5 with detailed breakdown.
    """
    pred_set = set(predicted)
    true_set = set(truth)
    
    is_singleton = len(true_set) == 0
    
    if not pred_set and not true_set:
        return {'f05': 1.0, 'precision': 1.0, 'recall': 1.0, 
                'tp': 0, 'fp': 0, 'fn': 0, 'singleton': True, 'singleton_correct': True}
    
    if pred_set and not true_set:
        return {'f05': 0.0, 'precision': 0.0, 'recall': 0.0,
                'tp': 0, 'fp': len(pred_set), 'fn': 0, 'singleton': True, 'singleton_correct': False}
    
    if not pred_set and true_set:
        return {'f05': 0.0, 'precision': 0.0, 'recall': 0.0,
                'tp': 0, 'fp': 0, 'fn': len(true_set), 'singleton': False, 'singleton_correct': False}
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        f05 = 0.0
    else:
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    return {'f05': f05, 'precision': precision, 'recall': recall,
            'tp': tp, 'fp': fp, 'fn': fn, 'singleton': False, 'singleton_correct': False}


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    verbose: bool = True,
) -> dict:
    """
    Full evaluation: macro-F0.5, singleton metrics, per-country breakdown.
    """
    entity_scores = []
    singleton_results = {'correct': 0, 'total_true': 0, 'total_pred': 0, 'false_pos': 0}
    
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, [])
        truth = ground_truth[s1_id]
        
        result = entity_f05(pred, truth)
        entity_scores.append(result)
        
        # Singleton tracking
        if len(truth) == 0:
            singleton_results['total_true'] += 1
            if len(pred) == 0:
                singleton_results['correct'] += 1
            else:
                singleton_results['false_pos'] += len(pred)
        
        if len(pred) == 0:
            singleton_results['total_pred'] += 1
    
    # Macro averages
    f05_scores = [r['f05'] for r in entity_scores]
    macro_f05 = np.mean(f05_scores)
    macro_precision = np.mean([r['precision'] for r in entity_scores])
    macro_recall = np.mean([r['recall'] for r in entity_scores])
    
    total_tp = sum(r['tp'] for r in entity_scores)
    total_fp = sum(r['fp'] for r in entity_scores)
    total_fn = sum(r['fn'] for r in entity_scores)
    
    # Micro F0.5
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if micro_p + micro_r > 0:
        micro_f05 = 1.25 * micro_p * micro_r / (0.25 * micro_p + micro_r)
    else:
        micro_f05 = 0.0
    
    results = {
        'macro_f05': macro_f05,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_f05': micro_f05,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'singleton_precision': (singleton_results['correct'] / singleton_results['total_pred'] 
                               if singleton_results['total_pred'] > 0 else 0.0),
        'singleton_recall': (singleton_results['correct'] / singleton_results['total_true']
                            if singleton_results['total_true'] > 0 else 0.0),
        'n_entities': len(ground_truth),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Entities evaluated:     {results['n_entities']:,}")
        print(f"  Macro F0.5 (primary):   {results['macro_f05']:.4f}")
        print(f"  Macro Precision:        {results['macro_precision']:.4f}")
        print(f"  Macro Recall:           {results['macro_recall']:.4f}")
        print(f"  Micro F0.5:             {results['micro_f05']:.4f}")
        print(f"  Total TP/FP/FN:         {total_tp:,} / {total_fp:,} / {total_fn:,}")
        print(f"  Singleton precision:    {results['singleton_precision']:.4f}")
        print(f"  Singleton recall:       {results['singleton_recall']:.4f}")
        print(f"  True singletons:        {singleton_results['total_true']:,}")
        print(f"  Predicted singletons:   {singleton_results['total_pred']:,}")
    
    return results


def evaluate_blocking(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    verbose: bool = True,
) -> dict:
    """
    Evaluate blocking quality: recall ceiling, reduction ratio, candidate counts.
    """
    total_recall = 0
    total_true_matches = 0
    total_candidates = 0
    candidate_counts = []
    missed_entities = 0
    
    for s1_id in ground_truth:
        true_matches = set(ground_truth[s1_id])
        cands = set(candidates.get(s1_id, []))
        
        total_candidates += len(cands)
        candidate_counts.append(len(cands))
        
        if true_matches:
            found = len(true_matches & cands)
            total_recall += found
            total_true_matches += len(true_matches)
            if found == 0:
                missed_entities += 1
    
    n_entities = len(ground_truth)
    recall_ceiling = total_recall / total_true_matches if total_true_matches > 0 else 0.0
    
    cc = np.array(candidate_counts)
    
    results = {
        'recall_ceiling': recall_ceiling,
        'total_true_matches': total_true_matches,
        'total_candidates': total_candidates,
        'mean_candidates': np.mean(cc),
        'median_candidates': np.median(cc),
        'p95_candidates': np.percentile(cc, 95),
        'max_candidates': np.max(cc) if len(cc) > 0 else 0,
        'missed_entities': missed_entities,
        'missed_entity_pct': missed_entities / n_entities * 100 if n_entities > 0 else 0,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BLOCKING QUALITY")
        print(f"{'='*60}")
        print(f"  Recall ceiling:         {results['recall_ceiling']:.4f}")
        print(f"  Total true matches:     {results['total_true_matches']:,}")
        print(f"  Total candidates:       {results['total_candidates']:,}")
        print(f"  Mean candidates/query:  {results['mean_candidates']:.1f}")
        print(f"  Median candidates/query:{results['median_candidates']:.1f}")
        print(f"  P95 candidates/query:   {results['p95_candidates']:.1f}")
        print(f"  Max candidates/query:   {results['max_candidates']}")
        print(f"  Completely missed:      {results['missed_entities']:,} ({results['missed_entity_pct']:.1f}%)")
    
    return results
