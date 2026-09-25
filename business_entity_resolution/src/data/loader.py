"""
Data loading and ground truth utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


def load_source(path: str) -> pd.DataFrame:
    """Load a source TSV file."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")
    return df


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """
    Load ground truth file into {s1_id: [matched_ids]} dict.
    Empty match list means singleton.
    """
    gt = pd.read_csv(path, sep="\t", dtype=str)
    result = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            result[s1_id] = []
        else:
            result[s1_id] = [m.strip() for m in str(matched).split(',')]
    return result


def load_all_train(train_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load all training data."""
    import os
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    return s1, s2, s3, gt


def load_all_test(test_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test data."""
    import os
    s1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(test_dir, "test_source3.tsv"))
    return s1, s2, s3


def build_id_lookup(s2: pd.DataFrame, s3: pd.DataFrame) -> Dict[str, dict]:
    """Build a lookup dict from entity_id to record dict for S2 and S3."""
    lookup = {}
    for df in [s2, s3]:
        for _, row in df.iterrows():
            lookup[row['entity_id']] = row.to_dict()
    return lookup


def build_reverse_gt(gt: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build reverse mapping: S2/S3 ID -> list of S1 IDs it matches.
    Used to check one-to-one property and for hard negative mining.
    """
    reverse = defaultdict(list)
    for s1_id, matches in gt.items():
        for mid in matches:
            reverse[mid].append(s1_id)
    return dict(reverse)


def stratified_val_split(
    gt: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split ground truth into train/val, stratified by country and match count.
    Returns (train_gt, val_gt).
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    s1_countries = dict(zip(s1_df['entity_id'], s1_df['country']))
    
    s1_ids = list(gt.keys())
    match_counts = [len(gt[sid]) for sid in s1_ids]
    countries = [s1_countries.get(sid, 'unknown') for sid in s1_ids]
    
    # Bin match counts: 0, 1, 2, 3+
    mc_bins = []
    for mc in match_counts:
        if mc == 0:
            mc_bins.append('0')
        elif mc == 1:
            mc_bins.append('1')
        elif mc == 2:
            mc_bins.append('2')
        else:
            mc_bins.append('3+')
    
    # Create stratification label
    strat_labels = [f"{c}_{m}" for c, m in zip(countries, mc_bins)]
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(s1_ids, strat_labels))
    
    train_gt = {s1_ids[i]: gt[s1_ids[i]] for i in train_idx}
    val_gt = {s1_ids[i]: gt[s1_ids[i]] for i in val_idx}
    
    return train_gt, val_gt


def write_matching_results(results: Dict[str, List[str]], path: str):
    """Write matching_results.tsv."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(results.keys()):
            matches = results[s1_id]
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")


def write_candidate_pairs(candidates: Dict[str, List[str]], path: str):
    """Write candidate_pairs.tsv."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cands = candidates[s1_id]
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
