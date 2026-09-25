"""
Pairwise feature extraction for the GBM matcher.
Feature groups: Name (A), Address (B), Semantic (C), Retrieval metadata (D).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re
import time


def _safe_str(val) -> str:
    """Safely convert to string, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


# ============================================================
# Feature Group A: Name Features
# ============================================================

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance (1 = identical)."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.normalized_similarity(s1, s2)
    except ImportError:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        max_len = max(len(s1), len(s2))
        # Simple implementation
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()


def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity."""
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.similarity(s1, s2)
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()


def token_sort_ratio(s1: str, s2: str) -> float:
    """RapidFuzz token sort ratio — handles word reordering."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    except ImportError:
        t1 = ' '.join(sorted(s1.lower().split()))
        t2 = ' '.join(sorted(s2.lower().split()))
        return levenshtein_ratio(t1, t2)


def token_set_ratio(s1: str, s2: str) -> float:
    """RapidFuzz token set ratio — handles subsets."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(s1, s2) / 100.0
    except ImportError:
        t1 = set(s1.lower().split())
        t2 = set(s2.lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / max(len(t1), len(t2))


def partial_ratio(s1: str, s2: str) -> float:
    """RapidFuzz partial ratio — handles substring matches."""
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(s1, s2) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()


def token_jaccard(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity."""
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return intersection / union if union > 0 else 0.0


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    
    ng1 = ngrams(s1.lower(), n)
    ng2 = ngrams(s2.lower(), n)
    
    if not ng1 or not ng2:
        return 0.0
    
    intersection = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return intersection / union if union > 0 else 0.0


def length_ratio(s1: str, s2: str) -> float:
    """Length ratio (shorter/longer)."""
    l1 = len(s1)
    l2 = len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def legal_suffix_match(name1_dict: dict, name2_dict: dict) -> float:
    """Check if legal suffixes match."""
    s1 = name1_dict.get('legal_suffix', '')
    s2 = name2_dict.get('legal_suffix', '')
    if not s1 and not s2:
        return 0.5  # both missing — neutral
    if not s1 or not s2:
        return 0.5  # one missing — neutral
    return 1.0 if s1 == s2 else 0.0


def acronym_match(name1_dict: dict, name2_dict: dict) -> float:
    """Check if acronyms match."""
    a1 = name1_dict.get('acronym', '')
    a2 = name2_dict.get('acronym', '')
    if not a1 or not a2 or len(a1) < 2 or len(a2) < 2:
        return 0.5  # can't determine
    return 1.0 if a1 == a2 else 0.0


def extract_name_features(name1: str, name2: str) -> dict:
    """Extract all name features for a pair."""
    n1 = _safe_str(name1).lower()
    n2 = _safe_str(name2).lower()
    
    return {
        'name_levenshtein': levenshtein_ratio(n1, n2),
        'name_jaro_winkler': jaro_winkler(n1, n2),
        'name_token_sort': token_sort_ratio(n1, n2),
        'name_token_set': token_set_ratio(n1, n2),
        'name_partial_ratio': partial_ratio(n1, n2),
        'name_token_jaccard': token_jaccard(n1, n2),
        'name_char3gram_jaccard': char_ngram_jaccard(n1, n2, 3),
        'name_char4gram_jaccard': char_ngram_jaccard(n1, n2, 4),
        'name_length_ratio': length_ratio(n1, n2),
    }


# ============================================================
# Feature Group B: Address Features
# ============================================================

def number_agreement(nums1: list, nums2: list) -> float:
    """
    Check if numeric components (house numbers, etc.) agree.
    Strong precision signal: mismatched numbers = near-certain non-match.
    """
    if not nums1 or not nums2:
        return 0.5  # can't determine
    
    # Check if any number appears in both
    s1 = set(nums1)
    s2 = set(nums2)
    
    common = s1 & s2
    if common:
        return len(common) / max(len(s1), len(s2))
    
    return 0.0


def pin_zip_match(pz1: str, pz2: str) -> float:
    """Check if PIN/ZIP codes match."""
    if not pz1 or not pz2:
        return 0.5  # can't determine
    return 1.0 if pz1 == pz2 else 0.0


def extract_address_features(addr1: str, addr2: str) -> dict:
    """Extract all address features for a pair."""
    try:
        from ..normalization.addresses import extract_numbers, extract_pin_zip
    except ImportError:
        from normalization.addresses import extract_numbers, extract_pin_zip
    
    a1 = _safe_str(addr1).lower()
    a2 = _safe_str(addr2).lower()
    
    # Numeric components
    nums1 = extract_numbers(a1)
    nums2 = extract_numbers(a2)
    pz1 = extract_pin_zip(a1) or ''
    pz2 = extract_pin_zip(a2) or ''
    
    # Handle empty addresses
    both_empty = (not a1.strip() and not a2.strip())
    one_empty = (not a1.strip()) != (not a2.strip())
    
    return {
        'addr_levenshtein': levenshtein_ratio(a1, a2),
        'addr_jaro_winkler': jaro_winkler(a1, a2),
        'addr_token_sort': token_sort_ratio(a1, a2),
        'addr_token_set': token_set_ratio(a1, a2),
        'addr_token_jaccard': token_jaccard(a1, a2),
        'addr_char3gram_jaccard': char_ngram_jaccard(a1, a2, 3),
        'addr_length_ratio': length_ratio(a1, a2),
        'addr_number_agreement': number_agreement(nums1, nums2),
        'addr_pin_zip_match': pin_zip_match(pz1, pz2),
        'addr_both_empty': float(both_empty),
        'addr_one_empty': float(one_empty),
    }


# ============================================================
# Feature Group D: Retrieval Metadata
# ============================================================

def extract_retrieval_features(
    retriever_results: Dict[str, Dict[str, List[Tuple[str, float]]]],
    query_id: str,
    candidate_id: str,
) -> dict:
    """
    Extract retrieval metadata features.
    Which retrievers found this candidate, their rank and score.
    """
    features = {}
    
    found_in = 0
    for name, results in retriever_results.items():
        query_results = results.get(query_id, [])
        
        rank = -1
        score = 0.0
        for r, (cid, s) in enumerate(query_results, start=1):
            if cid == candidate_id:
                rank = r
                score = s
                break
        
        features[f'retrieval_{name}_rank'] = rank if rank > 0 else 999
        features[f'retrieval_{name}_score'] = score
        features[f'retrieval_{name}_found'] = float(rank > 0)
        
        if rank > 0:
            found_in += 1
    
    features['retrieval_found_in_n'] = found_in
    
    return features


# ============================================================
# Combined feature extraction
# ============================================================

def extract_pair_features(
    s1_record: dict,
    candidate_record: dict,
    retriever_results: Optional[Dict] = None,
    query_id: str = '',
    candidate_id: str = '',
) -> dict:
    """
    Extract all features for a single (S1, candidate) pair.
    """
    features = {}
    
    # Name features (Group A)
    name_feats = extract_name_features(
        s1_record.get('name_clean', s1_record.get('business_name', '')),
        candidate_record.get('name_clean', candidate_record.get('business_name', '')),
    )
    features.update(name_feats)
    
    # Address features (Group B)
    addr_feats = extract_address_features(
        s1_record.get('addr_clean', s1_record.get('business_address', '')),
        candidate_record.get('addr_clean', candidate_record.get('business_address', '')),
    )
    features.update(addr_feats)
    
    # Country match (weak signal by design)
    c1 = _safe_str(s1_record.get('country_clean', s1_record.get('country', ''))).lower()
    c2 = _safe_str(candidate_record.get('country_clean', candidate_record.get('country', ''))).lower()
    features['country_match'] = float(c1 == c2) if c1 and c2 else 0.5
    
    # Source pair feature (S1-S2 vs S1-S3 might have different noise)
    cid = candidate_id or candidate_record.get('entity_id', '')
    features['source_is_s2'] = float(cid.startswith('S2-'))
    features['source_is_s3'] = float(cid.startswith('S3-'))
    
    # Retrieval metadata (Group D)
    if retriever_results:
        ret_feats = extract_retrieval_features(
            retriever_results, query_id, candidate_id
        )
        features.update(ret_feats)
    
    return features


def extract_features_batch(
    s1_records: Dict[str, dict],
    candidate_records: Dict[str, dict],
    pairs: List[Tuple[str, str]],
    retriever_results: Optional[Dict] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Extract features for a batch of (s1_id, candidate_id) pairs.
    Returns a DataFrame with one row per pair.
    """
    print(f"  [Features] Extracting features for {len(pairs):,} pairs...")
    t0 = time.time()
    
    all_features = []
    
    for i, (s1_id, cand_id) in enumerate(pairs):
        s1_rec = s1_records.get(s1_id, {})
        cand_rec = candidate_records.get(cand_id, {})
        
        feats = extract_pair_features(
            s1_rec, cand_rec,
            retriever_results=retriever_results,
            query_id=s1_id,
            candidate_id=cand_id,
        )
        feats['s1_id'] = s1_id
        feats['candidate_id'] = cand_id
        
        all_features.append(feats)
        
        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1:,}/{len(pairs):,} pairs ({rate:.0f}/s)")
    
    df = pd.DataFrame(all_features)
    
    elapsed = time.time() - t0
    print(f"  [Features] Done in {elapsed:.0f}s ({len(pairs)/elapsed:.0f} pairs/s)")
    
    return df


# Feature column names (for model training — exclude ID columns)
FEATURE_COLUMNS = [
    # Name features
    'name_levenshtein', 'name_jaro_winkler', 'name_token_sort', 
    'name_token_set', 'name_partial_ratio', 'name_token_jaccard',
    'name_char3gram_jaccard', 'name_char4gram_jaccard', 'name_length_ratio',
    # Address features
    'addr_levenshtein', 'addr_jaro_winkler', 'addr_token_sort',
    'addr_token_set', 'addr_token_jaccard', 'addr_char3gram_jaccard',
    'addr_length_ratio', 'addr_number_agreement', 'addr_pin_zip_match',
    'addr_both_empty', 'addr_one_empty',
    # Cross features
    'country_match', 'source_is_s2', 'source_is_s3',
]

RETRIEVAL_FEATURE_COLUMNS = [
    'retrieval_lexical_rank', 'retrieval_lexical_score', 'retrieval_lexical_found',
    'retrieval_embedding_rank', 'retrieval_embedding_score', 'retrieval_embedding_found',
    'retrieval_found_in_n',
]
