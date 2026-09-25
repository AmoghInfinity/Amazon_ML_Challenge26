"""Quick smoke test — verify imports and basic functionality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

from config import *
print("  [OK] config")

from data.loader import load_source, load_ground_truth, write_matching_results, write_candidate_pairs
print("  [OK] data.loader")

from normalization.names import normalize_name, normalize_name_simple
from normalization.addresses import normalize_address, normalize_address_simple
from normalization.pipeline import normalize_dataframe
print("  [OK] normalization")

from retrieval.lexical import LexicalRetriever
from retrieval.embeddings import EmbeddingRetriever
from retrieval.fusion import reciprocal_rank_fusion, apply_adaptive_k
print("  [OK] retrieval")

from features.pairwise import extract_pair_features, FEATURE_COLUMNS
print("  [OK] features")

from models.pair_matcher import PairMatcher
from models.hard_negatives import HardNegativeMiner
print("  [OK] models")

from decision.entity_resolution import EntityDecisionLayer, compute_macro_f05
print("  [OK] decision")

from evaluation.metrics import evaluate_predictions, evaluate_blocking
print("  [OK] evaluation")

# Test normalization
print("\nTesting name normalization...")
r1 = normalize_name("Custom Wealth Services LLC")
print(f"  Input:  'Custom Wealth Services LLC'")
print(f"  Output: {r1}")

r2 = normalize_name("Pvt. EFS Print Ventures Ltd.")
print(f"  Input:  'Pvt. EFS Print Ventures Ltd.'")
print(f"  Output: {r2}")

r3 = normalize_name("SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com")
print(f"  Input:  'SHIVSHAKTI VIDYALAYA...'")
print(f"  Output: {r3}")

print("\nTesting address normalization...")
a1 = normalize_address("1795 Westchester Dr, High Point, NC")
print(f"  Input:  '1795 Westchester Dr, High Point, NC'")
print(f"  Output: {a1}")

a2 = normalize_address("797, Lake Town Block A, Kolkata, Howrah, West Bengal")
print(f"  Input:  '797, Lake Town Block A, Kolkata...'")
print(f"  Output: {a2}")

# Test feature extraction
print("\nTesting feature extraction...")
s1_rec = {'name_clean': 'custom wealth services', 'addr_clean': '5559 orville avenue columbus oh', 'country_clean': 'us'}
s2_rec = {'name_clean': 'custom wealth svcs', 'addr_clean': '5559 orville ave columbus oh', 'country_clean': 'us'}
feats = extract_pair_features(s1_rec, s2_rec, query_id='S1-1', candidate_id='S2-1')
print(f"  Features: {len(feats)} extracted")
for k, v in feats.items():
    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

# Test decision layer
print("\nTesting decision layer...")
decision = EntityDecisionLayer(match_threshold=0.5, singleton_threshold=0.3)
scored = {
    'S1-1': [('S2-1', 0.9), ('S2-2', 0.8), ('S3-1', 0.1)],
    'S1-2': [('S2-3', 0.2)],
    'S1-3': [],
}
predictions = decision.decide(scored)
print(f"  Predictions: {predictions}")

# Test evaluation
print("\nTesting evaluation...")
gt = {'S1-1': ['S2-1', 'S2-2'], 'S1-2': [], 'S1-3': ['S3-1']}
pred = {'S1-1': ['S2-1', 'S2-2'], 'S1-2': [], 'S1-3': []}
results = evaluate_predictions(pred, gt, verbose=True)

print("\n[OK] All smoke tests passed!")
