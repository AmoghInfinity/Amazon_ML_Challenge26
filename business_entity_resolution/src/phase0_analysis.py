#!/usr/bin/env python3
"""
Phase 0 — Data-driven analysis before any modeling.
Answers key design questions from the PRD.
"""
import pandas as pd
import sys
from collections import Counter

DATA_DIR = r"c:\Users\LENOVO\Team_odyssey_Amazon\dataset"

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print("PHASE 0: DATA ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n--- Loading training data ---")
    s1 = pd.read_csv(f"{DATA_DIR}/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{DATA_DIR}/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{DATA_DIR}/train/train_source3.tsv", sep="\t")
    gt = pd.read_csv(f"{DATA_DIR}/train/train_ground_truth.tsv", sep="\t")

    print(f"Source 1: {len(s1):,} records")
    print(f"Source 2: {len(s2):,} records")
    print(f"Source 3: {len(s3):,} records")
    print(f"Ground truth: {len(gt):,} rows")

    # Country distribution
    print("\n--- Country Distribution ---")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"\n{name}:")
        print(df['country'].value_counts().to_string())

    # Load test data shapes
    print("\n--- Loading test data ---")
    t1 = pd.read_csv(f"{DATA_DIR}/test/test_source1.tsv", sep="\t")
    t2 = pd.read_csv(f"{DATA_DIR}/test/test_source2.tsv", sep="\t")
    t3 = pd.read_csv(f"{DATA_DIR}/test/test_source3.tsv", sep="\t")
    print(f"Test Source 1: {len(t1):,} records")
    print(f"Test Source 2: {len(t2):,} records")
    print(f"Test Source 3: {len(t3):,} records")
    print(f"\nTest S1 countries:")
    print(t1['country'].value_counts().to_string())
    
    # ============================================================
    # KEY QUESTION 1: One-to-one check
    # Does any S2/S3 ID appear in multiple S1 ground truth rows?
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY QUESTION 1: One-to-one relationship check")
    print("=" * 70)
    
    all_match_ids = []
    s1_to_matches = {}
    match_counts = []
    
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']) or str(row['matched_entity_ids']).strip() == '':
            match_counts.append(0)
            s1_to_matches[s1_id] = []
            continue
        ids = str(row['matched_entity_ids']).split(',')
        match_counts.append(len(ids))
        s1_to_matches[s1_id] = ids
        for mid in ids:
            all_match_ids.append((mid.strip(), s1_id))
    
    # Check if any S2/S3 ID appears in multiple S1 rows
    id_to_s1 = {}
    multi_match_ids = []
    for mid, s1_id in all_match_ids:
        if mid not in id_to_s1:
            id_to_s1[mid] = []
        id_to_s1[mid].append(s1_id)
    
    for mid, s1_list in id_to_s1.items():
        if len(s1_list) > 1:
            multi_match_ids.append((mid, s1_list))
    
    print(f"Total unique S2/S3 IDs in ground truth: {len(id_to_s1):,}")
    print(f"S2/S3 IDs matched to >1 S1 entity: {len(multi_match_ids)}")
    if multi_match_ids:
        print("Examples of multi-match S2/S3 IDs:")
        for mid, s1_list in multi_match_ids[:10]:
            print(f"  {mid} -> {s1_list}")
    else:
        print(">>> RESULT: Strict one-to-one relationship holds!")
        print("    Each S2/S3 entity maps to at most one S1 entity.")
    
    # ============================================================
    # KEY QUESTION 2: Match count distribution
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY QUESTION 2: Match count distribution per S1 entity")
    print("=" * 70)
    
    mc = Counter(match_counts)
    total = len(match_counts)
    print(f"Total S1 entities: {total:,}")
    print(f"\nDistribution:")
    for k in sorted(mc.keys()):
        pct = mc[k] / total * 100
        print(f"  {k} matches: {mc[k]:,} ({pct:.1f}%)")
    
    print(f"\nSingleton count (0 matches): {mc[0]:,} ({mc[0]/total*100:.1f}%)")
    print(f"Mean match count: {sum(match_counts)/len(match_counts):.2f}")
    print(f"Max match count: {max(match_counts)}")
    
    # ============================================================
    # KEY QUESTION 3: S2 vs S3 match distribution
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY QUESTION 3: S2 vs S3 match distribution")
    print("=" * 70)
    
    s2_match_counts = []
    s3_match_counts = []
    for s1_id, matches in s1_to_matches.items():
        s2_c = sum(1 for m in matches if m.startswith('S2-'))
        s3_c = sum(1 for m in matches if m.startswith('S3-'))
        s2_match_counts.append(s2_c)
        s3_match_counts.append(s3_c)
    
    s2_mc = Counter(s2_match_counts)
    s3_mc = Counter(s3_match_counts)
    
    print("S2 matches per S1 entity:")
    for k in sorted(s2_mc.keys()):
        print(f"  {k}: {s2_mc[k]:,}")
    
    print("\nS3 matches per S1 entity:")
    for k in sorted(s3_mc.keys()):
        print(f"  {k}: {s3_mc[k]:,}")
    
    # ============================================================
    # KEY QUESTION 4: Missing data analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY QUESTION 4: Missing/empty data fields")
    print("=" * 70)
    
    for name, df in [("S1-train", s1), ("S2-train", s2), ("S3-train", s3)]:
        print(f"\n{name}:")
        for col in ['business_name', 'business_address', 'country']:
            null_count = df[col].isna().sum()
            empty_count = (df[col].astype(str).str.strip() == '').sum()
            print(f"  {col}: {null_count} null, {empty_count} empty ({null_count+empty_count} total missing)")
    
    # ============================================================
    # KEY QUESTION 5: Name/address field lengths
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY QUESTION 5: Field length statistics")
    print("=" * 70)
    
    for name, df in [("S1-train", s1), ("S2-train", s2), ("S3-train", s3)]:
        print(f"\n{name}:")
        for col in ['business_name', 'business_address']:
            lengths = df[col].dropna().astype(str).str.len()
            print(f"  {col}: mean={lengths.mean():.0f}, median={lengths.median():.0f}, "
                  f"min={lengths.min()}, max={lengths.max()}")
    
    # ============================================================
    # Sample matched pairs for noise inspection
    # ============================================================
    print("\n" + "=" * 70)
    print("SAMPLE MATCHED PAIRS (for noise pattern inspection)")
    print("=" * 70)
    
    s1_dict = s1.set_index('entity_id').to_dict('index')
    s2_dict = s2.set_index('entity_id').to_dict('index')
    s3_dict = s3.set_index('entity_id').to_dict('index')
    
    count = 0
    for _, row in gt.iterrows():
        if count >= 15:
            break
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']) or str(row['matched_entity_ids']).strip() == '':
            continue
        ids = str(row['matched_entity_ids']).split(',')
        s1_rec = s1_dict.get(s1_id, {})
        print(f"\n  S1: {s1_id}")
        print(f"    Name: {s1_rec.get('business_name', '?')}")
        print(f"    Addr: {s1_rec.get('business_address', '?')}")
        print(f"    Country: {s1_rec.get('country', '?')}")
        for mid in ids[:3]:  # show first 3 matches
            mid = mid.strip()
            if mid.startswith('S2-'):
                rec = s2_dict.get(mid, {})
            else:
                rec = s3_dict.get(mid, {})
            print(f"  -> {mid}")
            print(f"    Name: {rec.get('business_name', '?')}")
            print(f"    Addr: {rec.get('business_address', '?')}")
            print(f"    Country: {rec.get('country', '?')}")
        count += 1

    print("\n\nPhase 0 analysis complete.")

if __name__ == "__main__":
    main()
