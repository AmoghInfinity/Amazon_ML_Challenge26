"""
Phase 2: Fine-Tuning a Cross-Encoder Reranker.
Trains a HuggingFace CrossEncoder on the hard-negative pairs to score
pairs with extremely high precision. The output of this cross-encoder
will be used as a dominant feature for the final LightGBM model.
"""
import os
import random
import pandas as pd
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample

def fine_tune_cross_encoder(
    model_name: str,
    training_pairs: List[Tuple[str, str, int]],
    s1_records: Dict[str, dict],
    candidate_records: Dict[str, dict],
    output_path: str,
    epochs: int = 2,
    batch_size: int = 16,
):
    print("=" * 70)
    print(f"PHASE 2: FINE-TUNING CROSS ENCODER ({model_name})")
    print("=" * 70)
    
    # We use num_labels=1 for binary classification (0 or 1)
    model = CrossEncoder(model_name, num_labels=1)
    
    print(f"  [CrossEncoder] Preparing {len(training_pairs):,} training pairs...")
    train_examples = []
    
    for s1_id, cand_id, label in training_pairs:
        s1_rec = s1_records.get(s1_id, {})
        cand_rec = candidate_records.get(cand_id, {})
        
        t1 = f"{s1_rec.get('name_clean', '')} {s1_rec.get('addr_clean', '')}"
        t2 = f"{cand_rec.get('name_clean', '')} {cand_rec.get('addr_clean', '')}"
        
        train_examples.append(InputExample(texts=[t1, t2], label=float(label)))
        
    random.shuffle(train_examples)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    print(f"  [CrossEncoder] Starting training for {epochs} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=output_path,
        use_amp=True  # Automatic Mixed Precision for speed
    )
    
    print(f"  [CrossEncoder] Fine-tuning complete. Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import sys
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.loader import load_source, load_ground_truth
    from normalization.pipeline import normalize_dataframe
    from retrieval.lexical import LexicalRetriever
    from models.hard_negatives import HardNegativeMiner
    from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output", type=str, default="models/finetuned_cross_encoder")
    args = parser.parse_args()
    
    print("Loading data...")
    s1 = load_source(TRAIN_S1)
    s2 = load_source(TRAIN_S2)
    s3 = load_source(TRAIN_S3)
    gt = load_ground_truth(TRAIN_GT)
    
    if args.sample_size:
        s1 = s1.head(args.sample_size)
        valid_ids = set(s1['entity_id'].values)
        gt = {k: v for k, v in gt.items() if k in valid_ids}
        
    s1 = normalize_dataframe(s1)
    candidates = pd.concat([normalize_dataframe(s2), normalize_dataframe(s3)], ignore_index=True)
    
    print("Generating hard negatives via lexical retrieval...")
    lexical = LexicalRetriever()
    lexical.fit(candidates)
    lexical_results = lexical.retrieve(s1, top_k=30)
    
    s1_records = {row['entity_id']: row.to_dict() for _, row in s1.iterrows()}
    cand_records = {row['entity_id']: row.to_dict() for _, row in candidates.iterrows()}
    
    miner = HardNegativeMiner()
    training_pairs = miner.generate_training_pairs(
        s1_records, cand_records, gt, lexical_results, neg_ratio=3.0
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fine_tune_cross_encoder(
        model_name="cross-encoder/stsb-distilroberta-base",
        training_pairs=training_pairs,
        s1_records=s1_records,
        candidate_records=cand_records,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
