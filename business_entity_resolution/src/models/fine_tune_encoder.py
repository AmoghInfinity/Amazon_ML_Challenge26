"""
Phase 2: Fine-Tuning the Dual-Encoder.
Takes the ground truth and hard negatives mined in Phase 1, and fine-tunes
the multilingual sentence transformer to push matches together and pull
hard negatives apart in the embedding space.
"""
import os
import torch
import random
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def fine_tune_dual_encoder(
    model_name: str,
    training_pairs: List[Tuple[str, str, int]],
    s1_records: Dict[str, dict],
    candidate_records: Dict[str, dict],
    output_path: str,
    epochs: int = 2,
    batch_size: int = 32,
    text_column: str = 'combined',
):
    """
    Fine-tune SentenceTransformer on generated hard-negative pairs.
    """
    print("=" * 70)
    print(f"PHASE 2: FINE-TUNING DUAL ENCODER ({model_name})")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  [FineTune] Loading model on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    # Create InputExamples
    print(f"  [FineTune] Preparing {len(training_pairs):,} training pairs...")
    train_examples = []
    
    for s1_id, cand_id, label in training_pairs:
        # Reconstruct the text
        # Assumes the records have been normalized and 'combined' or similar exists,
        # or we construct it on the fly.
        s1_rec = s1_records.get(s1_id, {})
        cand_rec = candidate_records.get(cand_id, {})
        
        # Fallback formatting if text_column doesn't exist directly
        t1 = s1_rec.get(text_column, f"{s1_rec.get('name_clean', '')} {s1_rec.get('addr_clean', '')}")
        t2 = cand_rec.get(text_column, f"{cand_rec.get('name_clean', '')} {cand_rec.get('addr_clean', '')}")
        
        # SentenceTransformers takes float labels for ContrastiveLoss
        train_examples.append(InputExample(texts=[t1, t2], label=float(label)))
        
    random.shuffle(train_examples)
    
    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Contrastive Loss pushes label=1 closer and label=0 further apart
    train_loss = losses.ContrastiveLoss(model=model)
    
    print(f"  [FineTune] Starting training for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=output_path,
        save_best_model=True
    )
    
    print(f"  [FineTune] Fine-tuning complete. Model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    pass
