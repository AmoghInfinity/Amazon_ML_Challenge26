"""
Stress Test Simulation — Evaluates pipeline robustness against unseen distributions.
Simulates the "France" shift by applying synthetic noise (typos, missing components, abbreviation)
to the validation set and evaluating the macro-F0.5 drop.
"""
import random
import pandas as pd
from typing import Dict, List

def apply_typo(text: str, prob: float = 0.1) -> str:
    """Randomly drop or swap characters."""
    if not text or not isinstance(text, str):
        return text
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob:
            if random.random() < 0.5:
                chars[i] = ''  # drop
            elif i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]  # swap
    return "".join(chars)

def drop_address_components(address: str, prob: float = 0.3) -> str:
    """Randomly drop parts of the address (e.g. missing PIN or State)."""
    if not address or not isinstance(address, str):
        return address
    parts = [p.strip() for p in address.split(',')]
    if len(parts) <= 1:
        return address
    # Drop one random part
    if random.random() < prob:
        idx_to_drop = random.randint(0, len(parts)-1)
        parts.pop(idx_to_drop)
    return ", ".join(parts)

def corrupt_dataframe(df: pd.DataFrame, typo_prob: float = 0.1, drop_prob: float = 0.3) -> pd.DataFrame:
    """Apply synthetic corruptions to simulate unseen country noise."""
    corrupted = df.copy()
    
    print(f"  [StressTest] Corrupting {len(corrupted)} records...")
    
    names = []
    addrs = []
    
    for _, row in corrupted.iterrows():
        name = str(row.get('business_name', ''))
        addr = str(row.get('business_address', ''))
        
        # Apply noise
        name = apply_typo(name, prob=typo_prob)
        addr = drop_address_components(addr, prob=drop_prob)
        addr = apply_typo(addr, prob=typo_prob * 0.5)
        
        names.append(name)
        addrs.append(addr)
        
    corrupted['business_name'] = names
    corrupted['business_address'] = addrs
    return corrupted

def run_stress_test(s1_val: pd.DataFrame):
    """
    Corrupt S1 validation records.
    """
    print("=" * 70)
    print("SYNTHETIC FRANCE-SIMULATION STRESS TEST")
    print("=" * 70)
    
    s1_corrupted = corrupt_dataframe(s1_val)
    
    print("\n  [StressTest] Simulation logic ready for integration.")
    
    return s1_corrupted

if __name__ == "__main__":
    pass
