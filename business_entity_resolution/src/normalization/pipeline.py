"""Normalization pipeline — applies name and address normalization to dataframes."""
import pandas as pd
from .names import normalize_name, normalize_name_simple
from .addresses import normalize_address, normalize_address_simple


def normalize_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply full normalization pipeline to a source dataframe.
    Adds normalized columns alongside originals.
    """
    if verbose:
        print(f"  Normalizing {len(df):,} records...")
    
    result = df.copy()
    
    from joblib import Parallel, delayed
    import numpy as np
    
    # Split the dataframe into chunks for parallel processing
    num_cores = -1  # Use all cores
    
    def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        c = chunk.copy()
        c['name_clean'] = c['business_name'].fillna('').apply(normalize_name_simple)
        c['addr_clean'] = c['business_address'].fillna('').apply(normalize_address_simple)
        c['combined'] = c['name_clean'] + ' ' + c['addr_clean']
        return c
    
    # Split into 20 chunks
    chunks = np.array_split(result, 20)
    processed_chunks = Parallel(n_jobs=num_cores)(delayed(process_chunk)(c) for c in chunks)
    
    result = pd.concat(processed_chunks)
    
    # Country lowercase
    result['country_clean'] = result['country'].fillna('').str.lower().str.strip()
    
    if verbose:
        print(f"  Done normalizing.")
    
    return result
