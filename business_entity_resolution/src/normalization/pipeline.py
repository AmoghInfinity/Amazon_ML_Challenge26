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
    
    # Simple normalized versions for retrieval
    result['name_clean'] = result['business_name'].fillna('').apply(normalize_name_simple)
    result['addr_clean'] = result['business_address'].fillna('').apply(normalize_address_simple)
    
    # Combined field for embedding
    result['combined'] = result['name_clean'] + ' ' + result['addr_clean']
    
    # Country lowercase
    result['country_clean'] = result['country'].fillna('').str.lower().str.strip()
    
    if verbose:
        print(f"  Done normalizing.")
    
    return result
