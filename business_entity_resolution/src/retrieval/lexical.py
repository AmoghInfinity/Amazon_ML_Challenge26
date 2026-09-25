"""
Lexical retrieval using TF-IDF with character n-grams.
Fast, scalable, works well for name/address matching with typos.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional
import time


class LexicalRetriever:
    """TF-IDF based retrieval with character n-grams."""
    
    def __init__(
        self,
        ngram_range: tuple = (2, 4),
        max_features: int = 100000,
        analyzer: str = 'char_wb',
        top_k: int = 100,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.analyzer = analyzer
        self.top_k = top_k
        
        self.name_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.addr_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        
        self.candidate_ids = None
        self.name_matrix = None
        self.addr_matrix = None
    
    def fit(self, candidates_df: pd.DataFrame):
        """
        Fit the TF-IDF vectorizers on the candidate pool (S2 + S3).
        """
        print("  [LexicalRetriever] Fitting TF-IDF on candidates...")
        t0 = time.time()
        
        self.candidate_ids = candidates_df['entity_id'].values
        
        names = candidates_df['name_clean'].fillna('').values
        addrs = candidates_df['addr_clean'].fillna('').values
        
        self.name_matrix = self.name_vectorizer.fit_transform(names)
        self.addr_matrix = self.addr_vectorizer.fit_transform(addrs)
        
        print(f"  [LexicalRetriever] Fit done in {time.time()-t0:.1f}s. "
              f"Name matrix: {self.name_matrix.shape}, Addr matrix: {self.addr_matrix.shape}")
    
    def retrieve(
        self,
        queries_df: pd.DataFrame,
        top_k: Optional[int] = None,
        batch_size: int = 50,  # Lower batch size to prevent dense array OOM (50 * 10M = 2GB)
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve top-k candidates for each query (S1 entity).
        Returns {s1_id: [(candidate_id, score), ...]} sorted by score descending.
        """
        if top_k is None:
            top_k = self.top_k
        
        query_ids = queries_df['entity_id'].values
        query_names = queries_df['name_clean'].fillna('').values
        query_addrs = queries_df['addr_clean'].fillna('').values
        
        results = {}
        total = len(query_ids)
        
        print(f"  [LexicalRetriever] Retrieving for {total:,} queries (batch_size={batch_size})...")
        t0 = time.time()
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = query_ids[start:end]
            batch_names = query_names[start:end]
            batch_addrs = query_addrs[start:end]
            
            # Transform query batch to dense and transpose (features x batch_size)
            # This makes matrix mult extremely fast: CSR (10M x features) @ Dense (features x batch) -> Dense (10M x batch)
            name_q = self.name_vectorizer.transform(batch_names).toarray().T
            addr_q = self.addr_vectorizer.transform(batch_addrs).toarray().T
            
            # Compute similarities (name + address combined)
            # Resulting shape: (batch_size, 10M)
            name_sim = self.name_matrix.dot(name_q).T
            addr_sim = self.addr_matrix.dot(addr_q).T
            
            # Weighted fusion: name is more important for identity
            combined_sim = 0.6 * name_sim + 0.4 * addr_sim
            
            # Extract top-k for each query
            for i, qid in enumerate(batch_ids):
                row = combined_sim[i]
                
                # Get top-k indices
                if len(row) <= top_k:
                    top_indices = np.argsort(row)[::-1]
                else:
                    top_indices = np.argpartition(row, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
                
                # Filter zero scores
                candidates = []
                for idx in top_indices:
                    score = float(row[idx])
                    if score > 0:
                        candidates.append((self.candidate_ids[idx], score))
                
                results[qid] = candidates
            
            if (start // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                progress = end / total * 100
                print(f"    Progress: {progress:.0f}% ({end:,}/{total:,}), "
                      f"elapsed: {elapsed:.0f}s")
        
        elapsed = time.time() - t0
        print(f"  [LexicalRetriever] Retrieval done in {elapsed:.0f}s")
        
        return results


class TokenBlocker:
    """
    Token-based blocking — generates candidate pairs based on shared tokens.
    Fast first-pass blocking before heavier retrieval.
    """
    
    def __init__(self, min_token_length: int = 3):
        self.min_token_length = min_token_length
        self.token_index = {}  # token -> set of candidate IDs
    
    def build_index(self, candidates_df: pd.DataFrame):
        """Build inverted index on candidate name + address tokens."""
        print("  [TokenBlocker] Building inverted index...")
        t0 = time.time()
        
        self.token_index = {}
        
        for _, row in candidates_df.iterrows():
            eid = row['entity_id']
            text = f"{row.get('name_clean', '')} {row.get('addr_clean', '')}"
            tokens = set(text.lower().split())
            
            for token in tokens:
                if len(token) >= self.min_token_length:
                    if token not in self.token_index:
                        self.token_index[token] = set()
                    self.token_index[token].add(eid)
        
        print(f"  [TokenBlocker] Index built in {time.time()-t0:.1f}s. "
              f"{len(self.token_index):,} unique tokens")
    
    def block(self, queries_df: pd.DataFrame, max_candidates: int = 500) -> Dict[str, set]:
        """Return candidate sets for each query via token overlap."""
        results = {}
        
        for _, row in queries_df.iterrows():
            qid = row['entity_id']
            text = f"{row.get('name_clean', '')} {row.get('addr_clean', '')}"
            tokens = set(text.lower().split())
            
            candidates = set()
            for token in tokens:
                if token in self.token_index and len(token) >= self.min_token_length:
                    # Skip very common tokens (appear in >10% of candidates)
                    if len(self.token_index[token]) < len(self.token_index) * 0.1:
                        candidates.update(self.token_index[token])
            
            # Limit candidate set size
            if len(candidates) > max_candidates:
                # Could prioritize by token frequency, but for now just truncate
                candidates = set(list(candidates)[:max_candidates])
            
            results[qid] = candidates
        
        return results
