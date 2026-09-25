"""
Embedding-based retrieval using sentence transformers + FAISS.
Multilingual encoder handles US, India, France without country-specific code.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import os


class EmbeddingRetriever:
    """Sentence-transformer embedding retrieval with FAISS ANN index."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 100,
        batch_size: int = 512,
        use_gpu: bool = True,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        self.model = None
        self.index = None
        self.candidate_ids = None
        self.embeddings = None
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            print(f"  [EmbeddingRetriever] Loading {self.model_name} on {device}...")
            self.model = SentenceTransformer(self.model_name, device=device)
    
    def _encode(self, texts: list, desc: str = "Encoding") -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()
        
        print(f"  [EmbeddingRetriever] {desc} {len(texts):,} texts...")
        t0 = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
            convert_to_numpy=True,
        )
        
        print(f"  [EmbeddingRetriever] {desc} done in {time.time()-t0:.0f}s, "
              f"shape: {embeddings.shape}")
        return embeddings.astype(np.float32)
    
    def fit(self, candidates_df: pd.DataFrame, text_column: str = 'combined'):
        """
        Build FAISS index on candidate embeddings.
        """
        import faiss
        
        self.candidate_ids = candidates_df['entity_id'].values
        texts = candidates_df[text_column].fillna('').tolist()
        
        self.embeddings = self._encode(texts, desc="Candidate encoding")
        
        dim = self.embeddings.shape[1]
        print(f"  [EmbeddingRetriever] Building FAISS index (dim={dim}, n={len(self.embeddings):,})...")
        t0 = time.time()
        
        # Use IVF index for large datasets
        n = len(self.embeddings)
        if n > 100000:
            nlist = min(int(np.sqrt(n)), 4096)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = min(nlist // 4, 64)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        
        print(f"  [EmbeddingRetriever] FAISS index built in {time.time()-t0:.1f}s")
    
    def retrieve(
        self,
        queries_df: pd.DataFrame,
        top_k: Optional[int] = None,
        text_column: str = 'combined',
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve top-k candidates for each query.
        Returns {s1_id: [(candidate_id, score), ...]} sorted by score desc.
        """
        if top_k is None:
            top_k = self.top_k
        
        query_ids = queries_df['entity_id'].values
        texts = queries_df[text_column].fillna('').tolist()
        
        query_embeddings = self._encode(texts, desc="Query encoding")
        
        print(f"  [EmbeddingRetriever] FAISS search for {len(query_ids):,} queries, top_k={top_k}...")
        t0 = time.time()
        
        scores, indices = self.index.search(query_embeddings, top_k)
        
        print(f"  [EmbeddingRetriever] Search done in {time.time()-t0:.1f}s")
        
        results = {}
        for i, qid in enumerate(query_ids):
            candidates = []
            for j in range(top_k):
                idx = indices[i, j]
                if idx >= 0:  # FAISS returns -1 for missing results
                    score = float(scores[i, j])
                    candidates.append((self.candidate_ids[idx], score))
            results[qid] = candidates
        
        return results
    
    def save(self, path: str):
        """Save the FAISS index and metadata."""
        import faiss
        import pickle
        
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "candidate_ids.pkl"), 'wb') as f:
            pickle.dump(self.candidate_ids, f)
    
    def load(self, path: str):
        """Load a saved FAISS index."""
        import faiss
        import pickle
        
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "candidate_ids.pkl"), 'rb') as f:
            self.candidate_ids = pickle.load(f)
