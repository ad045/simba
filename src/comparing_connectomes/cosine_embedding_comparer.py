import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class CosineEmbeddingEvaluator(NetworkEvaluator):
    """Evaluate networks using high-dimensional cosine-similarity embeddings."""
    
    def __init__(self, embedding_method: str = "spectral", embedding_dim: int = 128):
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
    
    def _spectral_embedding(self, A: torch.Tensor) -> torch.Tensor:
        """Compute spectral embedding of adjacency matrix."""
        # Compute Laplacian
        D = torch.diag(A.sum(dim=1))
        L = D - A
        
        # Compute eigenvectors (use numpy for stability)
        L_np = L.cpu().numpy()
        eigenvalues, eigenvectors = eigh(L_np)
        
        # Take first k eigenvectors
        k = min(self.embedding_dim, A.shape[0] - 1)
        embedding = torch.from_numpy(eigenvectors[:, :k]).float().to(A.device)
        
        return embedding.flatten()
    
    def _node2vec_embedding(self, A: torch.Tensor) -> torch.Tensor:
        """Simplified node embedding based on powers of adjacency matrix."""
        # Use powers of transition matrix as features
        D_inv = torch.diag(1.0 / (A.sum(dim=1) + 1e-8))
        P = D_inv @ A
        
        # Aggregate powers
        embedding = []
        P_k = torch.eye(A.shape[0], device=A.device)
        
        for _ in range(min(5, self.embedding_dim // A.shape[0])):
            P_k = P_k @ P
            embedding.append(P_k.flatten())
        
        embedding = torch.cat(embedding)
        
        # Truncate or pad to embedding_dim
        if embedding.shape[0] > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif embedding.shape[0] < self.embedding_dim:
            padding = torch.zeros(self.embedding_dim - embedding.shape[0], device=A.device)
            embedding = torch.cat([embedding, padding])
        
        return embedding
    
    def _cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between embeddings."""
        dot_product = torch.dot(emb1, emb2)
        norm1 = torch.norm(emb1)
        norm2 = torch.norm(emb2)
        
        similarity = (dot_product / (norm1 * norm2 + 1e-8)).item()
        return similarity
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute cosine similarity of embeddings for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_adj = generated[0]
        
        # Compute embedding for generated network
        if self.embedding_method == "spectral":
            gen_emb = self._spectral_embedding(gen_adj)
        elif self.embedding_method == "node2vec":
            gen_emb = self._node2vec_embedding(gen_adj)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
        
        for target_idx in range(n_targets):
            target_adj = target[target_idx]
            
            # Compute embedding for target network
            if self.embedding_method == "spectral":
                target_emb = self._spectral_embedding(target_adj)
            elif self.embedding_method == "node2vec":
                target_emb = self._node2vec_embedding(target_adj)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(gen_emb, target_emb)
            results[target_idx] = similarity
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"CosineEmb_{self.embedding_method}"