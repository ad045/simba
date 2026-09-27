# import torch
# import numpy as np
# from typing import Dict, List
# from scipy.linalg import eigh
# from scipy.spatial.distance import cosine
# from src.comparing_connectomes.base_comparer import NetworkEvaluator


# class GromovWassersteinEvaluator(NetworkEvaluator):
#     """Evaluate networks using Gromov-Wasserstein distance."""
    
#     def __init__(self, max_iter: int = 100, tol: float = 1e-7):
#         self.max_iter = max_iter
#         self.tol = tol
    
#     def _compute_gw_distance(self, C1: torch.Tensor, C2: torch.Tensor) -> float:
#         """Compute Gromov-Wasserstein distance between two cost matrices."""
#         print("C1 shape:", C1.shape, "C2 shape:", C2.shape) # C1 shape: torch.Size([100, 100]) C2 shape: torch.Size([100, 100])
#         n1, n2 = C1.shape[0], C2.shape[0]
        
#         # Uniform distributions
#         p = torch.ones(n1, device=C1.device) / n1
#         q = torch.ones(n2, device=C2.device) / n2
#         # print("p:", p) # p: tensor([0.0100, 0.0100,....])
#         # print("q:", q) # q: tensor([0.0100, 0.0100,....])
#         # Initialize transport plan
#         T = torch.outer(p, q)
#         # print("Initial T:", T) # Initial T: tensor([[1.0000e-04, 1.0000e-04, 1.0000e-04,  ..., 1.0000e-04, 1.0000e-04,
#                                 # 1.0000e-04],
#                                 # [1.0000e-04, 1.0000e-04, 1.0000e-04,  ..., 1.0000e-04, 1.0000e-04,
#                                 # 1.0000e-04],
        
#         # Entropic regularization parameter
#         epsilon = 0.01
        
#         for _ in range(self.max_iter):
#             # Compute loss matrix
#             f1 = C1 @ T @ torch.ones(n2, device=C1.device)
#             f2 = torch.ones(n1, device=C1.device) @ T @ C2
#             print("f1:", f1) # only nans
#             print("f2:", f2) # only nans 
#             L = (C1.pow(2) @ torch.ones((n1, n2), device=C1.device) + 
#                  torch.ones((n1, n2), device=C1.device) @ C2.pow(2).T - 
#                  2 * C1 @ T @ C2.T)

#             # print("L:", L) # only nans
#             # Sinkhorn iterations
#             K = torch.exp(-L / epsilon)
#             u = p / (K @ (q / (K.T @ p)))
#             T_new = torch.diag(u) @ K @ torch.diag(q / (K.T @ p))
            
#             # print("T_new:", T_new)
#             if torch.norm(T_new - T) < self.tol:
#                 break
#             T = T_new
#             # print("T updated:", T)
        
#         # Compute final distance
#         distance = torch.sum(L * T)
#         return distance.item()
    
#     def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
#         """Compute Gromov-Wasserstein distance for generated network against target batch."""
#         results = {}
#         n_targets = target.shape[0]
        
#         # Convert adjacency to distance matrix for generated network
#         gen_adj = generated[0]
        
#         for target_idx in range(n_targets):
#             target_adj = target[target_idx]
            
#             # Compute GW distance
#             distance = self._compute_gw_distance(gen_adj, target_adj)
#             results[target_idx] = distance
        
#         return results
    
#     @property
#     def metric_prefix(self) -> str:
#         return "GW"

import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class GromovWassersteinEvaluator(NetworkEvaluator):
    """Evaluate networks using Gromov-Wasserstein distance."""
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-7):
        self.max_iter = max_iter
        self.tol = tol
    
    def _compute_gw_distance(self, C1: torch.Tensor, C2: torch.Tensor) -> float:
        """Compute Gromov-Wasserstein distance between two cost matrices."""
        # Check for invalid inputs
        if torch.isnan(C1).any() or torch.isnan(C2).any():
            return float('nan')
        if torch.isinf(C1).any() or torch.isinf(C2).any():
            return float('nan')
        
        n1, n2 = C1.shape[0], C2.shape[0]
        
        # Convert adjacency matrices to distance matrices
        # Normalize first to avoid extreme values
        C1_norm = C1 / (C1.abs().max() + 1e-8)
        C2_norm = C2 / (C2.abs().max() + 1e-8)
        
        # Use normalized adjacency directly as cost (higher = more similar = lower cost)
        # So we invert: cost = 1 - similarity
        D1 = 1.0 - C1_norm.clamp(0, 1)
        D2 = 1.0 - C2_norm.clamp(0, 1)
        
        # Make symmetric and add small diagonal for numerical stability
        D1 = (D1 + D1.T) / 2 + torch.eye(n1, device=C1.device) * 1e-6
        D2 = (D2 + D2.T) / 2 + torch.eye(n2, device=C2.device) * 1e-6
        
        # Uniform distributions
        p = torch.ones(n1, device=C1.device, dtype=torch.float32) / n1
        q = torch.ones(n2, device=C2.device, dtype=torch.float32) / n2
        
        # Initialize transport plan
        T = torch.outer(p, q)
        
        # Entropic regularization parameter
        epsilon = 0.5  # Even higher for stability
        
        for iteration in range(self.max_iter):
            # Compute the cost tensor for GW
            constC1 = torch.outer(D1.pow(2).sum(dim=1), torch.ones(n2, device=C1.device))
            constC2 = torch.outer(torch.ones(n1, device=C1.device), D2.pow(2).sum(dim=1))
            constC = constC1 + constC2
            
            hC1 = D1 @ T
            hC2 = T @ D2.T
            L = constC - 2 * hC1 * hC2
            
            # Check for NaN in L
            if torch.isnan(L).any():
                return float('nan')
            
            # Clip extreme values in L
            L = L.clamp(-1e10, 1e10)
            
            # Log-stabilized Sinkhorn
            log_K = -L / epsilon
            log_K = log_K - log_K.max()  # Stabilization
            log_K = log_K.clamp(-50, 50)  # Prevent extreme values
            K = torch.exp(log_K)
            
            # Sinkhorn iterations with stabilization
            Kv = K @ q
            u = p / (Kv.clamp(min=1e-10))
            Ktu = K.T @ u
            v = q / (Ktu.clamp(min=1e-10))
            
            # Check for NaN/Inf in scaling factors
            if torch.isnan(u).any() or torch.isnan(v).any():
                return float('nan')
            if torch.isinf(u).any() or torch.isinf(v).any():
                return float('nan')
            
            T_new = torch.outer(u, v) * K
            
            # Check convergence
            err = torch.norm(T_new - T)
            if torch.isnan(err):
                return float('nan')
            if err < self.tol:
                break
            T = T_new
        
        # Compute final distance
        distance = torch.sum(L * T)
        
        # Final check
        if torch.isnan(distance) or torch.isinf(distance):
            return float('nan')
        
        return distance.item()
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Gromov-Wasserstein distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        # Convert adjacency to distance matrix for generated network
        gen_adj = generated[0]
        
        for target_idx in range(n_targets):
            target_adj = target[target_idx]
            
            # Compute GW distance
            distance = self._compute_gw_distance(gen_adj, target_adj)
            results[target_idx] = distance
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "GW"


class GraphKernelEvaluator(NetworkEvaluator):
    """Evaluate networks using advanced graph kernels."""
    
    def __init__(self, kernel_type: str = "weisfeiler_lehman", n_iterations: int = 5):
        self.kernel_type = kernel_type
        self.n_iterations = n_iterations
    
    def _weisfeiler_lehman_kernel(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute Weisfeiler-Lehman graph kernel."""
        n1, n2 = A1.shape[0], A2.shape[0]
        
        # Initialize node labels
        labels1 = torch.zeros(n1, dtype=torch.long)
        labels2 = torch.zeros(n2, dtype=torch.long)
        
        similarity = 0.0
        
        for _ in range(self.n_iterations):
            # Compute label histograms
            hist1 = torch.bincount(labels1, minlength=max(labels1.max(), labels2.max()) + 1)
            hist2 = torch.bincount(labels2, minlength=max(labels1.max(), labels2.max()) + 1)
            
            # Add kernel value (histogram intersection)
            similarity += torch.min(hist1.float(), hist2.float()).sum().item()
            
            # Update labels based on neighborhood
            new_labels1 = labels1.clone()
            new_labels2 = labels2.clone()
            
            for i in range(n1):
                neighbors = A1[i].nonzero(as_tuple=True)[0]
                neighbor_labels = labels1[neighbors].sort()[0]
                new_labels1[i] = hash(tuple(neighbor_labels.tolist())) % 10000
            
            for i in range(n2):
                neighbors = A2[i].nonzero(as_tuple=True)[0]
                neighbor_labels = labels2[neighbors].sort()[0]
                new_labels2[i] = hash(tuple(neighbor_labels.tolist())) % 10000
            
            labels1 = new_labels1
            labels2 = new_labels2
        
        return similarity
    
    def _random_walk_kernel(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute random walk graph kernel."""
        # Normalize adjacency matrices
        D1 = torch.diag(A1.sum(dim=1).pow(-0.5))
        D2 = torch.diag(A2.sum(dim=1).pow(-0.5))
        
        P1 = D1 @ A1 @ D1
        P2 = D2 @ A2 @ D2
        
        # Compute kernel via spectral decomposition
        lambda_param = 0.1
        n1, n2 = A1.shape[0], A2.shape[0]
        
        # Direct product kernel (simplified)
        kernel_val = torch.trace(
            torch.linalg.matrix_power(torch.eye(n1, device=A1.device) - lambda_param * P1, -1)
        ).item()
        
        return kernel_val
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute graph kernel similarity for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_adj = generated[0]
        
        for target_idx in range(n_targets):
            target_adj = target[target_idx]
            
            if self.kernel_type == "weisfeiler_lehman":
                similarity = self._weisfeiler_lehman_kernel(gen_adj, target_adj)
            elif self.kernel_type == "random_walk":
                similarity = self._random_walk_kernel(gen_adj, target_adj)
            else:
                raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
            results[target_idx] = similarity
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"GraphKernel_{self.kernel_type}"


class MultiplexLayerSimilarityEvaluator(NetworkEvaluator):
    """Evaluate multiplex networks using layer-similarity measures."""
    
    def __init__(self, similarity_type: str = "jaccard"):
        self.similarity_type = similarity_type
    
    def _jaccard_similarity(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute Jaccard similarity between edge sets."""
        A1_binary = (A1 > 0).float()
        A2_binary = (A2 > 0).float()
        
        intersection = (A1_binary * A2_binary).sum()
        union = ((A1_binary + A2_binary) > 0).float().sum()
        
        return (intersection / union).item() if union > 0 else 0.0
    
    def _overlap_coefficient(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute overlap coefficient between edge sets."""
        A1_binary = (A1 > 0).float()
        A2_binary = (A2 > 0).float()
        
        intersection = (A1_binary * A2_binary).sum()
        min_size = min(A1_binary.sum(), A2_binary.sum())
        
        return (intersection / min_size).item() if min_size > 0 else 0.0
    
    def _frobenius_similarity(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute normalized Frobenius similarity."""
        # Normalize matrices
        A1_norm = A1 / (torch.norm(A1, p='fro') + 1e-8)
        A2_norm = A2 / (torch.norm(A2, p='fro') + 1e-8)
        
        # Compute inner product
        similarity = torch.sum(A1_norm * A2_norm).item()
        return similarity
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute multiplex layer similarity for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_adj = generated[0]
        
        for target_idx in range(n_targets):
            target_adj = target[target_idx]
            
            if self.similarity_type == "jaccard":
                similarity = self._jaccard_similarity(gen_adj, target_adj)
            elif self.similarity_type == "overlap":
                similarity = self._overlap_coefficient(gen_adj, target_adj)
            elif self.similarity_type == "frobenius":
                similarity = self._frobenius_similarity(gen_adj, target_adj)
            else:
                raise ValueError(f"Unknown similarity type: {self.similarity_type}")
            
            results[target_idx] = similarity
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"MultiplexSim_{self.similarity_type}"


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