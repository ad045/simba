import torch
from typing import Dict, List, Optional
from src.comparing_connectomes.base_comparer import NetworkEvaluator


def ks_statistic(
    samples_1: torch.Tensor,
    samples_2: torch.Tensor,
) -> torch.Tensor:
    """Compute Kolmogorov-Smirnov statistics between all pairs of distributions.
    
    Args:
        samples_1: First batch of samples with shape [batch_1, num_samples_1]
        samples_2: Second batch of samples with shape [batch_2, num_samples_2]
    
    Returns:
        KS statistics for all pairs with shape [batch_1, batch_2]
    """
    # Sort samples for CDF computation
    sorted_1, _ = torch.sort(samples_1, dim=1)  # [batch_1, n_samples_1]
    sorted_2, _ = torch.sort(samples_2, dim=1)  # [batch_2, n_samples_2]
    
    # Get all unique values that could be CDF evaluation points
    all_values = torch.unique(
        torch.cat([sorted_1.reshape(-1), sorted_2.reshape(-1)])
    )  # [n_unique]
    
    # Compute CDFs for all distributions at these points
    cdf_1 = (
        (sorted_1.unsqueeze(-1) <= all_values.unsqueeze(0).unsqueeze(0))
        .float()
        .mean(dim=1)
    )
    
    cdf_2 = (
        (sorted_2.unsqueeze(-1) <= all_values.unsqueeze(0).unsqueeze(0))
        .float()
        .mean(dim=1)
    )
    
    # Compute absolute differences between all pairs of CDFs
    differences = torch.abs(
        cdf_1.unsqueeze(1) - cdf_2.unsqueeze(0)
    )  # [batch_1, batch_2, n_unique]
    
    # Get maximum difference for each pair
    ks_statistics = torch.max(differences, dim=2).values  # [batch_1, batch_2]
    
    return ks_statistics


def compute_degree_distribution(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Compute node degrees for binary networks.
    
    Args:
        adjacency_matrix: Shape [num_networks, num_nodes, num_nodes]
    
    Returns:
        Node degrees with shape [num_networks, num_nodes]
    """
    return adjacency_matrix.sum(dim=-1)


def compute_clustering_coefficients(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Compute clustering coefficients for binary networks.
    
    Args:
        adjacency_matrix: Shape [num_networks, num_nodes, num_nodes]
    
    Returns:
        Clustering coefficients with shape [num_networks, num_nodes]
    """
    degrees = adjacency_matrix.sum(dim=-1)
    number_of_pairs = degrees * (degrees - 1)
    
    # Count triangles: (A^3)_ii / 2
    number_of_triangles = torch.diagonal(
        torch.matmul(
            torch.matmul(adjacency_matrix, adjacency_matrix), adjacency_matrix
        ),
        dim1=-2,
        dim2=-1,
    )
    
    clustering = torch.zeros_like(number_of_triangles)
    mask = number_of_pairs > 0
    clustering[mask] = number_of_triangles[mask] / number_of_pairs[mask]
    
    return clustering


def compute_betweenness_centrality(
    adjacency_matrix: torch.Tensor,
    normalised: bool = True
) -> torch.Tensor:
    """Compute betweenness centrality for binary networks using Floyd-Warshall.
    
    Args:
        adjacency_matrix: Shape [num_networks, num_nodes, num_nodes]
        normalised: Whether to normalize by number of node pairs
    
    Returns:
        Betweenness centrality with shape [num_networks, num_nodes]
    """
    device = adjacency_matrix.device
    num_networks, num_nodes, _ = adjacency_matrix.shape
    
    # Initialize distance matrix
    distances = adjacency_matrix.clone().float()
    distances[distances == 0] = torch.inf
    diag_idx = torch.arange(num_nodes, device=device)
    distances[:, diag_idx, diag_idx] = 0
    
    # Initialize path count matrix
    path_counts = torch.zeros_like(distances)
    path_counts[adjacency_matrix > 0] = 1
    path_counts[:, diag_idx, diag_idx] = 1
    
    # Floyd-Warshall with path counting
    for k in range(num_nodes):
        dist_ik = distances[:, :, k].unsqueeze(-1)
        dist_kj = distances[:, k, :].unsqueeze(-2)
        new_dist = dist_ik + dist_kj
        
        count_ik = path_counts[:, :, k].unsqueeze(-1)
        count_kj = path_counts[:, k, :].unsqueeze(-2)
        new_count = count_ik * count_kj
        
        is_shorter = new_dist < distances
        is_equal = torch.isclose(new_dist, distances)
        
        distances[is_shorter] = new_dist[is_shorter]
        path_counts[is_shorter] = new_count[is_shorter]
        path_counts[is_equal] += new_count[is_equal]
    
    # Compute betweenness centrality
    betweenness = torch.zeros(num_networks, num_nodes, device=device)
    sigma_no_zeros = torch.where(
        path_counts == 0, torch.ones_like(path_counts), path_counts
    )
    
    for v in range(num_nodes):
        dist_sv = distances[:, :, v].unsqueeze(2)
        dist_vt = distances[:, v, :].unsqueeze(1)
        
        sigma_sv = path_counts[:, :, v].unsqueeze(2)
        sigma_vt = path_counts[:, v, :].unsqueeze(1)
        
        is_on_path = torch.isclose(distances, dist_sv + dist_vt)
        sigma_st_v = sigma_sv * sigma_vt
        pair_dependency = sigma_st_v / sigma_no_zeros
        
        dependency_v = torch.where(
            is_on_path, pair_dependency, torch.zeros_like(pair_dependency)
        )
        
        dependency_v.diagonal(dim1=-2, dim2=-1).fill_(0)
        dependency_v[:, v, :] = 0
        dependency_v[:, :, v] = 0
        
        betweenness[:, v] = dependency_v.sum(dim=(-1, -2))
    
    betweenness /= 2.0
    
    if normalised and num_nodes > 2:
        norm_factor = ((num_nodes - 1) * (num_nodes - 2)) / 2.0
        if norm_factor > 0:
            betweenness /= norm_factor
    
    return betweenness


class StandaloneKSEvaluator(NetworkEvaluator):
    """Evaluate networks using KS-based energy without GNM dependencies.
    
    Computes KS statistics for degree, clustering, and betweenness distributions.
    """
    
    def __init__(
        self,
        use_degree: bool = True,
        use_clustering: bool = True,
        use_betweenness: bool = True,
        aggregation: str = "max"  # "max", "mean", or "sum"
    ):
        """
        Args:
            use_degree: Whether to include degree distribution KS
            use_clustering: Whether to include clustering coefficient KS
            use_betweenness: Whether to include betweenness centrality KS
            aggregation: How to combine multiple KS statistics ("max", "mean", "sum")
        """
        self.use_degree = use_degree
        self.use_clustering = use_clustering
        self.use_betweenness = use_betweenness
        self.aggregation = aggregation
        
        # Build list of property names being used
        self.properties = []
        if use_degree:
            self.properties.append("degree")
        if use_clustering:
            self.properties.append("clustering")
        if use_betweenness:
            self.properties.append("betweenness")
    
    def __call__(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute KS energy for generated network against target batch.
        
        Args:
            generated: Generated network [1, num_nodes, num_nodes]
            target: Target networks [n_targets, num_nodes, num_nodes]
        
        Returns:
            Dictionary mapping target_idx to energy value
        """
        results = {}
        n_targets = target.shape[0]
        
        # Compute all KS statistics
        ks_values = []
        
        if self.use_degree:
            gen_degrees = compute_degree_distribution(generated)
            tgt_degrees = compute_degree_distribution(target)
            degree_ks = ks_statistic(gen_degrees, tgt_degrees)  # [1, n_targets]
            ks_values.append(degree_ks)
        
        if self.use_clustering:
            gen_clustering = compute_clustering_coefficients(generated)
            tgt_clustering = compute_clustering_coefficients(target)
            clustering_ks = ks_statistic(gen_clustering, tgt_clustering)  # [1, n_targets]
            ks_values.append(clustering_ks)
        
        if self.use_betweenness:
            gen_betweenness = compute_betweenness_centrality(generated)
            tgt_betweenness = compute_betweenness_centrality(target)
            betweenness_ks = ks_statistic(gen_betweenness, tgt_betweenness)  # [1, n_targets]
            ks_values.append(betweenness_ks)
        
        # Stack and aggregate
        ks_tensor = torch.stack(ks_values, dim=0)  # [n_properties, 1, n_targets]
        
        if self.aggregation == "max":
            energy = ks_tensor.max(dim=0).values  # [1, n_targets]
        elif self.aggregation == "mean":
            energy = ks_tensor.mean(dim=0)  # [1, n_targets]
        elif self.aggregation == "sum":
            energy = ks_tensor.sum(dim=0)  # [1, n_targets]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Extract per-target results
        for target_idx in range(n_targets):
            results[target_idx] = float(energy[0, target_idx].item())
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        """Return prefix for metric names."""
        props = "_".join(self.properties)
        return f"KS_{self.aggregation}_{props}"


# class DetailedKSEvaluator(NetworkEvaluator):
#     """Evaluate networks with separate KS scores for each property."""
    
#     def __init__(
#         self,
#         use_degree: bool = True,
#         use_clustering: bool = True,
#         use_betweenness: bool = True,
#     ):
#         """
#         Args:
#             use_degree: Whether to compute degree distribution KS
#             use_clustering: Whether to compute clustering coefficient KS
#             use_betweenness: Whether to compute betweenness centrality KS
#         """
#         self.use_degree = use_degree
#         self.use_clustering = use_clustering
#         self.use_betweenness = use_betweenness
    
#     def __call__(
#         self,
#         generated: torch.Tensor,
#         target: torch.Tensor
#     ) -> Dict[str, float]:
#         """Compute individual KS energies for each property.
        
#         Args:
#             generated: Generated network [1, num_nodes, num_nodes]
#             target: Target networks [n_targets, num_nodes, num_nodes]
        
#         Returns:
#             Dictionary with keys like 'target_0_degree', 'target_1_clustering', etc.
#         """
#         results = {}
#         n_targets = target.shape[0]
        
#         if self.use_degree:
#             gen_degrees = compute_degree_distribution(generated)
#             tgt_degrees = compute_degree_distribution(target)
#             degree_ks = ks_statistic(gen_degrees, tgt_degrees)  # [1, n_targets]
            
#             for target_idx in range(n_targets):
#                 results[f"target_{target_idx}_degree"] = float(
#                     degree_ks[0, target_idx].item()
#                 )
        
#         if self.use_clustering:
#             gen_clustering = compute_clustering_coefficients(generated)
#             tgt_clustering = compute_clustering_coefficients(target)
#             clustering_ks = ks_statistic(gen_clustering, tgt_clustering)
            
#             for target_idx in range(n_targets):
#                 results[f"target_{target_idx}_clustering"] = float(
#                     clustering_ks[0, target_idx].item()
#                 )
        
#         if self.use_betweenness:
#             gen_betweenness = compute_betweenness_centrality(generated)
#             tgt_betweenness = compute_betweenness_centrality(target)
#             betweenness_ks = ks_statistic(gen_betweenness, tgt_betweenness)
            
#             for target_idx in range(n_targets):
#                 results[f"target_{target_idx}_betweenness"] = float(
#                     betweenness_ks[0, target_idx].item()
#                 )
        
#         return results
    
#     @property
#     def metric_prefix(self) -> str:
#         """Return prefix for metric names."""
#         return "KS_detailed"