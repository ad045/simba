from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from numba import njit
import warnings
from functools import lru_cache
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Callable, Tuple, TypeVar, Any
import numpy.typing as npt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.metrics.pairwise import cosine_similarity
from kayson_generative import resistance_distance, shortest_path_distance, propagation_distance, topological_distance
import pandas as pd
import bct
from sklearn.decomposition import PCA
from collections import defaultdict


def evaluator(synthetic, empirical, euclidean_distance):
    """Network comparison using multiple metrics."""
    degrees_synthetic = np.sum(synthetic, axis=0)
    degrees_empirical = np.sum(empirical, axis=0)
    ks_degree = ks_2samp(degrees_synthetic, degrees_empirical)
    
    clustering_synthetic = nx.clustering(nx.from_numpy_array(synthetic))
    clustering_empirical = nx.clustering(nx.from_numpy_array(empirical))
    ks_clustering = ks_2samp(list(clustering_synthetic.values()), 
                            list(clustering_empirical.values()))
    
    betweenness_synthetic = nx.betweenness_centrality(nx.from_numpy_array(synthetic))
    betweenness_empirical = nx.betweenness_centrality(nx.from_numpy_array(empirical))
    ks_betweenness = ks_2samp(list(betweenness_synthetic.values()), 
                              list(betweenness_empirical.values()))
    
    distance_synthetic = euclidean_distance[np.triu(synthetic, 1) > 0]
    distance_empirical = euclidean_distance[np.triu(empirical, 1) > 0]
    ks_distance = ks_2samp(distance_synthetic, distance_empirical)
    
    results = {
        "energy": np.max([
            ks_degree[0], 
            ks_clustering[0], 
            ks_betweenness[0], 
            ks_distance[0]
        ]),
        "ks_degrees": ks_degree[0],
        "ks_clustering": ks_clustering[0],
        "ks_betweenness": ks_betweenness[0],
        "ks_distance": ks_distance[0]
    }
    
    return results

@njit
def fast_degrees(adj: np.ndarray) -> np.ndarray:
    """Compute degrees efficiently with numba."""
    return np.sum(adj, axis=0)

@njit
def fast_clustering(adj: np.ndarray) -> np.ndarray:
    """Compute clustering coefficients efficiently with numba."""
    n = len(adj)
    clustering = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            continue
            
        triangles = 0
        for j in range(k):
            for l in range(j + 1, k):
                if adj[neighbors[j], neighbors[l]] > 0:
                    triangles += 1
                    
        possible_triangles = k * (k - 1) / 2
        clustering[i] = triangles / possible_triangles if possible_triangles > 0 else 0
    
    return clustering

@njit
def fast_betweenness(adj: np.ndarray) -> np.ndarray:
    """Approximate betweenness centrality using shortest paths."""
    n = len(adj)
    betweenness = np.zeros(n)
    
    for s in range(n):
        dist = np.full(n, np.inf)
        sigma = np.zeros(n)
        dist[s] = 0
        sigma[s] = 1
        
        queue = np.zeros(n, dtype=np.int64)
        queue[0] = s
        q_start, q_end = 0, 1
        
        visited = np.zeros(n, dtype=np.int64)
        n_visited = 0
        
        while q_start < q_end:
            v = queue[q_start]
            q_start += 1
            visited[n_visited] = v
            n_visited += 1
            
            for w in range(n):
                if adj[v, w] == 0:
                    continue
                    
                if dist[w] == np.inf:
                    queue[q_end] = w
                    q_end += 1
                    dist[w] = dist[v] + 1
                
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        
        dependency = np.zeros(n)
        
        for i in range(n_visited - 1, -1, -1):
            v = visited[i]
            if v == s:
                continue
                
            coeff = (1 + dependency[v]) / sigma[v]
            
            for w in range(n):
                if adj[v, w] == 0:
                    continue
                if dist[w] == dist[v] - 1:
                    dependency[w] += sigma[w] * coeff
        
        betweenness += dependency
    
    betweenness = betweenness / (2 * ((n-1) * (n-2)))
    return betweenness

def fast_evaluator(
    synthetic: np.ndarray, 
    empirical: np.ndarray, 
    euclidean_distance: np.ndarray,
    cache_size: int = 128
) -> Dict[str, float]:
    """Optimized network comparison using KS tests on multiple metrics."""
    @lru_cache(maxsize=cache_size)
    def get_empirical_metrics(emp_key):
        emp = np.array(emp_key)
        return (
            fast_degrees(emp),
            fast_clustering(emp),
            fast_betweenness(emp),
            euclidean_distance[np.triu(emp, 1) > 0]
        )
    
    emp_key = tuple(map(tuple, empirical))
    
    (degrees_empirical,
     clustering_empirical,
     betweenness_empirical,
     distance_empirical) = get_empirical_metrics(emp_key)
    
    degrees_synthetic = fast_degrees(synthetic)
    clustering_synthetic = fast_clustering(synthetic)
    betweenness_synthetic = fast_betweenness(synthetic)
    distance_synthetic = euclidean_distance[np.triu(synthetic, 1) > 0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks_stats = [
            ks_2samp(degrees_synthetic, degrees_empirical)[0],
            ks_2samp(clustering_synthetic, clustering_empirical)[0],
            ks_2samp(betweenness_synthetic, betweenness_empirical)[0],
            ks_2samp(distance_synthetic, distance_empirical)[0]
        ]
    
    return {
        "energy": np.max(ks_stats),
        "ks_degrees": ks_stats[0],
        "ks_clustering": ks_stats[1],
        "ks_betweenness": ks_stats[2],
        "ks_distance": ks_stats[3]
    }

@njit
def density_distance(final_network, empirical, absolute=True):
    # Compute synthetic network density
    density_synthetic = np.sum(final_network) / (final_network.shape[0] * final_network.shape[0])
    density_empirical = np.sum(empirical) / (empirical.shape[0] * empirical.shape[0])
    
    if absolute:
        return np.abs(density_synthetic - density_empirical)
    else:
        return density_synthetic - density_empirical

def check_density(adj):
    return np.sum(adj)/(adj.shape[0] * (adj.shape[0]-1))

def calculate_wiring_cost(adj, euclidean_distance):
    return np.mean(adj * euclidean_distance)

def calculate_endpoint_similarity(synthetic_matrix, empirical_matrix):
    similarities = np.zeros(synthetic_matrix.shape[0])
    for i in range(synthetic_matrix.shape[0]):
        similarities[i] = cosine_similarity(synthetic_matrix[i].reshape(1, -1),
                                            empirical_matrix[i].reshape(1, -1))
    return similarities


def evaluate_adjacency(empirical_adj, simulated_adj):
    """
    Compute Accuracy and F1 score between two binary adjacency matrices.
    
    Args:
        empirical_adj (np.ndarray): Empirical adjacency (0/1), shape (n, n)
        simulated_adj (np.ndarray): Simulated adjacency (0/1), shape (n, n)
    
    Returns:
        (accuracy, f1): Tuple of floats
            - accuracy is in [0, 100] (percentage)
            - f1 is in [0, 1]
    """
    # Ensure inputs are numpy arrays and flattened (if we want to ignore diagonal, filter it out)
    # Here, we keep the entire matrix as is, including diagonal if present.
    A_emp = empirical_adj.astype(int).ravel()
    A_sim = simulated_adj.astype(int).ravel()

    # Calculate confusion matrix components:
    TP = np.sum((A_emp == 1) & (A_sim == 1))
    FP = np.sum((A_emp == 0) & (A_sim == 1))
    TN = np.sum((A_emp == 0) & (A_sim == 0))
    FN = np.sum((A_emp == 1) & (A_sim == 0))

    # Accuracy (percentage)
    total = TP + FP + TN + FN
    accuracy = 100.0 * (TP + TN) / total if total > 0 else 0.0

    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score (in [0, 1])
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return accuracy, f1

def randomize_graph(G, nswap=None, max_tries=None):
    """Randomizes the graph using double edge swap."""
    if nswap is None:
        nswap = 20 * G.number_of_edges()  # Adjust as needed
    if max_tries is None:
        max_tries = nswap * 20
    G_random = nx.double_edge_swap(G.copy(), nswap=nswap, max_tries=max_tries)
    return G_random

def compute_random_metrics(G, nrandomizations=10, nswap=None, max_tries=None):
    """
    Computes the average clustering and average shortest path length over multiple randomized versions
    of the input graph.
    """
    clustering_vals = []
    path_length_vals = []
    
    for _ in range(nrandomizations):
        Gr = randomize_graph(G, nswap, max_tries)
        clustering_vals.append(nx.average_clustering(Gr))
        try:
            path_length_vals.append(nx.average_shortest_path_length(Gr))
        except nx.NetworkXError:
            # In case the randomized graph is disconnected, compute the metric for the largest connected component.
            largest_cc = max(nx.connected_components(Gr), key=len)
            Gr_sub = Gr.subgraph(largest_cc)
            path_length_vals.append(nx.average_shortest_path_length(Gr_sub))
    
    return np.mean(clustering_vals), np.mean(path_length_vals)

def compute_sigma(adj_matrix, nrandomizations=32, nswap=None, max_tries=None):
    """
    Computes the smallworld index sigma for a graph given its adjacency matrix.
    
    Sigma is defined as:
        sigma = (C / C_rand) / (L / L_rand)
    where:
        C      = average clustering coefficient of the graph
        L      = characteristic path length of the graph
        C_rand = average clustering coefficient of randomized graphs
        L_rand = average characteristic path length of randomized graphs
    """
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Compute original graph metrics
    C = nx.average_clustering(G)
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # For disconnected graphs, compute L on the largest connected component.
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_sub)
    
    # Compute metrics for randomized graphs
    Crand, Lrand = compute_random_metrics(G, nrandomizations, nswap, max_tries)
    
    # Calculate and return sigma
    sigma = (C / Crand) / (L / Lrand)
    return sigma

def compute_omega(adj_matrix, nrandomizations=100, nswap=None, max_tries=None):
    """
    Computes the omega (ω) small-world metric for an undirected network given its adjacency matrix.
    
    ω is defined as:
        ω = (L_rand / L) - (C / C_latt)
    
    where:
        L     = average shortest path length of the original graph
        C     = average clustering coefficient of the original graph
        L_rand = average shortest path length of a randomized version of the graph (averaged over multiple realizations)
        C_latt = clustering coefficient of a lattice (regular) graph generated to approximate the equivalent lattice
    
    Parameters:
        adj_matrix (numpy.ndarray): The adjacency matrix.
        nrandomizations (int): Number of random networks to average for L_rand.
        nswap, max_tries: Parameters passed to the double edge swap (randomization).
    
    Returns:
        omega (float): The small-world omega metric.
    """
    # Create the original graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    n = G.number_of_nodes()
    
    # Compute clustering (C) and path length (L) for the original network.
    C = nx.average_clustering(G)
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # If the graph is disconnected, use the largest connected component.
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_sub)
    
    # Compute L_rand: average shortest path length of randomized networks.
    L_rand_values = []
    for _ in range(nrandomizations):
        Gr = randomize_graph(G, nswap, max_tries)
        try:
            L_rand_values.append(nx.average_shortest_path_length(Gr))
        except nx.NetworkXError:
            # Use the largest connected component if Gr is disconnected.
            largest_cc = max(nx.connected_components(Gr), key=len)
            Gr_sub = Gr.subgraph(largest_cc)
            L_rand_values.append(nx.average_shortest_path_length(Gr_sub))
    L_rand = np.mean(L_rand_values)
    
    laticized = bct.latmio_und(adj_matrix,itr=100)[0]
    G_lattice = nx.from_numpy_array(laticized)
    C_latt = nx.average_clustering(G_lattice)
    
    # Compute omega according to the formula.
    omega = (L_rand / L) - (C / C_latt)
    return omega

def compute_graph_metrics(simulated_tensor,
                          empirical_adjmat,
                          euclidean_distance,
                          coordinates):
    measure_labels = ["diffusion distance",
                      "shortest path distance",
                      "propagation distance",
                      "topological distance",
                      "density",
                      "wiring cost",
                      "average clustering",
                      "degree assortativity",
                      "small-worldness",
                      "endpoint similarity",
                      "accuracy",
                      "F1 score"]
    measures = np.zeros((len(measure_labels),
                         simulated_tensor.shape[2]))

    for timepoint in tqdm(range(simulated_tensor.shape[2]),desc="Computing graph metrics"):
        G = nx.from_numpy_array(simulated_tensor[:,:,timepoint])
        measures[0,timepoint] = resistance_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        measures[1,timepoint] = shortest_path_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        measures[2,timepoint] = propagation_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        measures[3,timepoint] = topological_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        
        measures[4,timepoint] = check_density(simulated_tensor[:,:,timepoint])
        measures[5,timepoint] = calculate_wiring_cost(simulated_tensor[:,:,timepoint],euclidean_distance)
        
        measures[6,timepoint] = nx.average_clustering(G)
        measures[7,timepoint] = nx.degree_assortativity_coefficient(G)
        measures[8,timepoint] = nx.smallworld.sigma(G)
        
        measures[9,timepoint] = calculate_endpoint_similarity(simulated_tensor[:,:,timepoint],empirical_adjmat).mean()
        measures[10,timepoint], measures[11,timepoint] = evaluate_adjacency(empirical_adjmat,simulated_tensor[:,:,timepoint])
    return pd.DataFrame(measures.T,columns=measure_labels)

def _compute_metrics_for_timepoint(timepoint, 
                                   simulated_tensor, 
                                   empirical_adjmat, 
                                   euclidean_distance, 
                                   coordinates):
    """
    Computes all metrics for a given timepoint and returns them as a 1D array.
    """
    sim = simulated_tensor[:, :, timepoint]
    G = nx.from_numpy_array(sim)
    
    metrics = np.zeros(12)
    metrics[0] = resistance_distance(sim, coordinates).mean()
    metrics[1] = shortest_path_distance(sim, coordinates).mean()
    metrics[2] = propagation_distance(sim, coordinates).mean()
    metrics[3] = topological_distance(sim, coordinates).mean()
    metrics[4] = check_density(sim)
    metrics[5] = calculate_wiring_cost(sim, euclidean_distance)
    metrics[6] = nx.average_clustering(G)
    metrics[7] = nx.degree_assortativity_coefficient(G)
    metrics[8] = compute_omega(sim)
    metrics[9] = calculate_endpoint_similarity(sim, empirical_adjmat).mean()
    metrics[10], metrics[11] = evaluate_adjacency(empirical_adjmat, sim)
    
    return metrics

def compute_graph_metrics_parallel(simulated_tensor, 
                                   empirical_adjmat, 
                                   euclidean_distance, 
                                   coordinates):
    measure_labels = [
        "diffusion distance",
        "shortest path distance",
        "propagation distance",
        "topological distance",
        "density",
        "wiring cost",
        "average clustering",
        "degree assortativity",
        "small-worldness",
        "endpoint similarity",
        "accuracy",
        "F1 score"
    ]
    
    n_timepoints = simulated_tensor.shape[2]
    
    # Use tqdm_joblib to wrap joblib.Parallel for a progress bar.
    with tqdm_joblib(tqdm(total=n_timepoints, desc="Computing graph metrics")):
        results = Parallel(n_jobs=-1)(
            delayed(_compute_metrics_for_timepoint)(tp, simulated_tensor, empirical_adjmat, euclidean_distance, coordinates)
            for tp in range(n_timepoints)
        )
    
    # Convert list of 1D arrays into a 2D NumPy array.
    # Each element in 'results' is of shape (12,), so we stack along axis 1.
    measures = np.array(results).T  # Shape: (12, n_timepoints)
    
    # Return a DataFrame with rows corresponding to timepoints.
    return pd.DataFrame(measures.T, columns=measure_labels)


@dataclass
class OptimizationResults:
    """Container for optimization results."""
    best_parameters: Dict[str, float]
    best_score: float
    all_parameters: List[Dict[str, float]]
    all_scores: List[float]
    convergence_info: Dict[str, Any]

class NetworkOptimizer:
    """
    Optimizer for network evolution parameters using Bayesian optimization.
    
    This class provides a flexible framework for optimizing network evolution parameters
    by allowing injection of custom simulation models and evaluation functions.
    """
    
    def __init__(
        self,
        simulation_model: Callable,
        evaluation_function: Callable,
        param_ranges: Dict[str, tuple],
        sim_kwargs: Dict[str, Any],
        eval_kwargs: Dict[str, Any],
        n_parallel: int = -1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            simulation_model: Function that simulates network evolution
            evaluation_function: Function that evaluates network quality
            param_ranges: Dictionary of parameter names and (min, max) ranges to optimize
            sim_kwargs: Fixed keyword arguments for simulation model
            eval_kwargs: Fixed keyword arguments for evaluation function
            n_parallel: Number of parallel jobs (-1 for all cores)
            random_seed: Random seed for reproducibility
        """
        self.simulation_model = simulation_model
        self.evaluation_function = evaluation_function
        self.param_ranges = param_ranges
        self.sim_kwargs = sim_kwargs
        self.eval_kwargs = eval_kwargs
        self.n_parallel = n_parallel
        self.random_seed = random_seed
        
    def _create_parameter_space(
        self,
        param_ranges: Dict[str, tuple]
    ) -> List[Real]:
        """Create skopt parameter space from ranges."""
        space = []
        for name, (low, high) in param_ranges.items():
            space.append(Real(low, high, name=name))
        return space
        
    def _simulate_with_params(
        self,
        params: Dict[str, float],
        batch_id: Optional[int] = None
    ) -> float:
        """
        Run simulation with given parameters and evaluate.
        
        Args:
            params: Dictionary of parameters to optimize
            batch_id: Optional batch identifier for parallel runs
        
        Returns:
            float: Evaluation score
        """
        # Update simulation kwargs with optimization parameters
        sim_kwargs = self.sim_kwargs.copy()
        sim_kwargs.update(params)
        
        # Add batch-specific random seed if needed
        if batch_id is not None and 'random_seed' in sim_kwargs:
            sim_kwargs['random_seed'] = sim_kwargs['random_seed'] + batch_id
            
        # Run simulation
        history = self.simulation_model(**sim_kwargs)
        
        # Get final network state
        final_network = history[:, :, -1] if history.ndim == 3 else history
        
        # Evaluate network
        score = self.evaluation_function(final_network, **self.eval_kwargs)
        
        # Handle both dict and float return types
        return score['energy'] if isinstance(score, dict) else score
        
    def optimize(
        self,
        n_calls: int = 50,
        n_initial_points: int = 10,
        n_parallel_samples: int = 16,
        acquisition_function: str = "gp_hedge",
        verbose: bool = True,
        **optimizer_kwargs
    ) -> OptimizationResults:
        """
        Run Bayesian optimization to find best parameters.
        
        Args:
            param_ranges: Dictionary of parameter names and (min, max) ranges
            n_calls: Number of optimization iterations
            n_initial_points: Number of initial random points
            n_parallel_samples: Number of parallel evaluations per iteration
            acquisition_function: Acquisition function for GP
            verbose: Whether to show progress bar
        
        Returns:
            OptimizationResults containing best parameters and optimization history
        """
        space = self._create_parameter_space(self.param_ranges)

        # Create objective function with named arguments
        @use_named_args(space)
        def objective(**params):
            # Run multiple samples in parallel
            scores = Parallel(n_jobs=self.n_parallel)(
            delayed(self._simulate_with_params)(params, i)
            for i in tqdm(range(n_parallel_samples), 
                        desc="Optimization Iteration",
                        position=1, 
                        leave=False))
            return np.mean([score for score in scores])
            
        # Run optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sklearn warnings
            
            if verbose:
                print("Running Bayesian optimization...")
                
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=acquisition_function,
                random_state=self.random_seed
            )
        
        # Collect results
        param_names = list(self.param_ranges.keys())
        best_params = dict(zip(param_names, result.x))
        
        all_params = [
            dict(zip(param_names, x))
            for x in result.x_iters
        ]
        
        return OptimizationResults(
            best_parameters=best_params,
            best_score=result.fun,
            all_parameters=all_params,
            all_scores=result.func_vals,
            convergence_info={
                'models': result.models,
                'space': result.space,
                'specs': result.specs
            }
        )


def brain_plotter(
    data: np.ndarray,
    coordinates: np.ndarray,
    axis: plt.Axes,
    view: Tuple[int, int] = (90, 180),
    size: int = 20,
    cmap: any = "viridis",
    scatter_kwargs=Optional[None],
) -> plt.Axes:
    """plots the 3D scatter plot of the brain. It's a simple function that takes the data, the coordinates, and the axis and plots the brain.
    It's a modified version the netneurotools python package but you can give it the axis to plot in. See here:
    https://netneurotools.readthedocs.io/en/latest/

    Args:
        data (np.ndarray): the values that need to be mapped to the nodes. Shape is (N,)
        coordinates (np.ndarray): 3D coordinates fo each node. Shape is (N, 3)
        axis (plt.Axes): Which axis to plot in. This means you have to already have a figure and an axis to plot in.
        view (Tuple[int, int], optional): Which view to look at. Defaults to (90, 180).
        size (int, optional): Size of the nodes. Defaults to 20.
        cmap (any, optional): Color map. Defaults to "viridis" which I don't like but you do you.
        scatter_kwargs (_type_, optional): kwargs for the dots. Defaults to Optional[None].

    Returns:
        plt.Axes: matplotlib axis with the brain plotted.
    """
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}

    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=data,
        cmap=cmap,
        # s=size,
        # **scatter_kwargs,
    )
    axis.view_init(*view)
    axis.axis("off")
    scaling = np.array([axis.get_xlim(), axis.get_ylim(), axis.get_zlim()])
    axis.set_box_aspect(tuple(scaling[:, 1] / 1.2 - scaling[:, 0]))
    return axis

def calculate_trajectories(
    simulation_tensors_list: List[np.ndarray],
    empirical_tensor: np.ndarray,
    n_components: int = 3,
    use_all_empirical: bool = True,
    empirical_indices: Optional[np.ndarray] = None
) -> Tuple[Dict[int, np.ndarray], np.ndarray, PCA]:
    """
    Create joint PCA embeddings for multiple simulation rule trajectories and empirical data.
    
    Parameters
    ----------
    simulation_tensors_list : List[np.ndarray]
        List of binary adjacency tensors for different simulation rules,
        each with shape (n_nodes, n_nodes, n_timepoints_i, n_sim_samples)
    empirical_tensor : np.ndarray
        Binary adjacency tensor with shape (n_nodes, n_nodes, n_emp_samples)
        where n_emp_samples may be different from n_sim_samples
    n_components : int, default=3
        Number of PCA components to use
    use_all_empirical : bool, default=True
        If True, use all empirical samples in the embedding
        If False, use only a subset matching the number of simulation samples
    empirical_indices : np.ndarray, optional
        Indices of empirical samples to use when use_all_empirical=False
        If None and use_all_empirical=False, uses first n_sim_samples
        
    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping rule indices to their embeddings
        Each embedding has shape (n_sim_samples, n_timepoints_i, n_components)
    np.ndarray
        Empirical embeddings with shape (n_used_emp_samples, n_components)
        where n_used_emp_samples is either n_emp_samples or n_sim_samples
    PCA
        Fitted PCA object for reference
    """
    n_nodes = simulation_tensors_list[0].shape[0]  # Assuming same number of nodes
    n_sim_samples = simulation_tensors_list[0].shape[3]  # Number of simulation samples
    n_emp_samples = empirical_tensor.shape[2]  # Number of empirical samples
    
    # Check consistency in simulation tensors
    for i, tensor in enumerate(simulation_tensors_list):
        if tensor.shape[0] != n_nodes or tensor.shape[1] != n_nodes:
            raise ValueError(f"Simulation tensor {i} has inconsistent node dimensions")
        if tensor.shape[3] != n_sim_samples:
            raise ValueError(f"Simulation tensor {i} has inconsistent number of samples")
    
    # Determine which empirical samples to use
    if use_all_empirical:
        emp_indices = np.arange(n_emp_samples)
    else:
        if empirical_indices is not None:
            emp_indices = empirical_indices
            if len(emp_indices) != n_sim_samples:
                raise ValueError("Number of specified empirical indices must match simulation samples")
        else:
            # Use first n_sim_samples by default
            if n_sim_samples > n_emp_samples:
                raise ValueError("Not enough empirical samples to match simulation samples")
            emp_indices = np.arange(n_sim_samples)
    
    n_used_emp_samples = len(emp_indices)
            
    # Extract upper triangular indices (without diagonal)
    triu_indices = np.triu_indices(n_nodes, k=1)
    
    # Flatten matrices into vectors (only upper triangle)
    flat_matrices = []
    
    # Keep track of trajectory info for reconstruction
    rule_trajectory_info = []
    
    # Process each simulation rule
    for rule_idx, sim_tensor in enumerate(simulation_tensors_list):
        n_timepoints = sim_tensor.shape[2]
        
        # Store info for reconstruction
        rule_trajectory_info.append({
            'rule_idx': rule_idx,
            'n_timepoints': n_timepoints,
            'start_idx': len(flat_matrices)
        })
        
        # Process simulation tensor for this rule - all samples
        for sample in range(n_sim_samples):
            for time in range(n_timepoints):
                # Extract upper triangle at this time point for this sample
                flat_matrices.append(
                    sim_tensor[triu_indices[0], triu_indices[1], time, sample]
                )
    
    # Store starting index for empirical data
    empirical_start_idx = len(flat_matrices)
    
    # Process empirical tensor - selected samples
    for emp_idx in emp_indices:
        # Extract upper triangle for this sample
        flat_matrices.append(
            empirical_tensor[triu_indices[0], triu_indices[1], emp_idx]
        )
    
    # Convert to array for PCA
    X = np.array(flat_matrices)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Split back into separate trajectories
    simulation_embeddings = {}
    empirical_embeddings = np.zeros((n_used_emp_samples, n_components))
    
    # Extract simulation embeddings for each rule
    for rule_info in rule_trajectory_info:
        rule_idx = rule_info['rule_idx']
        n_timepoints = rule_info['n_timepoints']
        start_idx = rule_info['start_idx']
        
        # Initialize embeddings for this rule - shape (n_sim_samples, n_timepoints, n_components)
        rule_embeddings = np.zeros((n_sim_samples, n_timepoints, n_components))
        
        # Extract embeddings for each sample
        for sample in range(n_sim_samples):
            sample_start = start_idx + sample * n_timepoints
            sample_end = sample_start + n_timepoints
            rule_embeddings[sample] = X_reduced[sample_start:sample_end]
        
        simulation_embeddings[rule_idx] = rule_embeddings
    
    # Extract empirical embeddings
    for i, emp_idx in enumerate(emp_indices):
        idx = empirical_start_idx + i
        empirical_embeddings[i] = X_reduced[idx]
    
    return simulation_embeddings, empirical_embeddings, pca

def process_labels(source_labels: List[str], 
                  gifti_labels_rh: Dict[int, str], 
                  gifti_labels_lh: Dict[int, str]) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Map source labels to gifti label indices, handling duplicates and hemispheres.
    
    Parameters:
    -----------
    source_labels : List[str]
        List of labels like 'rh_lateralorbitofrontal', 'lh_precentral', etc.
    gifti_labels_rh : Dict[int, str]
        Right hemisphere gifti labels dictionary
    gifti_labels_lh : Dict[int, str]
        Left hemisphere gifti labels dictionary
        
    Returns:
    --------
    label_mapping : Dict[str, int]
        Mapping from source labels to their positions in the new organization
    mask : np.ndarray
        Boolean mask for valid positions
    """
    # Initialize mapping dictionary
    label_mapping = {}
    
    # Create reverse lookup dictionaries for gifti labels
    rh_lookup = {}  # base_name -> list of indices
    lh_lookup = {}  # base_name -> list of indices
    
    # Process right hemisphere gifti labels
    for idx, label in gifti_labels_rh.items():
        if label != '???' and label != 'corpuscallosum':
            base_name = label.rsplit('_', 1)[0]  # Split on last underscore
            if base_name not in rh_lookup:
                rh_lookup[base_name] = []
            rh_lookup[base_name].append(idx)
            
    # Process left hemisphere gifti labels
    offset = len(gifti_labels_rh)  # Offset for left hemisphere indices
    for idx, label in gifti_labels_lh.items():
        if label != '???' and label != 'corpuscallosum':
            base_name = label.rsplit('_', 1)[0]  # Split on last underscore
            if base_name not in lh_lookup:
                lh_lookup[base_name] = []
            lh_lookup[base_name].append(idx + offset)
    
    # Counter for duplicate regions
    rh_counter = {k: 0 for k in rh_lookup.keys()}
    lh_counter = {k: 0 for k in lh_lookup.keys()}
    
    # Process source labels
    for label in source_labels:
        hemi, region = label.split('_', 1)  # Split on first underscore
        
        if hemi == 'rh':
            if region in rh_lookup:
                # Get next available index for this region
                if rh_counter[region] < len(rh_lookup[region]):
                    idx = rh_lookup[region][rh_counter[region]]
                    label_mapping[label] = idx
                    rh_counter[region] += 1
                
        elif hemi == 'lh':
            if region in lh_lookup:
                # Get next available index for this region
                if lh_counter[region] < len(lh_lookup[region]):
                    idx = lh_lookup[region][lh_counter[region]]
                    label_mapping[label] = idx
                    lh_counter[region] += 1
    
    # Create mask for valid positions
    total_length = len(gifti_labels_rh) + len(gifti_labels_lh)
    mask = np.zeros(total_length, dtype=bool)
    for idx in label_mapping.values():
        mask[idx] = True
        
    return label_mapping, mask


def enumerate_source_labels(source_labels: list) -> list:
    """
    Enumerate duplicate labels in the source list similar to gifti format.
    
    Parameters:
    -----------
    source_labels : list
        Original list of labels (e.g., ['rh_precentral', 'rh_precentral', ...])
    
    Returns:
    --------
    list
        Enumerated labels (e.g., ['rh_precentral_1', 'rh_precentral_2', ...])
    """
    # Keep track of counts for each base label
    counter = defaultdict(int)
    enumerated_labels = []
    
    for label in source_labels:
        counter[label] += 1
        enumerated_labels.append(f"{label}_{counter[label]}")
        
    return enumerated_labels

def reorganize_data_with_labels(data: np.ndarray, 
                              source_labels: List[str],
                              gifti_labels_rh: Dict[int, str],
                              gifti_labels_lh: Dict[int, str]) -> pd.DataFrame:
    """
    Reorganize data (matrix or vector) using pandas DataFrames and label matching.
    
    Parameters:
    -----------
    data : np.ndarray
        Original data, either a matrix (114x114) or vector (114,)
    source_labels : List[str]
        Original list of labels (rh_region and lh_region format)
    gifti_labels_rh : Dict[int, str]
        Right hemisphere gifti labels dictionary
    gifti_labels_lh : Dict[int, str]
        Left hemisphere gifti labels dictionary
    
    Returns:
    --------
    pd.DataFrame
        Reorganized data with zeros for missing regions
    """
    # Check if input is vector or matrix
    is_vector = data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1)
    
    # First enumerate the source labels
    enumerated_source_labels = enumerate_source_labels(source_labels)
    
    # Create the source DataFrame with enumerated labels
    if is_vector:
        df_source = pd.DataFrame(data, 
                               index=enumerated_source_labels,
                               columns=['value'],
                               dtype=data.dtype)
    else:
        df_source = pd.DataFrame(data, 
                               index=enumerated_source_labels,
                               columns=enumerated_source_labels,
                               dtype=data.dtype)
    
    # Create the full list of target labels in gifti order
    target_labels = []
    
    # Add RH labels preserving original gifti format
    for i in range(len(gifti_labels_rh)):
        target_labels.append(
            f"rh_{gifti_labels_rh[i]}" if gifti_labels_rh[i] != '???' 
            else gifti_labels_rh[i]
        )
    
    # Add LH labels preserving original gifti format
    for i in range(len(gifti_labels_lh)):
        target_labels.append(
            f"lh_{gifti_labels_lh[i]}" if gifti_labels_lh[i] != '???' 
            else gifti_labels_lh[i]
        )
    
    # Create empty target DataFrame
    if is_vector:
        df_target = pd.DataFrame(0, 
                               index=target_labels,
                               columns=['value'],
                               dtype=data.dtype)
    else:
        df_target = pd.DataFrame(0, 
                               index=target_labels,
                               columns=target_labels,
                               dtype=data.dtype)
    
    # Create mapping between source and target labels
    source_to_target = {}
    for source_label in enumerated_source_labels:
        # Split into components (e.g., 'rh_precentral_1' -> ['rh', 'precentral', '1'])
        hemi, region, num = source_label.rsplit('_', 1)[0].split('_', 1)[0], \
                           source_label.rsplit('_', 1)[0].split('_', 1)[1], \
                           source_label.rsplit('_', 1)[1]
        
        # Find matching target label
        target_base = f"{hemi}_{region}_{num}"
        matching_targets = [t for t in target_labels 
                          if t != '???' and t.startswith(f"{hemi}_{region}_")]
        
        if matching_targets:
            # Map to the corresponding numbered version
            try:
                source_to_target[source_label] = matching_targets[int(num) - 1]
            except IndexError:
                # If there aren't enough target labels, map to the last available one
                source_to_target[source_label] = matching_targets[-1]
    
    # Fill the target DataFrame with values from source
    if is_vector:
        for source_label, target_label in source_to_target.items():
            df_target.loc[target_label, 'value'] = df_source.loc[source_label, 'value']
    else:
        for source_i, target_i in source_to_target.items():
            for source_j, target_j in source_to_target.items():
                df_target.loc[target_i, target_j] = df_source.loc[source_i, source_j]
    
    return df_target

def data_to_surfdata(left_data, right_data, lh_surfaces, rh_surfaces):
    label_to_data = dict(enumerate(left_data.values.flatten()))
    left_surfdata = np.array([label_to_data.get(label, 0) for label in lh_surfaces])

    label_to_data = dict(enumerate(right_data.values.flatten()))
    right_surfdata = np.array([label_to_data.get(label, 0) for label in rh_surfaces])

    return left_surfdata, right_surfdata

def spectral_normalization(
    adjacency_matrix: np.ndarray, target_radius: float = 1.0
) -> np.ndarray:
    """Normalizes the adjacency matrix to have a spectral radius of the target_radius. Good to keep the system stable.

    Args:
        target_radius (float): A value below 1.0. It's the spectral radius that you want to achieve. But use 1.0 if you're planning to change the global coupling strength somewhere.
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.

    Returns:
        np.ndarray: Normalized adjacency matrix with the same shape as the input.
    """
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    return adjacency_matrix * target_radius / spectral_radius


def make_X_y(
    make_X: Callable,
    lags: np.ndarray,
    cut: int = 0,
) -> np.ndarray:
    """
    *** 
    Stolen from Fabrizio: 
    https://github.com/fabridamicelli/brain-connectivity-meets-reservoir-computing/blob/main/bioRNN/tasks/memory/capacity.py
    ***
    make_X: Callable
        Function to generate inputs. Should return a 1D np.ndarray.
        Note that the size of the output of make_X determines the size
        of X used for the task.
        Example:
        from functools import partial
        make_X = partial(np.random.uniform, low=-.5, high=.5, size=200)
    lags: np.ndarray
        Delays to be evaluated (memory capacity).
        For example: np.arange(1, 31, 5).
    cut: int, optional
        Number of initial steps to cut out.
        Make be at least larger than max(lags) if you want to avoid circle sequence.
    """
    X = make_X()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = _make_lagged_y(X, lags=lags)
    return X[cut:], y[cut:]

def _make_lagged_y(inputs: np.ndarray, lags: np.ndarray,) -> np.ndarray:
    """
    *** 
    Stolen from Fabrizio: 
    https://github.com/fabridamicelli/brain-connectivity-meets-reservoir-computing/blob/main/bioRNN/tasks/memory/capacity.py
    ***
    Generate delayed versions of inputs sequence.
    One sequence is generated for each lag value.

    Parameters
    ----------
    inputs: np.ndarray
        Signal to lag. It will be flattened before lagging,
        as it is supposed to be only one input chanel.
    lags: np.ndarray
        Delays to be evaluated (memory capacity).
        For example: np.arange(1, 31, 5).

    Returns
    -------
    lagged_inputs: np.ndarray of shape (len(inputs), len(lags))
        Array of lagged version of the inputs sequence.
        Each column represents U(t-k), where k is the lag.

    Examples
    --------
    >>> inputs = np.arange(5)
    >>> make_lagged_y(inputs, [1, 3])
    >>> array([[4., 2.],
               [0., 3.],
               [1., 4.],
               [2., 0.],
               [3., 1.]])

    >>> inputs = np.arange(5)
    >>> make_lagged_y(inputs, [1, 3])
    >>> array([[1., 4.],
               [2., 0.],
               [3., 1.]])
    """
    assert isinstance(inputs, np.ndarray), "inputs must be np.ndarray"
    inputs = inputs.flatten()
    inputs_lagged = np.zeros((len(inputs), len(lags)))
    for col, lag in enumerate(lags):
        inputs_lagged[:, col] = np.roll(inputs, lag)
    return inputs_lagged


def forgetting(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List, float]:
    """
    *** 
    Stolen from Fabrizio: 
    https://github.com/fabridamicelli/brain-connectivity-meets-reservoir-computing/blob/main/bioRNN/tasks/memory/capacity.py
    ***
    Scoring function.
    Return forgetting curve (r2(k) for all k) and Memory capacity
    (sum over all values of the curve).

    y_pred and y_true are compared column-wise,
    assuming each column contains the values for a
    give delay.
    """
    assert y_pred.shape == y_true.shape, "y_pred and y_true must have same shape"
    assert y_pred.shape[0] > 1, "Error while computing forgetting: y_pred has less than 1 sample. Increase the number of samples (steps)"
    r2s = []
    for true, pred in zip(y_true.T, y_pred.T):
        r2 = np.corrcoef(true, pred)[0, 1]
        r2s.append(0 if r2 is None else r2 ** 2)
    return r2s, np.sum(r2s)


def rewire_long_range_connections(adj_matrix, dist_matrix, distance_threshold, num_steps=10):
    """
    Progressively rewire long-range connections (exceeding distance_threshold) to reduce wiring cost.
    This implementation allows slight changes in degree distribution to maximize wiring efficiency.
    
    Parameters:
    -----------
    adj_matrix : numpy.ndarray
        Binary adjacency matrix (n x n)
    dist_matrix : numpy.ndarray
        Distance matrix between nodes (n x n)
    distance_threshold : float
        Minimum distance threshold to consider a connection as "long-range"
    num_steps : int
        Number of rewiring steps (default: 10)
        
    Returns:
    --------
    rewired_matrices : numpy.ndarray
        Array of rewired adjacency matrices at each step (n x n x (num_steps+1))
    total_distances : numpy.ndarray
        Total wiring distance at each step
    """
    n = adj_matrix.shape[0]
    
    # Initialize output arrays
    rewired_matrices = np.zeros((n, n, num_steps + 1))
    rewired_matrices[:, :, 0] = adj_matrix.copy()
    total_distances = np.zeros(num_steps + 1)
    
    # Calculate initial wiring distance
    total_distances[0] = np.sum(adj_matrix * dist_matrix) / 2  # Divide by 2 for undirected graph
    
    # Get long-range edges that exceed the distance threshold
    long_range_edges = []
    for i in range(n):
        for j in range(i+1, n):  # Upper triangle only (symmetric)
            if adj_matrix[i, j] > 0 and dist_matrix[i, j] >= distance_threshold:
                long_range_edges.append((i, j, dist_matrix[i, j]))
    
    # Sort long-range edges by distance (descending)
    long_range_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Check if there are any long-range edges to rewire
    total_edges = len(long_range_edges)
    if total_edges == 0:
        print(f"No connections exceed the distance threshold of {distance_threshold:.4f}.")
        # Return original matrix for all steps
        for step in range(1, num_steps + 1):
            rewired_matrices[:, :, step] = adj_matrix.copy()
            total_distances[step] = total_distances[0]
        return rewired_matrices, total_distances
        
    # Determine number of edges to rewire in each step
    edges_per_step = max(1, total_edges // num_steps)
    
    # For each step
    for step in range(1, num_steps + 1):
        current_adj = rewired_matrices[:, :, step-1].copy()
        
        # Determine edges to rewire in this step
        start_idx = (step - 1) * edges_per_step
        end_idx = min(start_idx + edges_per_step, total_edges)
        
        # Skip if no more edges to rewire
        if start_idx >= total_edges:
            rewired_matrices[:, :, step] = current_adj
            total_distances[step] = total_distances[step-1]
            continue
        
        edges_to_rewire = long_range_edges[start_idx:end_idx]
        
        # Rewire each edge
        for edge in edges_to_rewire:
            i, j, _ = edge
            
            # Skip if edge was already rewired in a previous step
            if current_adj[i, j] == 0:
                continue
                
            # Remove the long-range connection
            current_adj[i, j] = 0
            current_adj[j, i] = 0
            
            # Now for each node (i and j), find the closest disconnected node
            # Start with node i
            best_distance_i = float('inf')
            best_target_i = None
            for k in range(n):
                if k != i and current_adj[i, k] == 0:  # Disconnected from i
                    if dist_matrix[i, k] < best_distance_i:
                        best_distance_i = dist_matrix[i, k]
                        best_target_i = k
            
            # Then for node j
            best_distance_j = float('inf')
            best_target_j = None
            for k in range(n):
                if k != j and current_adj[j, k] == 0:  # Disconnected from j
                    if dist_matrix[j, k] < best_distance_j:
                        best_distance_j = dist_matrix[j, k]
                        best_target_j = k
            
            # Choose which node to connect (i or j) based on which has the closer target
            if best_target_i is not None and best_target_j is not None:
                if best_distance_i <= best_distance_j:
                    # Connect i to its closest disconnected neighbor
                    current_adj[i, best_target_i] = 1
                    current_adj[best_target_i, i] = 1
                else:
                    # Connect j to its closest disconnected neighbor
                    current_adj[j, best_target_j] = 1
                    current_adj[best_target_j, j] = 1
            elif best_target_i is not None:
                # Only i has a disconnected neighbor
                current_adj[i, best_target_i] = 1
                current_adj[best_target_i, i] = 1
            elif best_target_j is not None:
                # Only j has a disconnected neighbor
                current_adj[j, best_target_j] = 1
                current_adj[best_target_j, j] = 1
            # If neither has a disconnected neighbor, we simply remove the edge
            # (This will change the degree distribution)
            
        # Store the rewired matrix and calculate total distance
        rewired_matrices[:, :, step] = current_adj
        total_distances[step] = np.sum(current_adj * dist_matrix) / 2
    
    return rewired_matrices, total_distances


def compute_structural_complexity(A):
    """
    Compute the structural complexity C for a given square matrix A using the
    entropy of the singular values.

    Parameters
    ----------
    A : numpy.ndarray
        A square numpy array (n x n) representing the network's weight matrix.

    Returns
    -------
    float
        The complexity measure C in the range [0, 1].
    """
    # Ensure A is a numpy array
    A = np.asarray(A)
    n = A.shape[0]
    
    # Compute singular values of A
    # We don't need U and V, only the singular values (sigma)
    sigma = np.linalg.svd(A, compute_uv=False)
    
    # Compute normalized "probabilities" p_i = sigma_i^2 / sum of all sigma_j^2
    sigma_squared = sigma**2
    total = np.sum(sigma_squared)
    p = sigma_squared / total
    
    # Compute Shannon entropy: H = -sum_i p_i * log(p_i)
    # Use np.log for natural logarithm
    # Filter out zero p_i to avoid log(0) issues
    p_nonzero = p[p > 0]
    H = -np.sum(p_nonzero * np.log(p_nonzero))
    
    # Normalize by log(n) to get C in [0, 1]
    C = H / np.log(n)
    
    return C
