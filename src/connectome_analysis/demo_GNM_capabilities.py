"""
Complete examples demonstrating all GNM library features for your connectome analysis.
"""

import numpy as np
import torch
from pathlib import Path

# Import GNM library components
from gnm import (
    fitting, 
    generative_rules, 
    evaluation, 
    defaults,
    utils,
    weight_criteria
)
# from gnm.models import GNMBinary, GNMWeighted
from src.imported_libraries.GenerativeNetworkModels_2.src.gnm.models import GNMBinary, GNMWeighted

def example_1_basic_gnm_generation():
    """Example 1: Basic network generation using GNM library."""
    print("=" * 60)
    print("EXAMPLE 1: Basic GNM Network Generation")
    print("=" * 60)
    
    # Use GNM's default data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_matrix = defaults.get_distance_matrix(device=device)
    target_network = defaults.get_binary_network(device=device)
    
    print(f"Target network: {target_network.shape}, {int(target_network.sum()//2)} edges")
    
    # Create a GNM binary model with Matching Index rule
    model = GNMBinary(
        eta=-2.0,  # Distance penalty
        gamma=0.3,  # Homophily parameter
        lambdah=0.0,  # Time-dependency
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=generative_rules.MatchingIndex(),
        distance_matrix=distance_matrix,
        device=device
    )
    
    # Generate network with same number of edges as target
    n_edges = int(target_network.sum().item() // 2)
    model.run(n_edges)
    
    generated_network = model.get_adjacency_matrix()
    print(f"Generated network: {int(generated_network.sum()//2)} edges")
    
    # Evaluate the generated network
    degree_ks = evaluation.DegreeKS()
    clustering_ks = evaluation.ClusteringKS()
    edge_length_ks = evaluation.EdgeLengthKS(distance_matrix)
    
    print(f"Degree KS: {degree_ks(generated_network, target_network):.3f}")
    print(f"Clustering KS: {clustering_ks(generated_network, target_network):.3f}")
    print(f"Edge Length KS: {edge_length_ks(generated_network, target_network):.3f}")
    
    return generated_network


def example_2_parameter_sweep():
    """Example 2: Parameter sweep to find optimal eta and gamma."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: GNM Parameter Sweep")
    print("=" * 60)
    
    # Load default data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_matrix = defaults.get_distance_matrix(device=device)
    target_network = defaults.get_binary_network(device=device)
    n_edges = int(target_network.sum().item() // 2)
    
    # Define parameter ranges
    eta_values = torch.linspace(-4.0, -1.0, 10)
    gamma_values = torch.linspace(0.1, 0.6, 10)
    
    # Create sweep parameters
    binary_sweep_params = fitting.BinarySweepParameters(
        eta=eta_values,
        gamma=gamma_values,
        lambdah=torch.tensor([0.0]),
        distance_relationship_type=["powerlaw"],
        preferential_relationship_type=["powerlaw"],
        heterochronicity_relationship_type=["powerlaw"],
        generative_rule=[generative_rules.MatchingIndex()],
        num_iterations=[n_edges],
    )
    
    # Create sweep configuration
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_sweep_params,
        weighted_sweep_parameters=None,
        num_simulations=50,  # Run 50 simulations per parameter set
        distance_matrix=[distance_matrix]
    )
    
    # Define evaluation criteria
    criteria = [
        evaluation.DegreeKS(),
        evaluation.ClusteringKS(),
        evaluation.EdgeLengthKS(distance_matrix)
    ]
    energy_equation = evaluation.MaxCriteria(criteria)  # Use max of all criteria
    
    print(f"Running parameter sweep: {len(eta_values)} × {len(gamma_values)} = {len(eta_values)*len(gamma_values)} parameter combinations")
    print("This may take a few minutes...")
    
    # Run the sweep
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        binary_evaluations=[energy_equation],
        real_binary_matrices=target_network,
        save_model=False,
        save_run_history=False,
        verbose=False
    )
    
    # Find optimal parameters
    optimal_experiments, optimal_energies = fitting.optimise_evaluation(
        experiments=experiments,
        criterion=energy_equation,
    )
    
    if optimal_experiments:
        best_exp = optimal_experiments[0]
        best_energy = optimal_energies[0]
        
        print(f"\nBest parameters found:")
        print(f"  eta: {best_exp.run_config.binary_parameters.eta:.3f}")
        print(f"  gamma: {best_exp.run_config.binary_parameters.gamma:.3f}")
        print(f"  energy: {best_energy:.3f}")
    
    return optimal_experiments, optimal_energies


def example_3_compare_generative_rules():
    """Example 3: Compare different generative rules."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Comparing Generative Rules")
    print("=" * 60)
    
    # Load default data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_matrix = defaults.get_distance_matrix(device=device)
    target_network = defaults.get_binary_network(device=device)
    n_edges = int(target_network.sum().item() // 2)
    
    # Define rules to test
    rules_to_test = [
        ("Matching Index", generative_rules.MatchingIndex()),
        ("Neighbors", generative_rules.Neighbors()),
        ("Degree Product", generative_rules.DegreeProduct()),
        ("Clustering Coefficient", generative_rules.ClusteringCoefficient()),
        ("Spatial", generative_rules.Spatial()),
    ]
    
    # Fixed parameters for comparison
    eta = -2.0
    gamma = 0.3
    
    # Evaluation metric
    energy_equation = evaluation.MaxCriteria([
        evaluation.DegreeKS(),
        evaluation.ClusteringKS(),
        evaluation.EdgeLengthKS(distance_matrix)
    ])
    
    results = {}
    
    for rule_name, rule in rules_to_test:
        print(f"\nTesting {rule_name}...")
        
        # Create model
        model = GNMBinary(
            eta=eta,
            gamma=gamma,
            lambdah=0.0,
            distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw",
            heterochronicity_relationship_type="powerlaw",
            generative_rule=rule,
            distance_matrix=distance_matrix,
            device=device
        )
        
        # Generate multiple networks and average the energy
        energies = []
        for _ in range(10):
            model.reset()
            model.run(n_edges)
            generated = model.get_adjacency_matrix()
            energy = energy_equation(generated, target_network)
            energies.append(float(energy))
        
        avg_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        results[rule_name] = {
            "mean_energy": avg_energy,
            "std_energy": std_energy
        }
        
        print(f"  Energy: {avg_energy:.3f} ± {std_energy:.3f}")
    
    # Find best rule
    best_rule = min(results.keys(), key=lambda x: results[x]["mean_energy"])
    print(f"\nBest generative rule: {best_rule}")
    print(f"  Energy: {results[best_rule]['mean_energy']:.3f}")
    
    return results


def example_4_weight_optimization():
    """Example 4: Generate binary network and optimize weights."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Weight Optimization")
    print("=" * 60)
    
    # Load default data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_matrix = defaults.get_distance_matrix(device=device)
    target_network = defaults.get_binary_network(device=device)
    n_edges = int(target_network.sum().item() // 2)
    
    # First, generate a binary network
    print("Generating binary network...")
    binary_model = GNMBinary(
        eta=-2.0,
        gamma=0.3,
        lambdah=0.0,
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=generative_rules.MatchingIndex(),
        distance_matrix=distance_matrix,
        device=device
    )
    
    binary_model.run(n_edges)
    binary_network = binary_model.get_adjacency_matrix()
    
    print(f"Binary network generated: {int(binary_network.sum()//2)} edges")
    
    # Now optimize weights
    print("\nOptimizing edge weights...")
    
    # Create weight optimization criterion
    weight_criterion = weight_criteria.DistanceWeightedCommunicability(distance_matrix)
    
    # Create weighted model
    weighted_model = GNMWeighted(
        binary_adjacency_matrix=binary_network,
        alpha=0.01,  # Learning rate
        optimisation_criterion=weight_criterion,
        device=device
    )
    
    # Run weight optimization
    weighted_model.run(num_iterations=1000)
    
    # Get weighted network
    weighted_network = weighted_model.get_adjacency_matrix()
    
    # Compare binary and weighted networks
    print(f"\nNetwork statistics:")
    print(f"  Binary edges: {int(binary_network.sum()//2)}")
    print(f"  Weighted edges: {int((weighted_network > 0).sum()//2)}")
    print(f"  Mean weight: {weighted_network[weighted_network > 0].mean():.3f}")
    print(f"  Std weight: {weighted_network[weighted_network > 0].std():.3f}")
    
    return binary_network, weighted_network


def example_5_full_pipeline_with_custom_data():
    """Example 5: Full pipeline with custom connectome data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Full Pipeline with Custom Data")
    print("=" * 60)
    
    # Load your custom data (adjust paths as needed)
    data_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/data/preprocessed/01_first_analysises")
    
    try:
        # Load distance matrix
        dist_path = data_dir / "distance_matrix_68x68.npy"
        if dist_path.exists():
            distance_matrix_np = np.load(dist_path)
            print(f"Loaded distance matrix: {distance_matrix_np.shape}")
        else:
            # Use random data for demo
            print("Using random distance matrix for demo")
            coords = np.random.rand(68, 3) * 100
            distance_matrix_np = np.zeros((68, 68))
            for i in range(68):
                for j in range(68):
                    distance_matrix_np[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        # Load binary connectome
        binary_path = data_dir / "connectomes_binarized_68x68_density_15_percent.npy"
        if binary_path.exists():
            binary_connectomes = np.load(binary_path)
            target_network_np = binary_connectomes[:, :, 0]  # Use first subject
            print(f"Loaded connectome: {target_network_np.shape}")
        else:
            # Use random binary network for demo
            print("Using random binary network for demo")
            target_network_np = np.random.rand(68, 68) > 0.85
            target_network_np = ((target_network_np + target_network_np.T) > 0).astype(float)
            np.fill_diagonal(target_network_np, 0)
        
        # Convert to torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distance_matrix = torch.tensor(distance_matrix_np, dtype=torch.float32, device=device)
        target_network = torch.tensor(target_network_np, dtype=torch.float32, device=device)
        
        n_edges = int(target_network.sum().item() // 2)
        print(f"Target network has {n_edges} edges")
        
    except Exception as e:
        print(f"Could not load custom data: {e}")
        print("Using GNM defaults instead")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distance_matrix = defaults.get_distance_matrix(device=device)
        target_network = defaults.get_binary_network(device=device)
        n_edges = int(target_network.sum().item() // 2)
    
    # Step 1: Find best generative rule
    print("\nStep 1: Finding best generative rule...")
    rules = [
        generative_rules.MatchingIndex(),
        generative_rules.Neighbors(),
        generative_rules.DegreeProduct()
    ]
    
    best_rule = None
    best_energy = float('inf')
    
    for rule in rules:
        sweep_config = fitting.SweepConfig(
            binary_sweep_parameters=fitting.BinarySweepParameters(
                eta=torch.tensor([-2.0]),
                gamma=torch.tensor([0.3]),
                lambdah=torch.tensor([0.0]),
                distance_relationship_type=["powerlaw"],
                preferential_relationship_type=["powerlaw"],
                heterochronicity_relationship_type=["powerlaw"],
                generative_rule=[rule],
                num_iterations=[n_edges],
            ),
            num_simulations=10,
            distance_matrix=[distance_matrix]
        )
        
        energy_eq = evaluation.MaxCriteria([
            evaluation.DegreeKS(),
            evaluation.ClusteringKS()
        ])
        
        experiments = fitting.perform_sweep(
            sweep_config=sweep_config,
            binary_evaluations=[energy_eq],
            real_binary_matrices=target_network,
            save_model=False,
            save_run_history=False,
            verbose=False
        )
        
        if experiments:
            energy = experiments[0].evaluation_dict[energy_eq]
            if energy < best_energy:
                best_energy = energy
                best_rule = rule
    
    print(f"Best rule: {best_rule.__class__.__name__} (energy: {best_energy:.3f})")
    
    # Step 2: Optimize parameters for best rule
    print("\nStep 2: Optimizing parameters...")
    
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=fitting.BinarySweepParameters(
            eta=torch.linspace(-3.0, -1.0, 5),
            gamma=torch.linspace(0.1, 0.5, 5),
            lambdah=torch.tensor([0.0]),
            distance_relationship_type=["powerlaw"],
            preferential_relationship_type=["powerlaw"],
            heterochronicity_relationship_type=["powerlaw"],
            generative_rule=[best_rule],
            num_iterations=[n_edges],
        ),
        num_simulations=20,
        distance_matrix=[distance_matrix]
    )
    
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        binary_evaluations=[energy_eq],
        real_binary_matrices=target_network,
        save_model=False,
        save_run_history=False,
        verbose=False
    )
    
    optimal_experiments, optimal_energies = fitting.optimise_evaluation(
        experiments=experiments,
        criterion=energy_eq
    )
    
    if optimal_experiments:
        best_exp = optimal_experiments[0]
        print(f"Optimal eta: {best_exp.run_config.binary_parameters.eta:.3f}")
        print(f"Optimal gamma: {best_exp.run_config.binary_parameters.gamma:.3f}")
        print(f"Optimal energy: {optimal_energies[0]:.3f}")
    
    # Step 3: Generate networks with optimal parameters
    print("\nStep 3: Generating networks with optimal parameters...")
    
    model = GNMBinary(
        eta=float(best_exp.run_config.binary_parameters.eta),
        gamma=float(best_exp.run_config.binary_parameters.gamma),
        lambdah=0.0,
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=best_rule,
        distance_matrix=distance_matrix,
        device=device
    )
    
    # Generate 5 networks
    generated_networks = []
    for i in range(5):
        model.reset()
        model.run(n_edges)
        generated_networks.append(model.get_adjacency_matrix())
        print(f"  Generated network {i+1}: {int(generated_networks[-1].sum()//2)} edges")
    
    # Step 4: Evaluate generated networks
    print("\nStep 4: Evaluating generated networks...")
    
    all_metrics = {
        "degree_ks": evaluation.DegreeKS(),
        "clustering_ks": evaluation.ClusteringKS(),
        "edge_length_ks": evaluation.EdgeLengthKS(distance_matrix),
        "frobenius": evaluation.Frobenius()
    }
    
    for metric_name, metric in all_metrics.items():
        values = []
        for gen_net in generated_networks:
            values.append(float(metric(gen_net, target_network)))
        print(f"  {metric_name}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    
    print("\nPipeline completed!")
    
    return generated_networks


def main():
    """Run all examples."""
    print("GNM LIBRARY COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_1_basic_gnm_generation()
    example_2_parameter_sweep()
    example_3_compare_generative_rules()
    example_4_weight_optimization()
    example_5_full_pipeline_with_custom_data()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED!")


if __name__ == "__main__":
    main()
    
    
    
#     # Quick test with GNM defaults
# python main_pipeline.py test --config quick_test

# # Comprehensive analysis
# python main_pipeline.py comprehensive --compare-rules --fit-weights

# # Parameter sweep
# python main_pipeline.py sweep --config comprehensive

# # Full pipeline
# python main_pipeline.py full