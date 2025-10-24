"""
Example usage script showing how to run the metric update pipeline.
"""
import numpy as np
from pathlib import Path
from src.analysis.update_metrics import update_csv_with_metrics


def main():
    # ========== CONFIGURATION ==========
    
    # Animal/dataset configuration
    
    DATASET = "suarez_MaMI_dataset"
    EXPERIMENT = "60_generally_finer_search_animal_0" # 63_fine_grid_animal_0" # 61_testing_with_kayson_animal_0" # 70_mix_and_match_animal_0"
    BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
    OUTPUT_PATH = BASE_PATH / "output" / "gnm" / DATASET / EXPERIMENT
    generated_networks_dir = OUTPUT_PATH / "generated_networks"

    # Output CSV file
    OUTPUT_CSV = OUTPUT_PATH / f"further_metrics_exp_{EXPERIMENT}.csv"

    # Path to distance matrix 
    DISTANCE_MATRIX_PATH = BASE_PATH / f"data/preprocessed/{DATASET}/02_distance_matrices/distance_matrix_50.npy"



    # BASE_DIR = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{EXPERIMENT_ID}")
    CSV_PATH = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}.csv"
    
    # Define which metrics you want to compute. Only metrics that are missing from the CSV will be computed
    
    REQUESTED_METRICS = {
        'structural': [
            # 'small_world_omega',
            # 'structural_complexity',
            "density", 
            # 'hubness_gini',
            # 'degree_assortativity',
            # 'richclub_coefficient',
            # 'richclub_k_max',
            # 'distance_dependent_degree_assortativity',
        ],
        'dynamics': [
            # 'spectral_radius',
            # 'spectral_gap',
            # 'eigenspectrum_entropy',
            # 'diffusion_efficiency',
            # 'propagation_efficiency',
            # 'avg_controllability',
        ],
        'computation': [
            # 'kernel_rank',
            # 'effective_dimensionality',
            # 'memory_capacity_estimate',
            # 'multifunctionality_estimate',
        ]
    }
    
    # Create metric -> category mapping
    METRIC_CATEGORIES = {}
    for category, metrics in REQUESTED_METRICS.items():
        for metric in metrics:
            METRIC_CATEGORIES[metric] = category
    
    # ========== LOAD DISTANCE MATRIX ==========
    
    print("Loading distance matrix...")
    if DISTANCE_MATRIX_PATH.exists():
        distance_matrix = np.load(DISTANCE_MATRIX_PATH)
        print(f"  Loaded: {distance_matrix.shape}")
    else:
        print(f"  WARNING: Distance matrix not found at {DISTANCE_MATRIX_PATH}")
        print(f"  Creating dummy distance matrix for testing...")
        # For testing: create random distance matrix
        # Replace this with actual loading in production
        distance_matrix = np.random.rand(100, 100)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(distance_matrix, 0)
    
    # ========== RUN PIPELINE ==========
    
    print(f"\nStarting metric computation pipeline...")
    print(f"  Experiment: {EXPERIMENT}")
    print(f"  CSV path: {CSV_PATH}")
    print(f"  Requested metrics: {sum(len(m) for m in REQUESTED_METRICS.values())} total")
    
    # Update CSV with missing metrics
    update_csv_with_metrics(
        csv_path=str(CSV_PATH),
        generated_networks_dir=generated_networks_dir, # str(BASE_PATH),
        distance_matrix=distance_matrix,
        requested_metrics=REQUESTED_METRICS,
        metric_categories=METRIC_CATEGORIES
    )
    
    print("\n✓ Pipeline completed successfully!")
    print(f"  Updated CSV saved as: {CSV_PATH.parent / (CSV_PATH.stem + '_updated.csv')}")


# def batch_process_multiple_animals():
#     """
#     Example: Process multiple animals in batch.
#     """
#     ANIMAL_IDS = [
#         "70_mix_and_match_animal_0",
#         "70_mix_and_match_animal_1",
#         # Add more animal IDs here
#     ]
    
#     BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset")
#     DISTANCE_MATRIX_PATH = Path("/Users/adrian/path/to/distance_matrix.npy")
    
#     # Load distance matrix once
#     distance_matrix = np.load(DISTANCE_MATRIX_PATH)
    
#     # Define metrics once
#     REQUESTED_METRICS = {
#         'structural': ['small_world_omega', 'structural_complexity', 'hubness_gini'],
#         'dynamics': ['spectral_radius', 'spectral_gap'],
#         'computation': ['kernel_rank']
#     }
    
#     METRIC_CATEGORIES = {}
#     for category, metrics in REQUESTED_METRICS.items():
#         for metric in metrics:
#             METRIC_CATEGORIES[metric] = category
    
#     # Process each animal
#     for animal_id in ANIMAL_IDS:
#         print(f"\n{'='*60}")
#         print(f"Processing {animal_id}...")
#         print(f"{'='*60}")
        
#         base_dir = BASE_PATH / animal_id
#         csv_path = base_dir / f"all_metrics_for_{animal_id}.csv"
        
#         if not csv_path.exists():
#             print(f"  WARNING: CSV not found, skipping: {csv_path}")
#             continue
        
#         try:
#             update_csv_with_metrics(
#                 csv_path=str(csv_path),
#                 base_dir=str(base_dir),
#                 distance_matrix=distance_matrix,
#                 requested_metrics=REQUESTED_METRICS,
#                 metric_categories=METRIC_CATEGORIES
#             )
#             print(f"  ✓ Completed {animal_id}")
#         except Exception as e:
#             print(f"  ✗ Error processing {animal_id}: {e}")
#             import traceback
#             traceback.print_exc()


if __name__ == "__main__":
    # Run single animal
    main()
    
    # Or run batch processing
    # batch_process_multiple_animals()