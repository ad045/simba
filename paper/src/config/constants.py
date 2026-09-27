# """
# Configuration constants for the 14_4D_lab project.
# Centralizes magic numbers and commonly used values.
# """

# # Computational defaults
DEFAULT_APPEND_INTERVAL = 10

# # File naming patterns
CONNECTOMES_WEIGHTED_PATTERN = "connectomes_weighted_{resolution}x{resolution}.npy"
CONNECTOMES_BINARY_PATTERN = "connectomes_binarized_{resolution}x{resolution}_density_{density}_percent.npy"
DISTANCE_MATRIX_PATTERN = "distance_matrix_{resolution}x{resolution}.npy"

# # Tolerance values for numerical comparisons
NUMERICAL_TOLERANCE = 1e-10



#################################################

# # Default dataset + resolution for connectome data
# DEFAULT_DATASET_NAME = "shafiei_human_consensus_dataset"
# DEFAULT_RESOLUTION = 68

# # Random seed for reproducibility
# DEFAULT_RANDOM_SEED = 42

# # GNM parameter defaults
# DEFAULT_N_ETA = 200
# DEFAULT_N_GAMMA = 200
# DEFAULT_N_LAMBDA = 1
# DEFAULT_NUM_SIMULATIONS = 10

# # ESN parameter defaults
# DEFAULT_SPECTRAL_RADIUS = 0.99
# DEFAULT_INPUT_LENGTH = 4000
# DEFAULT_INPUT_SCALING = 0.0001 # 1.0
# DEFAULT_N_RUNS = 10
# DEFAULT_N_LAGS = 50
# DEFAULT_TEST_LENGTH = 1000
# DEFAULT_N_TRANSIENT = 0
# DEFAULT_LEAK_RATE = 1.0
# DEFAULT_BIAS = 1.0

# # Default density percentages for connectivity analysis
# DEFAULT_DENSITIES = [10, 12, 14, 16, 18, 20]

# # Default methods and algorithms
# DEFAULT_REGULARIZATION_METHOD = "pinv"
# DEFAULT_GENERATIVE_RULES = ["matching_index"]
# DEFAULT_EVALUATION_METRICS = ["degree_ks", "clustering_ks", "edge_length_ks"]
# DEFAULT_WEIGHT_CRITERION = "distance_weighted_communicability"

# # Default parameter ranges
# DEFAULT_ETA_RANGE = (-3.0, 0.0)
# DEFAULT_GAMMA_RANGE = (0.001, 0.6)
# DEFAULT_LAMBDA_RANGE = (0.0, 0.0)
# DEFAULT_ALPHA = 0.01

# # Plotting defaults
# DEFAULT_COLORMAP_CONTINUOUS = "viridis"
# DEFAULT_COLORMAP_DIVERGING = "RdBu_r"
# DEFAULT_FIGURE_FORMAT = "pdf"
# DEFAULT_DPI = 300

# # Default figure sizes
# DEFAULT_FIGSIZE_SINGLE = (8, 6)
# DEFAULT_FIGSIZE_DOUBLE = (12, 6)
# DEFAULT_FIGSIZE_LANDSCAPE = (10, 8)
# DEFAULT_FIGSIZE_HUGE = (18, 5)

