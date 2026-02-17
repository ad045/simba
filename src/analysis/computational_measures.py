import numpy as np

from src.analysis.from_fatemeh import compute_KR as compute_kernel_rank_fatemeh # needed for next script 

# Kernel Rank 
def kernel_rank(A, threshold=0.01):
    eigs = np.linalg.eigvals(A)
    return {
        "thresholded_and_summed_0.01": np.sum(np.abs(eigs) > threshold * np.abs(eigs).max()), 
        "max": np.abs(eigs).max(), 
        "phase_of_lambda_max": np.angle(eigs.max()), 
        "phase_diff_of_lambda_max_and_2nd": np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max()), 
    }

def kernel_rank_esn(A):
    # References Fatemeh's code. 
    return compute_kernel_rank_fatemeh(A)

# Effective dimensionality (Participation Ratio) (check) 
def effective_dimensionality(A):
    eigs = np.linalg.eigvals(A)
    eigs_abs = np.abs(eigs)
    return np.sum(eigs_abs)**2 / np.sum(eigs_abs**2)


# Multifunctionality (check)
# Source: https://www.nature.com/articles/s41467-018-03977-1
# def multifunctionality(A, n_functions=10):
#     eigs, vecs = np.linalg.eig(A)
#     idx = np.argsort(np.abs(eigs))[::-1]
#     return np.mean([np.linalg.norm(vecs[:, idx[i]]) for i in range(min(n_functions, len(eigs)))])


# More accurate multifunctionality (based on Rigotti et al.)
def multifunctionality(neural_responses, n_samples=1000):
    """
    neural_responses: NxM matrix (N neurons, M conditions/stimuli)
    Returns: fraction of random dichotomies that are linearly separable
    """
    from sklearn.svm import LinearSVC
    
    separable = 0
    for _ in range(n_samples):
        labels = np.random.choice([0, 1], size=neural_responses.shape[1])
        if len(np.unique(labels)) < 2:
            continue
        clf = LinearSVC()
        try:
            clf.fit(neural_responses.T, labels)
            separable += 1
        except:
            pass
    return separable / n_samples