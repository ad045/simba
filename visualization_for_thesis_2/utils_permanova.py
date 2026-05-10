
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.utils import shuffle


from skbio.stats.distance import permanova, DistanceMatrix
from sklearn.metrics import pairwise_distances


def run_permanova(X, labels, n=999):
    valid        = ~np.isnan(X).any(axis=1)
    X, labels    = X[valid], np.array(labels)[valid]
    dist         = pairwise_distances(X, metric="euclidean")
    print("Dist is symetric:", np.allclose(dist, dist.T))   
    dist = (dist + dist.T) / 2 # Ensure symmetry - there seem to be some minimal numerical issues with the distance matrix 
    np.fill_diagonal(dist, 0)
    dm           = DistanceMatrix(dist)
    res          = permanova(dm, labels, permutations=n)
    return res["test statistic"], res["p-value"]


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"



def print_latex_permanova_table(results, groups, sig_stars_fn):
    """Print a LaTeX table of pairwise PERMANOVA results."""
    
    short = {g: g[:4] for g in groups}
    
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Order A} & \textbf{Order B} & $F$ & $p$ & Cohen's $d$ \\")
    lines.append(r"\hline")
    
    for (a, b), vals in results.items():
        F     = vals["F"]
        p     = vals["p"]
        d     = vals["d"]
        stars = sig_stars_fn(p)
        p_str = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        relevant_element = f"{p_str} {stars}"
        if p < 0.01: 
            relevant_element = r"\textbf{" + relevant_element + "}"
        
        lines.append(
            f"{a} & {b} & {F:.2f} & {relevant_element} & {d:.2f} \\\\"
        )
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Pairwise PERMANOVA (999 permutations) on the first two "
        r"principal components. $F$: pseudo-$F$ statistic. "
        r"$p$: permutation $p$-value. "
        r"Cohen's $d$ computed on PC1. "
        r"Significance: *** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns $p\geq0.05$.}"
    )
    lines.append(r"\label{tab:permanova_orders}")
    lines.append(r"\end{table}")
    
    print("\n".join(lines))
    