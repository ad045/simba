"""
Figure 4 — Developmental and evolutionary gradients in the property morphospace.

Layout  (18 × 9 cm):
┌─────────────────────────────┬────────────────────┬──────────────┐
│  A  HCP-D individuals       │  B  MaMI mammals   │  C  Stats    │
│     colored by age (6–22)   │     colored by     │  top: age~PC1│
│     + consensus trajectory  │     taxon. order   │  bot: PERMANOVA│
│         (~50 % width)       │     (~30 % width)  │  (~20 %)     │
└─────────────────────────────┴────────────────────┴──────────────┘

PCA is fitted on the 25,000 GNM networks; all other datasets are
*transformed* (not re-fitted) into that same space so positions are
comparable across panels.
"""

# %% ── Imports ─────────────────────────────────────────────────────────────────
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from pathlib import Path
from scipy import stats
from scipy.interpolate import make_smoothing_spline
from scipy.spatial import ConvexHull, Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from vizman import viz
from itertools import combinations
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from config import PROPERTY_NAMES
from utils_permanova import run_permanova, sig_stars

# %% ── Paths ───────────────────────────────────────────────────────────────────

BASE       = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
PLOT_DIR   = Path("/Users/adrian/Documents/01_projects/14_4D_lab/latex_thesis/figures/results")
OUTPUT_DIR = BASE / "output/trade_off_analysis/04_age_and_phylogeny"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PKL_PATH      = BASE / "output/00_trade_off_analysis/all_datasets_precise_categories.pkl"
AGES_PATH     = BASE / "data/preprocessed/lexis_data_developing/04_further_info/00_ages.npy"
MAMI_META     = BASE / "data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv"
ETA_GAMMA_PKL = BASE / "output/00_trade_off_analysis/hcp_schaefer_100_dataset_gnm_eta_and_gamma.pkl"

# %% ── Constants ───────────────────────────────────────────────────────────────

AGE_CMAP  = LinearSegmentedColormap.from_list("age", ["#E6B212", "#C74800", "#BF003F"])

# Five main orders (≥20 animals) + "Other" for the rest
ORDER_COLORS = {
    "Carnivora":       "#2E86AB",   # steel blue
    "Primates":        "#A23B72",   # plum
    "Cetartiodactyla": "#F18F01",   # amber
    "Rodentia":        "#44BBA4",   # teal
    "Chiroptera":      "#E94F37",   # tomato
    "Other":           "#C8C8C8",   # light gray
}
MAIN_ORDERS = [o for o in ORDER_COLORS if o != "Other"]

# %% ── Load data ───────────────────────────────────────────────────────────────

with open(PKL_PATH, "rb") as f:
    datasets = pickle.load(f)

df_gnm   = datasets["hcp_schaefer_100_dataset_gnm"]
df_lexi  = datasets["lexis_data_developing"]      # 629 individuals, HCP-D
df_mami  = datasets["suarez_MaMI_dataset"]         # 224 mammals

# ── Age metadata (635 values; align to the 629 surviving rows via df index)
ages_all = np.load(AGES_PATH)                      # shape (635,)
# The pkl dataframe index encodes which original networks survived filtering
if len(ages_all) == len(df_lexi):
    ages = ages_all
else:
    ages = ages_all[df_lexi.index]                 # drop the filtered-out rows
assert len(ages) == len(df_lexi), \
    f"Age length mismatch: {len(ages)} vs {len(df_lexi)}"

# ── MaMI metadata (225 rows; align to 224 surviving rows via df index)
mami_meta_full = pd.read_csv(MAMI_META).reset_index(drop=True)
if len(mami_meta_full) == len(df_mami):
    mami_meta = mami_meta_full
else:
    mami_meta = mami_meta_full.iloc[df_mami.index].reset_index(drop=True)
assert len(mami_meta) == len(df_mami), \
    f"MaMI metadata length mismatch: {len(mami_meta)} vs {len(df_mami)}"

orders_raw = mami_meta["order"].fillna("Other").values
orders = np.where(np.isin(orders_raw, MAIN_ORDERS), orders_raw, "Other")

# %% ── Build shared feature set and fit PCA on GNM morphospace ────────────────
# Columns must be: (a) present in all three datasets, (b) fully non-NaN in GNM.

common_cols = (
    set(df_gnm.columns)
    .intersection(df_lexi.columns)
    .intersection(df_mami.columns)
)

# Drop columns that are NaN anywhere in GNM (e.g. wiring_cost can be constant)
gnm_sub = df_gnm[list(common_cols)].replace([np.inf, -np.inf], np.nan)
valid_cols = gnm_sub.columns[gnm_sub.notna().all()].tolist()
gnm_sub = gnm_sub[valid_cols]

print(f"Common valid features: {len(valid_cols)}")

scaler = StandardScaler()
gnm_scaled = scaler.fit_transform(gnm_sub)

pca = PCA(n_components=10, random_state=42)
pca.fit(gnm_scaled)

ev = pca.explained_variance_ratio_ * 100
print(f"PC1: {ev[0]:.1f}%  PC2: {ev[1]:.1f}%  PC3: {ev[2]:.1f}%")


def embed(df):
    """Project a dataset into the GNM PCA space."""
    sub = df[valid_cols].replace([np.inf, -np.inf], np.nan)
    # Per-row NaN imputation: fill with GNM column mean (scaler.mean_)
    # so that rows with sporadic NaNs don't drop entirely.
    arr = sub.values.astype(float)
    for j, col_mean in enumerate(scaler.mean_):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_mean
    return pca.transform(scaler.transform(arr))  # (n, 10)


gnm_pca  = pca.transform(gnm_scaled)             # already scaled
lexi_pca = embed(df_lexi)
mami_pca = embed(df_mami)

# %% ── Pareto-optimal GNM region ──────────────────────────────────────────────
# Identify which of the 2,500 (η, γ) cells are Pareto-optimal across three
# trade-off pairs (efficiency↑/memory↑/robustness↑ vs wiring-cost↓).
# The "Pareto region" is the union of cells on any front.

with open(ETA_GAMMA_PKL, "rb") as f:
    _gnm_eta_gamma = pickle.load(f)          # (25000, 2): eta, gamma

# Pareto objectives — all present in df_gnm via precise_categories
_obj_cols = [c for c in [
    "global_efficiency", "wiring_cost",
    "mc_input_scaling_0_1_mc_mean",
    "targeted_attack_robustness_rob_targeted_auc",
] if c in df_gnm.columns]

_gnm_for_pareto = df_gnm[_obj_cols].copy()
_gnm_for_pareto["eta"]   = _gnm_eta_gamma["eta"].values
_gnm_for_pareto["gamma"] = _gnm_eta_gamma["gamma"].values
_gnm_cell = _gnm_for_pareto.groupby(["eta", "gamma"])[_obj_cols].mean().reset_index()

def _is_pareto_efficient(costs):
    """Pareto efficiency for minimisation problem (negate to maximise)."""
    n = len(costs)
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if is_eff[i]:
            dom = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            dom[i] = False
            is_eff[i] = not dom.any()
    return is_eff

_pareto_union = np.zeros(len(_gnm_cell), dtype=bool)
_pair_cfgs = [
    ("global_efficiency",                        "wiring_cost", True, False),
    ("mc_input_scaling_0_1_mc_mean",             "wiring_cost", True, False),
    ("targeted_attack_robustness_rob_targeted_auc", "wiring_cost", True, False),
]
print("\nPareto front sizes per pair:")
for obj1, obj2, max1, max2 in _pair_cfgs:
    if obj1 not in _gnm_cell.columns or obj2 not in _gnm_cell.columns:
        continue
    _sub = _gnm_cell[[obj1, obj2]].dropna()
    c1   = _sub[obj1].values * (-1 if max1 else 1)
    c2   = _sub[obj2].values * (-1 if max2 else 1)
    _front = _is_pareto_efficient(np.column_stack([c1, c2]))
    _pareto_union[_sub.index[_front]] = True
    print(f"  {obj1} vs {obj2}: {_front.sum()} cells ({100*_front.sum()/len(_sub):.1f}%)")

_pareto_pairs_set = set(
    zip(_gnm_cell.loc[_pareto_union, "eta"].round(8),
        _gnm_cell.loc[_pareto_union, "gamma"].round(8))
)
print(f"Union Pareto cells: {_pareto_union.sum()} / {len(_gnm_cell)}")

# Map to individual GNM networks (row order matches gnm_pca)
_eta_r = _gnm_eta_gamma["eta"].round(8).values
_gam_r = _gnm_eta_gamma["gamma"].round(8).values
is_pareto_net = np.array([
    (e, g) in _pareto_pairs_set
    for e, g in zip(_eta_r, _gam_r)
])
print(f"Pareto-optimal GNM networks: {is_pareto_net.sum()} / {len(is_pareto_net)}")

# Convex hull around Pareto-optimal GNM networks in PC1–PC2 space
_pareto_pc   = gnm_pca[is_pareto_net, :2]
_hull        = ConvexHull(_pareto_pc)
_hull_verts  = _pareto_pc[_hull.vertices]   # ordered vertices for polygon

def _pts_in_hull(pts, hull_verts):
    """Boolean mask: which rows of pts lie inside the convex hull."""
    return Delaunay(hull_verts).find_simplex(pts) >= 0

pct_lexi_in = _pts_in_hull(lexi_pca[:, :2], _hull_verts).mean() * 100
pct_mami_in = _pts_in_hull(mami_pca[:, :2], _hull_verts).mean() * 100
print(f"\nHCP-D inside Pareto hull: {pct_lexi_in:.1f}%")
print(f"MaMI  inside Pareto hull: {pct_mami_in:.1f}%")

_pareto_pc_nonpareto = gnm_pca[~is_pareto_net, :2]
print(f"\nPareto GNM      — mean PC1={_pareto_pc[:,0].mean():+.3f}  "
      f"mean PC2={_pareto_pc[:,1].mean():+.3f}")
print(f"Non-Pareto GNM  — mean PC1={_pareto_pc_nonpareto[:,0].mean():+.3f}  "
      f"mean PC2={_pareto_pc_nonpareto[:,1].mean():+.3f}")

# %% ── Developmental consensus trajectory ─────────────────────────────────────

lexi_df_pca = pd.DataFrame({"PC1": lexi_pca[:, 0], "PC2": lexi_pca[:, 1], "age": ages})
consensus   = lexi_df_pca.groupby("age")[["PC1", "PC2"]].mean().reset_index()
# Sort by age (should be, but ensure)
consensus   = consensus.sort_values("age")

# Smooth spline trajectory
t = consensus["age"].values
x_spl = make_smoothing_spline(t, consensus["PC1"].values, lam=1.0)(t)
y_spl = make_smoothing_spline(t, consensus["PC2"].values, lam=1.0)(t)

# %% ── PERMANOVA per order (one-vs-rest in PC1/PC2 space) ─────────────────────

print("\nRunning PERMANOVA per taxon. order (999 permutations)...")
permanova_results = {}
X_mami_2d = mami_pca[:, :2]

for order in MAIN_ORDERS:
    labels_bin = np.where(orders == order, order, "other")
    F, p = run_permanova(X_mami_2d, labels_bin, n=999)
    permanova_results[order] = {"F": F, "p": p}
    print(f"  {order:20s}  F={F:.2f}  p={p:.3f} {sig_stars(p)}")

# %% ── Age ~ PC regression (all 10 PCs) ──────────────────────────────────────

print("\n── Age ~ PC correlations ──────────────────────────────────────────────────")
print(f"  {'PC':>2}  {'r':>7}  {'r²':>6}  {'p':>10}")
age_pc_corrs = []
for i in range(10):
    r_i, p_i = stats.pearsonr(ages, lexi_pca[:, i])
    age_pc_corrs.append((i, r_i, p_i))
    stars = "***" if p_i < 0.001 else "**" if p_i < 0.01 else "*" if p_i < 0.05 else "ns"
    print(f"  {i+1:>2}  {r_i:>+7.3f}  {r_i**2:>6.3f}  {p_i:>10.2e}  {stars}")

# Best PC for age gradient
best_age_pc = max(age_pc_corrs, key=lambda t: abs(t[1]))
print(f"\n  → strongest age signal: PC{best_age_pc[0]+1}  r={best_age_pc[1]:.3f}")

# %% ── PERMANOVA per PC (one-vs-rest, n=199 permutations for speed) ──────────

print("\n── PERMANOVA pseudo-F per PC (one-vs-rest) ────────────────────────────────")
header = f"  {'PC':>2}" + "".join(f"  {o[:8]:>12}" for o in MAIN_ORDERS)
print(header)

perm_per_pc: dict[int, dict[str, dict]] = {}
for i in range(10):
    X_1d = mami_pca[:, i:i+1]
    row_str = f"  {i+1:>2}"
    perm_per_pc[i] = {}
    for order in MAIN_ORDERS:
        labels_bin = np.where(orders == order, order, "other")
        F_i, p_i = run_permanova(X_1d, labels_bin, n=199)
        perm_per_pc[i][order] = {"F": F_i, "p": p_i}
        stars = "***" if p_i < 0.001 else "**" if p_i < 0.01 else " *" if p_i < 0.05 else "ns"
        row_str += f"  {F_i:>8.1f}{stars}"
    print(row_str)

# Mean pseudo-F across orders per PC
mean_F_per_pc = {
    i: np.mean([perm_per_pc[i][o]["F"] for o in MAIN_ORDERS])
    for i in range(10)
}
best_phylo_pc = max(mean_F_per_pc, key=mean_F_per_pc.get)
print(f"\n  → strongest phylogenetic signal: PC{best_phylo_pc+1}  mean-F={mean_F_per_pc[best_phylo_pc]:.1f}")

# %% ── Diagnostic figure: r and mean-F per PC ─────────────────────────────────

fig_diag, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=viz.cm_to_inch((14, 6)),
                                          layout="constrained")

pc_labels = [f"PC{i+1}" for i in range(10)]
r_vals  = [t[1] for t in age_pc_corrs]
p_vals_ = [t[2] for t in age_pc_corrs]
bar_colors = ["#C74800" if abs(r_vals[i]) == max(abs(v) for v in r_vals) else "#AAAAAA"
              for i in range(10)]

bars_d = ax_d1.bar(pc_labels, r_vals, color=bar_colors, edgecolor="black", linewidth=0.4)
ax_d1.axhline(0, color="black", lw=0.6)
for i, (rv, pv) in enumerate(zip(r_vals, p_vals_)):
    stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
    ax_d1.text(i, rv + (0.01 if rv >= 0 else -0.01), stars,
               ha="center", va="bottom" if rv >= 0 else "top", fontsize=6)
ax_d1.set_ylabel("Pearson r (age ~ PC score)", fontsize=7)
ax_d1.set_title("Age gradient per PC", fontsize=8)
ax_d1.tick_params(labelsize=6)
ax_d1.spines["top"].set_visible(False)
ax_d1.spines["right"].set_visible(False)

mean_F_vals = [mean_F_per_pc[i] for i in range(10)]
bar_colors2 = ["#2E86AB" if mean_F_vals[i] == max(mean_F_vals) else "#AAAAAA"
               for i in range(10)]
ax_d2.bar(pc_labels, mean_F_vals, color=bar_colors2, edgecolor="black", linewidth=0.4)
ax_d2.set_ylabel("Mean PERMANOVA pseudo-F (per order)", fontsize=7)
ax_d2.set_title("Phylogenetic order separation per PC", fontsize=8)
ax_d2.tick_params(labelsize=6)
ax_d2.spines["top"].set_visible(False)
ax_d2.spines["right"].set_visible(False)

diag_path = OUTPUT_DIR / "diagnostic_pc_age_phylo.pdf"
fig_diag.savefig(diag_path, bbox_inches="tight", dpi=150)
print(f"\nSaved diagnostic: {diag_path}")
plt.show()

# %% ── Age ~ PC1 regression (use for main figure) ────────────────────────────

r, p_age = stats.pearsonr(ages, lexi_pca[:, 0])
print(f"\nAge ~ PC1:  r = {r:.3f},  p = {p_age:.2e}")

# %% ── Helpers ─────────────────────────────────────────────────────────────────

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a covariance-based confidence ellipse for (x, y) data."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height,
        angle=theta,
        **kwargs
    )
    ax.add_patch(ell)


# %% ── Publication-quality PERMANOVA on PC2 alone (n=999) ────────────────────

print("\nFinal PERMANOVA on PC2 alone (999 permutations)…")
permanova_pc2 = {}
X_mami_pc2 = mami_pca[:, 1:2]
for order in MAIN_ORDERS:
    labels_bin = np.where(orders == order, order, "other")
    F, p = run_permanova(X_mami_pc2, labels_bin, n=999)
    permanova_pc2[order] = {"F": F, "p": p}
    print(f"  {order:20s}  F={F:.2f}  p={p:.3f} {sig_stars(p)}")

# %% ── MANOVA + univariate ANOVA (parametric supplement to PERMANOVA) ─────────
#
# "Other" is a heterogeneous grab-bag; exclude it so parametric assumptions
# are more tenable (groups should be biologically coherent).

mask_main   = np.isin(orders, MAIN_ORDERS)
pca_main    = mami_pca[mask_main]
orders_main = orders[mask_main]

# ── Overall MANOVA: PC1 + PC2 ~ order ────────────────────────────────────────
manova_df = pd.DataFrame(
    {"PC1": pca_main[:, 0], "PC2": pca_main[:, 1], "order": orders_main}
)
mv     = MANOVA.from_formula("PC1 + PC2 ~ order", data=manova_df)
mv_res = mv.mv_test()
print("\n── MANOVA (PC1 + PC2 ~ order, main orders only) ─────────────────────────")
print(mv_res.summary())

_order_stat = mv_res.results["order"]["stat"]
pillai_val  = _order_stat.loc["Pillai's trace", "Value"]
pillai_F    = _order_stat.loc["Pillai's trace", "F Value"]
pillai_p    = _order_stat.loc["Pillai's trace", "Pr > F"]
print(f"\nPillai's trace = {pillai_val:.4f},  F = {pillai_F:.2f},  p = {pillai_p:.2e}")

# ── Univariate ANOVA + partial η² per PC ─────────────────────────────────────
print("\n── Univariate ANOVA per PC (main orders only) ───────────────────────────")
print(f"  {'PC':>2}  {'F':>8}  {'p':>10}  {'η²_p':>6}")
anova_results = {}
for i in range(10):
    groups = [pca_main[orders_main == o, i] for o in MAIN_ORDERS]
    F_i, p_i = stats.f_oneway(*groups)
    grand_mean = pca_main[:, i].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within  = sum(((g - g.mean()) ** 2).sum() for g in groups)
    eta2_p     = ss_between / (ss_between + ss_within)
    anova_results[i] = {"F": F_i, "p": p_i, "eta2_p": eta2_p}
    stars = "***" if p_i < 0.001 else "**" if p_i < 0.01 else "*" if p_i < 0.05 else "ns"
    print(f"  {i+1:>2}  {F_i:>8.2f}  {p_i:>10.2e}  {eta2_p:>6.3f}  {stars}")

# ── Tukey HSD post-hoc on PC2 (phylogenetic axis) ────────────────────────────
print("\n── Tukey HSD post-hoc on PC2 ────────────────────────────────────────────")
tukey_res    = pairwise_tukeyhsd(pca_main[:, 1], orders_main)
print(tukey_res.summary())

_orders_uniq = list(tukey_res.groupsunique)
_n_ord       = len(_orders_uniq)
tukey_pmat   = np.ones((_n_ord, _n_ord))
tukey_reject = np.zeros((_n_ord, _n_ord), dtype=bool)
for k, (i1, i2) in enumerate(combinations(range(_n_ord), 2)):
    pv = tukey_res.pvalues[k]
    tukey_pmat[i1, i2]   = pv
    tukey_pmat[i2, i1]   = pv
    tukey_reject[i1, i2] = tukey_res.reject[k]
    tukey_reject[i2, i1] = tukey_res.reject[k]

# %% ── Build figures (one per panel) ─────────────────────────────────────────

# ── Shared axis limits ────────────────────────────────────────────────────────

_x_all = np.concatenate([gnm_pca[:, 0], lexi_pca[:, 0], mami_pca[:, 0]])
_y_all = np.concatenate([gnm_pca[:, 1], lexi_pca[:, 1], mami_pca[:, 1]])
x_pad, y_pad = 0.06 * np.ptp(_x_all), 0.06 * np.ptp(_y_all)
xlim = (_x_all.min() - x_pad, _x_all.max() + x_pad)
ylim = (_y_all.min() - y_pad, _y_all.max() + y_pad)

# ── Panel A: combined morphospace ─────────────────────────────────────────────

fig_A, ax_main = plt.subplots(figsize=viz.cm_to_inch((12, 10)), layout="constrained")

_pareto_poly = plt.Polygon(
    _hull_verts, closed=True,
    facecolor="#FFD700", edgecolor="#B8860B",
    alpha=0.18, linewidth=1.0, linestyle="--",
    zorder=0, label="Pareto-optimal region",
)
ax_main.add_patch(_pareto_poly)

ax_main.scatter(gnm_pca[:, 0], gnm_pca[:, 1],
                c="#E0E0E0", s=1, alpha=0.25, linewidths=0,
                rasterized=True, zorder=1)

mask_other = orders == "Other"
ax_main.scatter(mami_pca[mask_other, 0], mami_pca[mask_other, 1],
                c=ORDER_COLORS["Other"], s=6, alpha=0.45,
                linewidths=0, marker="^", zorder=2)

legend_patches = []
for order in MAIN_ORDERS:
    mask = orders == order
    col  = ORDER_COLORS[order]
    n    = mask.sum()
    ax_main.scatter(mami_pca[mask, 0], mami_pca[mask, 1],
                    c=col, s=8, alpha=0.6, linewidths=0,
                    marker="^", zorder=3)
    confidence_ellipse(mami_pca[mask, 0], mami_pca[mask, 1], ax_main,
                       n_std=2.0, edgecolor=col, facecolor="none",
                       lw=0.7, linestyle="--", zorder=2)
    legend_patches.append(mpatches.Patch(color=col, label=f"{order} (n={n})"))

sc_A = ax_main.scatter(lexi_pca[:, 0], lexi_pca[:, 1],
                       c=ages, cmap=AGE_CMAP,
                       s=5, alpha=0.55, linewidths=0,
                       vmin=ages.min(), vmax=ages.max(),
                       zorder=4, rasterized=True)

cbar = fig_A.colorbar(sc_A, ax=ax_main, shrink=0.45, pad=0.01, aspect=18)
cbar.set_label("Age (years)", fontsize=6)
cbar.ax.tick_params(labelsize=5)

pareto_patch = mpatches.Patch(
    facecolor="#FFD700", edgecolor="#B8860B",
    alpha=0.7, linestyle="--",
    label=f"Pareto-optimal region\n(HCP-D {pct_lexi_in:.0f}% inside, MaMI {pct_mami_in:.0f}%)",
)
ax_main.legend(handles=legend_patches + [pareto_patch],
               fontsize=4.5, loc="lower right",
               frameon=True, framealpha=0.75, edgecolor="none",
               title="Mammalian order (▲)", title_fontsize=4.5)
ax_main.text(0.97, 0.97, "HCP-D (humans, ●)", transform=ax_main.transAxes,
             ha="right", va="top", fontsize=5.5, color="#8B2500", style="italic")

ax_main.set_xlim(xlim); ax_main.set_ylim(ylim)
ax_main.set_xlabel(f"PC1 ({ev[0]:.1f}% var) — developmental axis  →", fontsize=6.5)
ax_main.set_ylabel(f"PC2 ({ev[1]:.1f}% var) — phylogenetic axis  ↑", fontsize=6.5)
ax_main.tick_params(labelsize=5.5)
ax_main.set_title("A  PC1 tracks development; PC2 tracks phylogeny", fontsize=7, loc="left")

# ── Panel B: PC2 distribution per order (horizontal strip plot) ───────────────

fig_B, ax_strip = plt.subplots(figsize=viz.cm_to_inch((8, 9)), layout="constrained")

orders_by_median = sorted(
    MAIN_ORDERS,
    key=lambda o: np.median(mami_pca[orders == o, 1])
)
rng_strip = np.random.default_rng(42)

for i, order in enumerate(orders_by_median):
    mask     = orders == order
    pc2_vals = mami_pca[mask, 1]
    col      = ORDER_COLORS[order]
    jitter   = rng_strip.uniform(-0.28, 0.28, mask.sum())
    ax_strip.scatter(pc2_vals, np.full(mask.sum(), i) + jitter,
                     c=col, s=7, alpha=0.7, linewidths=0, zorder=3)
    mn, sd = pc2_vals.mean(), pc2_vals.std()
    ax_strip.hlines(i, mn - sd, mn + sd, color=col, lw=1.8, zorder=4)
    ax_strip.scatter([mn], [i], c=col, s=22, zorder=5,
                     linewidths=0.5, edgecolors="black")

ax_strip.set_yticks(range(len(orders_by_median)))
ax_strip.set_yticklabels(orders_by_median, fontsize=5.5)
ax_strip.set_xlabel(f"PC2 score ({ev[1]:.1f}% var)", fontsize=6.5)
ax_strip.tick_params(labelsize=5)
ax_strip.set_title("B  PC2 per order\n(mean ± SD)", fontsize=7, loc="left")
ax_strip.spines["top"].set_visible(False)
ax_strip.spines["right"].set_visible(False)
ax_strip.axvline(0, color="gray", lw=0.5, linestyle=":", zorder=1)

# ── Panel C1: age ~ PC1 regression ───────────────────────────────────────────

fig_C1, ax_reg = plt.subplots(figsize=viz.cm_to_inch((8, 6)), layout="constrained")

ax_reg.scatter(ages, lexi_pca[:, 0],
               c=ages, cmap=AGE_CMAP,
               s=4, alpha=0.35, linewidths=0, rasterized=True)
slope, intercept = np.polyfit(ages, lexi_pca[:, 0], 1)
x_line = np.linspace(ages.min(), ages.max(), 100)
ax_reg.plot(x_line, slope * x_line + intercept, color="black", lw=1.0, zorder=5)

stars = "***" if p_age < 0.001 else "**" if p_age < 0.01 else "*" if p_age < 0.05 else "ns"
ax_reg.text(0.97, 0.05, f"$r$ = {r:.2f}{stars}",
            transform=ax_reg.transAxes, ha="right", va="bottom", fontsize=5.5)

ax_reg.set_xlabel("Age (years)", fontsize=6)
ax_reg.set_ylabel("PC1 score", fontsize=6)
ax_reg.tick_params(labelsize=5)
ax_reg.set_title("C  Age ~ PC1", fontsize=7, loc="left")
ax_reg.spines["top"].set_visible(False)
ax_reg.spines["right"].set_visible(False)

# ── Panel C2: PERMANOVA pseudo-F on PC2 per order ────────────────────────────

fig_C2, ax_perm = plt.subplots(figsize=viz.cm_to_inch((8, 6)), layout="constrained")

orders_sorted = sorted(permanova_pc2, key=lambda o: permanova_pc2[o]["F"])
F_vals  = [permanova_pc2[o]["F"] for o in orders_sorted]
colors  = [ORDER_COLORS[o] for o in orders_sorted]
p_vals  = [permanova_pc2[o]["p"] for o in orders_sorted]

ax_perm.barh(range(len(orders_sorted)), F_vals,
             color=colors, edgecolor="black", linewidth=0.3, height=0.65)

for i, (o, pv) in enumerate(zip(orders_sorted, p_vals)):
    ax_perm.text(F_vals[i] + 0.03 * max(F_vals), i,
                 sig_stars(pv), va="center", ha="left", fontsize=6)

ax_perm.set_yticks(range(len(orders_sorted)))
ax_perm.set_yticklabels(orders_sorted, fontsize=5)
ax_perm.set_xlabel("PERMANOVA pseudo-$F$ (PC2)", fontsize=6)
ax_perm.tick_params(labelsize=5)
ax_perm.set_title("  PC2 order separation", fontsize=7, loc="left")
ax_perm.spines["top"].set_visible(False)
ax_perm.spines["right"].set_visible(False)

# %% ── Pareto summary stats (for caption) ─────────────────────────────────────

print("\n── Pareto overlay summary ────────────────────────────────────────────────")
print(f"  Pareto-optimal cells (union, 3 pairs): {_pareto_union.sum()} / {len(_gnm_cell)} "
      f"({100*_pareto_union.sum()/len(_gnm_cell):.1f}%)")
print(f"  HCP-D inside Pareto hull: {pct_lexi_in:.1f}%")
print(f"  MaMI  inside Pareto hull: {pct_mami_in:.1f}%")
print(f"  Pareto    GNM — mean PC1={_pareto_pc[:,0].mean():+.3f}  "
      f"mean PC2={_pareto_pc[:,1].mean():+.3f}")
print(f"  Non-Pareto GNM — mean PC1={_pareto_pc_nonpareto[:,0].mean():+.3f}  "
      f"mean PC2={_pareto_pc_nonpareto[:,1].mean():+.3f}")
print("\nSuggested caption addition:")
print(f"  'The shaded region (gold dashed outline) marks the Pareto-optimal "
      f"zone of the GNM morphospace — ($\\eta$, $\\gamma$) cells on the Pareto front for "
      f"efficiency↑/memory↑/robustness↑ vs wiring-cost↓ trade-offs. "
      f"{pct_lexi_in:.0f}\\% of HCP-D individuals and {pct_mami_in:.0f}\\% of MaMI "
      f"mammals fall inside this region; Pareto-optimal GNM cells are shifted "
      f"toward lower PC2 (mean PC2 = {_pareto_pc[:,1].mean():+.2f}) relative to "
      f"non-Pareto cells (mean PC2 = {_pareto_pc_nonpareto[:,1].mean():+.2f}), "
      f"aligning the wiring-efficiency optimum with the human developmental "
      f"cluster rather than with phylogenetic diversity.'")

# %% ── Save ────────────────────────────────────────────────────────────────────

_panels = {
    "fig_4a_morphospace":    fig_A,
    "fig_4b_pc2_per_order":  fig_B,
    "fig_4c1_age_pc1":       fig_C1,
    "fig_4c2_permanova":     fig_C2,
}
for stem, fig in _panels.items():
    for out_dir in [OUTPUT_DIR, PLOT_DIR]:
        out_path = out_dir / f"{stem}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {out_path}")

plt.show()



# %% ── Supplementary: age trajectory in PCA space ────────────────────────────

age_vals   = consensus["age"].values
t_dense    = np.linspace(age_vals.min(), age_vals.max(), 300)

smooth_pc1 = make_smoothing_spline(age_vals, consensus["PC1"].values, lam=5)(t_dense)
smooth_pc2 = make_smoothing_spline(age_vals, consensus["PC2"].values, lam=5)(t_dense)

norm_age   = plt.Normalize(age_vals.min(), age_vals.max())

fig_traj, ax_traj = plt.subplots(figsize=viz.cm_to_inch((12, 9)))

points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc_traj  = LineCollection(segments, cmap=AGE_CMAP, norm=norm_age,
                           linewidth=4, alpha=1, zorder=2)
lc_traj.set_array(t_dense)
ax_traj.add_collection(lc_traj)

sc_traj = ax_traj.scatter(consensus["PC1"], consensus["PC2"],
                          c=consensus["age"], cmap=AGE_CMAP, norm=norm_age,
                          s=40, zorder=3, edgecolors="none", # white", 
                          linewidths=0) # .4)

# Label start and end ages
for age_label in [age_vals.min(), age_vals.max()]:
    idx = np.argmin(np.abs(age_vals - age_label))
    print(age_label)
    # Label with age in years; shifted slightly up for visibility
    ax_traj.text(consensus["PC1"].iloc[idx], consensus["PC2"].iloc[idx] + 0.02,
                 f"{age_label:.0f} yr", ha="center", va="bottom",
                 fontsize=7, color="#8B2500", style="italic")


ax_traj.autoscale()
# fig_traj.colorbar(sc_traj, ax=ax_traj, label="Age (years)")
ax_traj.set_title(f"HCP-D developmental trajectory ({age_vals.min():.0f}-{age_vals.max():.0f} years)")
ax_traj.set_xlabel(f"PC1 ({ev[0]:.1f}% var)")
ax_traj.set_ylabel(f"PC2 ({ev[1]:.1f}% var)")
ax_traj.spines["top"].set_visible(False)
ax_traj.spines["right"].set_visible(False)

fig_traj.tight_layout()

traj_path = OUTPUT_DIR / "lexis_data_developing_pca_age.pdf"
fig_traj.savefig(traj_path)
print(traj_path)
plt.show()