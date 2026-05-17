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
from pathlib import Path
from scipy import stats
from scipy.interpolate import make_smoothing_spline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from vizman import viz

from config import PROPERTY_NAMES
from utils_permanova import run_permanova, sig_stars

# %% ── Paths ───────────────────────────────────────────────────────────────────

BASE       = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
PLOT_DIR   = Path("/Users/adrian/Documents/01_projects/14_4D_lab/latex_thesis/figures/results")
OUTPUT_DIR = BASE / "output/trade_off_analysis/04_age_and_phylogeny"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PKL_PATH   = BASE / "output/00_trade_off_analysis/all_datasets_precise_categories.pkl"
AGES_PATH  = BASE / "data/preprocessed/lexis_data_developing/04_further_info/00_ages.npy"
MAMI_META  = BASE / "data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv"

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

# %% ── Age ~ PC1 regression ───────────────────────────────────────────────────

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


# %% ── Build figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=viz.cm_to_inch((18, 9)), layout="constrained")
gs  = gridspec.GridSpec(
    2, 4, figure=fig,
    width_ratios=[2, 2, 1.3, 1.3],
    height_ratios=[1, 1],
    wspace=0.08, hspace=0.05,
)

ax_age   = fig.add_subplot(gs[:, 0:2])     # Panel A: wide, both rows
ax_mami  = fig.add_subplot(gs[:, 2])       # Panel B: both rows
ax_reg   = fig.add_subplot(gs[0, 3])       # Panel C top: age regression
ax_perm  = fig.add_subplot(gs[1, 3])       # Panel C bottom: PERMANOVA

# ── Common axis limits (same for A and B) ─────────────────────────────────────

_x_all = np.concatenate([gnm_pca[:, 0], lexi_pca[:, 0], mami_pca[:, 0]])
_y_all = np.concatenate([gnm_pca[:, 1], lexi_pca[:, 1], mami_pca[:, 1]])
x_pad, y_pad = 0.06 * np.ptp(_x_all), 0.06 * np.ptp(_y_all)
xlim = (_x_all.min() - x_pad, _x_all.max() + x_pad)
ylim = (_y_all.min() - y_pad, _y_all.max() + y_pad)

# ── Panel A: HCP-D colored by age ─────────────────────────────────────────────

# GNM background
ax_age.scatter(gnm_pca[:, 0], gnm_pca[:, 1],
               c="#E0E0E0", s=1, alpha=0.3, linewidths=0, rasterized=True, zorder=1)

# HCP-D individuals colored by age
sc = ax_age.scatter(lexi_pca[:, 0], lexi_pca[:, 1],
                    c=ages, cmap=AGE_CMAP,
                    s=5, alpha=0.5, linewidths=0,
                    vmin=ages.min(), vmax=ages.max(),
                    zorder=2, rasterized=True)

# Consensus trajectory
ax_age.plot(x_spl, y_spl, color="black", lw=1.4, zorder=4)
# Arrowhead at the end
ax_age.annotate("", xy=(x_spl[-1], y_spl[-1]), xytext=(x_spl[-2], y_spl[-2]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))

# Year labels at 6, 10, 14, 18, 22
for yr in [6, 10, 14, 18, 22]:
    row = consensus[consensus["age"] == yr]
    if len(row):
        xc, yc = float(row["PC1"]), float(row["PC2"])
        ax_age.scatter(xc, yc, s=28, color="black", zorder=5, linewidths=0)
        ax_age.text(xc + 0.15, yc + 0.08, str(yr), fontsize=5, va="bottom", color="black")

# Colorbar for age
cbar = fig.colorbar(sc, ax=ax_age, shrink=0.55, pad=0.01, aspect=20)
cbar.set_label("Age (years)", fontsize=6)
cbar.ax.tick_params(labelsize=5)

ax_age.set_xlim(xlim);  ax_age.set_ylim(ylim)
ax_age.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)", fontsize=6.5)
ax_age.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)", fontsize=6.5)
ax_age.tick_params(labelsize=5.5)
ax_age.set_title("A  Developmental trajectory in the morphospace", fontsize=7, loc="left")

# ── Panel B: MaMI colored by order ────────────────────────────────────────────

# GNM background
ax_mami.scatter(gnm_pca[:, 0], gnm_pca[:, 1],
                c="#E0E0E0", s=1, alpha=0.3, linewidths=0, rasterized=True, zorder=1)

# "Other" orders first (back layer)
mask_other = orders == "Other"
ax_mami.scatter(mami_pca[mask_other, 0], mami_pca[mask_other, 1],
                c=ORDER_COLORS["Other"], s=6, alpha=0.5, linewidths=0, zorder=2)

# Main orders with ellipses
legend_patches = []
for order in MAIN_ORDERS:
    mask = orders == order
    col  = ORDER_COLORS[order]
    ax_mami.scatter(mami_pca[mask, 0], mami_pca[mask, 1],
                    c=col, s=8, alpha=0.75, linewidths=0, zorder=3)
    confidence_ellipse(mami_pca[mask, 0], mami_pca[mask, 1], ax_mami,
                       n_std=2.0, edgecolor=col, facecolor=col, alpha=0.12,
                       lw=0.8, zorder=2)
    n = mask.sum()
    legend_patches.append(mpatches.Patch(color=col,
                                         label=f"{order} (n={n})"))

ax_mami.legend(handles=legend_patches, fontsize=4.8, loc="upper right",
               frameon=False, markerscale=1)

ax_mami.set_xlim(xlim);  ax_mami.set_ylim(ylim)
ax_mami.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=6.5)
ax_mami.set_yticklabels([])
ax_mami.tick_params(labelsize=5.5)
ax_mami.set_title("B  Phylogenetic order separation", fontsize=7, loc="left")

# ── Panel C top: Age ~ PC1 regression ─────────────────────────────────────────

# Scatter with regression line
ax_reg.scatter(ages, lexi_pca[:, 0],
               c=ages, cmap=AGE_CMAP,
               s=4, alpha=0.35, linewidths=0, rasterized=True)
slope, intercept = np.polyfit(ages, lexi_pca[:, 0], 1)
x_line = np.linspace(ages.min(), ages.max(), 100)
ax_reg.plot(x_line, slope * x_line + intercept,
            color="black", lw=1.0, zorder=5)

stars = "***" if p_age < 0.001 else "**" if p_age < 0.01 else "*" if p_age < 0.05 else "ns"
ax_reg.text(0.97, 0.05, f"$r$ = {r:.2f}{stars}",
            transform=ax_reg.transAxes, ha="right", va="bottom", fontsize=5.5)

ax_reg.set_xlabel("Age (years)", fontsize=6)
ax_reg.set_ylabel("PC1 score", fontsize=6)
ax_reg.tick_params(labelsize=5)
ax_reg.set_title("C  Age encodes in PC1", fontsize=7, loc="left")

# ── Panel C bottom: PERMANOVA pseudo-F per order ──────────────────────────────

orders_sorted = sorted(permanova_results, key=lambda o: permanova_results[o]["F"], reverse=True)
F_vals  = [permanova_results[o]["F"] for o in orders_sorted]
colors  = [ORDER_COLORS[o] for o in orders_sorted]
p_vals  = [permanova_results[o]["p"] for o in orders_sorted]

bars = ax_perm.barh(range(len(orders_sorted)), F_vals,
                    color=colors, edgecolor="black", linewidth=0.3, height=0.65)

for i, (o, p) in enumerate(zip(orders_sorted, p_vals)):
    ax_perm.text(F_vals[i] + 0.05 * max(F_vals), i,
                 sig_stars(p), va="center", ha="left", fontsize=6)

ax_perm.set_yticks(range(len(orders_sorted)))
ax_perm.set_yticklabels(orders_sorted, fontsize=5)
ax_perm.set_xlabel("PERMANOVA pseudo-$F$", fontsize=6)
ax_perm.tick_params(labelsize=5)
ax_perm.set_title("  Order separation (PC1–PC2)", fontsize=7, loc="left")
ax_perm.spines["top"].set_visible(False)
ax_perm.spines["right"].set_visible(False)

# %% ── Save ────────────────────────────────────────────────────────────────────

for out_dir in [OUTPUT_DIR, PLOT_DIR]:
    out_path = out_dir / "fig_4_age_and_phylogeny.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_path}")

plt.show()
