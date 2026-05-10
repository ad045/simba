"""
Diagnostic: does the best-match GNM parameter (η, γ) for each empirical connectome
correlate with biological metadata?

  HCP-D: r(age, η_best)  vs  r(age, γ_best)  → is development a movement in η, γ, or both?
  MaMI:  ANOVA / strip plot: does order predict η_best or γ_best?

  Key output: 2D scatter of η_best vs γ_best, colored by age (HCP-D) or order (MaMI).
  If age creates a directional gradient along η or γ (not diagonally), the GNM
  decomposition is mechanistically interpretable.
"""

# %% ── Imports ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats
from vizman import viz
from utils_permanova import run_permanova, sig_stars

# %% ── Paths ───────────────────────────────────────────────────────────────────

GNM_OUT  = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm")
DATA_DIR = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed")
OUT_DIR  = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis/05_eta_gamma_biology")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEXI_CSV  = GNM_OUT / "lexis_data_developing/05_mst_animal_0_compared_with_lexis_data_developing/summary_indiv_delta_con_for_exp_05_mst_animal_0_compared_with_lexis_data_developing.csv"
MAMI_CSV  = GNM_OUT / "suarez_MaMI_dataset/05_mst_animal_0_compared_with_mami/summary_indiv_delta_con_for_exp_05_mst_animal_0_compared_with_mami.csv"
AGES_PATH = DATA_DIR / "lexis_data_developing/04_further_info/00_ages.npy"
MAMI_META = DATA_DIR / "suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv"

# %% ── Constants ───────────────────────────────────────────────────────────────

AGE_CMAP = LinearSegmentedColormap.from_list("age", ["#E6B212", "#C74800", "#BF003F"])

ORDER_COLORS = {
    "Carnivora":       "#2E86AB",
    "Primates":        "#A23B72",
    "Cetartiodactyla": "#F18F01",
    "Rodentia":        "#44BBA4",
    "Chiroptera":      "#E94F37",
    "Other":           "#C8C8C8",
}
MAIN_ORDERS = [o for o in ORDER_COLORS if o != "Other"]

# %% ── Helper: extract per-subject best-match (η, γ) ─────────────────────────

def best_match_eta_gamma(csv_path: Path, subj_col_pattern: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load DeltaCon CSV, group by (eta, gamma), find argmin per subject.
    Returns arrays (eta_best, gamma_best) of length n_subjects, plus subject col names.
    """
    df = pd.read_csv(csv_path)

    # De-duplicate: keep only the first occurrence of each subject
    # (HCP-D CSV has _x and _y duplicates from an upstream merge)
    subj_cols_all = [c for c in df.columns if subj_col_pattern in c]
    seen, subj_cols = set(), []
    for c in subj_cols_all:
        base = c.replace("_x", "").replace("_y", "").replace("_z", "")
        if base not in seen:
            seen.add(base)
            subj_cols.append(c)

    df_mean = df.groupby(["eta", "gamma"])[subj_cols].mean().reset_index()

    eta_best, gamma_best = [], []
    for col in subj_cols:
        idx = df_mean[col].idxmin()
        eta_best.append(df_mean.loc[idx, "eta"])
        gamma_best.append(df_mean.loc[idx, "gamma"])

    return np.array(eta_best), np.array(gamma_best), subj_cols


# %% ── Load HCP-D best-match parameters ──────────────────────────────────────

print("Loading HCP-D DeltaCon…")
eta_lexi, gamma_lexi, subj_cols_lexi = best_match_eta_gamma(LEXI_CSV, "DeltaCon_subject_")
n_lexi = len(eta_lexi)
print(f"  {n_lexi} subjects")

ages_all = np.load(AGES_PATH)
assert len(ages_all) == n_lexi, \
    f"Age length {len(ages_all)} ≠ n_subjects {n_lexi}. Check filtering alignment."
ages = ages_all

# %% ── Load MaMI best-match parameters ───────────────────────────────────────

print("Loading MaMI DeltaCon…")
eta_mami, gamma_mami, subj_cols_mami = best_match_eta_gamma(MAMI_CSV, "DeltaCon_subject_")
n_mami = len(eta_mami)
print(f"  {n_mami} subjects")

mami_meta = pd.read_csv(MAMI_META).reset_index(drop=True)
if len(mami_meta) > n_mami:
    # Trim to n_mami (first n_mami surviving networks)
    mami_meta = mami_meta.iloc[:n_mami].reset_index(drop=True)
assert len(mami_meta) == n_mami

orders_raw = mami_meta["order"].fillna("Other").values
orders = np.where(np.isin(orders_raw, MAIN_ORDERS), orders_raw, "Other")

# %% ── Print correlation table (HCP-D) ───────────────────────────────────────

print("\n── HCP-D: age ~ η_best and age ~ γ_best ──────────────────────────────────")
for label, arr in [("η_best", eta_lexi), ("γ_best", gamma_lexi)]:
    r, p = stats.pearsonr(ages, arr)
    rs, ps = stats.spearmanr(ages, arr)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  age ~ {label}:  Pearson r={r:+.3f} {stars}  |  Spearman ρ={rs:+.3f}")

# Multiple regression: age ~ η + γ
from sklearn.linear_model import LinearRegression
X_lexi = np.column_stack([eta_lexi, gamma_lexi])
lr = LinearRegression().fit(X_lexi, ages)
print(f"\n  Multiple R² (age ~ η + γ): {lr.score(X_lexi, ages):.3f}")
print(f"  Coefficients: η={lr.coef_[0]:+.3f}  γ={lr.coef_[1]:+.3f}")

# %% ── ANOVA: order → η_best and order → γ_best (MaMI) ──────────────────────

print("\n── MaMI: order → η_best and order → γ_best (one-way ANOVA) ──────────────")
for label, arr in [("η_best", eta_mami), ("γ_best", gamma_mami)]:
    groups = [arr[orders == o] for o in MAIN_ORDERS]
    F, p = stats.f_oneway(*groups)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  order → {label}:  F={F:.2f}  p={p:.3e}  {stars}")

# Per-order medians
print("\n  Per-order medians:")
print(f"  {'Order':20s}  {'η median':>10}  {'γ median':>10}  n")
for o in MAIN_ORDERS:
    mask = orders == o
    print(f"  {o:20s}  {np.median(eta_mami[mask]):>+10.3f}  {np.median(gamma_mami[mask]):>+10.3f}  {mask.sum()}")

# PERMANOVA for good measure
print("\n  PERMANOVA (η+γ space, 999 perm, one-vs-rest):")
X_mami_2d = np.column_stack([eta_mami, gamma_mami])
for o in MAIN_ORDERS:
    labels_bin = np.where(orders == o, o, "other")
    F_p, p_p = run_permanova(X_mami_2d, labels_bin, n=199)
    print(f"    {o:20s}  F={F_p:.2f}  p={p_p:.3f}  {sig_stars(p_p)}")

# %% ── Figure 1: η vs γ scatter colored by age (HCP-D) ───────────────────────

fig, axes = plt.subplots(1, 2, figsize=viz.cm_to_inch((16, 7)), layout="constrained")

# Left: η_best vs γ_best colored by age
sc = axes[0].scatter(eta_lexi, gamma_lexi, c=ages, cmap=AGE_CMAP,
                     s=8, alpha=0.55, linewidths=0, rasterized=True)
cbar = fig.colorbar(sc, ax=axes[0], shrink=0.8, aspect=20)
cbar.set_label("Age (years)", fontsize=7)
cbar.ax.tick_params(labelsize=6)

# Regression lines
slope_eta, intercept_eta = np.polyfit(ages, eta_lexi, 1)
slope_gam, intercept_gam = np.polyfit(ages, gamma_lexi, 1)
# Arrow from mean of youngest 20% to mean of oldest 20%
age_lo, age_hi = np.percentile(ages, 20), np.percentile(ages, 80)
lo_mask, hi_mask = ages < age_lo, ages > age_hi
mean_lo = (eta_lexi[lo_mask].mean(), gamma_lexi[lo_mask].mean())
mean_hi = (eta_lexi[hi_mask].mean(), gamma_lexi[hi_mask].mean())
axes[0].annotate("", xy=mean_hi, xytext=mean_lo,
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
axes[0].scatter(*mean_lo, s=60, color="#E6B212", zorder=5, edgecolors="black", lw=0.5)
axes[0].scatter(*mean_hi, s=60, color="#BF003F", zorder=5, edgecolors="black", lw=0.5)

r_eta, p_eta = stats.pearsonr(ages, eta_lexi)
r_gam, p_gam = stats.pearsonr(ages, gamma_lexi)
axes[0].text(0.03, 0.97,
             f"age ~ η: r={r_eta:+.2f}{'***' if p_eta<0.001 else '**' if p_eta<0.01 else '*' if p_eta<0.05 else ''}\n"
             f"age ~ γ: r={r_gam:+.2f}{'***' if p_gam<0.001 else '**' if p_gam<0.01 else '*' if p_gam<0.05 else ''}",
             transform=axes[0].transAxes, ha="left", va="top", fontsize=6,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"))

axes[0].set_xlabel("η (cost penalty)", fontsize=7)
axes[0].set_ylabel("γ (homophily)", fontsize=7)
axes[0].tick_params(labelsize=6)
axes[0].set_title("A  HCP-D: best-match GNM parameters by age\n(arrow: youngest 20% → oldest 20%)", fontsize=7, loc="left")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right: η_best vs γ_best colored by order (MaMI)
mask_other = orders == "Other"
axes[1].scatter(eta_mami[mask_other], gamma_mami[mask_other],
                c=ORDER_COLORS["Other"], s=8, alpha=0.4, linewidths=0, zorder=1)

legend_patches = []
for order in MAIN_ORDERS:
    mask = orders == order
    col = ORDER_COLORS[order]
    axes[1].scatter(eta_mami[mask], gamma_mami[mask],
                    c=col, s=10, alpha=0.7, linewidths=0, zorder=2)
    # Order centroid
    axes[1].scatter(np.median(eta_mami[mask]), np.median(gamma_mami[mask]),
                    c=col, s=60, zorder=4, edgecolors="black", lw=0.7,
                    marker="D")
    legend_patches.append(mpatches.Patch(color=col, label=f"{order} (n={mask.sum()})"))

axes[1].legend(handles=legend_patches, fontsize=5, loc="upper left",
               frameon=True, framealpha=0.8, edgecolor="none")
axes[1].set_xlabel("η (cost penalty)", fontsize=7)
axes[1].set_ylabel("γ (homophily)", fontsize=7)
axes[1].tick_params(labelsize=6)
axes[1].set_title("B  MaMI: best-match GNM parameters by order\n(diamond = order median)", fontsize=7, loc="left")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

out = OUT_DIR / "eta_gamma_vs_biology_scatter.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"\nSaved: {out}")
plt.show()

# %% ── Figure 2: marginal distributions — η and γ separately ─────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=viz.cm_to_inch((14, 10)), layout="constrained")

rng = np.random.default_rng(42)

# Top row: HCP-D age vs η and γ
for col_idx, (label, arr, r_val, p_val) in enumerate([
    ("η_best", eta_lexi, r_eta, p_eta),
    ("γ_best", gamma_lexi, r_gam, p_gam),
]):
    ax = axes2[0, col_idx]
    ax.scatter(ages, arr, c=ages, cmap=AGE_CMAP, s=4, alpha=0.35,
               linewidths=0, rasterized=True)
    m, b = np.polyfit(ages, arr, 1)
    x_line = np.linspace(ages.min(), ages.max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", lw=1.0)
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.text(0.97, 0.05, f"r = {r_val:+.2f}{stars}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6)
    ax.set_xlabel("Age (years)", fontsize=6.5)
    ax.set_ylabel(label, fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.set_title(f"HCP-D: age ~ {label}", fontsize=7, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Bottom row: MaMI order strip plots for η and γ
orders_by_eta_median = sorted(MAIN_ORDERS, key=lambda o: np.median(eta_mami[orders == o]))
orders_by_gam_median = sorted(MAIN_ORDERS, key=lambda o: np.median(gamma_mami[orders == o]))

for col_idx, (label, arr, order_sort) in enumerate([
    ("η_best", eta_mami, orders_by_eta_median),
    ("γ_best", gamma_mami, orders_by_gam_median),
]):
    ax = axes2[1, col_idx]
    F_val, p_val = stats.f_oneway(*[arr[orders == o] for o in MAIN_ORDERS])
    for i, order in enumerate(order_sort):
        mask = orders == order
        col = ORDER_COLORS[order]
        jitter = rng.uniform(-0.25, 0.25, mask.sum())
        ax.scatter(arr[mask], np.full(mask.sum(), i) + jitter,
                   c=col, s=7, alpha=0.65, linewidths=0, zorder=3)
        mn, sd = arr[mask].mean(), arr[mask].std()
        ax.hlines(i, mn - sd, mn + sd, color=col, lw=1.8, zorder=4)
        ax.scatter([mn], [i], c=col, s=22, zorder=5, linewidths=0.5, edgecolors="black")
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.set_yticks(range(len(order_sort)))
    ax.set_yticklabels(order_sort, fontsize=5.5)
    ax.set_xlabel(label, fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.set_title(f"MaMI: order ~ {label}  (ANOVA F={F_val:.1f}{stars})", fontsize=7, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="gray", lw=0.5, linestyle=":", zorder=1)

out2 = OUT_DIR / "eta_gamma_marginals.pdf"
fig2.savefig(out2, bbox_inches="tight", dpi=200)
print(f"Saved: {out2}")
plt.show()
