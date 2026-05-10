"""
Null model comparison: does the GNM *specifically* recover biological structure
in its PCA morphospace, or would any network morphospace do the same?

Comparison (both n=100, same 80 features — apples-to-apples):
  GNM morphospace  — PCA fitted on 100 GNM networks at best-fit parameters
  ER  morphospace  — PCA fitted on 100 Erdős-Rényi random networks

For each morphospace and each of PC1–10:
  - Pearson |r| between PC score and age (HCP-D, n=629)
  - One-way ANOVA F for taxonomic order → PC score (MaMI, n=224)

If GNM >> ER: the recovery is specific to generative model structure → strong claim.
If GNM ≈ ER: any morphospace would do it → finding is not surprising.
"""

# %% ── Imports ─────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from vizman import viz

# %% ── Paths ───────────────────────────────────────────────────────────────────

BASE     = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
OUT_DIR  = BASE / "output/trade_off_analysis/06_null_model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PKL_PATH   = BASE / "output/00_trade_off_analysis/all_datasets_precise_categories.pkl"
AGES_PATH  = BASE / "data/preprocessed/lexis_data_developing/04_further_info/00_ages.npy"
MAMI_META  = BASE / "data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv"

# %% ── Load data ───────────────────────────────────────────────────────────────

with open(PKL_PATH, "rb") as f:
    datasets = pickle.load(f)

df_gnm  = datasets["hcp_schaefer_100_dataset_gnm"]   # 100 × 80
df_er   = datasets["erdos_renyi_networks"]             # 100 × 86
df_lexi = datasets["lexis_data_developing"]            # 629 × 86
df_mami = datasets["suarez_MaMI_dataset"]              # 224 × 86

ages_all = np.load(AGES_PATH)
ages = ages_all if len(ages_all) == len(df_lexi) else ages_all[df_lexi.index]
assert len(ages) == len(df_lexi)

mami_meta_full = pd.read_csv(MAMI_META).reset_index(drop=True)
mami_meta = mami_meta_full if len(mami_meta_full) == len(df_mami) \
            else mami_meta_full.iloc[df_mami.index].reset_index(drop=True)

MAIN_ORDERS = ["Carnivora", "Primates", "Cetartiodactyla", "Rodentia", "Chiroptera"]
orders_raw = mami_meta["order"].fillna("Other").values
orders = np.where(np.isin(orders_raw, MAIN_ORDERS), orders_raw, "Other")

ORDER_COLORS = {
    "Carnivora": "#2E86AB", "Primates": "#A23B72",
    "Cetartiodactyla": "#F18F01", "Rodentia": "#44BBA4",
    "Chiroptera": "#E94F37",
}

print(f"GNM:  {df_gnm.shape}   ER: {df_er.shape}")
print(f"HCP-D: {df_lexi.shape}   MaMI: {df_mami.shape}")

# %% ── Build shared valid feature set (common to all four datasets) ────────────

common_cols = (
    set(df_gnm.columns) & set(df_er.columns)
    & set(df_lexi.columns) & set(df_mami.columns)
)

# Drop columns with NaN in either reference dataset
def valid_cols_for(df_ref, cols):
    sub = df_ref[list(cols)].replace([np.inf, -np.inf], np.nan)
    return set(sub.columns[sub.notna().all()].tolist())

valid = valid_cols_for(df_gnm, common_cols) & valid_cols_for(df_er, common_cols)
valid_cols = sorted(valid)
print(f"Common valid features: {len(valid_cols)}")

# %% ── Helper: fit morphospace + embed ────────────────────────────────────────

def build_morphospace(df_ref: pd.DataFrame, label: str):
    """Fit StandardScaler + PCA(10) on df_ref. Returns (scaler, pca, ref_pca)."""
    arr = df_ref[valid_cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    scaler = StandardScaler()
    pca    = PCA(n_components=10, random_state=42)
    scaled = scaler.fit_transform(arr)
    pca.fit(scaled)
    ev = pca.explained_variance_ratio_ * 100
    print(f"\n{label} morphospace (n={len(df_ref)}):")
    print(f"  PC1: {ev[0]:.1f}%  PC2: {ev[1]:.1f}%  PC3: {ev[2]:.1f}%  "
          f"  cumul 10: {ev.sum():.1f}%")
    return scaler, pca


def embed(df: pd.DataFrame, scaler, pca) -> np.ndarray:
    """Project df into a morphospace, imputing NaNs with training column means."""
    arr = df[valid_cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    for j, col_mean in enumerate(scaler.mean_):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_mean
    return pca.transform(scaler.transform(arr))


# %% ── Fit both morphospaces ───────────────────────────────────────────────────

scaler_gnm, pca_gnm = build_morphospace(df_gnm, "GNM")
scaler_er,  pca_er  = build_morphospace(df_er,  "ER")

lexi_gnm = embed(df_lexi, scaler_gnm, pca_gnm)
mami_gnm = embed(df_mami, scaler_gnm, pca_gnm)

lexi_er  = embed(df_lexi, scaler_er,  pca_er)
mami_er  = embed(df_mami, scaler_er,  pca_er)

# %% ── Compute r(age, PC_i) and ANOVA F(order → PC_i) for each morphospace ────

def bio_signal_per_pc(lexi_pca, mami_pca, label):
    print(f"\n── {label} morphospace ───────────────────────────────────────────────")
    print(f"  {'PC':>3}  {'|r| age':>8}  {'p_age':>10}  {'ANOVA F orders':>16}  {'p_ord':>10}")

    r_vals, F_vals = [], []
    for i in range(10):
        r, p_r = stats.pearsonr(ages, lexi_pca[:, i])
        groups = [mami_pca[orders == o, i] for o in MAIN_ORDERS]
        F, p_F = stats.f_oneway(*groups)
        r_vals.append(abs(r))
        F_vals.append(F)
        s_r = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else "ns"
        s_F = "***" if p_F < 0.001 else "**" if p_F < 0.01 else "*" if p_F < 0.05 else "ns"
        print(f"  {i+1:>3}  {abs(r):>8.3f}{s_r:3s}  {p_r:>10.2e}  {F:>16.2f}{s_F:3s}  {p_F:>10.2e}")

    return np.array(r_vals), np.array(F_vals)


r_gnm, F_gnm = bio_signal_per_pc(lexi_gnm, mami_gnm, "GNM")
r_er,  F_er  = bio_signal_per_pc(lexi_er,  mami_er,  "ER")

# %% ── Figure: comparison bar chart ───────────────────────────────────────────

pc_labels = [f"PC{i+1}" for i in range(10)]
x = np.arange(10)
w = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=viz.cm_to_inch((16, 7)), layout="constrained")

# ── Panel A: |r| age ──────────────────────────────────────────────────────────

ax1.bar(x - w/2, r_gnm, width=w, color="#232324", alpha=0.85,
        edgecolor="black", linewidth=0.4, label="GNM morphospace")
ax1.bar(x + w/2, r_er,  width=w, color="#B8B8B8", alpha=0.85,
        edgecolor="black", linewidth=0.4, label="ER morphospace")

# Highlight the strongest PC for each
best_gnm_r = np.argmax(r_gnm)
best_er_r  = np.argmax(r_er)
ax1.bar(best_gnm_r - w/2, r_gnm[best_gnm_r], width=w, color="#232324",
        edgecolor="#C74800", linewidth=1.2)
ax1.bar(best_er_r  + w/2, r_er[best_er_r],   width=w, color="#B8B8B8",
        edgecolor="#2E86AB", linewidth=1.2)

ax1.axhline(0, color="black", lw=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(pc_labels, fontsize=6)
ax1.set_ylabel("|Pearson r|  (age ~ PC score, HCP-D)", fontsize=7)
ax1.set_title("A  Developmental signal per PC\nGNM vs. Erdős–Rényi morphospace",
              fontsize=7, loc="left")
ax1.legend(fontsize=6, frameon=False)
ax1.tick_params(labelsize=6)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel B: ANOVA F orders ───────────────────────────────────────────────────

ax2.bar(x - w/2, F_gnm, width=w, color="#232324", alpha=0.85,
        edgecolor="black", linewidth=0.4, label="GNM morphospace")
ax2.bar(x + w/2, F_er,  width=w, color="#B8B8B8", alpha=0.85,
        edgecolor="black", linewidth=0.4, label="ER morphospace")

best_gnm_F = np.argmax(F_gnm)
best_er_F  = np.argmax(F_er)
ax2.bar(best_gnm_F - w/2, F_gnm[best_gnm_F], width=w, color="#232324",
        edgecolor="#C74800", linewidth=1.2)
ax2.bar(best_er_F  + w/2, F_er[best_er_F],   width=w, color="#B8B8B8",
        edgecolor="#2E86AB", linewidth=1.2)

ax2.axhline(1, color="gray", lw=0.6, linestyle=":")   # F=1 null expectation
ax2.set_xticks(x); ax2.set_xticklabels(pc_labels, fontsize=6)
ax2.set_ylabel("One-way ANOVA F  (order → PC score, MaMI)", fontsize=7)
ax2.set_title("B  Phylogenetic signal per PC\nGNM vs. Erdős–Rényi morphospace",
              fontsize=7, loc="left")
ax2.legend(fontsize=6, frameon=False)
ax2.tick_params(labelsize=6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

out = OUT_DIR / "null_model_comparison.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"\nSaved: {out}")
plt.show()

# %% ── Summary ─────────────────────────────────────────────────────────────────

print("\n── Summary ───────────────────────────────────────────────────────────────")
print(f"  Max |r| age:    GNM={r_gnm.max():.3f} (PC{r_gnm.argmax()+1})  "
      f"ER={r_er.max():.3f} (PC{r_er.argmax()+1})")
print(f"  Max ANOVA F:    GNM={F_gnm.max():.1f} (PC{F_gnm.argmax()+1})  "
      f"ER={F_er.max():.1f} (PC{F_er.argmax()+1})")
print(f"  GNM advantage:  r × {r_gnm.max()/max(r_er.max(),0.001):.1f}  "
      f"F × {F_gnm.max()/max(F_er.max(),0.001):.1f}")
