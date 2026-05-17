"""
Check: does IPC r=0.81 survive in the GNM-only morphospace?

Fits PCA exclusively on GNM networks using the curated 80-feature set
(same as Fig 4), then correlates PC1/PC2 with IPC measures.
Also re-checks age correlation and PERMANOVA in the same space.

Answers the critical question from analysis notes:
  - If IPC r≥0.75 in GNM-only PCA  → one consistent morphospace, submittable
  - If IPC r≈0.30 (like standard MC) → r=0.81 was an artefact of the mixed PCA
"""

# %% ── Imports ────────────────────────────────────────────────────────────────

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %% ── Paths ──────────────────────────────────────────────────────────────────

BASE     = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
PKL_PATH = BASE / "output/00_trade_off_analysis/all_datasets_precise_categories.pkl"

GNM_METRICS = (BASE / "output/gnm/hcp_schaefer_100_dataset"
               / "11_mst_2500_animal_0"
               / "all_metrics_for_11_mst_2500_animal_0_updated.csv")

AGES_PATH = BASE / "data/preprocessed/lexis_data_developing/04_further_info/00_ages.npy"
MAMI_META = (BASE / "data/preprocessed/suarez_MaMI_dataset"
             / "04_further_info"
             / "names_of_animals_with_preprocessed_connectomes_50_processed.csv")

MAIN_ORDERS = ["Carnivora", "Primates", "Cetartiodactyla", "Rodentia", "Chiroptera"]

IPC_COLS = ["ipc_ipc_deg1_mean", "ipc_ipc_deg2_mean",
            "ipc_ipc_linear_mean", "ipc_ipc_total_mean"]
MC_COLS  = ["mc_input_scaling_0_1_mc_mean",
            "mc_nonlinear_input_scaling_0_1_mc_mean"]

# %% ── Load curated feature set ───────────────────────────────────────────────

with open(PKL_PATH, "rb") as f:
    datasets = pickle.load(f)

df_gnm_feats = datasets["hcp_schaefer_100_dataset_gnm"]   # already filtered to curated cols
df_lexi      = datasets["lexis_data_developing"]
df_mami      = datasets["suarez_MaMI_dataset"]

curated_cols = list(df_gnm_feats.columns)
print(f"Curated features: {len(curated_cols)}")

# %% ── Fit GNM-only PCA ───────────────────────────────────────────────────────

# Restrict to features present in all three datasets, then drop any with NaN in GNM
common = (set(df_gnm_feats.columns)
          & set(df_lexi.columns)
          & set(df_mami.columns))
df_gnm_clean = df_gnm_feats[sorted(common)].replace([np.inf, -np.inf], np.nan)
valid_cols = df_gnm_clean.columns[df_gnm_clean.notna().all()].tolist()
df_gnm_clean = df_gnm_clean[valid_cols]

arr_gnm = df_gnm_clean.values.astype(float)
n_inf = np.isinf(arr_gnm).sum()
n_nan = np.isnan(arr_gnm).sum()
print(f"Features after dropping NaN cols: {len(curated_cols)} → {len(valid_cols)}")
print(f"GNM feature matrix: {arr_gnm.shape}  |  inf={n_inf}  nan={n_nan}")

sc  = StandardScaler()
pca = PCA(n_components=10, random_state=42)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    gnm_scores = pca.fit_transform(sc.fit_transform(arr_gnm))
    if w:
        for warning in w:
            print(f"  WARNING during GNM PCA fit: {warning.message}")

print(f"Explained variance PC1–5: {pca.explained_variance_ratio_[:5].round(3)}")
print(f"PC1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%")

# %% ── Project empirical datasets ─────────────────────────────────────────────

def safe_project(df, cols, sc, pca):
    arr = df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    # Mean-impute per column using GNM training mean
    for j, m in enumerate(sc.mean_):
        arr[np.isnan(arr[:, j]), j] = m
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scores = pca.transform(sc.transform(arr))
        if w:
            for warning in w:
                print(f"  WARNING during projection: {warning.message}")
    n_inf = np.isinf(scores).sum()
    if n_inf > 0:
        print(f"  !! {n_inf} inf values in projected scores — these rows are unreliable")
    return scores

lexi_scores = safe_project(df_lexi, valid_cols, sc, pca)
mami_scores = safe_project(df_mami, valid_cols, sc, pca)

# %% ── Load IPC + MC for GNM networks ────────────────────────────────────────

df_gnm_full = pd.read_csv(GNM_METRICS)
print(f"\nGNM full metrics: {df_gnm_full.shape}")

# Align index with df_gnm_clean (which may have been filtered for connectivity)
shared_idx = df_gnm_clean.index
df_gnm_func = df_gnm_full.iloc[shared_idx] if shared_idx.max() < len(df_gnm_full) else df_gnm_full

# Check IPC availability
missing_ipc = [c for c in IPC_COLS if c not in df_gnm_func.columns]
missing_mc  = [c for c in MC_COLS  if c not in df_gnm_func.columns]
if missing_ipc:
    print(f"  Missing IPC cols: {missing_ipc}")
if missing_mc:
    print(f"  Missing MC cols:  {missing_mc}")

ipc_cols_avail = [c for c in IPC_COLS if c in df_gnm_func.columns]
mc_cols_avail  = [c for c in MC_COLS  if c in df_gnm_func.columns]

# %% ── Correlate GNM PC scores with IPC / MC ─────────────────────────────────

print("\n" + "═"*60)
print("IPC / MC  ×  GNM morphospace (GNM-only PCA, curated features)")
print("═"*60)
print(f"{'Feature':<45} {'PC1 r':>7}  {'PC1 p':>8}  {'PC2 r':>7}  {'PC2 p':>8}")
print("-"*60)

results = {}
for col in ipc_cols_avail + mc_cols_avail:
    vals = df_gnm_func[col].values
    mask = np.isfinite(vals) & np.isfinite(gnm_scores[:len(vals), 0])

    r1, p1 = stats.pearsonr(gnm_scores[:len(vals), 0][mask], vals[mask])
    r2, p2 = stats.pearsonr(gnm_scores[:len(vals), 1][mask], vals[mask])

    sig1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else "ns"
    sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"

    short = col.replace("ipc_ipc_", "ipc_").replace("mc_input_scaling_0_1_", "mc_lin_").replace("mc_nonlinear_input_scaling_0_1_", "mc_nonlin_")
    print(f"{short:<45} {r1:>+7.3f}  {sig1:>8}  {r2:>+7.3f}  {sig2:>8}")
    results[col] = dict(r_pc1=r1, p_pc1=p1, r_pc2=r2, p_pc2=p2)

# %% ── Age correlation (HCP-D) ────────────────────────────────────────────────

ages_all = np.load(AGES_PATH)
ages = ages_all if len(ages_all) == len(df_lexi) else ages_all[df_lexi.index]

print("\n" + "═"*60)
print("Age correlation  (HCP-D developmental, GNM-only morphospace)")
print("═"*60)
print(f"{'PC':<6} {'r':>7}  {'p':>8}")
for i in range(5):
    mask = np.isfinite(lexi_scores[:, i]) & np.isfinite(ages)
    r, p = stats.pearsonr(lexi_scores[mask, i], ages[mask])
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"PC{i+1:<4} {r:>+7.3f}  {sig:>8}")

# %% ── PERMANOVA-style ANOVA on MaMI taxonomic orders ─────────────────────────

mami_meta_full = pd.read_csv(MAMI_META).reset_index(drop=True)
mami_meta = (mami_meta_full if len(mami_meta_full) == len(df_mami)
             else mami_meta_full.iloc[df_mami.index].reset_index(drop=True))
orders_raw = mami_meta["order"].fillna("Other").values
orders = np.where(np.isin(orders_raw, MAIN_ORDERS), orders_raw, "Other")

print("\n" + "═"*60)
print("ANOVA F (taxonomic orders, MaMI, GNM-only morphospace)")
print("═"*60)
print(f"{'PC':<6} {'F':>8}  {'p':>8}")
for i in range(5):
    groups = [mami_scores[orders == o, i] for o in MAIN_ORDERS]
    groups = [g for g in groups if len(g) > 1]
    mask   = np.isfinite(mami_scores[:, i])
    groups_clean = [mami_scores[(orders == o) & mask, i] for o in MAIN_ORDERS]
    groups_clean = [g for g in groups_clean if len(g) > 1]
    F, p = f_oneway(*groups_clean)
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"PC{i+1:<4} {F:>8.2f}  {sig:>8}")

# %% ── Summary verdict ────────────────────────────────────────────────────────

r_ipc1_pc1 = results.get("ipc_ipc_deg1_mean", {}).get("r_pc1", float("nan"))
r_ipc2_pc1 = results.get("ipc_ipc_deg2_mean", {}).get("r_pc1", float("nan"))

print("\n" + "═"*60)
print("VERDICT")
print("═"*60)
print(f"  ipc_deg1 × PC1:  r = {r_ipc1_pc1:+.3f}")
print(f"  ipc_deg2 × PC1:  r = {r_ipc2_pc1:+.3f}")
if abs(r_ipc2_pc1) >= 0.75:
    print("  → r≥0.75 HOLDS in GNM-only morphospace. One consistent space. Submittable.")
elif abs(r_ipc2_pc1) >= 0.50:
    print("  → r is moderate (0.50–0.75). Meaningful but weaker than the mixed-PCA result.")
else:
    print("  → r<0.50. The r=0.81 was an artefact of the mixed-PCA training set.")
