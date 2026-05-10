"""
figures_diagnostic_spectral_radius_confound.py

Evaluates whether the PC1 ~ IPC correlation in the GNM morphospace is a
tautology driven by spectral_radius (a primary determinant of reservoir-
computing dynamics by construction).

Data loading and feature selection mirror the main PCA script exactly:
  USE_REMAINING_CATEGORIES_ONLY = True
  INCLUDE_MC_IN_PCA             = False
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ── local imports (same directory as config.py / utils.py) ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from vizman import viz
from config import COLOR_SCHEME, remaining_categories
from utils import get_combined_colors

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/"
    "14_4D_lab_code/output/00_trade_off_analysis"
)
DATA_PATH = OUTPUT_FOLDER / "all_datasets_precise_categories.pkl"

ORDERED_DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
]
GNM_DATASET = "hcp_schaefer_100_dataset_gnm"
IPC_COL     = "ipc_ipc_deg2_mean"


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_datasets():
    """Mirrors load_and_merge_datasets from the main PCA script exactly."""
    with open(DATA_PATH, "rb") as f:
        datasets_dict = pickle.load(f)

    huge_df = pd.DataFrame()
    for dataset_name in ORDERED_DATASETS:
        if dataset_name not in datasets_dict:
            continue
        df = datasets_dict[dataset_name].copy()
        df["dataset"] = dataset_name
        if dataset_name in ["kaysons_generated_networks_topology", "hcp_schaefer_100_dataset"]:
            continue
        if dataset_name == GNM_DATASET and "eta" in df.columns and "gamma" in df.columns:
            df["color_dataset"] = get_combined_colors(df)
        else:
            df["color_dataset"] = COLOR_SCHEME.get(dataset_name, "#000000")
        huge_df = pd.concat([huge_df, df], ignore_index=True)
    return huge_df


def get_structural_features():
    """
    Mirrors USE_REMAINING_CATEGORIES_ONLY=True + INCLUDE_MC_IN_PCA=False.
    Returns the list of structural (non-MC) columns from remaining_categories.
    """
    mc_keywords = ["computational_capacity", "mc_", "ipc_"]
    all_cols = [col for cat_cols in remaining_categories.values() for col in cat_cols]
    return [c for c in all_cols if not any(kw in c for kw in mc_keywords)]


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of y after linearly regressing out x."""
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    return y - reg.predict(x.reshape(-1, 1))


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² for OLS; X may be 1-D or 2-D."""
    X2 = X.reshape(-1, 1) if X.ndim == 1 else X
    return LinearRegression().fit(X2, y).score(X2, y)


# ════════════════════════════════════════════════════════════════════════════
# Load data & fit PCA
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SPECTRAL RADIUS CONFOUND DIAGNOSTIC")
print("=" * 70)

huge_df = load_datasets()
structural_cols = [c for c in get_structural_features() if c in huge_df.columns]
print(f"\nStructural features available: {len(structural_cols)}")

# Preprocess — identical to run_pca_and_correlate in main script
df_pca = huge_df[structural_cols].copy()
df_pca.replace([np.inf, -np.inf], np.nan, inplace=True)
df_pca.dropna(axis=1, inplace=True)
used_features = df_pca.columns.tolist()
print(f"Features retained after NaN-drop: {len(used_features)}")

if "spectral_radius" not in used_features:
    raise RuntimeError("spectral_radius dropped during NaN cleaning — cannot proceed.")

scaler = StandardScaler()
pca    = PCA(n_components=10)
pca_result = pca.fit_transform(scaler.fit_transform(df_pca))

print("\nPCA explained variance (all datasets):")
for i, ev in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {ev*100:.1f}%")

# Build analysis vectors — GNM rows with valid IPC only
gnm_mask  = (huge_df["dataset"] == GNM_DATASET).values
ipc_all   = huge_df[IPC_COL].values if IPC_COL in huge_df.columns else np.full(len(huge_df), np.nan)
valid     = gnm_mask & ~np.isnan(ipc_all)
n_valid   = valid.sum()
print(f"\nGNM rows with valid {IPC_COL}: {n_valid}")

ipc        = ipc_all[valid]
pc1        = pca_result[valid, 0]
sr_col_idx = used_features.index("spectral_radius")
sr         = huge_df["spectral_radius"].values[valid]
colors_gnm = huge_df.loc[valid, "color_dataset"].tolist()


# ════════════════════════════════════════════════════════════════════════════
# 1.  Head-to-head: every structural feature vs IPC
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("1. EACH STRUCTURAL FEATURE vs IPC  (Pearson r, GNM rows only)")
print("─" * 70)

rows = []
for feat in used_features:
    vals = huge_df[feat].values[valid]
    if np.isnan(vals).any() or np.std(vals) < 1e-12:
        continue
    r, p = pearsonr(vals, ipc)
    rows.append({"feature": feat, "r": r, "p": p})

feat_df = (
    pd.DataFrame(rows)
    .sort_values("r", key=abs, ascending=False)
    .reset_index(drop=True)
)
feat_df.index += 1                       # 1-based rank
print(feat_df.to_string())

sr_rank = feat_df[feat_df["feature"] == "spectral_radius"].index[0]
print(f"\nspectral_radius rank by |r|: #{sr_rank} of {len(feat_df)}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  PC1 loading of spectral_radius
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("2. PC1 LOADINGS")
print("─" * 70)

loadings_pc1 = pca.components_[0]          # shape (n_features,)
sr_loading   = loadings_pc1[sr_col_idx]
print(f"spectral_radius loading on PC1: {sr_loading:+.4f}")
if abs(sr_loading) > 0.3:
    print("  → spectral_radius is a dominant PC1 contributor (|loading| > 0.3)")

top5 = np.argsort(np.abs(loadings_pc1))[::-1][:5]
print("\nTop 5 features by |PC1 loading|:")
for rank, idx in enumerate(top5, 1):
    flag = "  ← spectral_radius" if used_features[idx] == "spectral_radius" else ""
    print(f"  {rank}. {used_features[idx]:55s}  {loadings_pc1[idx]:+.4f}{flag}")


# ════════════════════════════════════════════════════════════════════════════
# 3.  Direct comparison: r(SR, IPC) vs r(PC1, IPC)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("3. DIRECT COMPARISON")
print("─" * 70)

r_sr_ipc,  p_sr_ipc  = pearsonr(sr,  ipc)
r_pc1_ipc, p_pc1_ipc = pearsonr(pc1, ipc)

print(f"r(spectral_radius, IPC) = {r_sr_ipc:+.4f}   p = {p_sr_ipc:.2e}")
print(f"r(PC1,             IPC) = {r_pc1_ipc:+.4f}   p = {p_pc1_ipc:.2e}")
print(f"Δr (PC1 − SR)           = {r_pc1_ipc - r_sr_ipc:+.4f}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  Partial correlations
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("4. PARTIAL CORRELATIONS")
print("─" * 70)

# r(PC1, IPC | spectral_radius)
resid_pc1_on_sr  = residualize(pc1, sr)
resid_ipc_on_sr  = residualize(ipc, sr)
r_pc1_ipc_gsr, p_pc1_ipc_gsr = pearsonr(resid_pc1_on_sr, resid_ipc_on_sr)

# r(spectral_radius, IPC | PC1)
resid_sr_on_pc1  = residualize(sr,  pc1)
resid_ipc_on_pc1 = residualize(ipc, pc1)
r_sr_ipc_gpc1, p_sr_ipc_gpc1 = pearsonr(resid_sr_on_pc1, resid_ipc_on_pc1)

print(f"r(PC1, IPC | spectral_radius) = {r_pc1_ipc_gsr:+.4f}   p = {p_pc1_ipc_gsr:.2e}")
print(f"r(SR,  IPC | PC1)             = {r_sr_ipc_gpc1:+.4f}   p = {p_sr_ipc_gpc1:.2e}")

retention_pct = abs(r_pc1_ipc_gsr) / abs(r_pc1_ipc) * 100 if r_pc1_ipc != 0 else 0
print(f"\nPC1~IPC partial r retains {retention_pct:.1f}% of the raw r(PC1, IPC).")


# ════════════════════════════════════════════════════════════════════════════
# 5.  Variance partitioning
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("5. VARIANCE PARTITIONING  (R² in IPC)")
print("─" * 70)

r2_A = ols_r2(ipc, sr)                               # Model A: SR only
r2_B = ols_r2(ipc, pc1)                              # Model B: PC1 only
r2_C = ols_r2(ipc, np.column_stack([sr, pc1]))       # Model C: SR + PC1

unique_sr  = r2_C - r2_B
unique_pc1 = r2_C - r2_A
shared     = r2_A + r2_B - r2_C

print(f"  R²_A  (SR only)      : {r2_A*100:5.1f}%")
print(f"  R²_B  (PC1 only)     : {r2_B*100:5.1f}%")
print(f"  R²_C  (SR + PC1)     : {r2_C*100:5.1f}%")
print(f"  ─────────────────────────────")
print(f"  Unique to SR         : {unique_sr*100:5.1f}%")
print(f"  Unique to PC1        : {unique_pc1*100:5.1f}%")
print(f"  Shared SR ∩ PC1      : {shared*100:5.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# 6.  Summary figure
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=viz.cm_to_inch((36, 28)), dpi=150)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38,
                        left=0.14, right=0.97, top=0.95, bottom=0.07)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

# ── A: r(feature, IPC) ranked bar ───────────────────────────────────────
# Sort by r value for a waterfall view; spectral_radius highlighted in red
feat_sorted_by_r = feat_df.sort_values("r", ascending=True)   # bottom → top in barh
r_vals   = feat_sorted_by_r["r"].values
feat_nms = feat_sorted_by_r["feature"].values
bar_cols = ["#E84653" if f == "spectral_radius" else "#394D73" for f in feat_nms]

y_pos = np.arange(len(feat_nms))
ax_A.barh(y_pos, r_vals, color=bar_cols, height=0.75)
ax_A.axvline(0, color="black", lw=0.6)
ax_A.set_yticks(y_pos)
ax_A.set_yticklabels(feat_nms, fontsize=5)
ax_A.set_xlabel("Pearson r with IPC", fontsize=8)
ax_A.set_title("A  All structural features vs IPC\n(spectral_radius = red)", fontsize=8, loc="left")
ax_A.spines[["top", "right"]].set_visible(False)

# ── B: spectral_radius vs IPC ───────────────────────────────────────────
ax_B.scatter(sr, ipc, c=colors_gnm, s=8, alpha=0.55, edgecolors="none", rasterized=True)
ax_B.set_xlabel("spectral_radius", fontsize=8)
ax_B.set_ylabel(IPC_COL, fontsize=8)
ax_B.set_title(f"B  spectral_radius vs IPC   r = {r_sr_ipc:+.3f}", fontsize=8, loc="left")
ax_B.spines[["top", "right"]].set_visible(False)

# ── C: PC1 vs IPC ──────────────────────────────────────────────────────
ax_C.scatter(pc1, ipc, c=colors_gnm, s=8, alpha=0.55, edgecolors="none", rasterized=True)
ax_C.set_xlabel("PC1 score", fontsize=8)
ax_C.set_ylabel(IPC_COL, fontsize=8)
ax_C.set_title(f"C  PC1 vs IPC   r = {r_pc1_ipc:+.3f}", fontsize=8, loc="left")
ax_C.spines[["top", "right"]].set_visible(False)

# ── D: partial correlation — PC1 residual vs IPC residual (both | SR) ──
ax_D.scatter(resid_pc1_on_sr, resid_ipc_on_sr,
             c=colors_gnm, s=8, alpha=0.55, edgecolors="none", rasterized=True)
ax_D.axhline(0, color="grey", lw=0.5, alpha=0.5)
ax_D.axvline(0, color="grey", lw=0.5, alpha=0.5)
ax_D.set_xlabel("PC1 residual  (SR removed)", fontsize=8)
ax_D.set_ylabel("IPC residual  (SR removed)", fontsize=8)
ax_D.set_title(
    f"D  PC1 vs IPC | SR   r = {r_pc1_ipc_gsr:+.3f}   ({retention_pct:.0f}% retained)",
    fontsize=8, loc="left"
)
ax_D.spines[["top", "right"]].set_visible(False)

out_path = OUTPUT_FOLDER / "diagnostic_spectral_radius_confound.pdf"
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Plain-English verdict
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

abs_partial = abs(r_pc1_ipc_gsr)
if abs_partial < 0.20:
    verdict = "CONFOUND CONFIRMED"
    interp  = (
        f"r(PC1, IPC | SR) = {r_pc1_ipc_gsr:+.3f} ≈ 0. "
        "Spectral radius explains essentially the entire PC1→IPC link. "
        "PC1 adds no meaningful predictive power beyond SR alone. "
        "The morphospace result is a structural tautology."
    )
elif abs_partial < 0.50:
    verdict = "PARTIAL CONFOUND"
    interp  = (
        f"r(PC1, IPC | SR) = {r_pc1_ipc_gsr:+.3f} "
        f"({retention_pct:.0f}% of raw r = {r_pc1_ipc:+.3f} retained). "
        "Spectral radius is the dominant driver, but PC1 still captures "
        "additional structural signal that independently predicts IPC. "
        "A reviewer's concern is partially valid but does not invalidate the result."
    )
else:
    verdict = "CONFOUND REJECTED"
    interp  = (
        f"r(PC1, IPC | SR) = {r_pc1_ipc_gsr:+.3f} "
        f"({retention_pct:.0f}% of raw r = {r_pc1_ipc:+.3f} retained). "
        "PC1 predicts IPC largely independently of spectral radius. "
        "The PC1→IPC relationship reflects genuine multi-dimensional "
        "structural variation, not merely a spectral-radius proxy."
    )

print(f"\n>>> {verdict}\n")
print(interp)
print(f"""
Key numbers
───────────────────────────────────────────────────
r(SR,  IPC)               = {r_sr_ipc:+.4f}
r(PC1, IPC)               = {r_pc1_ipc:+.4f}
r(PC1, IPC | SR)          = {r_pc1_ipc_gsr:+.4f}   ({retention_pct:.0f}% of raw r retained)
r(SR,  IPC | PC1)         = {r_sr_ipc_gpc1:+.4f}
spectral_radius PC1 load  = {sr_loading:+.4f}   (rank #{sr_rank} by |r| with IPC)
R² unique to PC1          = {unique_pc1*100:.1f}%
R² unique to SR           = {unique_sr*100:.1f}%
R² shared PC1 ∩ SR        = {shared*100:.1f}%
───────────────────────────────────────────────────
""")
