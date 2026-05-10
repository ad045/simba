"""
Two diagnostics to test whether the GNM morphospace has specific value:

PATH 2 — Mantel test
  Does pairwise distance in the morphospace correlate with taxonomic distance?
  Compare GNM morphospace vs ER morphospace.
  Taxonomic distance uses full hierarchy: species < genus < family < order < super_order.
  If GNM Mantel r >> ER Mantel r → GNM provides a biologically more faithful space.

PATH 3 — Feature selection bootstrap
  Does using the curated 80-feature set outperform random subsets of the same size?
  For N=1000 random draws of k features from the 80 available:
    - Fit PCA on GNM, transform HCP-D + MaMI
    - Record max |r|(age) and max ANOVA-F(orders) across PC1–10
  Compare distribution to the full curated 80-feature result.
  If curated >> random → feature curation adds value.
  If curated ≈ random → the signal is robust to feature choice.
"""

# %% ── Imports ─────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from vizman import viz

# %% ── Paths ───────────────────────────────────────────────────────────────────

BASE    = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
OUT_DIR = BASE / "output/trade_off_analysis/07_mantel_and_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PKL_PATH  = BASE / "output/00_trade_off_analysis/all_datasets_precise_categories.pkl"
AGES_PATH = BASE / "data/preprocessed/lexis_data_developing/04_further_info/00_ages.npy"
MAMI_META = BASE / "data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv"

MAIN_ORDERS = ["Carnivora", "Primates", "Cetartiodactyla", "Rodentia", "Chiroptera"]

# %% ── Load data ───────────────────────────────────────────────────────────────

with open(PKL_PATH, "rb") as f:
    datasets = pickle.load(f)

df_gnm  = datasets["hcp_schaefer_100_dataset_gnm"]
df_er   = datasets["erdos_renyi_networks"]
df_lexi = datasets["lexis_data_developing"]
df_mami = datasets["suarez_MaMI_dataset"]

ages_all = np.load(AGES_PATH)
ages = ages_all if len(ages_all) == len(df_lexi) else ages_all[df_lexi.index]

mami_meta_full = pd.read_csv(MAMI_META).reset_index(drop=True)
mami_meta = (mami_meta_full if len(mami_meta_full) == len(df_mami)
             else mami_meta_full.iloc[df_mami.index].reset_index(drop=True))

orders_raw = mami_meta["order"].fillna("Other").values
orders = np.where(np.isin(orders_raw, MAIN_ORDERS), orders_raw, "Other")

# %% ── Common valid features ───────────────────────────────────────────────────

common = set(df_gnm.columns) & set(df_er.columns) & set(df_lexi.columns) & set(df_mami.columns)

def _valid(df, cols):
    s = df[list(cols)].replace([np.inf, -np.inf], np.nan)
    return set(s.columns[s.notna().all()])

valid_cols = sorted(_valid(df_gnm, common) & _valid(df_er, common))
n_features = len(valid_cols)
print(f"Common valid features: {n_features}")

# %% ── Morphospace helpers ─────────────────────────────────────────────────────

def fit_pca(df_ref, cols):
    arr = df_ref[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    sc  = StandardScaler().fit(arr)
    pc  = PCA(n_components=10, random_state=42).fit(sc.transform(arr))
    return sc, pc

def project(df, cols, sc, pc):
    arr = df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    for j, m in enumerate(sc.mean_):
        arr[np.isnan(arr[:, j]), j] = m
    return pc.transform(sc.transform(arr))

def bio_signals(lexi_pca, mami_pca):
    """Return max |r|(age) and max ANOVA-F(orders) across PC1-10."""
    r_max = max(abs(stats.pearsonr(ages, lexi_pca[:, i])[0]) for i in range(10))
    F_max = max(
        stats.f_oneway(*[mami_pca[orders == o, i] for o in MAIN_ORDERS])[0]
        for i in range(10)
    )
    return r_max, F_max

# Fit GNM and ER morphospaces on full feature set
sc_gnm, pc_gnm = fit_pca(df_gnm, valid_cols)
sc_er,  pc_er  = fit_pca(df_er,  valid_cols)

lexi_gnm = project(df_lexi, valid_cols, sc_gnm, pc_gnm)
mami_gnm = project(df_mami, valid_cols, sc_gnm, pc_gnm)
lexi_er  = project(df_lexi, valid_cols, sc_er,  pc_er)
mami_er  = project(df_mami, valid_cols, sc_er,  pc_er)

r_gnm_full, F_gnm_full = bio_signals(lexi_gnm, mami_gnm)
r_er_full,  F_er_full  = bio_signals(lexi_er,  mami_er)
print(f"GNM (all {n_features} features): max|r|={r_gnm_full:.3f}  max-F={F_gnm_full:.1f}")
print(f"ER  (all {n_features} features): max|r|={r_er_full:.3f}  max-F={F_er_full:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PATH 2 — Mantel test
# ═══════════════════════════════════════════════════════════════════════════════

# %% ── Build taxonomic distance matrix (MaMI, n=224) ──────────────────────────
#
# Levels (higher = more distant):
#   same species   → 0
#   same genus     → 1
#   same family    → 2
#   same order     → 3
#   same superorder→ 4
#   different      → 5

def taxonomic_distance(meta: pd.DataFrame) -> np.ndarray:
    n = len(meta)
    D = np.zeros((n, n), dtype=float)
    levels = ["species", "genus", "family", "order", "super_order"]
    for i in range(n):
        for j in range(i + 1, n):
            dist = len(levels) + 1          # default: completely different
            for k, lvl in enumerate(levels):
                vi, vj = meta[lvl].iloc[i], meta[lvl].iloc[j]
                if pd.notna(vi) and pd.notna(vj) and vi == vj:
                    dist = k                # match at level k → distance = k
                    break
            D[i, j] = D[j, i] = dist
    return D

print("\nBuilding taxonomic distance matrix…")
D_taxo = taxonomic_distance(mami_meta)
print(f"  Done. Unique distances: {np.unique(D_taxo)}")
print(f"  Mean dist: {D_taxo[np.triu_indices(len(D_taxo),1)].mean():.2f}")

# %% ── Mantel test helper ──────────────────────────────────────────────────────

def mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int = 9999,
           rng: np.random.Generator = None) -> tuple[float, float]:
    """Pearson Mantel test on lower-triangle vectors. Returns (r, p)."""
    if rng is None:
        rng = np.random.default_rng(42)
    idx  = np.tril_indices(len(D1), -1)
    v1, v2 = D1[idx], D2[idx]
    r_obs, _ = stats.pearsonr(v1, v2)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(D2))
        D2p  = D2[np.ix_(perm, perm)]
        r_p, _ = stats.pearsonr(v1, D2p[idx])
        if r_p >= r_obs:
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)

# %% ── Morphospace distance matrices (PC1–2) ───────────────────────────────────

def dist_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

D_gnm = dist_matrix(mami_gnm[:, :2])   # PC1–2 only
D_er  = dist_matrix(mami_er[:, :2])

print("\nRunning Mantel tests (9999 permutations)…")
rng = np.random.default_rng(42)
r_mantel_gnm, p_mantel_gnm = mantel(D_taxo, D_gnm, n_perm=9999, rng=rng)
r_mantel_er,  p_mantel_er  = mantel(D_taxo, D_er,  n_perm=9999, rng=rng)

print(f"  GNM morphospace:  Mantel r = {r_mantel_gnm:.4f}  p = {p_mantel_gnm:.4f}")
print(f"  ER  morphospace:  Mantel r = {r_mantel_er:.4f}  p = {p_mantel_er:.4f}")
print(f"  GNM advantage:    Δr = {r_mantel_gnm - r_mantel_er:+.4f}")

# Also test using all 10 PCs
D_gnm_10 = dist_matrix(mami_gnm)
D_er_10  = dist_matrix(mami_er)
r_m_gnm_10, p_m_gnm_10 = mantel(D_taxo, D_gnm_10, n_perm=9999, rng=rng)
r_m_er_10,  p_m_er_10  = mantel(D_taxo, D_er_10,  n_perm=9999, rng=rng)
print(f"\n  GNM (10 PCs):  Mantel r = {r_m_gnm_10:.4f}  p = {p_m_gnm_10:.4f}")
print(f"  ER  (10 PCs):  Mantel r = {r_m_er_10:.4f}  p = {p_m_er_10:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PATH 3 — Feature selection bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

# %% ── Bootstrap: random k features from the 80 available ────────────────────

N_ITER  = 1000
K_VALUES = [10, 20, 30, 40, 60, n_features]   # last = all curated features
rng_bs  = np.random.default_rng(0)

print(f"\nFeature bootstrap ({N_ITER} iterations per k, GNM morphospace)…")

results = {k: {"r": [], "F": []} for k in K_VALUES}

for k in K_VALUES:
    if k == n_features:
        # Full curated set — no random sampling
        r_val, F_val = r_gnm_full, F_gnm_full
        results[k]["r"].append(r_val)
        results[k]["F"].append(F_val)
        print(f"  k={k:>3} (all features): |r|={r_val:.3f}  F={F_val:.1f}")
        continue

    for _ in range(N_ITER):
        sampled = list(rng_bs.choice(valid_cols, size=k, replace=False))
        # Only use features valid in GNM (no NaN)
        gnm_sub = df_gnm[sampled].replace([np.inf, -np.inf], np.nan)
        ok_cols = list(gnm_sub.columns[gnm_sub.notna().all()])
        if len(ok_cols) < 2:
            continue
        sc_k, pc_k = fit_pca(df_gnm, ok_cols)
        n_comps = min(10, len(ok_cols), len(df_gnm) - 1)
        sc_k2 = StandardScaler().fit(df_gnm[ok_cols].replace([np.inf, -np.inf], np.nan).values)
        pc_k2 = PCA(n_components=n_comps, random_state=42)
        arr_ref = sc_k2.transform(df_gnm[ok_cols].replace([np.inf, -np.inf], np.nan).values)
        pc_k2.fit(arr_ref)

        def _proj(df):
            a = df[ok_cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
            for j, m in enumerate(sc_k2.mean_):
                a[np.isnan(a[:, j]), j] = m
            return pc_k2.transform(sc_k2.transform(a))

        lp = _proj(df_lexi)
        mp = _proj(df_mami)
        r_v = max(abs(stats.pearsonr(ages, lp[:, i])[0]) for i in range(n_comps))
        F_v = max(
            stats.f_oneway(*[mp[orders == o, i] for o in MAIN_ORDERS])[0]
            for i in range(n_comps)
        )
        results[k]["r"].append(r_v)
        results[k]["F"].append(F_v)

    r_arr = np.array(results[k]["r"])
    F_arr = np.array(results[k]["F"])
    print(f"  k={k:>3}: |r| mean={r_arr.mean():.3f} ±{r_arr.std():.3f}  "
          f"F mean={F_arr.mean():.1f} ±{F_arr.std():.1f}")

# %% ── Figures ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=viz.cm_to_inch((16, 12)), layout="constrained")

# ── PATH 2: Mantel scatter (taxonomic dist vs morphospace dist) ───────────────

idx_lower = np.tril_indices(len(D_taxo), -1)
d_tax = D_taxo[idx_lower]
d_gnm_v = D_gnm[idx_lower]
d_er_v  = D_er[idx_lower]

for ax, d_morph, label, r_val, p_val, color in [
    (axes[0, 0], d_gnm_v, "GNM", r_mantel_gnm, p_mantel_gnm, "#232324"),
    (axes[0, 1], d_er_v,  "ER",  r_mantel_er,  p_mantel_er,  "#B8B8B8"),
]:
    # Hex-bin for density (224×223/2 = 24976 pairs)
    ax.hexbin(d_tax, d_morph, gridsize=30, cmap="Blues" if color == "#232324" else "Greys",
              linewidths=0.2, mincnt=1)
    # Per-level means
    for lvl in np.unique(d_tax):
        mask = d_tax == lvl
        ax.scatter(lvl, d_morph[mask].mean(), s=40, color=color,
                   edgecolors="black", lw=0.6, zorder=5)
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.text(0.97, 0.05, f"Mantel r = {r_val:.3f}{stars}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"))
    tax_labels = {0: "species", 1: "genus", 2: "family", 3: "order",
                  4: "super_order", 5: "other", 6: "unmatched"}
    xticks = sorted(np.unique(d_tax))
    ax.set_xticks(xticks)
    ax.set_xticklabels([tax_labels.get(int(x), str(int(x))) for x in xticks],
                       fontsize=5.5, rotation=30, ha="right")
    ax.set_xlabel("Taxonomic distance", fontsize=7)
    ax.set_ylabel("Morphospace distance (PC1–2)", fontsize=7)
    ax.set_title(f"{'A' if label=='GNM' else 'B'}  {label} morphospace\nvs taxonomic distance (Mantel test)",
                 fontsize=7, loc="left")
    ax.tick_params(labelsize=5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── PATH 3: Feature bootstrap distributions ───────────────────────────────────

k_plot = [k for k in K_VALUES if k < n_features]   # exclude "all features" (single point)

for ax, metric, ylabel, title_letter in [
    (axes[1, 0], "r", "|Pearson r|  (age ~ best PC)", "C"),
    (axes[1, 1], "F", "ANOVA F  (order → best PC)",   "D"),
]:
    data_bp = [results[k][metric] for k in k_plot]
    positions = list(range(len(k_plot)))

    bp = ax.boxplot(data_bp, positions=positions, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    flierprops=dict(marker=".", ms=2, alpha=0.3))
    for patch in bp["boxes"]:
        patch.set_facecolor("#AAAAAA")
        patch.set_alpha(0.7)

    # Full curated set reference line
    ref_val = results[n_features][metric][0]
    ax.axhline(ref_val, color="#C74800", lw=1.4, linestyle="--",
               label=f"All {n_features} curated features ({ref_val:.3f})" if metric=="r"
                     else f"All {n_features} curated features (F={ref_val:.1f})")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"k={k}" for k in k_plot], fontsize=6)
    ax.set_xlabel("Number of randomly sampled features", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(f"{title_letter}  Feature bootstrap (GNM morphospace, n={N_ITER} draws per k)",
                 fontsize=7, loc="left")
    ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

out = OUT_DIR / "mantel_and_features.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"\nSaved: {out}")
plt.show()

# %% ── Final summary ──────────────────────────────────────────────────────────

print("\n═══ Summary ════════════════════════════════════════════════════════════════")
print(f"Mantel test (PC1–2):")
print(f"  GNM r={r_mantel_gnm:.4f} p={p_mantel_gnm:.4f}  |  ER r={r_mantel_er:.4f} p={p_mantel_er:.4f}")
print(f"  GNM advantage: Δr = {r_mantel_gnm - r_mantel_er:+.4f}")
print(f"\nMantel test (all 10 PCs):")
print(f"  GNM r={r_m_gnm_10:.4f} p={p_m_gnm_10:.4f}  |  ER r={r_m_er_10:.4f} p={p_m_er_10:.4f}")
print(f"\nFeature bootstrap: random k=30 vs all {n_features} features:")
r30 = np.array(results[30]["r"]); F30 = np.array(results[30]["F"])
print(f"  k=30: |r| {r30.mean():.3f}±{r30.std():.3f}  F {F30.mean():.1f}±{F30.std():.1f}")
print(f"  full: |r| {r_gnm_full:.3f}  F {F_gnm_full:.1f}")
print(f"  Signal retained with k=30: r {r30.mean()/r_gnm_full*100:.0f}%  "
      f"F {F30.mean()/F_gnm_full*100:.0f}%")
