# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vizman import viz

from config import COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES
from utils import get_combined_colors

# ── Data ─────────────────────────────────────────────────────────────────────

with open(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis/all_datasets_precise_categories.pkl",
    "rb",
) as f:
    dict_with_all_datasets = pickle.load(f)

output_folder = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis/03_trade_offs"
)
output_folder.mkdir(parents=True, exist_ok=True)

# ── Dataset selection ─────────────────────────────────────────────────────────

INCLUDE_LEXIS = True

datasets_to_plot = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
    "ring_lattice_networks",
    "erdos_renyi_networks",
]

# all_datasets_precise_categories.pkl dropped several columns (mc_input_scaling,
# proportion_long_range_connections_0.3956, repertoire, eta, gamma) from the GNM entry.
# Replace it with the full GNM from all_datasets.pkl which has everything.
with open(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis/all_datasets.pkl",
    "rb",
) as _f:
    _dict_full = pickle.load(_f)
dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"] = _dict_full["hcp_schaefer_100_dataset_gnm"]

df_gnm = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]
combined_colors = get_combined_colors(df_gnm)

# ── Panel definitions ─────────────────────────────────────────────────────────

scatter_panels = {
    "A": {
        "x": "nct_control_std",
        "y": "targeted_attack_robustness_rob_random_auc",
        "title": "Controllability vs. Robustness",
    },
    "B": {
        "x": "repertoire_sweep_weighted_by_distances_T_critical",
        "y": "mc_input_scaling_0_1_mc_mean",
        "title": "Repertoire vs. Computation",
    },
    "C": {
        "x": "mc_input_scaling_0_1_mc_mean",
        "y": "targeted_attack_robustness_rob_random_auc",
        "title": "Memory vs. Random Failure Robustness",
    },
    "D": {
        "x": "mc_input_scaling_0_1_mc_mean",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Memory vs. Targeted Attack Robustness",
    },
}

ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "targeted_attack_robustness_rob_targeted_auc",
    "computation": "mc_input_scaling_0_1_mc_mean",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}

# ── Ternary helpers ───────────────────────────────────────────────────────────

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0, 0.0],                    # vertex 0: economy     (bottom-left)
    [1.0, 0.0],                    # vertex 1: robustness  (bottom-right)
    [0.5, np.sqrt(3) / 2],        # vertex 2: computation (top)
])


def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])  # (N, 3)
    return (pts @ _TRI)[:, 0], (pts @ _TRI)[:, 1]


def _draw_ternary_frame(ax):
    """Draw triangle boundary, gridlines, and vertex labels."""
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"], ha="center", va="top", fontsize=6.5)
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"], ha="center", va="top", fontsize=6.5)
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"], ha="center", va="bottom", fontsize=6.5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Build figure ──────────────────────────────────────────────────────────────

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ─────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_plot:
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            continue

        df_merged = dict_with_all_datasets[dataset]
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
        is_kaysons = "kaysons" in dataset
        is_reference = dataset in ("ring_lattice_networks", "erdos_renyi_networks")

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset],
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.4 if is_reference else 0.25 if not is_gnm else 0,
            s=12 if is_reference else 10 if is_kaysons else 5,
            alpha=0.15 if is_gnm else 0.6 if is_reference else 1.0 if is_kaysons else 0.4,
            marker="D" if is_reference else "o",
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=5 if is_reference else 4 if is_kaysons else 1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, loc="left")

# ── 2. Ternary panel E ────────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7, loc="left", pad=8)

# Pre-compute global min/max from the GNM dataset — it spans the full morphospace
# and must be the reference range for all datasets so positions are comparable.
_gnm_df = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"][list(ternary_cols.values())].dropna()
_global_min = np.array([
    1.0 - _gnm_df[ternary_cols["economy"]].max(),   # economy = 1 - f_LR
    _gnm_df[ternary_cols["robustness"]].min(),
    _gnm_df[ternary_cols["computation"]].min(),
])
_global_max = np.array([
    1.0 - _gnm_df[ternary_cols["economy"]].min(),
    _gnm_df[ternary_cols["robustness"]].max(),
    _gnm_df[ternary_cols["computation"]].max(),
])

for dataset in datasets_to_plot:
    if (not INCLUDE_LEXIS) and ("lexi" in dataset):
        continue

    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Normalize using global GNM bounds so all datasets are on the same scale
    for col_idx in range(3):
        rng = _global_max[col_idx] - _global_min[col_idx]
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - _global_min[col_idx]) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0
    stack = np.clip(stack, 0.0, 1.0)  # clip empirical values that exceed the GNM range

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
    is_kaysons = "kaysons" in dataset
    is_reference = dataset in ("ring_lattice_networks", "erdos_renyi_networks")
    ax_t.scatter(
        x_cart, y_cart,
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.4 if is_reference else 0.2 if not is_gnm else 0,
        s=12 if is_reference else 10 if is_kaysons else 5,
        marker="D" if is_reference else "o",
        alpha=0.1 if is_gnm else 0.7 if is_reference else 1.0,
        zorder=5 if is_reference else 4 if is_kaysons else 1 if is_gnm else 2,
        rasterized=is_gnm,
    )

# ── Legend ────────────────────────────────────────────────────────────────────

handles, labels = axes["A"].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=len(labels),
           fontsize=6, markerscale=1.5, frameon=False)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = output_folder / "trade_off_grid_4_with_triangle.pdf"
plt.savefig(out_path, bbox_inches="tight", dpi=200)
print(out_path)
plt.show()
