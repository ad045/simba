"""
viz_utils.py — Reusable functions for connectome trade-off analysis.

Covers:
  - Correlation matrix plotting (category-ordered, cluster-ordered)
  - Spider / radar plots (per-network, category-level)
  - PCA scatter plots and quiver plots
  - PERMANOVA helper
  - Misc geometry helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle


# ─────────────────────────────────────────────────────────────────────────────
# 1. CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

# Return columns sorted by CATEGORY_ORDER, within-category hierarchical clustering 
def sort_cols_by_category(
    cols_in_data: list[str],
    corr_full: pd.DataFrame,
    meta: pd.DataFrame,
    category_order: list[str],
) -> list[str]:
    """
    Return columns sorted by CATEGORY_ORDER, with within-category hierarchical
    clustering by average-linkage on (1 - r) distances.
    """
    def _sort_key(col):
        cat = meta.loc[col, "Category"] if col in meta.index else "ZZZ"
        cat = cat if pd.notna(cat) else "ZZZ"
        idx = category_order.index(cat) if cat in category_order else len(category_order)
        return (idx, str(meta.loc[col, "section"]) if col in meta.index else "")

    cols_sorted = sorted(cols_in_data, key=_sort_key)

    final_order: list[str] = []
    for cat in category_order:
        cat_cols = [c for c in cols_sorted if c in meta.index and meta.loc[c, "Category"] == cat]
        if len(cat_cols) == 0:
            continue
        if len(cat_cols) == 1:
            final_order.extend(cat_cols)
            continue
        sub_corr = corr_full.loc[cat_cols, cat_cols].fillna(0)
        dist = np.clip(1 - sub_corr.values, 0, 2)
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist, checks=False), method="average")
        final_order.extend([cat_cols[i] for i in leaves_list(Z)])

    # Append any uncategorised variables
    final_order.extend([c for c in cols_in_data if c not in final_order])
    return final_order


def _draw_category_brackets(
    ax,
    final_order: list[str],
    meta: pd.DataFrame,
    category_colours: dict[str, str],
    bracket_x: float = -0.70,
    serif_w: float = 0.005,
    label_x: float = -0.71,
    fontsize: int = 8,
):
    """Draw coloured bracket + label for each category on the left of a heatmap."""
    n = len(final_order)
    trans = ax.transAxes

    cat_row_spans: dict[str, list[int]] = {}
    for row_i, c in enumerate(final_order):
        cat = meta.loc[c, "Category"] if c in meta.index else "Other"
        cat_row_spans.setdefault(cat, [row_i, row_i])[1] = row_i

    def _row_to_y(row_i):
        return 1.0 - (row_i + 0.5) / n

    eps = 0.00085
    for cat, (r0, r1) in cat_row_spans.items():
        colour = category_colours.get(cat, "#444444")
        y_top = _row_to_y(r0) + 0.5 / n + eps
        y_bot = _row_to_y(r1) - 0.5 / n - eps
        y_mid = (y_top + y_bot) / 2
        kw = dict(
            xycoords=trans, textcoords=trans, annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=colour, lw=1.5),
        )
        ax.annotate("", xy=(bracket_x, y_bot), xytext=(bracket_x, y_top), **kw)
        ax.annotate("", xy=(bracket_x, y_top), xytext=(bracket_x + serif_w, y_top), **kw)
        ax.annotate("", xy=(bracket_x, y_bot), xytext=(bracket_x + serif_w, y_bot), **kw)
        ax.text(
            label_x, y_mid, cat,
            transform=trans, ha="right", va="center",
            fontsize=fontsize, fontweight="bold", color=colour, clip_on=False,
        )


def plot_corr_matrix(
    df_properties: pd.DataFrame,
    final_order: list[str],
    meta: pd.DataFrame,
    category_colours: dict[str, str],
    property_names: dict[str, str],
    *,
    cmap: str = "RdBu_r",
    figsize_cm: tuple[float, float] = (18, 12),
    label_fontsize: int = 8,
    title: str | None = None,
    savepath: Path | None = None,
    viz=None,                    # optional vizman handle for cm_to_inch
) -> plt.Figure:
    """
    Plot a correlation heatmap ordered by `final_order` with category brackets.
    """
    corr = df_properties[final_order].corr()
    new_labels = [property_names.get(c, c) for c in final_order]
    n = len(final_order)

    _to_inch = viz.cm_to_inch if viz is not None else lambda x: (x[0] / 2.54, x[1] / 2.54)
    fig = plt.figure(figsize=_to_inch(figsize_cm))
    gs = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
    ax = fig.add_subplot(gs[0])

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="equal", interpolation="none")

    # White dividers between categories
    prev_cat = meta.loc[final_order[0], "Category"] if final_order[0] in meta.index else None
    for i, c in enumerate(final_order[1:], start=1):
        curr_cat = meta.loc[c, "Category"] if c in meta.index else None
        if curr_cat != prev_cat:
            ax.axhline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
            ax.axvline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
        prev_cat = curr_cat

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(new_labels, fontsize=label_fontsize)
    ax.set_xticks([])
    ax.tick_params(axis="both", which="both", length=0)

    _draw_category_brackets(ax, final_order, meta, category_colours)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Pearson r", fontsize=9)
    if title:
        ax.set_title(title)

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPIDER / RADAR PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def draw_axis_group_arcs(
    ax,
    feature_labels: list[str],
    angles: list[float],
    arc_radius: float,
    groups: list[tuple[str, str, str]],
):
    """
    Draw a bold arc connecting pairs of spokes on a polar axis.
    groups: list of (label_a, label_b, color)
    """
    for label_a, label_b, color in groups:
        if label_a not in feature_labels or label_b not in feature_labels:
            continue
        i_a, i_b = feature_labels.index(label_a), feature_labels.index(label_b)
        angle_a, angle_b = angles[i_a], angles[i_b]

        if angle_b < angle_a:
            angle_a, angle_b = angle_b, angle_a
        arc_angles = (
            np.linspace(angle_b, angle_a + 2 * np.pi, 80)
            if angle_b - angle_a > np.pi
            else np.linspace(angle_a, angle_b, 80)
        )
        ax.plot(arc_angles, np.full_like(arc_angles, arc_radius),
                color=color, linewidth=3.5, solid_capstyle="round", zorder=0, alpha=0.85)
        ax.scatter([arc_angles[0], arc_angles[-1]], [arc_radius, arc_radius],
                   color=color, s=18, zorder=7, alpha=0.85)


def make_spider(
    values: np.ndarray,
    label: str,
    color: str,
    feature_labels: list[str],
    title: str,
    ylim: tuple[float, float],
    filepath: Path,
    neighbour_vals_list: list[np.ndarray] | None = None,
    axis_colors: str | list[str] = "gray",
    ring_vals: list[float] | None = None,
    std_vals: np.ndarray | None = None,
    viz=None,
):
    """
    Save a standalone spider/radar plot to `filepath`.

    Parameters
    ----------
    values : z-scored category-level values for the focal network.
    neighbour_vals_list : list of z-scored rows for nearest neighbours
                         (drawn as faint background traces).
    axis_colors : single color string or list matching feature_labels.
    ring_vals : y-positions for the concentric reference rings.
    std_vals : per-axis standard deviations; when provided a ±std band is
               drawn around the focal trace.
    """
    ring_vals = ring_vals or [-2, 0, 2, 4]
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + angles[:1]
    vals_c = list(values) + [values[0]]

    _to_inch = viz.cm_to_inch if viz is not None else lambda x: (x[0] / 2.54, x[1] / 2.54)
    fig, ax = plt.subplots(figsize=_to_inch((6, 6)), subplot_kw=dict(polar=True), dpi=100)

    ax.set_ylim(*ylim)
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    theta = np.linspace(0, 2 * np.pi, 300)
    _ac = axis_colors if isinstance(axis_colors, str) else "gray"

    # Reference rings
    for r in ring_vals:
        ls = "-" if r == 0 else "--"
        lw = 1.0 if r == 0 else 0.5
        ax.plot(theta, np.full(300, r), color=_ac, linewidth=lw, linestyle=ls, zorder=0)

    # Spokes
    for angle in angles:
        ax.plot([angle, angle], list(ylim), color=_ac, linewidth=0.5, linestyle="--", zorder=0)

    # Neighbour traces (background)
    if neighbour_vals_list:
        for nb in neighbour_vals_list:
            nb_c = list(nb) + [nb[0]]
            ax.plot(angles_c, nb_c, color=color, linewidth=1, alpha=0.8, zorder=3)
            ax.fill(angles_c, nb_c, color=color, alpha=0.05, zorder=3)

    # ±std band
    if std_vals is not None:
        upper_c = list(values + std_vals) + [values[0] + std_vals[0]]
        lower_c = list(values - std_vals) + [values[0] - std_vals[0]]
        ax.fill_between(angles_c, lower_c, upper_c, color=color, alpha=0.20, zorder=3)

    # Focal trace
    ax.fill(angles_c, vals_c, color=color, alpha=0.05, zorder=4)
    ax.plot(angles_c, vals_c, color="black", linewidth=1, zorder=5)

    ax.set_thetagrids([])
    ax.tick_params(axis="y", labelcolor=_ac)
    ax.set_yticks(ring_vals)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath.name}")


def make_spider_legend(
    feature_labels: list[str],
    angles: list[float],
    ylim: tuple[float, float],
    ring_vals: list[float],
    filepath: Path,
    label_color: str = "gray",
    category_colours: dict[str, str] | None = None,
    axis_groups: list[tuple] | None = None,
    viz=None,
):
    """Empty radar showing only spoke labels and reference rings — for use as legend."""
    n = len(feature_labels)
    angles_plot = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    _to_inch = viz.cm_to_inch if viz is not None else lambda x: (x[0] / 2.54, x[1] / 2.54)
    fig, ax = plt.subplots(figsize=_to_inch((9, 6)), subplot_kw=dict(polar=True), dpi=150)
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    theta = np.linspace(0, 2 * np.pi, 300)
    for r in ring_vals:
        ls = "-" if r == 0 else "--"
        lw = 1.8 if r == 0 else 0.5
        ax.plot(theta, np.full(300, r), color=label_color, linewidth=lw, linestyle=ls, zorder=0)
    for angle in angles_plot:
        ax.plot([angle, angle], list(ylim), color=label_color, linewidth=0.5, linestyle="--")

    # Optional arcs
    if axis_groups:
        arc_r = ylim[1] + (ylim[1] - ylim[0]) * 0.12
        draw_axis_group_arcs(ax, list(feature_labels), angles_plot, arc_r, axis_groups)
        ax.set_ylim(ylim[0], arc_r + (ylim[1] - ylim[0]) * 0.05)

    # Spoke labels
    labels_nl = [lbl.replace(" ", "\n") for lbl in feature_labels]
    font_colors = (
        [category_colours.get(lbl, label_color) for lbl in feature_labels]
        if category_colours else [label_color] * n
    )
    ax.set_thetagrids(np.degrees(angles_plot), labels_nl)
    for i, txt in enumerate(ax.get_xticklabels()):
        txt.set_color(font_colors[i])
        txt.set_horizontalalignment("right" if i in [2, 3, 4] else "left")

    plt.title("Legend")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_corners(pca_result_2d: np.ndarray) -> dict[str, int]:
    """
    Return indices (into pca_result_2d) of three 'corner' networks:
      top_left     → highest PC2
      bottom_left  → most negative (PC1 + PC2)
      bottom_right → highest PC1, lowest PC2
    """
    return {
        "top_left":     int(np.argmax(pca_result_2d[:, 1])),
        "bottom_left":  int(np.argmax(-pca_result_2d[:, 0] - pca_result_2d[:, 1])),
        "bottom_right": int(np.argmax(pca_result_2d[:, 0] - pca_result_2d[:, 1])),
    }


def get_neighbours(
    orig_idx: int,
    embed: np.ndarray,
    n: int = 5,
    exclude_mask: np.ndarray | None = None,
) -> list[int]:
    """
    Return indices of the n nearest neighbours of `orig_idx` in `embed`.
    Rows where exclude_mask is True are excluded from results.
    """
    dists = pairwise_distances(embed[orig_idx : orig_idx + 1], embed)[0]
    dists[orig_idx] = np.inf
    if exclude_mask is not None:
        dists[exclude_mask] = np.inf
    return np.argsort(dists)[:n].tolist()


def get_cat_values(
    row_vals: np.ndarray,
    feature_cols: list[str],
    cat_order: list[str],
    meta: pd.DataFrame,
) -> np.ndarray:
    """Mean of z-scored feature values within each category — one value per category."""
    result = []
    for cat in cat_order:
        cols = [c for c in feature_cols if c in meta.index and meta.loc[c, "Category"] == cat]
        if not cols:
            result.append(np.nan)
            continue
        idxs = [feature_cols.index(c) for c in cols]
        result.append(np.nanmean(row_vals[idxs]))
    return np.array(result)


def build_cat_scaled(
    scaled_data: np.ndarray,
    feature_cols: list[str],
    cat_order: list[str],
    meta: pd.DataFrame,
) -> np.ndarray:
    """
    Build a (n_networks × n_categories) matrix of category-mean z-scores,
    then re-z-score each category column across all networks.
    """
    cat_scaled = np.array([
        get_cat_values(scaled_data[i], feature_cols, cat_order, meta)
        for i in range(len(scaled_data))
    ])
    return np.apply_along_axis(zscore, 0, cat_scaled)


def build_cat_quiver_vectors(
    loading_and_cat_df: pd.DataFrame,
    loading_labels: list[str],
    meta: pd.DataFrame,
    category_colours: dict[str, str],
    pc_cols: tuple[str, str] = ("PC1", "PC2"),
) -> dict[str, tuple[float, float]]:
    """
    For each category, compute a 2-D loading vector using L2-norm magnitude
    with mean-loading direction.

    Magnitude = sqrt(sum of squared loadings across all features and both PC axes),
    so cancellation between opposite-sign features cannot reduce the arrow length.
    Direction = unit vector of the mean loading, i.e. the average direction across
    category members.  If the mean direction is degenerate (near-zero), the arrow
    is placed along the axis with the larger total squared loading.

    Returns dict[category_name → (vx, vy)].
    """
    vecs: dict[str, tuple[float, float]] = {}
    for cat in category_colours:
        cat_vars = [
            c for c in loading_labels
            if c in meta.index and meta.loc[c, "Category"] == cat
        ]
        if not cat_vars:
            continue
        sub = loading_and_cat_df.loc[cat_vars]
        loads_pc1 = sub[pc_cols[0]].values
        loads_pc2 = sub[pc_cols[1]].values

        magnitude = np.sqrt((loads_pc1 ** 2).sum() + (loads_pc2 ** 2).sum())
        mean_dir = np.array([loads_pc1.mean(), loads_pc2.mean()])
        norm = np.linalg.norm(mean_dir)
        if norm > 1e-10:
            unit = mean_dir / norm
        else:
            # degenerate mean direction: point along the dominant axis
            unit = np.array([1.0, 0.0]) if (loads_pc1 ** 2).sum() >= (loads_pc2 ** 2).sum() else np.array([0.0, 1.0])

        vecs[cat] = (float(unit[0] * magnitude), float(unit[1] * magnitude))
    return vecs


def plot_quiver(
    cat_vectors: dict[str, tuple[float, float]],
    category_colours: dict[str, str],
    *,
    with_labels: bool = True,
    savepath: Path | None = None,
    figsize_cm: tuple[float, float] = (12, 12),
    viz=None,
) -> plt.Figure:
    """Plot category-level PCA loading vectors as quiver arrows."""
    _to_inch = viz.cm_to_inch if viz is not None else lambda x: (x[0] / 2.54, x[1] / 2.54)
    fig, ax = plt.subplots(figsize=_to_inch(figsize_cm), dpi=150)

    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.scatter([0], [0], color="black", s=20, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = category_colours[cat]
        ax.quiver(0, 0, vx, vy, angles="xy", scale_units="xy", scale=1,
                  color=colour, width=0.012, headwidth=4, headlength=5,
                  headaxislength=4, zorder=5)
        if with_labels:
            ax.text(vx * 1.15, vy * 1.15, cat, ha="center", va="center",
                    color=colour, clip_on=False)

    max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.4
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATISTICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def permanova(
    X: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 999,
) -> tuple[float, float]:
    """
    Permutation-based MANOVA (PERMANOVA) using a pseudo-F statistic.
    Returns (F_observed, p_value).
    """
    def _f(X, y):
        groups = [X[y == g] for g in np.unique(y)]
        grand_m = X.mean(axis=0)
        ss_b = sum(len(g) * ((g.mean(0) - grand_m) ** 2).sum() for g in groups)
        ss_w = sum(((g - g.mean(0)) ** 2).sum() for g in groups)
        k, n = len(groups), len(X)
        return (ss_b / (k - 1)) / (ss_w / (n - k))

    f_obs = _f(X, labels)
    null = [_f(X, shuffle(labels)) for _ in range(n_permutations)]
    p = (np.sum(np.array(null) >= f_obs) + 1) / (n_permutations + 1)
    return float(f_obs), float(p)
