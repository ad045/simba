"""Empty radar (spider) plot template.

Draws a blank radar grid in the same visual style as the representatives
spider plots: dashed gray rings + spokes, manual axis labels, no data.

Axes:   accuracy, agreement, biological plausibility,
        computational efficiency, sensitivity
Rings:  0, 1, 2, 3, 4, 5
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def cm_to_inch(size):
    return tuple(s / 2.54 for s in size)


FEATURE_LABELS = [
    "accuracy",
    "agreement",
    "biological plausibility",
    "computational efficiency",
    "sensitivity",
]
RING_VALS = [0, 1, 2, 3, 4, 5]
YLIM = (0, 5)


def make_empty_spider(feature_labels, ring_vals, ylim, filepath,
                      axis_colors="gray", title=None):
    n             = len(feature_labels)
    angles        = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    theta_ring    = np.linspace(0, 2 * np.pi, 300)

    fig, ax = plt.subplots(figsize=cm_to_inch((6, 6)),
                           subplot_kw=dict(polar=True), dpi=100)
    ax.set_ylim(*ylim)
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Rotate so that one spoke points straight up (vertical).
    ax.set_theta_offset(np.pi / 2)

    # Rings: dashed gray, with the baseline ring drawn solid.
    # Rings at 3 and 5 are emphasised (thicker).
    thick_rings = {3, 5}
    for r_val in ring_vals:
        if r_val == ylim[0]:
            ax.plot(theta_ring, np.full(300, r_val), color=axis_colors,
                    linewidth=0.75, # linestyle="-", 
                    zorder=10)
        else:
            lw = 1 if r_val in thick_rings else 0.5
            style = "-" if r_val in thick_rings else "--"
            ax.plot(theta_ring, np.full(300, r_val), color="gray",
                    linewidth=lw, linestyle=style, zorder=10)

    # Spokes
    spoke_outer = max(ring_vals)
    for angle in angles:
        ax.plot([angle, angle], [ylim[0], spoke_outer],
                color=axis_colors, linewidth=0.5, linestyle="--", zorder=10)

    # Ring tick labels: small, sitting just left of the downward radial.
    ax.set_yticks(ring_vals)
    # 180 + 90deg theta_offset => labels sit on the downward radial.
    ax.set_rlabel_position(180)
    ax.yaxis.set_ticklabels(
        labels=ring_vals,
        fontdict={"verticalalignment": "center",
                  "horizontalalignment": "right"},
    )
    ax.tick_params(axis="y", pad=1, labelcolor=axis_colors,
                   labelsize=5, zorder=12)

    # Axis (feature) labels placed just outside the outer ring.
    label_r = ylim[1] * 1.3
    for angle, lbl in zip(angles, feature_labels):
        ha = "right" if np.pi / 2 < angle < 3 * np.pi / 2 else "left"
        if abs(np.sin(angle)) < 0.15:
            ha = "center"
        va = "top" if np.pi < angle < 2 * np.pi else "bottom"
        if abs(np.cos(angle)) < 0.15:
            va = "center"
        ax.text(angle, label_r, lbl.replace(" ", "\n"),
                ha=ha, va=va, fontsize=6, color=axis_colors, zorder=20)

    ax.set_thetagrids([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    out = Path(__file__).parent / "empty_radar.pdf"
    make_empty_spider(FEATURE_LABELS, RING_VALS, YLIM, out)
    print(f"{out}")
