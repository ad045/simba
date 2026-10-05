"""
Parameter-recovery animation for social media
=============================================

A network is generated at known (eta, gamma); each of the 16 measures searches
the 10 x 10 recovery grid for the grid point it finds most similar. Every
panel shows one measure's landscape (rank of the distance to each grid point,
bright = closest), the true parameters (star) and the measure's pick (dot).
Several targets are cycled through, then the whole-morphospace recovery error
of every measure is shown against chance.

Everything on screen comes from the shipped bundle (tables/recovery_wide,
networks/recovery_wide) and `published_readouts()`; nothing is typed in.

    conda activate ma_thesis
    python visualization/animate_parameter_recovery.py     # from paper/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))                              # the simba_networks package
os.environ.setdefault("SIMBA_DATA", str(ROOT / "publication_data" / "simba-networks-data"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.stats import rankdata

from simba_networks import benchmark, data
from simba_networks.measures import NAMES, SIMILARITIES

OUT = ROOT / "figures" / "animations" / "parameter_recovery.mp4"
N_TARGETS = 6
FPS = 20
MOVE, HOLD, FINAL = 14, 34, 90          # frames: transition, hold per target, closing card
BG, FG = "#111111", "#eeeeee"

readouts = benchmark.published_readouts()
order = readouts.loc["recovery_wide"].sort_values().index.tolist()
chance = benchmark.CHANCE["wide"]

rec = data.load_networks("recovery_wide")
table = data.load_table("recovery_wide")
eta_axis, gamma_axis = np.unique(rec["grid_eta"]), np.unique(rec["grid_gamma"])
n = len(eta_axis)


def to_grid(eta, gamma):
    """Continuous grid coordinates (x = eta step, y = gamma step)."""
    return ((eta - eta_axis[0]) / (eta_axis[1] - eta_axis[0]),
            (gamma - gamma_axis[0]) / (gamma_axis[1] - gamma_axis[0]))


# targets spread over the morphospace: greedy farthest-point sampling
pts = np.column_stack(to_grid(rec["target_eta"], rec["target_gamma"]))
targets = [int(np.argmin(np.hypot(*(pts - pts.mean(0)).T)))]
while len(targets) < N_TARGETS:
    targets.append(int(np.argmax(np.min(np.linalg.norm(pts[:, None] - pts[targets], axis=2), axis=1))))

# per measure, per shown target: rank landscape (0 = closest) and the pick
cols = [f"dist_to_grid_{i}" for i in range(n * n)]
land, pick, err = {}, {}, {}
for m in order:
    t = table[table.measure == m].set_index("gt_idx").loc[targets, cols].to_numpy()
    d = -t if m in SIMILARITIES else t
    r = np.array([rankdata(row, nan_policy="omit") for row in d])
    land[m] = ((r - 1) / (n * n - 1)).reshape(-1, n, n)            # gamma-outer / eta-inner
    p = np.nanargmin(d, axis=1)
    pick[m] = np.column_stack([p % n, p // n]).astype(float)
    true_cell = np.round(pts[targets])
    err[m] = np.hypot(*(pick[m] - true_cell).T)

# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG})
fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
fig.text(0.5, 0.965, "Can a similarity measure find\nthe parameters that built a brain network?",
         ha="center", va="top", fontsize=26, weight="bold")
sub = fig.text(0.5, 0.875, "", ha="center", va="top", fontsize=16, color="#bbbbbb")

# target network inset
ax_net = fig.add_axes([0.05, 0.80, 0.11, 0.088])
ax_net.set_xticks([]), ax_net.set_yticks([])
net_img = ax_net.imshow(rec["targets"][targets[0]], cmap="gray_r", interpolation="nearest")
ax_net.set_title("target network", fontsize=11, color="#bbbbbb")

axes, imgs, stars, dots, links, labels = [], [], [], [], [], []
for k, m in enumerate(order):
    r, c = divmod(k, 4)
    ax = fig.add_axes([0.07 + c * 0.235, 0.62 - r * 0.18, 0.17, 0.136])
    ax.set_facecolor(BG)
    imgs.append(ax.imshow(land[m][0], origin="lower", cmap="magma_r", vmin=0, vmax=1,
                          extent=(-0.5, n - 0.5, -0.5, n - 0.5)))
    links.append(ax.plot([], [], color="white", lw=1.5, ls="--")[0])
    stars.append(ax.plot([], [], marker="*", ms=20, color="#00e5ff", mec="black", mew=1.2)[0])
    dots.append(ax.plot([], [], marker="o", ms=11, color="white", mec="black", mew=1.5)[0])
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444444")
    ax.set_title(NAMES[m], fontsize=11.5, pad=4)
    labels.append(ax.text(0.5, -0.08, "", transform=ax.transAxes, ha="center", va="top", fontsize=11))
    axes.append(ax)
legend = fig.text(0.5, 0.025, "x: η (distance penalty)   y: γ (homophily)   ★ true   ● recovered   bright = judged most similar",
         ha="center", fontsize=12.5, color="#999999")

# closing card: recovery error of every measure (hidden until the end)
ax_bar = fig.add_axes([0.33, 0.1, 0.6, 0.7], facecolor=BG, visible=False)
vals = readouts.loc["recovery_wide", order].to_numpy()
bars = ax_bar.barh(range(len(order)), np.zeros(len(order)), color="#00e5ff")
ax_bar.set_yticks(range(len(order)), [NAMES[m] for m in order], fontsize=13)
ax_bar.invert_yaxis()
ax_bar.axvline(chance, color="#ff5a5a", ls="--", lw=2)
ax_bar.text(chance, -1.2, f"chance {chance:.2f}", color="#ff5a5a", ha="center", fontsize=13)
ax_bar.set_xlim(0, max(vals.max(), chance) * 1.08)
ax_bar.set_xlabel("mean recovery error (grid steps, lower is better)", fontsize=14)
for s in ("top", "right"):
    ax_bar.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax_bar.spines[s].set_color("#666666")


def ease(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))


PER = MOVE + HOLD
N_FRAMES = N_TARGETS * PER + FINAL


def frame(f):
    if f >= N_TARGETS * PER:                                        # closing card
        for ax in axes + [ax_net]:
            ax.set_visible(False)
        legend.set_visible(False)
        ax_bar.set_visible(True)
        g = ease((f - N_TARGETS * PER) / (FINAL * 0.4))
        for b, v in zip(bars, vals):
            b.set_width(v * g)
        sub.set_text(f"Over 100 targets, the best measure misses by {vals.min():.1f} grid steps on average "
                     f"(chance: {chance:.2f})")
        return
    i, t = divmod(f, PER)
    a = ease(t / MOVE)
    prev = max(i - 1, 0)
    star = pts[targets[prev]] + a * (pts[targets[i]] - pts[targets[prev]])
    net_img.set_data(rec["targets"][targets[i]])
    eta, gamma = rec["target_eta"][targets[i]], rec["target_gamma"][targets[i]]
    sub.set_text(f"Target {i + 1}/{N_TARGETS}: generated at η = {eta:.2f}, γ = {gamma:.2f}")
    for k, m in enumerate(order):
        imgs[k].set_data((1 - a) * land[m][prev] + a * land[m][i])
        dot = pick[m][prev] + a * (pick[m][i] - pick[m][prev])
        stars[k].set_data([star[0]], [star[1]])
        dots[k].set_data([dot[0]], [dot[1]])
        settled = t >= MOVE
        links[k].set_data(*(([star[0], dot[0]], [star[1], dot[1]]) if settled else ([], [])))
        e = err[m][i]
        labels[k].set_text(f"off by {e:.1f} steps" if settled else "")
        labels[k].set_color("#7CFC00" if e <= 1 else ("#ffcc00" if e <= 3 else "#ff5a5a"))


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, frame, frames=N_FRAMES).save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=6000),
                                                     savefig_kwargs={"facecolor": BG})
    print(f"wrote {OUT} ({N_FRAMES / FPS:.1f} s)")
