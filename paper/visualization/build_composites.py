"""
Rebuild the hand-assembled composite figures from the generated panels
======================================================================

Most manuscript figures were arranged in a vector editor: the scripts write
individual panels, and the composite is laid out by hand and renamed. That step
could not be re-run, so after a regeneration the panels were current while the
composites still showed the old data.

This reproduces those composites programmatically. Panel positions were measured
off the published PDFs (ink bounding boxes at 200 dpi), so the layout follows the
originals closely; it is a faithful reconstruction, not a byte-identical one -
see LIMITS at the bottom of this docstring.

    conda activate ma_thesis
    python visualization/build_composites.py            # all of them
    python visualization/build_composites.py --list
    python visualization/build_composites.py --only fig2 pca

LIMITS
  * Panel letters are drawn in Helvetica-Bold (a PDF base-14 font), not the
    IBM Plex Sans of the rest of the manuscript. They will look slightly
    different from the published letters.
  * Source panels are placed preserving their aspect ratio and centred in the
    measured slot, so a panel whose aspect changed between runs sits in the same
    slot but does not fill it identically.
  * Anything the editor drew by hand and no script produces - the divider rule in
    the hub figure, nudged tick labels - is not reproduced.
"""

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments_config import MANUSCRIPT_FIGURES  # noqa: E402
from experiments_config import MORPHO_DIR, ROOT_DIR, TIMING_DIR

PAPER = MANUSCRIPT_FIGURES
HUB = ROOT_DIR / "output" / "hub_topography"
GT = MORPHO_DIR / "ground_truth_analysis"
# Runtime panels come from the reference timing run, not from MORPHO_EXP:
# wall-clock timings move with the machine, so the manuscript quotes one
# measurement throughout (see experiments_config.TIMING_EXP).
TIME_FM = TIMING_DIR / "figures_manuscript"
TIME_GT = TIMING_DIR / "ground_truth_analysis"
FM = MORPHO_DIR / "figures_manuscript"

LABEL_SIZE = 9.0
LABEL_FONT = "hebo"          # Helvetica-Bold; see LIMITS


@dataclass
class Panel:
    src: Path
    rect: tuple                      # (x0, y0, x1, y1) on the output page, points
    clip: tuple | None = None        # crop in source coordinates
    fill: bool = False               # stretch instead of preserving aspect
    autocrop: bool = False           # clip to the source's ink, ignoring padding


def ink_box(pdf: Path, dpi: int = 200) -> tuple:
    """Bounding box of the drawn content, in source points.

    Legends are saved with generous padding, so placing the whole page shrinks
    the content to fit; cropping to the ink first makes the slot mean what it
    looks like it means.
    """
    import numpy as np
    pg = fitz.open(pdf)[0]
    pix = pg.get_pixmap(dpi=dpi)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    mask = a[..., :3].min(axis=2) < 245
    if not mask.any():
        return tuple(pg.rect)
    ys, xs = np.where(mask)
    sc = 72.0 / dpi
    return (xs.min() * sc, ys.min() * sc, (xs.max() + 1) * sc, (ys.max() + 1) * sc)


@dataclass
class Composite:
    name: str
    out: Path
    size: tuple
    panels: list
    labels: list = field(default_factory=list)   # (letter, x, y)


# Slots below are ink bounding boxes measured off the published composites.
COMPOSITES = [
    Composite(
        "fig2", PAPER / "fig_huge_02_landscapes_corr_network_timing_plausible.pdf",
        (510.2, 439.4),
        [Panel(FM / "sixteen_subplots_colored.pdf", (23.4, 0.4, 173.9, 179.6)),
         Panel(FM / "correlation_heatmap_clustered.pdf", (196.6, 0.4, 341.0, 180.7)),
         Panel(FM / "correlation_graph.pdf", (345.0, 18.7, 493.2, 160.2)),
         # panel D is the boxplot variant plus its colour strip, not the violin one
         Panel(TIME_FM / "timing_16_no_cbar.pdf", (0.0, 207.4, 333.4, 353.9)),
         Panel(TIME_FM / "cbar_timing_16.pdf", (62.0, 354.5, 333.4, 369.0)),
         Panel(FM / "sixteen_subplots_colored_with_100_mins.pdf", (345.2, 193.7, 499.0, 374.8)),
         Panel(FM / "method_legend_colored.pdf", (60.0, 389.0, 460.0, 433.0), autocrop=True)],
        [("A", 5.4, 9.0), ("B", 188.0, 9.0), ("C", 348.0, 9.0),
         ("D", 5.4, 202.0), ("E", 348.0, 202.0)],
    ),
    Composite(
        "pca", PAPER / "appendix" / "pca_eight" / "fig_pca_landscapes_loadings_2.pdf",
        (510.2, 234.5),
        [Panel(PAPER / "appendix" / "pca_eight" / "fig_pca_scree.pdf", (12.6, 8.0, 174.6, 82.8)),
         Panel(PAPER / "appendix" / "pca_eight" / "fig_pca_landscapes.pdf", (22.9, 92.9, 174.6, 151.6)),
         Panel(PAPER / "appendix" / "pca_eight" / "fig_pca_loadings_strip.pdf",
               (192.2, 10.4, 198.7, 142.6), fill=True, autocrop=True),
         Panel(PAPER / "appendix" / "pca_eight" / "fig_pca_loadings_pc1.pdf", (200.0, 8.0, 334.4, 163.1)),
         Panel(PAPER / "appendix" / "pca_eight" / "fig_pca_loadings_pc2.pdf", (348.5, 8.0, 491.0, 163.1)),
         Panel(PAPER / "legend_8_measures.pdf", (33.8, 185.0, 475.6, 205.9), autocrop=True)],
        [("A", 60.0, 9.0), ("B", 185.0, 9.0), ("C", 342.0, 9.0)],
    ),
    Composite(
        "hubs", PAPER / "appendix" / "hubs_landscapes_violins_2.pdf",
        (461.0, 424.4),
        [Panel(HUB / "fig_hub_topography_maps.pdf", (0.0, 25.6, 461.0, 205.6)),
         Panel(HUB / "fig_hub_topography_landscape.pdf", (6.1, 222.1, 167.4, 366.1)),
         # the distributions panel carries three violins; the composite uses the
         # first two, so crop to the left block measured at x < 335
         Panel(HUB / "fig_hub_topography_distributions.pdf", (182.2, 222.1, 457.9, 366.1),
               clip=(0.0, 0.0, 335.0, 177.4)),
         Panel(PAPER / "legend_8_measures.pdf", (60.0, 389.2, 420.0, 409.0), autocrop=True)],
        [("A", 2.0, 9.0), ("B", 2.0, 218.0), ("C", 176.0, 218.0), ("D", 326.0, 218.0)],
    ),
]

# Composites that turned out to be plain renames of a generated panel: the
# published file and the script output have identical page geometry.
COPIES = [
    (GT / "components_of_ks_energy_grayscale.pdf",
     PAPER / "appendix" / "components_of_ks_energy_grayscale.pdf"),
    (GT / "method_legend.pdf",
     PAPER / "appendix" / "method_legend.pdf"),
    (ROOT / "figures" / "appendix" / "fig_rewiring_robustness.pdf",
     PAPER / "appendix" / "drift_of_recovered_parameters.pdf"),
    (HUB / "fig_hub_topography_sa.pdf",
     PAPER / "appendix" / "hub_topography_sa_axis.pdf"),
    (TIME_GT / "similarity_methods_timing_comparison.pdf",
     PAPER / "appendix" / "similarity_methods_timing_comparison.pdf"),
    (ROOT_DIR / "output" / "cost_hemisphere_landscapes" / "fig_cost_hemisphere_landscapes.pdf",
     PAPER / "appendix" / "fig_cost_hemisphere_landscapes.pdf"),
    # these two write into this repo's own figures/ rather than the manuscript's
    (ROOT / "figures" / "legend_8_measures.pdf", PAPER / "legend_8_measures.pdf"),
    (ROOT / "figures" / "appendix" / "grid_and_variance.pdf",
     PAPER / "appendix" / "grid_and_variance.pdf"),
]


def place(page, panel: Panel) -> None:
    src = fitz.open(panel.src)
    sp = src[0]
    clip = panel.clip
    if panel.autocrop and clip is None:
        clip = ink_box(panel.src)
    box = fitz.Rect(*clip) if clip else sp.rect
    x0, y0, x1, y1 = panel.rect
    target = fitz.Rect(x0, y0, x1, y1)
    if not panel.fill:
        s = min(target.width / box.width, target.height / box.height)
        w, h = box.width * s, box.height * s
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        target = fitz.Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    page.show_pdf_page(target, src, 0, clip=fitz.Rect(*clip) if clip else None)


def build(c: Composite) -> tuple[bool, list]:
    missing = [p.src for p in c.panels if not p.src.exists()]
    if missing:
        return False, missing
    doc = fitz.open()
    page = doc.new_page(width=c.size[0], height=c.size[1])
    for p in c.panels:
        place(page, p)
    for letter, x, y in c.labels:
        page.insert_text((x, y), letter, fontname=LABEL_FONT,
                         fontsize=LABEL_SIZE, color=(0.137, 0.137, 0.141))
    c.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(c.out, garbage=3, deflate=True)
    return True, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-copies", action="store_true")
    args = ap.parse_args()

    todo = [c for c in COMPOSITES if not args.only or c.name in args.only]
    if args.list:
        for c in COMPOSITES:
            print(f"  {c.name:<6} {c.size[0]:6.1f} x {c.size[1]:6.1f}  {len(c.panels)} panels -> {c.out.name}")
        for s, d in COPIES:
            print(f"  copy   {d.name}  <- {s.name}")
        return

    skipped = []
    for c in todo:
        ok, missing = build(c)
        if ok:
            print(f"  built  {c.out.name}")
        else:
            skipped.append(c.out.name)
            print(f"  SKIP   {c.out.name} - missing: {[m.name for m in missing]}")

    if not args.no_copies and not args.only:
        for src, dst in COPIES:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  copied {dst.name}")
            else:
                skipped.append(dst.name)
                print(f"  SKIP   {dst.name} - no {src}")

    if skipped:
        # a missing panel must not pass as success: the composite silently keeps
        # whatever version was there before, which is how a stale figure survives
        raise SystemExit(f"{len(skipped)} composite(s) not rebuilt: {', '.join(skipped)}")

if __name__ == "__main__":
    main()
