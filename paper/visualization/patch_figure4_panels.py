"""
Rebuild the assembled precision/robustness figure of the manuscript.

The composite comes out of a vector editor; this script drops the regenerated
panels D, F and H into it without touching the rest, so the manuscript can be
built before the editor file is updated:

  D  response curves against the EFFECTIVE degeneration (was: attempted swaps)
  F  MAE against that panel's diagonal          (follows from D)
  H  scale-invariant iSNR                       (was: normalised numerator,
                                                 raw denominator)

Sources: run_degeneration_effective_panels.py, run_isnr_consistent_panel.py.

Run:  conda activate ma_thesis && python patch_figure4_panels.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments_config import MANUSCRIPT_FIGURES  # noqa: E402

import fitz

PAPER = MANUSCRIPT_FIGURES
SRC = PAPER / "fig_3_total_variation_degree_distance_2.pdf"     # editor export
DST = PAPER / "fig_3_total_variation_degree_distance_4.pdf"     # what the tex includes
PANELS = Path("output/gnm/"
              "hcp_schaefer_100_dataset")

# (panel pdf, its own axes-frame box, the frame box it has to land on, area to blank)
JOBS = [
    (PANELS / "chaos_analysis" / "fig_panel_D_effective.pdf",
     (32.16, 6.72, 234.24, 128.88), (48.28, 212.61, 249.88, 334.17), (10, 207, 256, 362)),
    (PANELS / "chaos_analysis" / "fig_panel_F_effective.pdf",
     (42.24, 9.36, 158.88, 126.00), (50.40, 388.38, 166.45, 504.42), (6, 384, 190, 530)),   # right edge stops short of panel G's tick labels
    (PANELS / "106_distance_metrics_mst_animal_0_density10" / "fig_panel_H_isnr_consistent.pdf",
     (42.24, 9.36, 158.88, 126.00), (378.15, 388.38, 494.20, 504.42), (338, 384, 505, 530)),
]


def main():
    doc = fitz.open(SRC)
    page = doc[0]
    for _, _, _, blank in JOBS:
        page.draw_rect(fitz.Rect(*blank), color=None, fill=(1, 1, 1), overlay=True)
    for panel, (fx0, fy0, fx1, fy1), (ox0, oy0, ox1, oy1), _ in JOBS:
        src = fitz.open(panel)
        r = src[0].rect
        sx, sy = (ox1 - ox0) / (fx1 - fx0), (oy1 - oy0) / (fy1 - fy0)
        page.show_pdf_page(
            fitz.Rect(ox0 - fx0 * sx, oy0 - fy0 * sy,
                      ox0 + (r.width - fx0) * sx, oy0 + (r.height - fy0) * sy), src, 0)
        print(f"placed {panel.name}")
    doc.save(DST, garbage=3, deflate=True)
    print("wrote", DST)


if __name__ == "__main__":
    main()
