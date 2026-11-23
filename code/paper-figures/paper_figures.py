"""Paper-ready multi-panel figures (layout of the CDC clearance round 2 figures), built from the notebooks' own
data-loading and drawing functions. Run from the repository root (after the notebooks):
    python code/paper-figures/paper_figures.py [fig3 fig4 fig5 copy ...]      (default: all)
Outputs: figures/main/<FigureN_main>.pdf|png and figures/supplementary/<FigureSN_supplementary>.pdf|png.
"copy" collects the figures that come straight from the notebooks (Fig 1 panel, S1, S2, S4, S5, S8-S12).
"""
import ast, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.font_manager as fm

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = {"main": os.path.join(ROOT, "figures", "main"), "supplementary": os.path.join(ROOT, "figures", "supplementary")}
for _d in OUT_DIR.values():
    os.makedirs(_d, exist_ok=True)


def out_path(name):
    return OUT_DIR["main" if name.endswith("_main") or name.startswith("Figure1") else "supplementary"]
SEASONS = {"2526": "season-2025-26", "2425": "season-2024-25"}
NAMES = {  # (season, figure) -> output name
    ("2526", "fig3"): "Figure3_main", ("2425", "fig3"): "FigureS3_supplementary",
    ("2526", "fig4"): "Figure4_main", ("2425", "fig4"): "FigureS6_supplementary",
    ("2526", "fig5"): "Figure5_main", ("2425", "fig5"): "FigureS7_supplementary",
}


def code_cells(nb):
    return ["".join(c["source"]) for c in json.load(open(nb))["cells"] if c["cell_type"] == "code"]


def defs_only(src):
    """Function definitions and simple assignments of a cell (no plotting / saving at top level)."""
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    return compile(ast.Module(body=keep, type_ignores=[]), "defs", "exec")


def fonts(code_dir):
    fp = os.path.join(code_dir, "font-folder")
    return {k: fm.FontProperties(fname=os.path.join(fp, f), size=s) for k, f, s in
            [("lab", "Roboto-Light.ttf", 15), ("small", "Roboto-Light.ttf", 13), ("tick", "Roboto-Light.ttf", 12),
             ("title", "Roboto-Bold.ttf", 16), ("letter", "Roboto-Bold.ttf", 24), ("hub", "Roboto-Bold.ttf", 20),
             ("leg", "Roboto-Light.ttf", 15)]}


def letter(fig, ax, s, F, dx=-0.055, dy=0.035):
    p = ax.get_position()
    fig.text(p.x0 + dx, p.y1 + dy, s, fontproperties=F["letter"], va="bottom", ha="left")


def boxed_legend(fig, handles, labels, F, y=0.012, ncol=None):
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
                     frameon=True, fancybox=True, prop=F["leg"], handletextpad=0.5, columnspacing=1.6,
                     borderpad=0.7)
    leg.get_frame().set_edgecolor("#BBBBBB"); leg.get_frame().set_linewidth(1.0)
    return leg


def sep(fig, x0, y0, x1, y1):
    fig.add_artist(Line2D([x0, x1], [y0, y1], transform=fig.transFigure, color="#CCCCCC", lw=2))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_path(name), f"{name}.{ext}"), dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Figure 3 / S3 (cleveland + N wins)
def fig3(tag):
    code = os.path.join(ROOT, SEASONS[tag], "code")
    os.chdir(code)
    ns = {}
    cells = code_cells("05_cleveland_plot_locations.ipynb")
    for i in range(3):
        exec(cells[i], ns)
    exec(defs_only(cells[3]), ns)
    F = fonts(code)
    ns["font_prop"], ns["font_prop_small"], ns["font_prop_title"] = F["lab"], F["tick"], F["title"]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 10.5))
    gs = GridSpec(2, 5, width_ratios=[4, 1.1, 1.45, 4, 1.1], wspace=0.05, hspace=0.42,
                  left=0.085, right=0.985, top=0.93, bottom=0.12)
    panels = [("A", "FluSight", "WIS", 0, 0), ("B", "FluSight", "AE", 0, 3),
              ("C", "CovidHub", "WIS", 1, 0), ("D", "CovidHub", "AE", 1, 3)]
    col = {"WIS": "wis_scaled_relative_skill", "AE": "ae_median_scaled_relative_skill"}
    data = {(h, m): (ns[f"df_{'flu' if h == 'FluSight' else 'covid'}_{m.lower()}"],
                     ns[f"order_{'flu' if h == 'FluSight' else 'covid'}_{m.lower()}"]) for _, h, m, _, _ in panels}
    axes, bars = [], []
    for L, hub, m, r, c in panels:
        ax, axb = fig.add_subplot(gs[r, c]), fig.add_subplot(gs[r, c + 1])
        df, order = data[(hub, m)]
        ns["plot_cleveland"](ax, df, order, f"{hub} – {m}", col[m])
        ns["plot_win_bar_h"](axb, df, order, col[m])
        ax.set_title(f"{hub} – {m}", fontproperties=F["title"], loc="left", x=0.0, pad=12)
        ax.set_xlabel("Relative score (log)", fontproperties=F["lab"])
        axes.append(ax); bars.append(axb)
    xmin = min(a.get_xlim()[0] for a in axes); xmax = max(a.get_xlim()[1] for a in axes)
    wmax = 1.22 * max(max(p.get_width() for p in b.patches) for b in bars)
    for a in axes: a.set_xlim(xmin, xmax)
    for b in bars: b.set_xlim(0, wmax)
    for (L, *_), ax in zip(panels, axes):
        letter(fig, ax, L, F, dx=-0.085, dy=0.012)
    # separators
    pA, pB, pC = axes[0].get_position(), axes[1].get_position(), axes[2].get_position()
    xmid = bars[0].get_position().x1 + 0.018
    ymid = (pC.y1 + pA.y0) / 2 + 0.005
    sep(fig, xmid, 0.075, xmid, 0.98)
    sep(fig, 0.02, ymid, xmid - 0.01, ymid); sep(fig, xmid + 0.01, ymid, 0.99, ymid)
    g = "#808080"
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=g, markersize=8),
         Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor=g, markersize=9),
         Line2D([0], [0], marker="D", color="w", markerfacecolor=g, markersize=8),
         Line2D([0], [0], color=g, lw=1, ls="--")]
    boxed_legend(fig, h, ["US states", "US national", "Median", "hub-baseline (1.0)"], F, y=-0.035)
    save(fig, NAMES[(tag, "fig3")])



# ---------------------------------------------------------------- Figure 4 / S6 (top/bottom 30% maps)
def fig4(tag):
    code = os.path.join(ROOT, SEASONS[tag], "code")
    os.chdir(code)
    ns = {}
    cells = code_cells("07b_top_bottom_map_plot.ipynb")
    exec(cells[0] + "\n" + cells[1], ns)
    exec(defs_only(cells[2]), ns)
    F = fonts(code)
    ns["font_prop"], ns["font_prop_small"], ns["font_prop_title"] = F["lab"], F["small"], F["title"]
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    flu, cov = pd.read_csv(ns["FLU_FILE"]), pd.read_csv(ns["COVID_FILE"])
    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 2, wspace=0.22, hspace=0.32, left=0.08, right=0.98, top=0.95, bottom=0.12)
    panels = [("A", "FluSight", "WIS", flu, "wis"), ("B", "FluSight", "AE", flu, "ae"),
              ("C", "CovidHub", "WIS", cov, "wis"), ("D", "CovidHub", "AE", cov, "ae")]
    axes = []
    for k, (L, hub, m, d, col) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ns["plot_panel"](ax, ns["compute_summary"](d, col, 0.30), f"{hub} \u2013 {m}", hub, 0.30,
                         xlabel=k >= 2, ylabel=k % 2 == 0)
        ax.set_title(f"{hub} \u2013 {m}", fontproperties=F["title"], loc="left", x=0.0, pad=10)
        if k >= 2:
            ax.set_xlabel("% of forecasts in the bottom 30%", fontproperties=F["lab"])
        if k % 2 == 0:
            ax.set_ylabel("% of forecasts in the top 30%", fontproperties=F["lab"])
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(F["lab"])
        axes.append(ax)
    for (L, *_), ax in zip(panels, axes):
        letter(fig, ax, L, F, dx=-0.055, dy=0.008)
    pA, pB, pC = axes[0].get_position(), axes[1].get_position(), axes[2].get_position()
    xmid = (pA.x1 + pB.x0) / 2 - 0.012
    ymid = (pC.y1 + pA.y0) / 2 + 0.01
    sep(fig, xmid, 0.07, xmid, 0.98)
    sep(fig, 0.02, ymid, xmid - 0.012, ymid); sep(fig, xmid + 0.012, ymid, 0.99, ymid)
    colors = ns["get_model_colors"]("FluSight")
    order = ["TimeGPT", "TimesFM", "Chronos-2", "TabPFN-TS", "Moirai-2", "Lag-Llama", "FluSight-baseline", "FluSight-ensemble"]
    h = [Line2D([0], [0], marker=ns["get_model_marker"](m, "FluSight"), color="w", markerfacecolor=colors[m],
                markeredgecolor="white", markersize=10) for m in order]
    h.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markeredgecolor="gray", markersize=10))
    labs = [ns["get_display_name"](m, "FluSight") for m in order] + ["Other models"]
    boxed_legend(fig, h, labs, F, y=-0.005)
    save(fig, NAMES[(tag, "fig4")])


# ---------------------------------------------------------------- Figure 5 / S7 (epidemic phases)
def fig5(tag):
    code = os.path.join(ROOT, SEASONS[tag], "code")
    os.chdir(code)
    ns = {}
    cells = code_cells("08b_trend_combined_panel_plot.ipynb")
    exec("import matplotlib\nmatplotlib.use('Agg')\n" + cells[0], ns)
    for i in (1, 2, 3, 6):
        exec(cells[i], ns)
    exec(defs_only(cells[7]), ns)
    F = fonts(code)
    ns["font_prop"], ns["font_prop_title"] = F["small"], F["title"]
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    cats = ["large_decrease", "decrease", "stable", "increase", "large_increase"]
    fig = plt.figure(figsize=(20, 11.5))
    outer = GridSpec(1, 2, wspace=0.27, left=0.065, right=0.962, top=0.90, bottom=0.17)
    hubs = [("FluSight", "Flu epidemic dynamic", ns["pivot_flu"], ns["df_us_flu"], "ABCD"),
            ("CovidHub", "COVID-19 epidemic dynamic", ns["pivot_covid"], ns["df_us_covid"], "BEF")]
    letters_ax = []
    for k, (hub, ttl, pivot, df_us, _) in enumerate(hubs):
        g = GridSpecFromSubplotSpec(4, 2, subplot_spec=outer[0, k], height_ratios=[1.25, 0.95, 0.62, 1.75],
                                    hspace=0.08, wspace=0.42)
        ax_bar = fig.add_subplot(g[0, :]); ax_line = fig.add_subplot(g[1, :], sharex=ax_bar)
        n_before = len(fig.axes)
        ns["plot_trend_panel"](ax_bar, ax_line, pivot, df_us, ttl, ns["surv"][hub], add_legend=False)
        ax2 = fig.axes[n_before]          # twin axis with the state curves
        for a in (ax_line, ax2):
            for t in list(a.texts):
                t.remove()
        ax_line.text(0.02, 0.88, "US national", transform=ax_line.transAxes, fontproperties=F["small"], color="#2C3E50", va="top")
        ax_line.text(0.48, 0.62, "US states", transform=ax_line.transAxes, fontproperties=F["small"], color="#AAAAAA", va="top")
        ax_bar.set_title(ttl, fontproperties=F["title"], loc="left", pad=8)
        pos_line = ax_line.get_position()
        gc = GridSpecFromSubplotSpec(1, 2, subplot_spec=g[3, :], wspace=0.42)
        axC, axD = fig.add_subplot(gc[0, 0]), fig.add_subplot(gc[0, 1])
        ns["plot_cleveland_panel"](axC, hub, "wis_relative", "WIS")
        ns["plot_cleveland_panel"](axD, hub, "ae_median_relative", "AE")
        for a in (axC, axD):
            a.set_xlabel("Median ratio (log)", fontproperties=F["small"])
            a.set_title(a.get_title(loc="left"), fontproperties=F["title"], loc="left", pad=8)
        if k == 1:
            axC.set_yticklabels([])
        axD.set_yticklabels([])
        letters_ax += [(ax_bar, "AB"[k]), (axC, "CE"[k]), (axD, "DF"[k])]
    # same x range on the four dot panels, wide enough for every point
    import numpy as np
    dots = [a for a in fig.axes if a.get_title(loc="left") in ("WIS", "AE")]
    xs = np.concatenate([c.get_offsets()[:, 0] for a in dots for c in a.collections if len(c.get_offsets())])
    lo, hi = min(0.42, xs.min() / 1.08), max(3.2, xs.max() * 1.08)
    import matplotlib.ticker as mt
    ticks = [t for t in [0.5, 0.75, 1, 1.5, 2, 3, 4, 5] if lo <= t <= hi]
    for a in dots:
        a.set_xlim(lo, hi)
        a.xaxis.set_major_locator(mt.FixedLocator(ticks))
        for lbl in a.get_xticklabels():
            lbl.set_fontproperties(F["small"])
    fig.canvas.draw()
    boxes = []
    for k in range(len(hubs)):
        cs = outer[0, k].get_position(fig)
        boxes.append((cs.x0 - 0.052, cs.x1 + 0.033))
    # keep the y tick labels of panel C (the only dot panel with labels) inside its box
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    axC0 = letters_ax[1][0]
    left = inv.transform(axC0.get_tightbbox(rend).p0)[0]
    need = boxes[0][0] + 0.012 - left
    if need > 0:
        p = axC0.get_position()
        axC0.set_position([p.x0 + need, p.y0, p.width - need, p.height])
        fig.canvas.draw()
    for a, L in letters_ax:
        if L in "AC":
            p = a.get_position(); fig.text(boxes[0][0] + 0.012, p.y1 + 0.006, L, fontproperties=F["letter"], va="bottom")
        elif L in "BE":
            p = a.get_position(); fig.text(boxes[1][0] + 0.012, p.y1 + 0.006, L, fontproperties=F["letter"], va="bottom")
        else:
            letter(fig, a, L, F, dx=-0.03, dy=0.006)
    # rounded boxes and hub headers
    for k, (hub, *_rest) in enumerate(hubs):
        x0, x1 = boxes[k]
        box = FancyBboxPatch((x0, 0.115), x1 - x0, 0.84, boxstyle="round,pad=0,rounding_size=0.025",
                             transform=fig.transFigure, fill=False, edgecolor="#CCCCCC", lw=2.2)
        fig.add_artist(box)
        fig.text((x0 + x1) / 2, 0.97, hub, ha="center", va="bottom", fontproperties=F["hub"])
        ysep = [a for a in fig.axes if a.get_title(loc="left") in ("WIS", "AE")][0].get_position().y1 + 0.045
        sep(fig, x0 + 0.02, ysep, x1 - 0.02, ysep)
    cc = ns["cat_colors"]
    from matplotlib.legend_handler import HandlerTuple
    h = [(Patch(facecolor=cc[c]), Line2D([0], [0], marker="o", color="w", markerfacecolor=cc[c], markersize=10))
         for c in cats] + [Line2D([0], [0], ls="--", color="#888888")]
    labs = [c.replace("_", " ") for c in cats] + ["Baseline (1.0)"]
    leg = fig.legend(h, labs, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=6, frameon=True, fancybox=True,
                     prop=F["leg"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=3.2,
                     columnspacing=1.8, borderpad=0.7)
    leg.get_frame().set_edgecolor("#BBBBBB")
    save(fig, NAMES[(tag, "fig5")])

# ---------------------------------------------------------------- figures taken as produced by the notebooks
COPIES = [  # (season, path under model-evaluation/figures, output name)
    ("season-2025-26", "forecasts_visualizations/concept_figure_all_forecasts_panel_US", "Figure1_panel_end_of_season"),
    ("season-2025-26", "plot_locations/heatmap_locations_WIS_log", "FigureS1_supplementary"),
    ("season-2025-26", "plot_locations/heatmap_locations_AE_log", "FigureS2_supplementary"),
    ("season-2024-25", "plot_locations/heatmap_locations_WIS_log_2024-2025", "FigureS4_supplementary"),
    ("season-2024-25", "plot_locations/heatmap_locations_AE_log_2024-2025", "FigureS5_supplementary"),
    ("season-2025-26", "negative_values_summary_combined", "FigureS8_supplementary"),
    ("season-2024-25", "negative_values_summary_combined", "FigureS9_supplementary"),
    ("season-2025-26", "monothonicity-violation-quantiles_combined", "FigureS10_supplementary"),
    ("season-2024-25", "monothonicity-violation-quantiles_combined", "FigureS11_supplementary"),
    ("season-2025-26", "trend_classification_2x2", "FigureS12_supplementary"),
]


def copy():
    import shutil
    for season, src, name in COPIES:
        for ext in ("pdf", "png", "svg"):
            f = os.path.join(ROOT, season, "model-evaluation", "figures", f"{src}.{ext}")
            if os.path.exists(f):
                shutil.copy(f, os.path.join(out_path(name), f"{name}.{ext}"))
        print("copied", name)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig3", "fig4", "fig5", "copy"]
    for f in todo:
        if f == "copy":
            copy()
            continue
        for tag in ("2526", "2425"):
            globals()[f](tag)
