"""Figure 2 (main text): aggregated performance of the TSFMs, hub ensemble and baseline (2025/26).

Run from the repository root after 03a/03b. Usage: python code/paper-figures/make_figure2.py [season-folder] [out-basename]
Ranks: relative log WIS / AE ascending; coverage by distance from the nominal level (0.50, 0.90),
rank method "min" (as in the SI tables). Tertiles (Methods 4.3): with M models, each tertile holds
M // 3 models and the remainder goes to the top, then to the middle tertile. Bold = best TSFM per metric.
"""
import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEASON = sys.argv[1] if len(sys.argv) > 1 else "season-2025-26"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("figures", "main", "Figure2_main")
FILES = {"season-2025-26": ("df_evaluation_table_log_{h}.csv", "coverage_table_log_{h}.csv"),
         "season-2024-25": ("df_evaluation_{h}_2024-2025_log.csv", "coverage_{h}_2024-2025_log.csv")}
TSFMS = ["Chronos-2", "Lag-Llama", "Moirai-2", "TabPFN-TS", "TimeGPT", "TimesFM"]
COLORS = {0: "#C9E7F0", 1: "#DEDEDE", 2: "#FBD9B3"}   # top, middle, bottom
METRICS = [("wis_scaled_relative_skill", "r_wis", "Rel. log WIS"),
           ("ae_median_scaled_relative_skill", "r_ae", "Rel. log AE"),
           ("interval_coverage_50", "r_c50", "50% Cov."),
           ("interval_coverage_90", "r_c90", "90% Cov.")]


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def tertile(rank, m):
    base, rem = divmod(m, 3)
    top = base + (1 if rem >= 1 else 0)
    mid = base + (1 if rem >= 2 else 0)
    return 0 if rank <= top else 1 if rank <= top + mid else 2


def load(hub):
    f_w, f_c = FILES[SEASON]
    base = os.path.join(SEASON, "model-evaluation", "evaluation-table")
    df = pd.read_csv(os.path.join(base, f_w.format(h=hub))).merge(
        pd.read_csv(os.path.join(base, f_c.format(h=hub))), on="model", how="left")
    df["r_wis"] = df.wis_scaled_relative_skill.rank(method="min").astype(int)
    df["r_ae"] = df.ae_median_scaled_relative_skill.rank(method="min").astype(int)
    df["r_c50"] = (df.interval_coverage_50 - 0.5).abs().rank(method="min").astype(int)
    df["r_c90"] = (df.interval_coverage_90 - 0.9).abs().rank(method="min").astype(int)
    return df.set_index("model")


def panel(ax, letter, hub):
    df = load(hub)
    m = len(df)
    ax.set_axis_off()
    rows = [("TSFM models", None)] + [(t, t) for t in TSFMS] + [("Hub models", None),
            (f"{hub}-ensemble", f"{hub}-ensemble"), (f"{hub}-baseline", f"{hub}-baseline")]
    n = len(rows) + 2
    h = 1.0 / n
    xs = [0.0, 0.27, 0.45, 0.63, 0.81, 1.0]
    ax.text(0.01, 1 - 0.5 * h, letter, weight="bold", fontsize=15, va="center")
    ax.text(0.05, 1 - 0.5 * h, f"{hub} (N = {m})", fontsize=14, va="center")
    ax.plot([0, 1], [1 - h, 1 - h], color="#9E9E9E", lw=3)
    for j, (_, _, lab) in enumerate(METRICS):
        ax.text(xs[j + 1] + 0.01, 1 - 1.5 * h, lab, weight="bold", fontsize=12, va="center")
    ax.plot([0, 1], [1 - 2 * h, 1 - 2 * h], color="#D0D0D0", lw=2)
    best = {r: df.loc[TSFMS, r].min() for _, r, _ in METRICS}
    for i, (label, model) in enumerate(rows):
        y = 1 - (i + 2.5) * h
        if model is None:
            ax.text(0.01, y, label, style="italic", weight="bold", fontsize=12, va="center")
            continue
        ax.text(0.01, y, label, fontsize=12, va="center")
        for j, (col, rcol, _) in enumerate(METRICS):
            rank = int(df.loc[model, rcol])
            ax.add_patch(plt.Rectangle((xs[j + 1] + 0.002, y - 0.46 * h), xs[j + 2] - xs[j + 1] - 0.004,
                                       0.92 * h, color=COLORS[tertile(rank, m)], lw=0))
            bold = model in TSFMS and rank == best[rcol]
            ax.text(xs[j + 1] + 0.01, y, f"{df.loc[model, col]:.2f} ({ordinal(rank)})", fontsize=12,
                    va="center", weight="bold" if bold else "normal")


fig, axes = plt.subplots(2, 1, figsize=(9, 10.5))
panel(axes[0], "A", "FluSight")
panel(axes[1], "B", "CovidHub")
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("saved", OUT)
