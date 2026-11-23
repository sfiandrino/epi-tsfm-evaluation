import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

font_path      = 'font-folder/Roboto-Light.ttf'
font_path_bold = 'font-folder/Roboto-Bold.ttf'
font_prop         = fm.FontProperties(fname=font_path, size=20)
font_prop_small = fm.FontProperties(fname=font_path, size=18)
font_prop_numbers = fm.FontProperties(fname=font_path, size=16)
font_prop_title = fm.FontProperties(fname=font_path, size=22)

LOCATION = 'US'

tsfm_models = ['Chronos-2', 'Lag-Llama', 'Moirai-2', 'TabPFN-TS', 'TimeGPT', 'TimesFM']
colors_dict = {
    'Chronos-2':  '#4ECED4',
    'Lag-Llama':  '#3E977E',
    'Moirai-2':   '#967ECD',
    'TabPFN-TS':  '#CE5858',
    'TimeGPT':    '#1F77B4',
    'TimesFM':    '#F2B134',
}

folder_to_model = {
    'UMass-flusion': 'UMass-flusion',
    'OHT_JHU-nbxd': 'OHT_JHU-nbxd',
    'Chronos-2': 'Chronos-2',
    'Lag-Llama': 'Lag-Llama',
    'Moirai-2':  'Moirai-2',
    'TabPFN-TS': 'TabPFN-TS',
    'TimeGPT':   'TimeGPT',
    'TimesFM':   'TimesFM',
}

BASE_DIR    = '../model-output/FluSight'
TARGET_FILE = '../target-data/FluSight/surveillance_weekly.csv'
OUTPUT_DIR  = '../model-evaluation/figures/forecasts_visualizations/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_90 = 0.12
ALPHA_50 = 0.28

# Load target data for FluSight
df_target = pd.read_csv(TARGET_FILE)
df_target['truth_date'] = pd.to_datetime(df_target['truth_date'])
df_target['target_week'] = (
    df_target['truth_date'].dt.isocalendar().year.astype(str) + '-' +
    df_target['truth_date'].dt.isocalendar().week.astype(str).str.zfill(2)
)
df_target = df_target.rename(columns={'value': 'obs'})

loc_target = (
    df_target[df_target['location'] == LOCATION]
    .sort_values('truth_date')
    .copy()
)
loc_target = loc_target[loc_target['truth_date'] >= '2025-11-22']
obs_map = loc_target.drop_duplicates('target_week').set_index('target_week')['obs']
print(f'Surveillance: {len(loc_target)} rows')

def load_all_forecasts(folder_name, location=LOCATION):
    folder = os.path.join(BASE_DIR, folder_name)
    files  = sorted([f for f in os.listdir(folder)
                     if (f.endswith('.csv') or f.endswith('.gz'))
                     and '2025-11-15' not in f])
    forecasts = {}
    for fname in files:
        ref_date_str = fname[:10]
        try:
            pd.Timestamp(ref_date_str)
        except Exception:
            continue
        fpath = os.path.join(folder, fname)
        df = pd.read_csv(fpath)
        df = df[df['location'] == location].copy()
        df['value'] = df['value'].clip(lower=0)
        if df.empty:
            continue
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
        df['output_type_id']  = pd.to_numeric(df['output_type_id'], errors='coerce')
        df['value']           = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['output_type_id', 'value', 'target_end_date'])
        if 'target' in df.columns:
            df = df[df['target'].str.contains('flu', case=False, na=False)]
        if df.empty:
            continue
        df['output_type_id'] = df['output_type_id'].round(3)
        df['target_week'] = (
            df['target_end_date'].dt.isocalendar().year.astype(str) + '-' +
            df['target_end_date'].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        df['obs'] = df['target_week'].map(obs_map)
        forecasts[ref_date_str] = df
    print(f'  {folder_name}: {len(forecasts)} submission dates loaded')
    return forecasts


model_to_folder = {v: k for k, v in folder_to_model.items()}
all_forecasts   = {}
for model_display in tsfm_models:
    folder = model_to_folder[model_display]
    all_forecasts[model_display] = load_all_forecasts(folder)

all_ref_dates = sorted(set(
    d for fc in all_forecasts.values() for d in fc.keys()
))
x_start = pd.Timestamp(all_ref_dates[0]) - pd.Timedelta(weeks=1)
x_end   = pd.Timestamp(all_ref_dates[-1]) + pd.Timedelta(weeks=5)

loc_hist_full = loc_target[
    (loc_target['truth_date'] >= x_start) &
    (loc_target['truth_date'] <= x_end)
].copy()

print(f'Date range: {x_start.date()} → {x_end.date()}')

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_q(df_sorted, d, q):
    sub = df_sorted[df_sorted['target_end_date'] == d].copy()
    sub['output_type_id'] = sub['output_type_id'].round(3)
    s   = sub.set_index('output_type_id')['value']
    val = s.get(round(q, 3), s.get(0.5, np.nan))
    return float(val) if val is not None else np.nan

def plot_model_on_ax(ax, model_display, forecasts_dict, color,
                     show_xlabel=False, show_ylabel=False):
    # ── observations ────────────────────────────────────────
    ax.plot(
        loc_hist_full['truth_date'], loc_hist_full['obs'],
        'o-', color='black', linewidth=1.2, markersize=3,
        zorder=10
    )
    
    # ── 4 weeks ahead ─
    for ref_date_str in sorted(forecasts_dict.keys()):
        df       = forecasts_dict[ref_date_str]
        ref_date = pd.Timestamp(ref_date_str)

        df_sorted = df.sort_values('target_end_date')
        dates_fc  = [pd.Timestamp(d)
                     for d in sorted(df_sorted['target_end_date'].unique())]
        if not dates_fc:
            continue

        dates_cone = dates_fc
        med_cone   = [get_q(df_sorted, d, 0.50) for d in dates_fc]
        lo25_cone  = [get_q(df_sorted, d, 0.25) for d in dates_fc]
        hi75_cone  = [get_q(df_sorted, d, 0.75) for d in dates_fc]
        lo05_cone  = [get_q(df_sorted, d, 0.05) for d in dates_fc]
        hi95_cone  = [get_q(df_sorted, d, 0.95) for d in dates_fc]

        ax.fill_between(dates_cone, lo05_cone, hi95_cone,
                        color=color, alpha=ALPHA_90, zorder=2)
        ax.fill_between(dates_cone, lo25_cone, hi75_cone,
                        color=color, alpha=ALPHA_50, zorder=3)
        ax.plot(dates_cone, med_cone,
                '-', color=color, linewidth=0.8, alpha=0.7, zorder=4)

    # ── styling ──────────────────────────────────────────────
    ax.set_xlim(x_start, x_end)
    ax.set_title(model_display, fontproperties=font_prop_title,
                 loc='left', x=0.01, pad=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right',
             fontproperties=font_prop_numbers)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(font_prop_numbers)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}')
    )
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    if show_ylabel:
        pass
    if show_xlabel:
        ax.set_xlabel(' ', fontproperties=font_prop)


# ============================================================
# FIGURE: PANEL  3 cols x 2 rows
# ============================================================
fig, axes = plt.subplots(
    2, 3,
    figsize=(20, 9),
    dpi=300,
    sharey=False,   # each model on its own y scale
    sharex=True
)

for idx, model_display in enumerate(tsfm_models):
    row, col = divmod(idx, 3)
    ax       = axes[row, col]
    color    = colors_dict[model_display]
    plot_model_on_ax(
        ax, model_display,
        all_forecasts[model_display],
        color,
        show_xlabel=(row == 1),
        show_ylabel=(col == 0)
    )

# ── shared legend ────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black', markersize=5, label='Observed data'),
    Line2D([0], [0], color='gray', linewidth=1.0, label='Forecast median'),
    mpatches.Patch(facecolor='gray', alpha=ALPHA_50, label='IQR (25–75%)'),
    mpatches.Patch(facecolor='gray', alpha=ALPHA_90, label='90% CI (5–95%)'),
]
fig.legend(
    handles=legend_handles,
    loc='lower center', ncol=4,
    frameon=False,
    prop=font_prop_small,
    bbox_to_anchor=(0.5, -0.02)
)

fig.supylabel('Weekly hospital admissions (flu)', fontproperties=font_prop, x=0.005)
plt.tight_layout(rect=[0.015, 0.04, 1, 1])

out_base = os.path.join(OUTPUT_DIR, f'concept_figure_all_forecasts_panel_{LOCATION}')
fig.savefig(out_base + '.svg', bbox_inches='tight')
fig.savefig(out_base + '.pdf', bbox_inches='tight')
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')

print(f'Saved: {out_base}')