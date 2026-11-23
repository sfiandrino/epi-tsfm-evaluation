#!/usr/bin/env bash
# Reproduce scores, tables and figures of the paper. Run from the repository root.
#   Full rerun (needs the hub forecasts: run 01a_download_hub_forecasts.ipynb in each season first):
#       PY=python3 KERNEL=python3 bash code/paper-figures/run_all.sh
#   Figures and tables only, from the scores included in the repository:
#       PY=python3 KERNEL=python3 bash code/paper-figures/run_all.sh --skip-scoring
set -uo pipefail
PY=${PY:-python3}; KERNEL=${KERNEL:-python3}
SKIP_SCORING=0; [ "${1:-}" = "--skip-scoring" ] && SKIP_SCORING=1
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)

run() {  # notebook, executed in place; prints the error and goes on if it fails
  echo "--- $1"
  out=$($PY -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=$KERNEL --ExecutePreprocessor.timeout=3600 "$1" 2>&1) \
    && echo "ok" || { echo "FAILED"; echo "$out" | grep -E "Error|error" | tail -3; }
}

cd "$ROOT"
if [ $SKIP_SCORING -eq 0 ]; then
  echo "===== eligibility (75% rule)"; $PY code/eligibility/compute_eligibility.py
fi
for s in season-2024-25 season-2025-26; do
  echo "===== $s"; cd "$ROOT/$s/code"
  if [ $SKIP_SCORING -eq 0 ]; then
    for hub in FluSight CovidHub; do
      echo "--- 02 scoring $hub"
      Rscript 02_forecast_evaluation_pairwisecomparison.R --hub_name $hub --baseline_name $hub-baseline > /dev/null 2>&1 \
        && echo ok || echo FAILED
    done
  fi
  run 01d_negative_values.ipynb                 # Figs S8, S9
  run 01e_quantile_corrections.ipynb            # Figs S10, S11
  for nb in 03a_wis_ae_evaluation_table 03b_coverage_evaluation_table; do   # Fig 2, Tables S3-S6
    run $nb.ipynb                               # CovidHub (as saved in the notebook)
    $PY "$HERE/set_challenge.py" $nb.ipynb _${nb}_flu.ipynb FluSight
    run _${nb}_flu.ipynb; rm -f _${nb}_flu.ipynb
  done
  run 05_cleveland_plot_locations.ipynb         # Fig 3, Fig S3
  run 06_heatmap_plot_locations.ipynb           # Figs S1/S2, S4/S5 (metric saved in the notebook)
  $PY "$HERE/swap_metric.py" 06_heatmap_plot_locations.ipynb _06_other.ipynb
  run _06_other.ipynb; rm -f _06_other.ipynb    # the other metric
  echo "--- 07a"; PATH=$(dirname "$(command -v $PY)"):$PATH bash 07a_compute_ranks.sh 2>&1 | tail -1
  run 07b_top_bottom_map_plot.ipynb             # Fig 4, Fig S6
  [ -f 08a_trend_classification_CDC_official.ipynb ] && run 08a_trend_classification_CDC_official.ipynb
  run 08b_trend_combined_panel_plot.ipynb       # Fig 5, Fig S7
  [ -f 08c_compare_CDC_official_adapted_methods.ipynb ] && run 08c_compare_CDC_official_adapted_methods.ipynb   # Fig S12
  [ -f 04_concept_figure_forecasts_panel.py ] && { echo "--- 04"; MPLBACKEND=Agg $PY 04_concept_figure_forecasts_panel.py > /dev/null && echo ok; }  # Fig 1 panel
  # notebooks are stored without outputs
  $PY -m jupyter nbconvert --clear-output --inplace *.ipynb > /dev/null 2>&1
done
cd "$ROOT"
echo "===== paper figures"
$PY code/paper-figures/make_figure2.py
$PY code/paper-figures/paper_figures.py
