#!/usr/bin/env bash
# Per-forecast ranks used by 07b_top_bottom_map_plot.ipynb. Run from code/.
set -euo pipefail
S=../model-evaluation/forecast_scores
python3 preprocessing.py --input $S/forecast_scores_wis_ae_coverage_log_FluSight.csv --output $S/flusight_preprocessed.csv
python3 scoring.py --input $S/flusight_preprocessed.csv --output $S/flusight_all_ranks.csv --output_horizon $S/flusight_ranks_by_horizon.csv --output_location $S/flusight_ranks_by_location.csv
python3 preprocessing.py --input $S/forecast_scores_wis_ae_coverage_log_CovidHub.csv --output $S/covidhub_preprocessed.csv
python3 scoring.py --input $S/covidhub_preprocessed.csv --output $S/covidhub_all_ranks.csv --output_horizon $S/covidhub_ranks_by_horizon.csv --output_location $S/covidhub_ranks_by_location.csv
