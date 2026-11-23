# Are time series foundation models the new frontier for operational epidemic forecasting?

Forecasts, code and evaluation results for the manuscript *"Are time series foundation models the new frontier for operational epidemic forecasting?"*

We evaluate six time series foundation models (TSFMs): Chronos-2, Lag-Llama, Moirai-2, TabPFN-TS, TimeGPT and TimesFM. They forecast weekly laboratory-confirmed influenza (FluSight) and COVID-19 (CovidHub) hospital admissions for the U.S. national level and 52 U.S. jurisdictions, 0 to 3 weeks ahead, as 23 quantiles. The TSFMs are benchmarked against the models submitted to the two CDC-coordinated hubs. The 2025/26 season is a prospective evaluation (main text), and the 2024/25 season is a retrospective evaluation (Supplementary Information, Sec. S3).

## Repository structure

```
code/
  forecast-generation/     code that produced the weekly TSFM forecasts (+ the GitHub Actions that ran it)
  eligibility/             75% eligibility rule for the hub models (script and result)
  paper-figures/           run_all.sh (full pipeline) and the scripts that assemble the paper figures
figures/
  main/                    Figures 1-5
  supplementary/           Figures S1-S12
season-2025-26/            prospective evaluation (main text)
season-2024-25/            retrospective evaluation (Supplementary Information, Sec. S3)
  code/                    evaluation scripts, run in order from this folder (see below)
  model-output/<hub>/<TSFM>/   TSFM forecasts, one hub-format file per round: <reference_date>-<model>.csv
  model-metadata/          TSFM metadata (2025-26)
  target-data/<hub>/       surveillance_weekly.csv (evaluation data) and snapshots/ (versioned model inputs)
  model-evaluation/        scores, evaluation tables and figure panels produced by the scripts
  auxiliary-files/         location codes and populations
```

## Evaluation rounds and date convention

- **2025/26:** 27 weekly rounds, November 22, 2025 – May 23, 2026, for both hubs.
- **2024/25:** 25 weekly rounds, November 30, 2024 – May 24, 2025, for both hubs (no hub round on 2025-01-18).
- **Date convention.** Files are named after the last week of data (`reference_date` = last `truth_date`; horizon 0 = the following week). The hubs use the following Saturday as reference date, so the hub forecasts are shifted back by 7 days when downloaded (`01a`). For example, the TSFM round `2026-04-25` uses the data released on Wednesday 2026-04-29 and corresponds to the hub round with reference date 2026-05-02.
- Only official hub rounds are scored (reference dates with a hub ensemble).

## TSFM forecasts and inputs

`season-*/model-output/<hub>/<model>/` contains one quantile file per round in the hub format: 53 locations × 4 horizons × 23 quantiles.

The input series given to the models each week are in `season-2025-26/target-data/<hub>/snapshots/`:
- `surveillance_weekly_up_to_<date>.csv` is the series as reported.
- `surveillance_weekly_up_to_<date>_interpolated.csv` has missing values filled by linear interpolation within each location's series. These are the inputs of Moirai-2, TabPFN-TS and TimeGPT, which need a gap-free sequence.

For 2024/25 the snapshots are the hub target-data versions available at each round.

`target-data/<hub>/surveillance_weekly.csv` is the series used for evaluation and trend classification (latest NHSN data).

## Hub forecasts

The forecasts submitted to FluSight and CovidHub are not included. They are public in [cdcepi/FluSight-forecast-hub](https://github.com/cdcepi/FluSight-forecast-hub) and [CDCgov/covid19-forecast-hub](https://github.com/CDCgov/covid19-forecast-hub). `season-*/code/01a_download_hub_forecasts.ipynb` downloads them into `model-output/` (they are ignored by git). You only need them to rerun the scoring (`02`) and the eligibility check.

## Reproducing the results

**A. Figures and tables from the included scores** (no download needed):
```
PY=python3 KERNEL=python3 bash code/paper-figures/run_all.sh --skip-scoring
```

**B. Full rerun:**
1. Run `01a_download_hub_forecasts.ipynb` in `season-2024-25/code/` and `season-2025-26/code/`.
2. Run `PY=python3 KERNEL=python3 bash code/paper-figures/run_all.sh`.

This computes eligibility, runs the R scoring (`02`, about 1 hour), all notebooks, and the paper figures.

| Script (in `season-*/code/`) | What it does | Paper |
|---|---|---|
| `01a_download_hub_forecasts.ipynb` | Downloads the hub forecasts of the evaluation rounds | – |
| `01b_prepare_covid19_target_data.ipynb`, `01c_save_snapshots_covid_data.R` | Build the versioned COVID-19 target data from the Delphi API files | – |
| `01d_negative_values.ipynb` | Negative quantile values produced by each TSFM | Figs S8, S9 |
| `01e_quantile_corrections.ipynb` | Quantile monotonicity corrections | Figs S10, S11 |
| `02_forecast_evaluation_pairwisecomparison.R` | WIS, AE, coverage on the log scale and pairwise relative scores (eligible hub models + TSFMs) | input to all results |
| `03a_wis_ae_evaluation_table.ipynb`, `03b_coverage_evaluation_table.ipynb` | Aggregated evaluation tables | Fig. 2, Tables S3–S6 |
| `04_concept_figure_forecasts_panel.py` (2025-26) | End-of-season forecast panel of the overview figure | Fig. 1 |
| `05_cleveland_plot_locations.ipynb` | Performance across jurisdictions | Fig. 3, Fig. S3 |
| `06_heatmap_plot_locations.ipynb` | Relative log WIS or AE by jurisdiction | Figs S1, S2, S4, S5 |
| `07a_compute_ranks.sh` (uses `preprocessing.py`, `scoring.py`), `07b_top_bottom_map_plot.ipynb` | Per-forecast ranks and top/bottom 30% maps | Fig. 4, Fig. S6 |
| `08a_trend_classification_CDC_official.ipynb` (2025-26) | Trend classification with the official CDC thresholds | Sec. S5 |
| `08b_trend_combined_panel_plot.ipynb` | Adapted trend classification and performance by epidemic phase | Fig. 5, Fig. S7 |
| `08c_compare_CDC_official_adapted_methods.ipynb` (2025-26) | Agreement between the official and adapted classifications | Fig. S12 |

`code/paper-figures/make_figure2.py` and `paper_figures.py` assemble the final multi-panel figures in `figures/`. They reuse the notebooks' data and drawing functions. The infographic of Figure 1 was assembled in a graphics program around the panel produced by `04`.

## Eligibility of hub models

`code/eligibility/compute_eligibility.py` (run from the repository root, after `01a`) computes, for every hub model, the share of all possible forecast instances it submitted. An instance is an official round × 53 locations × horizons 0–3, with all 23 quantiles. Models with at least 75% are evaluated, together with the hub ensemble and baseline. The result is `code/eligibility/eligibility_75pct.csv`, which is read by `02`. Details are in `code/eligibility/README.md`.

## Forecast generation

`code/forecast-generation/` contains the code that produced the weekly TSFM forecasts. Run it from `season-2025-26/`:
```
python ../code/forecast-generation/data_import.py --target totalconfflunewadm          # or totalconfc19newadm
python ../code/forecast-generation/run_forecasts.py --target totalconfflunewadm --model_name Chronos-2
```

- `data_import.py` downloads the NHSN data from data.cdc.gov and saves the series as reported and interpolated, plus a dated snapshot of each.
- `run_forecasts.py` gives each model its input series (interpolated for Moirai-2, TabPFN-TS and TimeGPT) and writes `model-output/<hub>/<model>/<reference_date>-<model>.csv`.
- `reinterpolate.py` regenerates the interpolated series from the as-reported ones.

Install `code/forecast-generation/requirements.txt` and the package of each model: `chronos-forecasting` v2.2.1, `uni2ts`, `tabpfn-time-series` v1.0.7, `nixtla`, `timesfm`, `lag-llama`. TimeGPT and TabPFN-TS need API tokens in the `TIMEGPT_ACCESS_TOKEN` and `TABPFN_ACCESS_TOKEN` environment variables.

The GitHub Actions that ran these scripts every Wednesday are in `code/forecast-generation/github-workflows/`. They are not active in this repository.

## Requirements

- Python ≥ 3.9: `pip install -r requirements.txt`
- R with `scoringutils`, `hubData`, `dplyr`, `optparse`, `readr`, `purrr`, `stringr`, `tidyr`, `zoo`, `arrow`, `data.table`, `MMWRweek`, `epidatr`

## Data sources

- Target data: CDC Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction, National Healthcare Safety Network (NHSN); COVID-19 versions from the Delphi Epidata API.
- Hub forecasts: CDC FluSight and COVID-19 Forecast Hub repositories (see above).

## License

MIT (see `LICENSE`).
