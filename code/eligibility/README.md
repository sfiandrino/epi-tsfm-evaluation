# Eligibility (75% rule)

Run from the repository root, after downloading the hub forecasts (`01a` in each season): `python code/eligibility/compute_eligibility.py` (needs pandas and pyarrow). Output: `code/eligibility/eligibility_75pct.csv`, read by `season-*/code/02_forecast_evaluation_pairwisecomparison.R`.

- Forecast instance: (reference date, location, horizon) for the incident hospitalization target, horizons 0–3, U.S. national + 52 jurisdictions, with all 23 quantiles.
- Evaluation window: 2024/25 = 2024-11-30 to 2025-05-24; 2025/26 = 2025-11-22 to 2026-05-23 (last FluSight round; CovidHub cut at the same date).
- Strict rule. Denominator = official hub rounds in the window × 53 locations × 4 horizons. Official rounds = reference dates with a hub ensemble: 25 in 2024/25 for both hubs (no round on 2025-01-18), 27 in 2025/26 for both hubs.
- Eligible: submitted ≥ 75% of the denominator. Hub-generated variants (e.g. FluSight-lop_norm, FluSight-trained_mean) are listed but never eligible; only the hub ensemble and baseline are kept.
- Both csv and parquet submissions are read.

Output `eligibility_75pct.csv`: season, hub, model, hub_variant, rounds, total_rounds, pct_rounds, locations, instances, expected_instances, pct_instances, eligible.
