"""Produce one week of TSFM forecasts in the FluSight / COVID-19 Forecast Hub format.

Run from the season folder, e.g.
    python ../forecast-generation/run_forecasts.py --target totalconfflunewadm --model_name Chronos-2
Output: model-output/<hub>/<model>/<reference_date>-<model>.csv
"""
from utils.plot_utils import plot_forecast
import argparse
import pandas as pd
import numpy as np
import os

HUB = {
    "totalconfflunewadm": "FluSight",
    "totalconfc19newadm": "CovidHub",
}

# Input series for each model: models that need a gap-free sequence use the interpolated series
INPUT_FILE = {
    "Chronos-2": "surveillance_weekly.csv",
    "Lag-Llama": "surveillance_weekly.csv",
    "TimesFM":   "surveillance_weekly.csv",
    "Moirai-2":  "surveillance_weekly_interpolated.csv",
    "TabPFN-TS": "surveillance_weekly_interpolated.csv",
    "TimeGPT":   "surveillance_weekly_interpolated.csv",
}


def load_model(model_name):
    if model_name == "Chronos-2":
        from utils.chronos_utils import chronos_forecasting as f
    elif model_name == "Lag-Llama":
        from utils.lagllama_utils import lagllama_forecasting as f
    elif model_name == "Moirai-2":
        from utils.moirai_utils import moirai_forecasting as f
    elif model_name == "TabPFN-TS":
        from utils.tabpfnts_utils import tabPFNTS_forecasting as f
    elif model_name == "TimeGPT":
        from utils.timegpt_utils import timeGPT_forecasting as f
    elif model_name == "TimesFM":
        from utils.timesfm_utils import timeFM_forecasting as f
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, choices=list(HUB))
    parser.add_argument("--model_name", type=str, required=True, choices=list(INPUT_FILE))
    args = parser.parse_args()

    hub, model = HUB[args.target], args.model_name
    out_dir = os.path.join("model-output", hub, model)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    df = pd.read_csv(os.path.join("target-data", hub, INPUT_FILE[model]))

    # skip if this week's forecast already exists
    last_update = pd.to_datetime(df["truth_date"].max()).strftime("%Y-%m-%d")
    if os.path.exists(os.path.join(out_dir, f"{last_update}-{model}.csv")):
        print(f"Forecast for {model} on {last_update} already exists, skipping")
        return

    # horizons 0-3 weeks ahead of the reference week (Saturday)
    pred_df = load_model(model)(df, horizon_delta=0, freq="W-SAT")
    pred_df["output_type_id"] = np.round(pred_df["output_type_id"], 3)
    pred_df["value"] = np.round(pred_df["value"], 3)

    ref_date = pd.to_datetime(pred_df["reference_date"].iloc[0]).strftime("%Y-%m-%d")
    pred_df.to_csv(os.path.join(out_dir, f"{ref_date}-{model}.csv"), index=False)

    locations = pd.read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/auxiliary-data/locations.csv")
    locations_dict = {row.location: row.abbreviation for _, row in locations.iterrows()}
    plot_forecast(pred_df, df, savefig=True, location_names_dict=locations_dict,
                  save_path=os.path.join(out_dir, "plots", f"{ref_date}-{model}.png"))


if __name__ == "__main__":
    main()
