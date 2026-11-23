import pandas as pd
import argparse
import os

def load_and_prepare_data(input_file):
    df = pd.read_csv(input_file)

    df = df[[
        "model",
        "reference_date",
        "target_end_date",
        "horizon",
        "location",
        "wis",
        "ae_median"
    ]].copy()

    df = df.rename(columns={"ae_median": "ae"})

    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    df["horizon"] = df["horizon"].astype(int)
    df["location"] = df["location"].astype(str)
    df["wis"] = pd.to_numeric(df["wis"])
    df["ae"] = pd.to_numeric(df["ae"])

    df = df.sort_values(["reference_date", "target_end_date", "horizon", "location", "model"]).reset_index(drop=True)

    return df


def calculate_relative_performance_for_all_weeks(input_path, output_path, baseline_model="FluSight-baseline"):
    df = pd.read_csv(input_path)

    keys = ["target_end_date", "location", "horizon"]

    baseline_df = (
        df[df["model"] == baseline_model][keys + ["wis", "ae_median"]]
        .rename(columns={
            "wis": "baseline_wis",
            "ae_median": "baseline_ae"
        })
        .copy()
    )

    merged = df.merge(baseline_df, on=keys, how="left")

    merged = merged.dropna(subset=["baseline_wis", "baseline_ae"]).copy()

    merged = merged[
        (merged["wis"] >= 0) &
        (merged["ae_median"] >= 0) &
        (merged["baseline_wis"] > 0) &
        (merged["baseline_ae"] > 0)
    ].copy()

    merged["relative_wis"] = merged["wis"] / merged["baseline_wis"]
    merged["relative_ae"] = merged["ae_median"] / merged["baseline_ae"] 

    merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input csv file")
    parser.add_argument("--output", required=True, help="Path to output csv file")
    args = parser.parse_args()

    df = load_and_prepare_data(args.input)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("All results are saved in data/processed")