import pandas as pd
import argparse
import os

def calculate_scores(input_file):
    df = pd.read_csv(input_file)

    group_cols = ["reference_date", "target_end_date", "horizon", "location"]

    # std rank score for WIS
    df["rank_wis"] = df.groupby(group_cols)["wis"].rank(method="average", ascending=True)
    df["N"] = df.groupby(group_cols)["model"].transform("count")
    df["std_rank_wis"] = 1 - (df["rank_wis"] - 1) / (df["N"] - 1)

    # std rank score for AE
    df["rank_ae"] = df.groupby(group_cols)["ae"].rank(method="average", ascending=True)
    df["std_rank_ae"] = 1 - (df["rank_ae"] - 1) / (df["N"] - 1)

    # value-based relative rank score for WIS
    df["max_wis"] = df.groupby(group_cols)["wis"].transform("max")
    df["min_wis"] = df.groupby(group_cols)["wis"].transform("min")

    df["relative_score_wis"] = (
        (df["max_wis"] - df["wis"]) / (df["max_wis"] - df["min_wis"])
    )

    # value-based relative rank score for AE
    df["max_ae"] = df.groupby(group_cols)["ae"].transform("max")
    df["min_ae"] = df.groupby(group_cols)["ae"].transform("min")

    df["relative_score_ae"] = (
        (df["max_ae"] - df["ae"]) / (df["max_ae"] - df["min_ae"])
    )

    df = df.drop(columns=[
        "rank_wis", "rank_ae", "N",
        "max_wis", "min_wis",
        "max_ae", "min_ae"
    ])
    
    df = df.sort_values(["reference_date", "target_end_date", "horizon", "location", "std_rank_wis"],ascending=[True, True, True, True, False]).reset_index(drop=True)
    return df

def aggregate_by_horizon(df):
    df_horizon = (
        df.groupby(["model", "horizon"])[
            ["std_rank_wis", "std_rank_ae", "relative_score_wis", "relative_score_ae"]
        ]
        .mean()
        .reset_index()
    )
    
    df_horizon = df_horizon.sort_values(["horizon", "std_rank_wis"],ascending=[True, False]).reset_index(drop=True)

    return df_horizon

def aggregate_by_location(df):
    df_location = (
        df.groupby(["model", "location"])[
            ["std_rank_wis", "std_rank_ae", "relative_score_wis", "relative_score_ae"]
        ]
        .mean()
        .reset_index()
    )
    
    df_location = df_location.sort_values(["location", "std_rank_wis"],ascending=[True, False]).reset_index(drop=True)
    
    return df_location


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input csv file")
    parser.add_argument("--output", required=True, help="Path to output csv file")
    parser.add_argument("--output_horizon", required=True, help="Path to horizon aggregated output csv file")
    parser.add_argument("--output_location", required=True, help="Path to location aggregated output csv file")


    args = parser.parse_args()

    df = calculate_scores(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    df_horizon = aggregate_by_horizon(df)
    os.makedirs(os.path.dirname(args.output_horizon), exist_ok=True)
    df_horizon.to_csv(args.output_horizon, index=False)
    
    df_location = aggregate_by_location(df)
    os.makedirs(os.path.dirname(args.output_location), exist_ok=True)
    df_location.to_csv(args.output_location, index=False)
    
    print("All results are saved in outputs/scores")