import pandas as pd
import numpy as np
import os 
from nixtla import NixtlaClient

token = os.environ["TIMEGPT_ACCESS_TOKEN"]  
nixtla_client = NixtlaClient(api_key = token)
nixtla_client.validate_api_key()


def make_timegpt_quantile_output(
    df, 
    horizon_delta = 0,
):
    """
    Make the TimeGPT quantile output into a long dataframe
    """

    id_cols = ["location", "truth_date", "predictions"]
    quantile_cols = [c for c in df.columns if c not in id_cols]
    submission_df = df.melt(
        id_vars=["location", "truth_date"],
        value_vars=quantile_cols,
        var_name="output_type_id",
        value_name="value",
    )
    # rename / add columns 
    submission_df = submission_df.rename(
        columns={
            "truth_date": "target_end_date",
        }
    )

    submission_df["output_type"] = "quantile"
    submission_df["output_type_id"] = submission_df["output_type_id"].astype(float)

    # add horizon
    submission_df["horizon"] = (
        (submission_df["target_end_date"] - submission_df["target_end_date"].min()).dt.days // 7 + horizon_delta
    ).astype(int)

    # reorder columns 
    submission_df = submission_df[["horizon", "target_end_date", "location", "output_type", "output_type_id", "value"]].sort_values(["horizon", "location", "output_type_id"])

    return submission_df


def timeGPT_forecasting(df, horizon=4,  REQ_Q=None, horizon_delta=0, freq="W-SAT"):
    """
    Forecast with TimeGPT
    """

    if REQ_Q is None: 
        REQ_Q = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    # ensure truth_date is datetime
    df = df.copy()
    df["truth_date"] = pd.to_datetime(df["truth_date"])

    
    # forecast
    pred_df = nixtla_client.forecast(
        df=df,
        h=horizon,  
        quantiles=REQ_Q,  
        id_col="location",  
        time_col="truth_date",  
        target_col="value",  
    )
    pred_df.rename(columns={'TimeGPT': 'predictions'}, inplace=True) 
    quantile_cols = [c for c in pred_df.columns if str(c).startswith("TimeGPT-q-")] 
    rename_map = {old: str(REQ_Q[i]) for i, old in enumerate(quantile_cols)}
    pred_df.rename(columns=rename_map, inplace=True)

    # transform into submission df
    pred_df = make_timegpt_quantile_output(pred_df, horizon_delta=horizon_delta) 

    # add extra columns 
    ref_date = df.truth_date.max()
    target = df.target.unique()[0]
    pred_df.insert(0, "reference_date", ref_date)
    pred_df.insert(1, "target", target)

    return pred_df
