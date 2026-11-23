import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline


def make_chronos_quantile_output(
    df, 
    horizon_delta = 0,
):
    """
    Make the Chronos quantile output into a long dataframe
    """

    df["truth_date"] = pd.to_datetime(df["truth_date"])

    # columns that are quantiles (everything except these id vars)
    id_cols = ["location", "truth_date", "target_name", "predictions"]
    quantile_cols = [c for c in df.columns if c not in id_cols]

    # wide → long
    long = df.melt(
        id_vars=["location", "truth_date", "target_name"],
        value_vars=quantile_cols,
        var_name="output_type_id",
        value_name="value",
    )

    # rename / add columns
    long = long.rename(
        columns={
            "truth_date": "target_end_date",
            "abbreviation": "location",
            "target_name": "target",
        }
    )
    long["output_type"] = "quantile"
    long["output_type_id"] = long["output_type_id"].astype(float)

    # add horizon 
    long["horizon"] = (
        (long["target_end_date"] - long["target_end_date"].min()).dt.days // 7 + horizon_delta
    ).astype(int)

    # reorder columns 
    long = long[["horizon", "target_end_date", "location", "output_type", "output_type_id", "value"]].sort_values(["horizon", "location", "output_type_id"])

    return long


def chronos_forecasting(df, horizon=4,  REQ_Q=None, horizon_delta=0, freq="W-SAT"):
    """
    Forecast with Chronos
    """
    # pipeline
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    # required quantiles
    if REQ_Q is None: 
        REQ_Q = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    # generate predictions
    pred_df = pipeline.predict_df(
        df,
        prediction_length=horizon,  
        quantile_levels=REQ_Q,
        id_column="location",  
        timestamp_column="truth_date",  
        target="value",  
    )

    # transform into submission df
    pred_df = make_chronos_quantile_output(pred_df, horizon_delta=horizon_delta)

    # add extra columns 
    ref_date = df.truth_date.max()
    target = df.target.unique()[0]
    pred_df.insert(0, "reference_date", ref_date)
    pred_df.insert(1, "target", target)

    return pred_df
