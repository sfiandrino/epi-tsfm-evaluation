import pandas as pd
import numpy as np
import os 
import tabpfn_client
from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode
from tabpfn_time_series import TimeSeriesDataFrame
from tabpfn_time_series import FeatureTransformer
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

token = os.environ["TABPFN_ACCESS_TOKEN"]  
tabpfn_client.set_access_token(token)

def make_tabpfnts_quantile_output(
    df, 
    horizon_delta = 0,
):
    """
    Make the TabPFN-TS quantile output into a long dataframe
    """

    df["truth_date"] = pd.to_datetime(df["truth_date"])

    # columns that are quantiles (everything except these id vars)
    id_cols = ["location", "truth_date", "level_0",	"index", "target"]
    quantile_cols = [c for c in df.columns if c not in id_cols]

    # wide → long
    long = df.melt(
        id_vars=["location", "truth_date"],
        value_vars=quantile_cols,
        var_name="output_type_id",
        value_name="value",
    )

    # rename / add columns 
    long = long.rename(
        columns={
            "truth_date": "target_end_date",
            "location": "location",
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


def tabPFNTS_forecasting(df, horizon=4,  REQ_Q=None, horizon_delta=0, freq="W-SAT"):
    """
    Forecast with Chronos
    """

    # build prediction test rows: for each state and 4 horizons (weeks), with values = NaN
    last_date = pd.to_datetime(df["truth_date"].max())

    new_rows = []
    for location in df["location"].unique():
        for h in np.arange(1, horizon + 1):
            new_date = last_date + pd.DateOffset(weeks=h)
            new_rows.append({
                "truth_date": new_date.date(),
                "location": location,    
                "value": pd.NA        
            })
    pred_df = pd.DataFrame(new_rows)

    # append to original dataframe
    surv_data_with_pred = pd.concat([df, pred_df], ignore_index=True)

    # sort by state and date
    surv_data_with_pred = surv_data_with_pred.sort_values(
        ["location", "truth_date"]
    ).reset_index(drop=True)

    # prepare data for forecasting
    surv_data_with_pred = surv_data_with_pred[['location', 'truth_date', 'value']]
    surv_data_with_pred = surv_data_with_pred.rename(
        columns={
            "location": "item_id",
            "truth_date": "timestamp",
            "value": "target",
        }
    )
    surv_data_with_pred["timestamp"] = pd.to_datetime(surv_data_with_pred["timestamp"])
    tsdf = TimeSeriesDataFrame(surv_data_with_pred)

    # split off the last `prediction_length` points per series
    train_tsdf, _ = tsdf.train_test_split(prediction_length=horizon)

    # build the test feature frame for the next steps
    test_tsdf = generate_test_X(train_tsdf, horizon)

    # Add features
    features = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]

    train_tsdf, test_tsdf = FeatureTransformer(features).transform(
        train_tsdf,
        test_tsdf,
    )

    # predict 
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.CLIENT,
    )
    pred = predictor.predict(train_tsdf, test_tsdf)

    # interpolate missing quantile
    pred = pd.DataFrame(pred).reset_index() 
    quantile_cols, quantile_levels = [], []

    for c in pred.columns:
        if c in ["item_id", "timestamp", "target"]:
            continue
        try:
            q = float(c)
            quantile_cols.append(c)
            quantile_levels.append(q)
        except:
            continue

    if not quantile_cols:
        raise RuntimeError("Could not find any quantile columns in pred")

    # sort quantiles
    order = np.argsort(quantile_levels)
    quantile_cols = [quantile_cols[i] for i in order]
    quantile_levels = np.array(sorted(quantile_levels))

    # apply interpolation
    if REQ_Q is None: 
        REQ_Q = np.array([
            0.01, 0.025, 0.05,
            0.10, 0.15, 0.20, 0.25,
            0.30, 0.35, 0.40, 0.45,
            0.50,
            0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85,
            0.90, 0.95, 0.975, 0.99,
        ])

    def interpolate_quantiles(row):
        vals = row[quantile_cols].to_numpy(dtype=float)
        interp = np.interp(REQ_Q, quantile_levels, vals)
        colnames = [f"{q:.3f}".rstrip("0").rstrip(".") for q in REQ_Q]
        return pd.Series(interp, index=colnames)

    extra_quantiles = pred.apply(interpolate_quantiles, axis=1)

    # build final dataframe including item_id and timestamp
    pred_23 = pd.concat(
        [
            pred[["item_id", "timestamp", "target"]],
            extra_quantiles
        ],
        axis=1
    )

    pred_23 = pred_23.reset_index().rename(
        columns={
            "item_id": "location",
            "timestamp": "truth_date"
        }
    )

    submission_df = make_tabpfnts_quantile_output(pred_23, horizon_delta=horizon_delta)

    # add extra columns 
    ref_date = df.truth_date.max()
    target = df.target.unique()[0]
    submission_df.insert(0, "reference_date", ref_date)
    submission_df.insert(1, "target", target)

    return submission_df
