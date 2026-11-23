import torch
from torch.library import Library
import pandas as pd 
import numpy as np 
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# Define a dummy torchvision::nms operator so torchvision's meta-registration doesn't crash
lib = Library("torchvision", "DEF")
lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")


def make_moirai_quantile_output(
    df, 
    horizon_delta = 0,
):
    """
    Make the Moirai quantile output into a long dataframe
    """
    
    df["truth_date"] = pd.to_datetime(df["truth_date"])

    # columns that are quantiles (everything except these id vars)
    id_cols = ["location", "truth_date", "index", "target"]
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


def moirai_forecasting(df, horizon=4,  REQ_Q=None, horizon_delta=0, freq="W-SAT"):
    """
    Forecast with Moirai
    """

    # Config
    CTX = df['truth_date'].nunique()  
    BSZ = 32  
    PRED_LEN = horizon  
    NUM_SAMPLES = 1000

    # Sort consistently
    df = df.sort_values(["location", "truth_date"])

    # map each state abbreviation to a numeric id
    state_to_id = {
        abbr: i for i, abbr in enumerate(sorted(df["location"].unique()))
    }

    test_data = ListDataset(
        [
            {
                "start": group["truth_date"].iloc[0],
                "target": group["value"].astype("float32").to_numpy(),
                "feat_static_cat": [state_to_id[state]],  
                "item_id": state, 
            }
            for state, group in df.groupby("location")
        ],
        freq=freq, 
    )

    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
        prediction_length=PRED_LEN,
        context_length=CTX,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    # List-ify test_data so we can iterate multiple times
    input_list = list(test_data)

    # Run predictions
    fc_list = list(predictor.predict(test_data, num_samples=NUM_SAMPLES))

    # Known base quantiles for Moirai2
    base_quantile_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    base_quantile_cols = [f"{q:.1f}" for q in base_quantile_levels]
    rows = []

    for inp, fc in zip(input_list, fc_list):
        state = inp["item_id"]          
        timestamps = [p.end_time.normalize() for p in fc.index]

        for t_idx, ts in enumerate(timestamps):
            row = {
                "item_id": state,
                "timestamp": ts,
                "target": np.nan,      
            }

            # fill 0.1..0.9 quantiles from QuantileForecast
            for q, col in zip(base_quantile_levels, base_quantile_cols):
                q_vals = fc.quantile(q)            
                row[col] = float(q_vals[t_idx])    
            rows.append(row)

    pred = pd.DataFrame(rows).reset_index(drop=True)

    # transform into 23 quantile
    pred = pred.reset_index(drop=True)

    # 1. Detect quantile columns and their numeric levels
    quantile_cols = []
    quantile_levels = []

    for c in pred.columns:
        if c in ["item_id", "timestamp", "target"]:
            continue
        try:
            q = float(c)  # e.g. "0.1" -> 0.1
            quantile_cols.append(c)
            quantile_levels.append(q)
        except Exception:
            continue

    if not quantile_cols:
        raise RuntimeError("Could not find any quantile columns in pred")

    # Sort quantiles
    order = np.argsort(quantile_levels)
    quantile_cols = [quantile_cols[i] for i in order]
    quantile_levels = np.array(sorted(quantile_levels))

    print("Base quantiles detected:", quantile_levels)

    # desired quantiles
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
        vals = row[quantile_cols].to_numpy(dtype=float)  # base 0.1–0.9
        interp = np.interp(REQ_Q, quantile_levels, vals)
        colnames = [f"{q:.3f}".rstrip("0").rstrip(".") for q in REQ_Q]
        return pd.Series(interp, index=colnames)

    # apply interpolation row-wise
    extra_quantiles = pred.apply(interpolate_quantiles, axis=1)

    # build final DF including item_id, timestamp, target, and 23 quantiles
    pred_23 = pd.concat(
        [
            pred[["item_id", "timestamp", "target"]],
            extra_quantiles,
        ],
        axis=1
    )

    pred_23 = pred_23.reset_index(drop=True).rename(
        columns={
            "item_id": "location",
            "timestamp": "truth_date"
        }
    )

    submission_df = make_moirai_quantile_output(pred_23, horizon_delta=horizon_delta)

    # add extra columns 
    ref_date = df.truth_date.max()
    target = df.target.unique()[0]
    submission_df.insert(0, "reference_date", ref_date)
    submission_df.insert(1, "target", target)

    return submission_df


