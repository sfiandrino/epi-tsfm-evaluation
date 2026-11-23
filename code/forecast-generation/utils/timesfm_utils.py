import pandas as pd
import numpy as np
import timesfm
import torch


def make_timesfm_quantile_output(
    df,
    quantile_fcst,
    required_quantiles= [0.1, 0.9],
    horizon_delta = 0,
):
    """
    Make the TimesFM quantile output into a long dataframe
    """

    # quantiles in timesfm output
    HAVE_Q = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], dtype=np.float32)
    d = df.copy()
    d["truth_date"] = pd.to_datetime(d["truth_date"])
    d = d.sort_values(["location", "truth_date"])

    # states order must match the order used to create the inputs passed to TimesFM
    states = sorted(d["location"].unique())
    if quantile_fcst.shape[0] != len(states):
        raise ValueError(
            f"quantile_fcst has N={quantile_fcst.shape[0]} series, but dataframe has {len(states)} states. "
            "Make sure you used the same state ordering when forecasting."
        )

    # last observed date per state
    last_dates = (
        d.groupby("location")["truth_date"]
         .max()
         .reindex(states)
         .to_list()
    )

    # Extract q10..q90 from indices 1..9 (skip mean at 0)
    qvals_have = quantile_fcst[:, :, 1:10].astype(np.float32)  # (N,H,9)

    # Interpolate to required quantiles (vectorized over N,H)
    N, H, _ = qvals_have.shape
    qvals_2d = qvals_have.reshape(N * H, -1)

    req = np.array(required_quantiles, dtype=np.float32)
    out_2d = np.empty((N * H, len(req)), dtype=np.float32)
    for i in range(N * H):
        out_2d[i, :] = np.interp(req, HAVE_Q, qvals_2d[i, :]).astype(np.float32)

    out = out_2d.reshape(N, H, len(req))  # (N,H,Qreq)

    rows = []
    for i, st in enumerate(states):
        last = last_dates[i]
        for h in range(0, H):
            target_end_date = last + pd.Timedelta(days=7 * (h+1))
            for qi, q in enumerate(req):
                rows.append({
                    "horizon": h + horizon_delta,
                    "target_end_date": target_end_date,
                    "location": st,
                    "output_type": "quantile",
                    "output_type_id": float(q),
                    "value": float(out[i, h, qi]),
                })

    return pd.DataFrame(rows)


def timeFM_forecasting(df, horizon=4, REQ_Q=None, horizon_delta=0, freq="W-SAT"):
    """
    Forecast with TimesFM
    """

    # create series for timesfm
    state_order, series = sorted(df["location"].unique()), []
    for st in state_order:
        s = df.loc[df["location"] == st].sort_values(by="truth_date")
        y = s["value"].to_numpy(dtype=np.float32)
        series.append(y)

    # timesfm forecasts
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context= df.truth_date.nunique(),  
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,  
            fix_quantile_crossing=True,
        )
    )

    point_fcst, quant_fcst = model.forecast(horizon=horizon, inputs=series)

    # required quantiles
    if REQ_Q is None: 
        REQ_Q = np.array(
            [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
            dtype=np.float32
        )

    # format forecasts
    pred_df = make_timesfm_quantile_output(
        df=df,
        quantile_fcst=quant_fcst,
        required_quantiles=REQ_Q,
        horizon_delta=horizon_delta,
    )

    # add extra columns 
    ref_date = df.truth_date.max()
    target = df.target.unique()[0]
    pred_df.insert(0, "reference_date", ref_date)
    pred_df.insert(1, "target", target)

    return pred_df

