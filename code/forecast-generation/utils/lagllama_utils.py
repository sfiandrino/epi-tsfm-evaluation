import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
import torch
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from pathlib import Path
import types
import sys
from types import ModuleType

# Create dummy module hierarchy
def create_dummy_module(module_path):
    """
    Create a dummy module hierarchy for the given path.
    Returns the leaf module.
    """
    parts = module_path.split('.')
    current = ''
    parent = None

    for part in parts:
        current = current + '.' + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current

    return sys.modules[module_path]

# Create the dummy gluonts module hierarchy
gluonts_module = create_dummy_module('gluonts.torch.modules.loss')

# Create dummy classes for the specific loss functions
class DistributionLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class NegativeLogLikelihood:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Add the specific classes to the module
gluonts_module.DistributionLoss = DistributionLoss
gluonts_module.NegativeLogLikelihood = NegativeLogLikelihood

#CKPT_PATH = Path(__file__).resolve().parent / "lag-llama.ckpt"
CKPT_PATH = (Path(__file__).resolve().parents[2] / "lag-llama" / "lag-llama.ckpt")


def get_lag_llama_predictions(dataset, prediction_length, device, context_length=640, use_rope_scaling=False, num_samples=100):
    """
    Get the predictions from the Lag-Llama model
    """
    
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=str(CKPT_PATH),
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling="mean",
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


def make_lagllama_quantile_output(
    df, 
    horizon_delta = 0,
):
    """
    Make the Lag-Llama quantile output into a long dataframe
    """

    # columns that are quantiles (everything except these id vars)
    id_cols = ["location", "truth_date", "predictions"]
    quantile_cols = [c for c in df.columns if c not in id_cols]

    # wide → long
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


def compute_quantiles(forecasts, last_update, REQ_Q=None):
    """Compute the quantiles from the forecasts"""

    if REQ_Q is None: 
        REQ_Q = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    rows = []

    for f in forecasts:
        abbreviation = getattr(f, "item_id", None)

        samples = f.samples                  
        pred_len = samples.shape[1]
        offset = to_offset(f.freq)          

        # Iterate over each forecasted step
        for t in range(pred_len):
            forecast_date = last_update + (t + 1) * offset   
            qvals = np.quantile(samples, q=REQ_Q, axis=0)[:, t]

            row = {
                "location": abbreviation,
                "truth_date": forecast_date,
                "predictions": samples[:, t].mean(),
            }

            for ql, qv in zip(REQ_Q, qvals):
                row[str(ql)] = qv

            rows.append(row)

    return pd.DataFrame(rows)


def lagllama_forecasting(df, horizon=4, REQ_Q=None, horizon_delta=0, freq="W-SAT", num_samples=100):
    """
    Forecast with Lag-Llama
    """

    df["truth_date"] = pd.to_datetime(df["truth_date"])
    last_update = df['truth_date'].max()

    # Create the Pandas
    df = df.set_index('truth_date')
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')

    dataset = PandasDataset.from_long_dataframe(df, target="value", item_id="location", freq=freq)

    # forecast with samples
    device = torch.device("cpu")
    forecasts, tss = get_lag_llama_predictions(
        dataset=dataset,
        prediction_length=horizon,
        device=device,
        num_samples=num_samples,
    )

    # make quantiles fro msamples
    pred_df = compute_quantiles(forecasts, last_update, REQ_Q=REQ_Q)
    pred_df = make_lagllama_quantile_output(pred_df, horizon_delta=horizon_delta) 

    # add extra columns 
    target = df.target.unique()[0]
    pred_df.insert(0, "reference_date", last_update)
    pred_df.insert(1, "target", target)

    return pred_df