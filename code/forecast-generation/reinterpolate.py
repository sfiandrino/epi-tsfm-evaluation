import argparse
import glob
import os
import pandas as pd

path_switch = {
    "ILI_incidence": "Respicast-Syndromic-ILI",
    "ARI_incidence": "Respicast-Syndromic-ARI",
    "hospital_admissions": "Respicast-Covid",
    "totalconfflunewadm": "FluSight",
    "totalconfc19newadm": "CovidHub",
}


def reindex_data(df, freq="W-SUN", interpol=True, method="linear"):
    """Copied from utils/data_utils.py — avoids importing sodapy."""
    all_dates = pd.date_range(df["truth_date"].min(), df["truth_date"].max(), freq=freq)
    all_countries = df["location"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_dates, all_countries], names=["truth_date", "location"]
    )
    df = df.set_index(["truth_date", "location"]).reindex(full_index).reset_index()
    df = df.sort_values(by=["location", "truth_date"])
    if interpol:
        df["value"] = df["value"].interpolate(method=method, limit_direction="both")
    df["target"] = df["target"].ffill()
    return df


def reinterpolate_file(src_path: str, dst_path: str, freq: str):
    df = pd.read_csv(src_path)
    df["truth_date"] = pd.to_datetime(df["truth_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = reindex_data(df, freq=freq, interpol=True, method="linear")
    df.to_csv(dst_path, index=False)
    print(f"  {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=str, required=True, help="EU or US")
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()

    hub = path_switch[args.target]
    freq = "W-SUN" if args.challenge == "EU" else "W-SAT"
    base = f"target-data/{hub}"

    # --- snapshots ---
    snapshot_dir = os.path.join(base, "snapshots")
    itpF_snapshots = sorted(glob.glob(os.path.join(snapshot_dir, "*_itpF_*.csv")))
    print(f"Found {len(itpF_snapshots)} itpF snapshots in {snapshot_dir}")
    for src in itpF_snapshots:
        dst = src.replace("_itpF_", "_itpT_")
        reinterpolate_file(src, dst, freq)

    # --- current files ---
    itpF_current = sorted(glob.glob(os.path.join(base, "surveillance_weekly_itpF_*.csv")))
    print(f"Found {len(itpF_current)} current itpF files in {base}")
    for src in itpF_current:
        dst = src.replace("_itpF_", "_itpT_")
        reinterpolate_file(src, dst, freq)

    print("Done.")


if __name__ == "__main__":
    main()
