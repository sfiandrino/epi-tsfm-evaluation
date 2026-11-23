"""Download the NHSN weekly hospital admissions used as TSFM input and save a dated snapshot.

Two versions of the series are saved each week:
- surveillance_weekly.csv               as reported (Chronos-2, Lag-Llama, TimesFM)
- surveillance_weekly_interpolated.csv  missing values filled by linear interpolation
                                        (Moirai-2, TabPFN-TS, TimeGPT, which need a gap-free sequence)

Run from the season folder, e.g.  python ../forecast-generation/data_import.py --target totalconfflunewadm
"""
from utils.data_utils import import_data_US
import argparse
import os

HUB = {
    "totalconfflunewadm": "FluSight",
    "totalconfc19newadm": "CovidHub",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, choices=list(HUB))
    args = parser.parse_args()

    out_dir = os.path.join("target-data", HUB[args.target])
    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    for interpol, suffix in [(False, ""), (True, "_interpolated")]:
        data = import_data_US(args.target, interpol=interpol, freq="W-SAT")
        last_update = data["truth_date"].max().strftime("%Y-%m-%d")
        snapshot = os.path.join(snap_dir, f"surveillance_weekly_up_to_{last_update}{suffix}.csv")
        if os.path.exists(snapshot):
            print(f"{snapshot} already exists, skipping")
            continue
        data.to_csv(snapshot, index=False)
        data.to_csv(os.path.join(out_dir, f"surveillance_weekly{suffix}.csv"), index=False)
        print(f"Saved {snapshot}")


if __name__ == "__main__":
    main()
