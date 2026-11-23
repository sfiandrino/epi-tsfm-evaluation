"""Rebuild the TSFM input series as they were available on a past Wednesday (data release),
from the hub target-data git history, in the format written by forecast-generation/data_import.py.

Recent weeks come from the hub target data at the given commit; older weeks (before the start of
the hub file) come from an earlier real snapshot, which does not change between releases.
Used for the retrospective round 2026-04-25 (data released on 2026-04-29), the only round not produced in real time.
Usage (from season-2025-26/):
  python ../code/forecast-generation/retrospective/build_snapshot.py --hub FluSight --sha <commit> \
      --base_snapshot 2026-04-04 --out_dir <dir>
"""
import argparse, io, os, sys
import pandas as pd, requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.data_utils import remove_countries_with_skipped_last_data_update, reindex_data

SRC = {"FluSight": ("cdcepi/FluSight-forecast-hub", "target-data/target-hospital-admissions.csv", "wk inc flu hosp"),
       "CovidHub": ("CDCgov/covid19-forecast-hub", "target-data/covid-hospital-admissions.csv", "wk inc covid hosp")}

p = argparse.ArgumentParser()
p.add_argument("--hub", required=True, choices=SRC); p.add_argument("--sha", required=True)
p.add_argument("--base_snapshot", required=True); p.add_argument("--out_dir", required=True)
a = p.parse_args()
repo, path, target = SRC[a.hub]

hub = pd.read_csv(io.StringIO(requests.get(f"https://raw.githubusercontent.com/{repo}/{a.sha}/{path}").text),
                  dtype={"location": str})
hub = pd.DataFrame({"truth_date": pd.to_datetime(hub["date" if "date" in hub else "target_end_date"]), "location": hub["location"],
                    "target": target, "value": pd.to_numeric(hub["value"], errors="coerce")})
base = pd.read_csv(os.path.join("target-data", a.hub, "snapshots",
                                f"surveillance_weekly_up_to_{a.base_snapshot}.csv"), dtype={"location": str})
base["truth_date"] = pd.to_datetime(base["truth_date"])
old = base[base.truth_date < hub.truth_date.min()]
df = pd.concat([old, hub[hub.location.isin(base.location.unique())]], ignore_index=True)
df = df.sort_values(["location", "truth_date"])
df = remove_countries_with_skipped_last_data_update(df)

os.makedirs(a.out_dir, exist_ok=True)
for interpol, suffix in [(False, ""), (True, "_interpolated")]:
    out = reindex_data(df.copy(), freq="W-SAT", interpol=interpol, method="linear")
    last = out.truth_date.max().strftime("%Y-%m-%d")
    f = os.path.join(a.out_dir, f"surveillance_weekly_up_to_{last}{suffix}.csv")
    out.to_csv(f, index=False); print("saved", f)
