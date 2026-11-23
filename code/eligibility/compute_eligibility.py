"""Eligibility of hub models under the 75% rule (run from the repository root).

A forecast instance is (reference_date, location, horizon) with all 23 quantiles of the
incident-hospitalization target, horizons 0-3, U.S. national + 52 jurisdictions.
Strict rule: denominator = all possible instances in the official hub rounds of the evaluation
window = rounds x 53 locations x 4 horizons. Official rounds = reference dates on which the hub
ensemble was produced (this drops 2025-01-18, when neither hub ran a round). The 2025/26 window
ends on 2026-05-23, the last FluSight round; CovidHub is cut at the same date (27 rounds each).
Eligible if the model submitted >= 75% of them. Reads csv and parquet submissions.
"""
import os, sys
import pandas as pd

WINDOWS = {"season-2024-25": ("2024-11-30", "2025-05-24"),
           "season-2025-26": ("2025-11-22", "2026-05-23")}
HUBS = {"FluSight": ("wk inc flu hosp", "FluSight-ensemble", "FluSight-baseline"),
        "CovidHub": ("wk inc covid hosp", "CovidHub-ensemble", "CovidHub-baseline")}
TSFMS = {"Chronos-2", "Lag-Llama", "Moirai-2", "TabPFN-TS", "TimeGPT", "TimesFM"}
# hub-generated variants, not contributing models (only the hub ensemble and baseline are kept)
HUB_VARIANT = lambda m: m.startswith(("FluSight-", "CovidHub-")) and m not in {v[1] for v in HUBS.values()} | {v[2] for v in HUBS.values()}
LOC = {"US", "72"} | {f"{i:02d}" for i in range(1, 57) if i not in (3, 7, 14, 43, 52)}   # US, 50 states, DC, PR = 53
THRESHOLD = 0.75

def read(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype=str)
    for c in ("location", "horizon", "output_type", "target", "reference_date"):
        df[c] = df[c].astype(str)
    df["location"] = df["location"].where(df["location"] == "US", df["location"].str.zfill(2))
    df["horizon"] = df["horizon"].str.replace(r"\.0$", "", regex=True)
    return df

def instances(path, target):
    df = read(path)
    df = df[(df.target == target) & (df.output_type == "quantile")
            & df.horizon.isin(["0", "1", "2", "3"]) & df.location.isin(LOC)]
    n = df.groupby(["reference_date", "location", "horizon"]).size()
    return set(n[n >= 23].index)

rows = []
for season, (start, end) in WINDOWS.items():
    for hub, (target, ens, base) in HUBS.items():
        root = os.path.join(season, "model-output", hub)
        per_model = {}
        for m in sorted(os.listdir(root)):
            if m in TSFMS or not os.path.isdir(os.path.join(root, m)):
                continue
            files = {}
            for f in sorted(os.listdir(os.path.join(root, m))):
                if f.endswith((".csv", ".parquet")) and start <= f[:10] <= end:
                    files.setdefault(f[:10], f)          # one file per round (csv first)
            inst = set()
            for d, f in files.items():
                inst |= instances(os.path.join(root, m, f), target)
            per_model[m] = inst
        ens_rounds = {i[0][:10] for i in per_model[ens]}           # official hub rounds
        expected = len(ens_rounds) * len(LOC) * 4
        for m, inst in per_model.items():
            if not inst:
                continue
            covered = {i for i in inst if i[0][:10] in ens_rounds}
            rounds = {i[0][:10] for i in covered}
            share = len(covered) / expected
            rows.append(dict(season=season.replace("season-", "").replace("-", "/"), hub=hub, model=m,
                             hub_variant=HUB_VARIANT(m),
                             rounds=len(rounds), total_rounds=len(ens_rounds),
                             pct_rounds=round(100 * len(rounds) / len(ens_rounds), 1),
                             locations=len({i[1] for i in covered}),
                             instances=len(covered), expected_instances=expected,
                             pct_instances=round(100 * share, 1),
                             eligible=(share >= THRESHOLD) and not HUB_VARIANT(m)))
        print(season, hub, "done", file=sys.stderr)

out = pd.DataFrame(rows)
out.to_csv("code/eligibility/eligibility_75pct.csv", index=False)
print(out.groupby(["season", "hub"]).agg(contributing=("hub_variant", lambda s: (~s).sum()),
                                          eligible=("eligible", "sum")))
