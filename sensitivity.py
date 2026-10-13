"""
Sensitivity of annual means to missing data (Critical Reflection).
For each well-covered year 2012-2025, the script removes the same calendar
periods that were missing in 2005 and compares the resulting annual mean with
the mean of all valid days. It also removes 25 % of the days at random
(200 repetitions) as a reference.
Run: python sensitivity.py   (reads dataSetDanmarksPlass.txt in the same folder)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "dataSetDanmarksPlass.txt"),
                 parse_dates=["Time"], index_col="Time")
df = df.rename(columns={"PM2_5": "PM2.5"})

# The gap each pollutant had in 2005 (month-day of first and last missing day)
GAPS = {"PM10": ("06-01", "08-16"),
        "NO2": ("05-31", "07-27"),
        "PM2.5": ("04-30", "10-07")}
YEARS = range(2012, 2026)

rows = []
for y in YEARS:
    d = df.loc[str(y)]
    for p, (start, end) in GAPS.items():
        full = d[p].mean()
        gap = (d.index >= f"{y}-{start}") & (d.index <= f"{y}-{end}")
        rows.append({"year": y, "pollutant": p, "full": full,
                     "with_gap": d.loc[~gap, p].mean()})
res = pd.DataFrame(rows)
res["bias_%"] = (res["with_gap"] / res["full"] - 1) * 100
print(res.round(1).to_string(index=False))
print("\nBias range (%):\n", res.groupby("pollutant")["bias_%"].agg(["min", "max"]).round(0))

# Reference: random removal of 25 % of the days
rng_std = []
for y in YEARS:
    s = df.loc[str(y), "PM10"].dropna()
    means = [s.sample(frac=0.75, random_state=k).mean() for k in range(200)]
    rng_std.append(np.std(means) / s.mean() * 100)
print("\nRandom 25 % removal, relative std of annual PM10 mean: "
      f"{min(rng_std):.1f}-{max(rng_std):.1f} %")
