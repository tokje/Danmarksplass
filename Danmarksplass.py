"""
Group work 3 - Air pollution at Danmarksplass (Bergen), 2003-2025
Starter script: load data, coverage, annual stats, high-pollution days,
meteorology correlations. Adapt the style/choices and justify them in your report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE, "dataSetDanmarksPlass.txt")
OUT = os.path.join(BASE, "figures") + os.sep
os.makedirs(OUT, exist_ok=True)
MIN_VALID = 274                     # ~75 % of a year; a choice you must justify

# ------------------------------------------------------------------ 1. Load
df = pd.read_csv(FILE, parse_dates=["Time"], index_col="Time")   # 'NaN' is read as missing
df = df.rename(columns={"PM2_5": "PM2.5"})
poll = ["PM10", "PM2.5", "NO2", "NO", "CO"]
met = ["AirTemperature", "RelativeHumidity", "WindSpeed",
       "SurfaceAirPressure", "Precipitation24h"]
print(df.shape, df.index.min().date(), "to", df.index.max().date())

# Negative concentrations are physically impossible (instrument offset/noise).
# Here they are counted and kept; decide and document what you do with them.
print("Negative values:\n", (df[poll] < 0).sum())

# ------------------------------------------------- 1b. Coverage (valid days)
valid = df.groupby(df.index.year).count()
print(valid)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(valid[poll + met].T / 365 * 100, aspect="auto", cmap="viridis",
               extent=[valid.index[0] - .5, valid.index[-1] + .5, len(poll + met), 0])
ax.set_yticks(np.arange(len(poll + met)) + .5, poll + met)
plt.colorbar(im, label="Valid days per year (%)")
ax.set_title("Data coverage"); fig.tight_layout()
fig.savefig(OUT + "fig_coverage.png", dpi=200)

# ------------------------------------- 1c. Daily time series (one panel each)
fig, axs = plt.subplots(len(poll), 1, figsize=(10, 9), sharex=True)
units = {"PM10": "µg m$^{-3}$", "PM2.5": "µg m$^{-3}$", "NO2": "µg m$^{-3}$",
         "NO": "µg m$^{-3}$", "CO": "mg m$^{-3}$"}
for ax, p in zip(axs, poll):
    ax.plot(df.index, df[p], lw=.4)
    ax.set_ylabel(f"{p}\n({units[p]})")
fig.tight_layout(); fig.savefig(OUT + "fig_timeseries.png", dpi=200)

# --------------------------------------------- 2. Annual mean/median + n
g = df[["PM10", "PM2.5", "NO2"]].groupby(df.index.year)
annual = pd.concat({"mean": g.mean(), "median": g.median(), "n_valid": g.count()}, axis=1)
annual = annual.round(1)
print(annual)
annual.to_csv("annual_stats.csv")

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, p in zip(axs, ["PM10", "PM2.5", "NO2"]):
    ax.plot(annual.index, annual[("mean", p)], "o-", label="Mean")
    ax.plot(annual.index, annual[("median", p)], "s--", label="Median")
    low = annual[("n_valid", p)] < MIN_VALID           # flag poorly covered years
    ax.plot(annual.index[low], annual[("mean", p)][low], "rx", ms=10, label=f"< {MIN_VALID} valid days")
    ax.set_ylabel(f"{p} (µg m$^{{-3}}$)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "fig_annual.png", dpi=200)

# --------------------------- 3. High / very high days (Table 1 thresholds)
def count_classes(s, hi_lo, hi_up, vh):
    yr = s.index.year
    high = ((s >= hi_lo) & (s <= hi_up)).groupby(yr).sum()
    vhigh = (s > vh).groupby(yr).sum()
    n = s.groupby(yr).count()
    return pd.DataFrame({"high": high, "very_high": vhigh, "n_valid": n})

cls = {"PM10": count_classes(df["PM10"], 50, 150, 150),
       "PM2.5": count_classes(df["PM2.5"], 25, 75, 75)}
for k, v in cls.items():
    print(k, "\n", v)

fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, (k, v) in zip(axs, cls.items()):
    ax.bar(v.index, v["high"], label="High")
    ax.bar(v.index, v["very_high"], bottom=v["high"], label="Very high")
    for x, tot, n in zip(v.index, v["high"] + v["very_high"], v["n_valid"]):
        ax.text(x, tot + .3, str(n), ha="center", fontsize=6, rotation=90)  # valid days
    ax.set_ylabel(f"{k}: days per year"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "fig_highdays.png", dpi=200)

# -------------------------------------------- 4. Meteorology vs pollution
rows = []
for p in ["PM10", "PM2.5", "NO2"]:
    for m in met:
        pair = df[[p, m]].dropna()
        rows.append({"pollutant": p, "met": m, "N": len(pair),
                     "pearson": pair[p].corr(pair[m]),
                     "spearman": pair[p].rank().corr(pair[m].rank())})
corr = pd.DataFrame(rows).pivot(index="pollutant", columns="met", values=["pearson", "spearman", "N"])
print(corr.round(2).to_string())
corr.round(2).to_csv("correlations.csv")

fig, axs = plt.subplots(3, 5, figsize=(16, 9))
for i, p in enumerate(["PM10", "PM2.5", "NO2"]):
    for j, m in enumerate(met):
        axs[i, j].scatter(df[m], df[p], s=2, alpha=.3)
        axs[i, j].set_xlabel(m if i == 2 else "")
        axs[i, j].set_ylabel(p if j == 0 else "")
fig.tight_layout(); fig.savefig(OUT + "fig_scatter.png", dpi=200)

# Seasonal cycle (useful for the critical reflection)
seasonal = df[["PM10", "PM2.5", "NO2"]].groupby(df.index.month).median()
seasonal.plot(marker="o", figsize=(7, 4), xlabel="Month", ylabel="Median (µg m$^{-3}$)")
plt.tight_layout(); plt.savefig(OUT + "fig_seasonal.png", dpi=200)
plt.show()
