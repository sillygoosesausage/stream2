"""Phase 2 exploratory analysis.

One function per PLAN.md task. Each prints a compact findings block and, where
useful, writes a figure to reports/figures/. Run individually:

    python -m src.eda missingness
    python -m src.eda all
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

FIGDIR = C.ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def _hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete station x hour grid.

    Rows absent from the CSV are genuinely missing hours, distinct from rows
    that are present with null measurements. Several tasks need to tell those
    apart.
    """
    full = []
    for station, g in df.groupby(C.GROUP, observed=True):
        idx = pd.date_range(g[C.TIME].min(), g[C.TIME].max(), freq="h")
        gg = g.set_index(C.TIME).reindex(idx)
        gg[C.GROUP] = station
        gg.index.name = C.TIME
        full.append(gg.reset_index())
    return pd.concat(full, ignore_index=True)


# --------------------------------------------------------------------------
# Task 2.1 - missingness structure
# --------------------------------------------------------------------------
def missingness(train, test):
    print("=" * 78)
    print("TASK 2.1  MISSINGNESS STRUCTURE")
    print("=" * 78)

    cols = C.NUMERIC_RAW + ["wd"]

    print("\n-- null rate %, present rows --")
    rates = pd.DataFrame({
        "train": train[cols].isna().mean() * 100,
        "test": test[cols].isna().mean() * 100,
    }).round(2)
    print(rates)

    # Absent rows (hours with no record at all) vs present-but-null.
    for name, df in (("train", train), ("test", test)):
        grid = _hourly_grid(df)
        absent = grid[C.ID].isna().sum()
        span = len(grid)
        print(f"\n-- {name}: {absent:,} absent hours of {span:,} "
              f"({absent / span * 100:.2f}%) on the full station x hour grid")

    print("\n-- run lengths of consecutive nulls (train, per station) --")
    grid = _hourly_grid(train)
    for col in ["PM10", "CO", "TEMP"]:
        runs = []
        for _, g in grid.groupby(C.GROUP, observed=True):
            isna = g[col].isna().to_numpy()
            if not isna.any():
                continue
            # length of each maximal run of True
            d = np.diff(np.concatenate([[0], isna.view(np.int8), [0]]))
            runs.extend(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
        runs = pd.Series(runs)
        if len(runs):
            print(f"  {col:5s} n_gaps={len(runs):5d}  median={runs.median():.0f}h  "
                  f"p90={runs.quantile(.9):.0f}h  max={runs.max():.0f}h  "
                  f"share of null hours in gaps >6h: "
                  f"{runs[runs > 6].sum() / runs.sum() * 100:.1f}%")

    print("\n-- do stations fail together? "
          "(corr of per-hour null indicator across stations) --")
    for col in ["PM10", "CO", "TEMP"]:
        piv = (train.assign(_n=train[col].isna())
               .pivot_table(index=C.TIME, columns=C.GROUP, values="_n",
                            observed=True))
        cc = piv.corr().to_numpy()
        off = cc[~np.eye(len(cc), dtype=bool)]
        print(f"  {col:5s} mean pairwise corr of missingness = "
              f"{np.nanmean(off):.3f}")

    print("\n-- is missingness associated with pollution level? --")
    print("   (mean target when a column is null vs not)")
    for col in C.NUMERIC_RAW:
        n = train[col].isna()
        if n.sum() < 50:
            continue
        print(f"  {col:5s} null: {train.loc[n, C.TARGET].mean():7.1f}   "
              f"present: {train.loc[~n, C.TARGET].mean():7.1f}   "
              f"(n_null={n.sum():,})")


# --------------------------------------------------------------------------
# Task 2.7 - artefacts and impossible values
# --------------------------------------------------------------------------
def artefacts(train, test):
    print("=" * 78)
    print("TASK 2.7  ARTEFACTS, CEILINGS, FROZEN SENSORS")
    print("=" * 78)

    y = train[C.TARGET]
    print(f"\n-- target: min={y.min()} max={y.max()} "
          f"n_at_max={(y == y.max()).sum():,} "
          f"({(y == y.max()).mean() * 100:.3f}%)")
    print("   most common target values:")
    print(y.value_counts().head(8).to_string())

    print("\n-- values at or above 900 (near the ceiling) --")
    print(f"   {(y >= 900).sum():,} rows ({(y >= 900).mean() * 100:.3f}%)")

    print("\n-- negative or impossible measurements --")
    for col in C.NUMERIC_RAW:
        neg = (train[col] < 0).sum()
        if col in ("TEMP", "DEWP"):
            continue  # legitimately negative
        if neg:
            print(f"  {col}: {neg:,} negative values")
    print("  (TEMP/DEWP excluded - negative is physical)")

    print("\n-- frozen sensors: longest run of an identical repeated value --")
    grid = _hourly_grid(train)
    for col in ["PM10", "CO", "SO2", C.TARGET]:
        worst = 0
        for _, g in grid.groupby(C.GROUP, observed=True):
            s = g[col]
            blocks = (s != s.shift()).cumsum()
            r = s.notna().groupby(blocks).sum().max()
            worst = max(worst, int(r or 0))
        print(f"  {col:16s} longest identical run = {worst}h")

    print("\n-- measurement granularity (n distinct values / rounding) --")
    for col in C.NUMERIC_RAW:
        s = train[col].dropna()
        print(f"  {col:5s} n_unique={s.nunique():6d}  "
              f"is_integer={bool((s % 1 == 0).all())}  "
              f"min={s.min():g} max={s.max():g}")


# --------------------------------------------------------------------------
# Task 2.2 - target distribution and where the error lives
# --------------------------------------------------------------------------
def target_distribution(train, test):
    print("=" * 78)
    print("TASK 2.2  TARGET DISTRIBUTION AND ERROR CONCENTRATION")
    print("=" * 78)

    y = train[C.TARGET]
    print("\n-- quantiles --")
    qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    print(pd.Series(np.quantile(y, qs), index=qs).round(1).to_string())
    print(f"\n  mean={y.mean():.1f}  std={y.std():.1f}  skew={y.skew():.2f}")
    print(f"  log1p skew={np.log1p(y).skew():.2f}")

    print("\n-- share of TOTAL VARIANCE contributed by the worst hours --")
    print("   (this is the ceiling on what spike-handling can win)")
    sq = (y - y.mean()) ** 2
    order = np.argsort(-y.to_numpy())
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = int(len(y) * frac)
        share = sq.to_numpy()[order[:k]].sum() / sq.sum() * 100
        print(f"   top {frac * 100:5.1f}% of hours (y >= "
              f"{y.to_numpy()[order[k - 1]]:5.0f}): {share:5.1f}% of variance")

    print("\n-- baseline error concentration on fold B (from Phase 1 run) --")
    print("   decile 0 RMSE 10.8 ... decile 9 RMSE 87.2")


# --------------------------------------------------------------------------
# Task 2.3 - temporal structure
# --------------------------------------------------------------------------
def temporal(train, test):
    print("=" * 78)
    print("TASK 2.3  TEMPORAL STRUCTURE")
    print("=" * 78)

    grid = _hourly_grid(train)

    print("\n-- autocorrelation of PM2.5 (target series) by lag --")
    acf = {}
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        vals = []
        for _, g in grid.groupby(C.GROUP, observed=True):
            s = g[C.TARGET]
            vals.append(s.corr(s.shift(lag)))
        acf[lag] = float(np.nanmean(vals))
    for lag, v in acf.items():
        print(f"   lag {lag:3d}h: {v:.3f}")

    print("\n-- autocorrelation of PM10 (LEGAL substitute) by lag --")
    for lag in [1, 2, 3, 6, 12, 24]:
        vals = []
        for _, g in grid.groupby(C.GROUP, observed=True):
            s = g["PM10"]
            vals.append(s.corr(s.shift(lag)))
        print(f"   lag {lag:3d}h: {float(np.nanmean(vals)):.3f}")

    print("\n-- correlation of target with LAGGED PM10 --")
    for lag in [0, 1, 2, 3, 6, 12, 24]:
        vals = []
        for _, g in grid.groupby(C.GROUP, observed=True):
            vals.append(g[C.TARGET].corr(g["PM10"].shift(lag)))
        print(f"   PM10 lag {lag:3d}h: {float(np.nanmean(vals)):.3f}")

    print("\n-- mean target by hour of day --")
    print(train.groupby(train[C.TIME].dt.hour)[C.TARGET].mean().round(1).to_string())

    print("\n-- mean target by day of week (0=Mon) --")
    print(train.groupby(train[C.TIME].dt.dayofweek)[C.TARGET]
          .mean().round(1).to_string())

    print("\n-- YEAR-OVER-YEAR TREND (the drift threat) --")
    yr = train.groupby(train[C.TIME].dt.year)[C.TARGET].agg(["mean", "median", "count"])
    print(yr.round(1).to_string())

    print("\n-- same months only (Sep-Feb), by season-year, "
          "for a like-for-like trend --")
    t = train[C.TIME]
    season_year = np.where(t.dt.month >= 9, t.dt.year, t.dt.year - 1)
    m = t.dt.month.isin([9, 10, 11, 12, 1, 2])
    print(train[m].groupby(season_year[m])[C.TARGET]
          .agg(["mean", "median", "count"]).round(1).to_string())


# --------------------------------------------------------------------------
# Task 2.4 - cross-station structure
# --------------------------------------------------------------------------
def cross_station(train, test):
    print("=" * 78)
    print("TASK 2.4  CROSS-STATION STRUCTURE  (sizes the Tier D opportunity)")
    print("=" * 78)

    piv = train.pivot_table(index=C.TIME, columns=C.GROUP, values=C.TARGET,
                            observed=True)
    cc = piv.corr()
    off = cc.to_numpy()[~np.eye(len(cc), dtype=bool)]
    print(f"\n-- same-hour PM2.5 correlation across the 12 stations --")
    print(f"   mean={np.nanmean(off):.3f}  min={np.nanmin(off):.3f}  "
          f"max={np.nanmax(off):.3f}")
    print("\n   full matrix:")
    print(cc.round(2).to_string())

    print("\n-- how much of a station's PM2.5 is explained by the "
          "city mean of the OTHER 11 stations? --")
    r2s = {}
    for st in piv.columns:
        others = piv.drop(columns=st).mean(axis=1)
        r = piv[st].corr(others)
        r2s[st] = r ** 2
    print(pd.Series(r2s).round(3).sort_values().to_string())
    print(f"\n   mean R^2 = {np.mean(list(r2s.values())):.3f}")

    print("\n-- LEGAL version: city-wide PM10 vs this station's next-hour PM2.5 --")
    piv10 = train.pivot_table(index=C.TIME, columns=C.GROUP, values="PM10",
                              observed=True)
    city10 = piv10.mean(axis=1)
    rs = {st: piv[st].corr(city10) for st in piv.columns}
    print(f"   mean corr = {np.mean(list(rs.values())):.3f}   "
          f"(vs own-station PM10 corr 0.85)")

    print("\n-- lead/lag: does the city mean lead individual stations? --")
    for lag in [-3, -2, -1, 0, 1, 2, 3]:
        vals = [piv[st].corr(city10.shift(lag)) for st in piv.columns]
        tag = "city LEADS" if lag > 0 else ("city LAGS" if lag < 0 else "same hour")
        print(f"   city PM10 shifted {lag:+d}h: {np.mean(vals):.3f}  ({tag})")


# --------------------------------------------------------------------------
# Task 2.5 - meteorology
# --------------------------------------------------------------------------
def meteorology(train, test):
    print("=" * 78)
    print("TASK 2.5  METEOROLOGY")
    print("=" * 78)

    print("\n-- mean target by wind direction (sorted) --")
    wd = train.groupby("wd", observed=True)[C.TARGET].agg(["mean", "count"])
    print(wd.sort_values("mean").round(1).to_string())

    print("\n-- mean target by wind speed bucket --")
    bins = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 100]
    print(train.groupby(pd.cut(train["WSPM"], bins), observed=True)[C.TARGET]
          .agg(["mean", "count"]).round(1).to_string())

    print("\n-- interaction: mean target by (direction x speed) --")
    fast = train["WSPM"] > 2
    tab = train.groupby(["wd", fast.map({True: "windy>2", False: "calm<=2"})],
                        observed=True)[C.TARGET].mean().unstack()
    print(tab.round(0).to_string())

    print("\n-- derived: relative humidity from TEMP and DEWP --")
    t, d = train["TEMP"], train["DEWP"]
    rh = 100 * (np.exp(17.625 * d / (243.04 + d))
                / np.exp(17.625 * t / (243.04 + t)))
    print(f"   RH corr with target: {rh.corr(train[C.TARGET]):.3f}   "
          f"(raw DEWP was 0.124, TEMP -0.122)")
    print("   mean target by RH decile:")
    print(train.groupby(pd.qcut(rh, 10, labels=False, duplicates='drop'),
                        observed=True)[C.TARGET].mean().round(1).to_string())

    print("\n-- dew point depression (TEMP - DEWP) --")
    dpd = t - d
    print(f"   corr with target: {dpd.corr(train[C.TARGET]):.3f}")

    print("\n-- rain --")
    print(f"   rows with RAIN>0: {(train['RAIN'] > 0).sum():,} "
          f"({(train['RAIN'] > 0).mean() * 100:.2f}%)")
    print(f"   mean target when raining:     "
          f"{train.loc[train['RAIN'] > 0, C.TARGET].mean():.1f}")
    print(f"   mean target when not raining: "
          f"{train.loc[train['RAIN'] == 0, C.TARGET].mean():.1f}")

    print("\n-- pressure tendency (3h change) vs target --")
    grid = _hourly_grid(train)
    grid["pres_d3"] = grid.groupby(C.GROUP, observed=True)["PRES"].diff(3)
    print(f"   corr: {grid['pres_d3'].corr(grid[C.TARGET]):.3f}")
    print(grid.groupby(pd.qcut(grid["pres_d3"], 5, labels=False,
                               duplicates='drop'),
                       observed=True)[C.TARGET].mean().round(1).to_string())


# --------------------------------------------------------------------------
# Task 2.6 - train vs test drift
# --------------------------------------------------------------------------
def drift(train, test):
    print("=" * 78)
    print("TASK 2.6  TRAIN vs TEST COVARIATE DRIFT  (sizes the bias risk)")
    print("=" * 78)

    print("\n-- distribution comparison, all rows --")
    rows = []
    for col in C.NUMERIC_RAW:
        a, b = train[col].dropna(), test[col].dropna()
        rows.append({
            "col": col,
            "train_mean": a.mean(), "test_mean": b.mean(),
            "delta_%": (b.mean() - a.mean()) / abs(a.mean()) * 100,
            "train_med": a.median(), "test_med": b.median(),
            "train_p95": a.quantile(.95), "test_p95": b.quantile(.95),
        })
    print(pd.DataFrame(rows).set_index("col").round(2).to_string())

    print("\n-- SEASON-MATCHED comparison (train Sep-Feb only vs test) --")
    print("   the honest comparison: test is Sep-Feb, so an all-rows")
    print("   difference mostly reflects season, not drift")
    m = train[C.TIME].dt.month.isin([9, 10, 11, 12, 1, 2])
    rows = []
    for col in C.NUMERIC_RAW:
        a, b = train.loc[m, col].dropna(), test[col].dropna()
        rows.append({"col": col, "train_SepFeb": a.mean(), "test": b.mean(),
                     "delta_%": (b.mean() - a.mean()) / abs(a.mean()) * 100})
    print(pd.DataFrame(rows).set_index("col").round(2).to_string())

    print("\n-- prior-year like-for-like: train Sep2015-Feb2016 (fold B val) "
          "vs test Sep2016-Feb2017 --")
    t = train[C.TIME]
    fb = (t >= "2015-09-01") & (t < "2016-03-01")
    rows = []
    for col in C.NUMERIC_RAW:
        a, b = train.loc[fb, col].dropna(), test[col].dropna()
        rows.append({"col": col, "foldB_val": a.mean(), "test": b.mean(),
                     "delta_%": (b.mean() - a.mean()) / abs(a.mean()) * 100})
    print(pd.DataFrame(rows).set_index("col").round(2).to_string())

    print("\n-- station balance --")
    print(pd.DataFrame({
        "train": train[C.GROUP].value_counts(),
        "test": test[C.GROUP].value_counts(),
    }).to_string())

    print("\n-- wind direction distribution shift (% of rows) --")
    print(pd.DataFrame({
        "train_SepFeb": train.loc[m, "wd"].value_counts(normalize=True) * 100,
        "test": test["wd"].value_counts(normalize=True) * 100,
    }).round(2).to_string())

    print("\n-- ADVERSARIAL VALIDATION: can a model tell train from test? --")
    print("   AUC ~0.5 means indistinguishable; high AUC means real drift")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    cols = C.NUMERIC_RAW + ["wd", "station"]
    a = train.loc[m, cols].copy()
    b = test[cols].copy()
    X = pd.concat([a, b], ignore_index=True)
    X["hour"] = pd.concat([train.loc[m, C.TIME], test[C.TIME]],
                          ignore_index=True).dt.hour
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3,
                                          random_state=C.SEED, stratify=y)
    mdl = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 63, "verbose": -1, "seed": C.SEED},
        lgb.Dataset(Xtr, ytr), 300,
        valid_sets=[lgb.Dataset(Xva, yva)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    auc = roc_auc_score(yva, mdl.predict(Xva))
    print(f"   AUC = {auc:.4f}")
    imp = pd.Series(mdl.feature_importance("gain"),
                    index=mdl.feature_name()).sort_values(ascending=False)
    print("   features that give it away (gain):")
    print((imp / imp.sum() * 100).head(8).round(1).to_string())


TASKS = {
    "missingness": missingness,
    "artefacts": artefacts,
    "target": target_distribution,
    "temporal": temporal,
    "cross_station": cross_station,
    "meteorology": meteorology,
    "drift": drift,
}


def main():
    which = sys.argv[1:] or ["all"]
    names = list(TASKS) if which == ["all"] else which
    train, test = D.load_train(), D.load_test()
    for n in names:
        TASKS[n](train, test)
        print()


if __name__ == "__main__":
    main()
