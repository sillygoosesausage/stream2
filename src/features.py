"""Feature builders (PLAN.md Phase 5).

Everything is computed on a *panel*: a complete station x hour grid spanning
train and test together. Two reasons:

1. Lags and leads must mean "one real hour", not "the previous available row".
   ~2% of hours are absent from the CSVs entirely.
2. Building across the train/test boundary gives the first test hours their
   proper lag history. This is legal -- it uses only covariates, which are
   given for every test row, and never touches PM2_5_next_hour.

Tiers are column-prefixed (raw_, A_, B_, C0_, ...) so a variant selects tiers by
filtering columns on an already-built superset. Building once and slicing keeps
`validate.compare` cheap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Tier prefixes, in build order.
TIERS = ["raw", "A", "B", "C0", "C", "D", "E", "F"]

_CACHE: dict[tuple, pd.DataFrame] = {}


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------
def _panel(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Complete station x hour grid over train+test, covariates only."""
    cols = [C.TIME, C.GROUP] + C.NUMERIC_RAW + ["wd"]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)

    hours = pd.date_range(both[C.TIME].min(), both[C.TIME].max(), freq="h")
    idx = pd.MultiIndex.from_product(
        [C.STATION_CATEGORIES, hours], names=[C.GROUP, C.TIME]
    )
    panel = (both.set_index([C.GROUP, C.TIME])
                 .reindex(idx)
                 .sort_index())
    panel["wd"] = pd.Categorical(panel["wd"], categories=C.WD_CATEGORIES)
    return panel


def _impute(panel: pd.DataFrame) -> pd.DataFrame:
    """D7: short-gap interpolation, then cross-station fill, then leave NaN.

    Justified by the Phase 2 finding that gaps are bimodal (median 1-2h but up
    to 1517h) and that stations fail semi-independently (missingness correlation
    ~0.25), so other stations usually have the reading when one does not.
    """
    p = panel.copy()
    g = p.groupby(level=0, observed=True)

    # 1. interpolate gaps of <= 6h, interior only (never extrapolate the ends)
    for col in C.NUMERIC_RAW:
        p[col] = g[col].transform(
            lambda s: s.interpolate(method="linear", limit=6, limit_area="inside")
        )
        p[f"E_wasnull_{col}"] = panel[col].isna().astype("int8")

    # 2. cross-station fill: city median for the hour, scaled by this station's
    #    long-run ratio to the city, so a chronically dirty site is not filled
    #    with a city-average value
    for col in C.NUMERIC_RAW:
        city = p[col].groupby(level=1, observed=True).transform("median")
        ratio = (p[col] / city).groupby(level=0, observed=True).transform("median")
        p[col] = p[col].fillna(city * ratio)

    # 3. anything still missing (the whole city is out) stays NaN -- LightGBM
    #    routes it, and the "was null" flags above record that it happened
    return p


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------
def _tier_raw(p: pd.DataFrame) -> dict:
    out = {f"raw_{c}": p[c] for c in C.NUMERIC_RAW}
    out["raw_wd"] = p["wd"]
    out["raw_station"] = pd.Categorical(
        p.index.get_level_values(0), categories=C.STATION_CATEGORIES
    )
    return out


def _tier_A(p: pd.DataFrame) -> dict:
    """Calendar and cyclical."""
    t = p.index.get_level_values(1)
    hour, doy, month = t.hour, t.dayofyear, t.month
    out = {
        "A_hour": hour.astype("int16"),
        "A_dow": t.dayofweek.astype("int16"),
        "A_doy": doy.astype("int16"),
        "A_month": month.astype("int16"),
        "A_is_weekend": (t.dayofweek >= 5).astype("int8"),
        "A_hour_sin": np.sin(2 * np.pi * hour / 24),
        "A_hour_cos": np.cos(2 * np.pi * hour / 24),
        "A_doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "A_doy_cos": np.cos(2 * np.pi * doy / 365.25),
        # Beijing central heating: roughly 15 Nov - 15 Mar. A step change in
        # coal burning, not a smooth seasonal effect.
        "A_heating": (((month == 11) & (t.day >= 15)) | month.isin([12, 1, 2])
                      | ((month == 3) & (t.day <= 15))).astype("int8"),
    }
    return out


def _tier_B(p: pd.DataFrame) -> dict:
    """Meteorology, including derived humidity (Phase 2: much stronger than raw)."""
    temp, dewp, wspm = p["TEMP"], p["DEWP"], p["WSPM"]

    deg = p["wd"].cat.codes.astype("float32") * 22.5
    deg = deg.where(p["wd"].notna())
    rad = np.deg2rad(deg)

    # Magnus formula. RH corr 0.374 and dew point depression -0.389, versus
    # raw DEWP 0.124 / TEMP -0.122.
    rh = 100 * (np.exp(17.625 * dewp / (243.04 + dewp))
                / np.exp(17.625 * temp / (243.04 + temp)))

    return {
        # Wind as a vector: speed and direction jointly. Phase 2 showed strong
        # wind only cleans the air from the north-west (mean 20 vs 90 from ESE).
        "B_wind_u": (wspm * np.sin(rad)).astype("float32"),
        "B_wind_v": (wspm * np.cos(rad)).astype("float32"),
        "B_wd_sin": np.sin(rad).astype("float32"),
        "B_wd_cos": np.cos(rad).astype("float32"),
        "B_rh": rh.clip(0, 100).astype("float32"),
        "B_dewpoint_depression": (temp - dewp).astype("float32"),
        "B_is_raining": (p["RAIN"] > 0).astype("int8"),
        "B_stagnation": (1.0 / (wspm + 0.1)).astype("float32"),
    }


def _shift_by_station(p: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Shift within station. k>0 = lag (past), k<0 = lead (future)."""
    return p.groupby(level=0, observed=True)[col].shift(k)


def _tier_C0(p: pd.DataFrame) -> dict:
    """LEAD features -- the hour being predicted (PLAN.md Fact 4).

    Worth ~9 RMSE alone. For a row at t the target is PM2.5 at t+1, and the
    t+1 row's covariates are in the test file for 99.25% of rows.
    """
    out = {}
    for col in C.NUMERIC_RAW:
        lead1 = _shift_by_station(p, col, -1)
        out[f"C0_lead1_{col}"] = lead1.astype("float32")
        # Change across the prediction boundary: how the atmosphere actually
        # moved during the hour being predicted. Observed, not guessed.
        out[f"C0_delta1_{col}"] = (lead1 - p[col]).astype("float32")

    for col in ["PM10", "CO", "NO2", "O3", "TEMP", "WSPM", "DEWP"]:
        out[f"C0_lead2_{col}"] = _shift_by_station(p, col, -2).astype("float32")
        out[f"C0_lead3_{col}"] = _shift_by_station(p, col, -3).astype("float32")

    # Wind at t+1 as a vector.
    deg = p["wd"].cat.codes.astype("float32") * 22.5
    deg = deg.where(p["wd"].notna())
    rad = np.deg2rad(deg)
    u = (p["WSPM"] * np.sin(rad))
    v = (p["WSPM"] * np.cos(rad))
    out["C0_lead1_wind_u"] = u.groupby(level=0, observed=True).shift(-1).astype("float32")
    out["C0_lead1_wind_v"] = v.groupby(level=0, observed=True).shift(-1).astype("float32")
    out["C0_has_lead"] = _shift_by_station(p, "PM10", -1).notna().astype("int8")
    return out


def _tier_C(p: pd.DataFrame) -> dict:
    """Lags and rolling windows. PM10 gets the most attention -- Phase 2 showed
    it is the usable substitute for the banned PM2.5 history (acf 0.93 at 1h)."""
    out = {}
    lag_cols = ["PM10", "CO", "NO2", "SO2", "O3", "TEMP", "PRES", "DEWP", "WSPM"]

    for col in lag_cols:
        for k in (1, 2, 3, 6, 12, 24):
            out[f"C_lag{k}_{col}"] = _shift_by_station(p, col, k).astype("float32")
        for k in (1, 3, 24):
            out[f"C_diff{k}_{col}"] = (
                p[col] - _shift_by_station(p, col, k)
            ).astype("float32")

    g = p.groupby(level=0, observed=True)
    for col in ["PM10", "CO", "NO2", "O3", "WSPM", "TEMP"]:
        for w in (3, 6, 12, 24, 72):
            r = g[col].rolling(w, min_periods=max(2, w // 3))
            out[f"C_rmean{w}_{col}"] = (
                r.mean().reset_index(level=0, drop=True).astype("float32")
            )
            if w in (6, 24):
                out[f"C_rstd{w}_{col}"] = (
                    r.std().reset_index(level=0, drop=True).astype("float32")
                )

    # Where does PM10 sit relative to its own recent history? A normalised
    # measure of "is this an episode building or clearing".
    rm24 = out["C_rmean24_PM10"]
    out["C_pm10_vs_24h"] = (p["PM10"] / (rm24 + 1)).astype("float32")
    return out


def _tier_D(p: pd.DataFrame) -> dict:
    """Cross-station / regional. Phase 2: mean inter-station correlation 0.88,
    and a station's PM2.5 is 87.8% explained by the other 11."""
    out = {}
    by_hour = lambda s: s.groupby(level=1, observed=True)

    for col in ["PM10", "CO", "NO2", "SO2", "O3"]:
        city_mean = by_hour(p[col]).transform("mean")
        out[f"D_city_mean_{col}"] = city_mean.astype("float32")
        out[f"D_city_med_{col}"] = by_hour(p[col]).transform("median").astype("float32")
        # How anomalous is this station right now, relative to the city?
        out[f"D_anom_{col}"] = (p[col] - city_mean).astype("float32")

    for col in ["PM10", "CO"]:
        out[f"D_city_std_{col}"] = by_hour(p[col]).transform("std").astype("float32")
        out[f"D_city_max_{col}"] = by_hour(p[col]).transform("max").astype("float32")
        # Spread across the city: is a front moving through?
        out[f"D_city_range_{col}"] = (
            by_hour(p[col]).transform("max") - by_hour(p[col]).transform("min")
        ).astype("float32")

    # City aggregates at t+1 and their lags. The city-wide value at the
    # predicted hour is legal for the same reason as Tier C0.
    for col in ["PM10", "CO", "NO2"]:
        city = by_hour(p[col]).transform("mean")
        gc = city.groupby(level=0, observed=True)
        out[f"D_city_lead1_{col}"] = gc.shift(-1).astype("float32")
        out[f"D_city_lag1_{col}"] = gc.shift(1).astype("float32")
        out[f"D_city_diff1_{col}"] = (gc.shift(-1) - city).astype("float32")

    # City-wide weather (mostly shared, but smooths out station sensor noise)
    for col in ["WSPM", "TEMP"]:
        out[f"D_city_mean_{col}"] = by_hour(p[col]).transform("mean").astype("float32")
    return out


def _tier_F(p: pd.DataFrame) -> dict:
    """Interactions and source-signature ratios."""
    eps = 1.0
    temp, dewp = p["TEMP"], p["DEWP"]
    rh = 100 * (np.exp(17.625 * dewp / (243.04 + dewp))
                / np.exp(17.625 * temp / (243.04 + temp)))
    return {
        # Humid + high particulate = hygroscopic growth, the classic haze setup
        "F_pm10_x_rh": (p["PM10"] * rh.clip(0, 100) / 100).astype("float32"),
        # Particulate load weighted by how badly the air is ventilated
        "F_pm10_per_wspm": (p["PM10"] / (p["WSPM"] + 0.1)).astype("float32"),
        "F_co_per_wspm": (p["CO"] / (p["WSPM"] + 0.1)).astype("float32"),
        # Source signatures: combustion vs traffic, coal vs traffic
        "F_co_over_no2": (p["CO"] / (p["NO2"] + eps)).astype("float32"),
        "F_so2_over_no2": (p["SO2"] / (p["NO2"] + eps)).astype("float32"),
        "F_no2_over_pm10": (p["NO2"] / (p["PM10"] + eps)).astype("float32"),
        # Secondary vs primary pollution regime
        "F_o3_x_temp": (p["O3"] * temp).astype("float32"),
    }


def _tier_G(p: pd.DataFrame) -> dict:
    """Extended leads. Tier C0 was worth -9.15 RMSE; this pushes the same idea
    further, since everything at t+1..t+3 is present in the test file."""
    out = {}
    g = lambda col, k: p.groupby(level=0, observed=True)[col].shift(k)
    by_hour = lambda s: s.groupby(level=1, observed=True)

    # Deeper leads for the pollutants that matter most.
    for col in ["PM10", "CO", "NO2", "O3", "WSPM", "TEMP", "PRES", "DEWP"]:
        for k in (4, 5, 6):
            out[f"G_lead{k}_{col}"] = g(col, -k).astype("float32")

    # Changes measured across and beyond the prediction boundary.
    for col in ["PM10", "CO", "NO2", "WSPM", "PRES"]:
        out[f"G_delta2_{col}"] = (g(col, -2) - p[col]).astype("float32")
        out[f"G_delta3_{col}"] = (g(col, -3) - p[col]).astype("float32")
        # Acceleration: is the change itself speeding up?
        out[f"G_accel_{col}"] = (g(col, -2) - 2 * g(col, -1) + p[col]).astype("float32")

    # Derived meteorology AT the predicted hour. Phase 2 found RH and dew point
    # depression far stronger than their raw inputs (0.374 / -0.389 vs ~0.12).
    t1, d1 = g("TEMP", -1), g("DEWP", -1)
    rh1 = 100 * (np.exp(17.625 * d1 / (243.04 + d1))
                 / np.exp(17.625 * t1 / (243.04 + t1)))
    out["G_lead1_rh"] = rh1.clip(0, 100).astype("float32")
    out["G_lead1_dpd"] = (t1 - d1).astype("float32")
    out["G_rh_delta"] = (rh1.clip(0, 100) - 100 * (
        np.exp(17.625 * p["DEWP"] / (243.04 + p["DEWP"]))
        / np.exp(17.625 * p["TEMP"] / (243.04 + p["TEMP"]))).clip(0, 100)
    ).astype("float32")

    # City-wide aggregates at the predicted hour, for every pollutant. Tier D
    # only carried PM10/CO/NO2 leads.
    for col in ["PM10", "SO2", "NO2", "CO", "O3", "WSPM", "TEMP"]:
        city = by_hour(p[col]).transform("mean")
        gc = city.groupby(level=0, observed=True)
        out[f"G_city_lead1_{col}"] = gc.shift(-1).astype("float32")
        if col in ("PM10", "CO"):
            out[f"G_city_lead2_{col}"] = gc.shift(-2).astype("float32")
            out[f"G_city_lead3_{col}"] = gc.shift(-3).astype("float32")
        # How anomalous is this station at the predicted hour?
        out[f"G_anom_lead1_{col}"] = (g(col, -1) - gc.shift(-1)).astype("float32")

    # Spread across the city at t+1: a wide spread means a front is moving
    # through and the city mean is a poor summary.
    for col in ["PM10", "CO"]:
        gmax = by_hour(p[col]).transform("max").groupby(level=0, observed=True)
        gmin = by_hour(p[col]).transform("min").groupby(level=0, observed=True)
        out[f"G_city_range_lead1_{col}"] = (gmax.shift(-1) - gmin.shift(-1)).astype("float32")

    # PM10 at t+1 relative to this station's recent history -- normalises away
    # chronically dirty vs clean sites.
    roll = (p.groupby(level=0, observed=True)["PM10"]
             .rolling(24, min_periods=4).mean().reset_index(level=0, drop=True))
    out["G_pm10_lead1_vs_roll24"] = (g("PM10", -1) / (roll + 1)).astype("float32")
    out["G_co_lead1_vs_roll24"] = (
        g("CO", -1) / (p.groupby(level=0, observed=True)["CO"]
                        .rolling(24, min_periods=4).mean()
                        .reset_index(level=0, drop=True) + 1)
    ).astype("float32")

    # Ratios at the predicted hour.
    out["G_lead1_co_over_no2"] = (g("CO", -1) / (g("NO2", -1) + 1)).astype("float32")
    out["G_lead1_pm10_x_rh"] = (g("PM10", -1) * rh1.clip(0, 100) / 100).astype("float32")
    out["G_lead1_pm10_per_wspm"] = (g("PM10", -1) / (g("WSPM", -1) + 0.1)).astype("float32")
    return out


def _tier_H(p: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """The feature families with no analogue anywhere else in the file.

    Everything here is a covariate of the *predicted* hour or its immediate
    neighbourhood, so it is computable on the test frame exactly as on train.
    `raw` is the PRE-imputation panel, used only for the observation flags.
    """
    out = {}
    g = lambda col, k: p.groupby(level=0, observed=True)[col].shift(k)
    by_hour = lambda s: s.groupby(level=1, observed=True)

    # --- F2: windows CENTRED on the predicted hour ------------------------
    # Every window in tier C is trailing, and the trailing rollings lost
    # (+0.89). A window spanning t-1..t+3 is legal -- all of it is observed in
    # the test file -- and describes the episode the target hour sits INSIDE
    # rather than the one leading up to it. Centre them; do not also re-admit
    # the trailing ones.
    for col in ["PM10", "CO", "NO2", "WSPM", "TEMP"]:
        s3 = g(col, -3)                     # so rolling(5) spans t-1 .. t+3
        r = (s3.groupby(level=0, observed=True)
               .rolling(5, min_periods=2).agg(["mean", "std", "min", "max"])
               .reset_index(level=0, drop=True))
        for stat in ("mean", "std", "min", "max"):
            out[f"H_cw5_{stat}_{col}"] = r[stat].astype("float32")
        # Where does the predicted hour sit inside its own local window?
        out[f"H_cw5_pos_{col}"] = (
            (g(col, -1) - r["mean"]) / (r["std"] + 1e-3)
        ).astype("float32")

    # --- A5: is the dominant feature observed, or fabricated? -------------
    # C0_has_lead is built on the IMPUTED panel, so it is ~always 1 and carries
    # nothing. Built from `raw` it becomes the signal the model most needs:
    # whether C0_lead1_PM10 -- half of all feature gain -- is a real reading.
    rg = lambda col, k: raw.groupby(level=0, observed=True)[col].shift(k)
    obs = {}
    for col in ["PM10", "CO", "NO2", "SO2", "O3", "WSPM", "TEMP"]:
        o = rg(col, -1).notna()
        obs[col] = o.astype("int8")
        out[f"H_obs_lead1_{col}"] = obs[col]
    out["H_obs_lead1_n"] = sum(obs.values()).astype("int8")
    # The UN-imputed lead, NaN preserved. Measured on fold B: the 430 rows
    # (0.84%) whose lead1_PM10 was fabricated carry 16.6% of all squared error
    # -- RMSE 74.0 against 15.2 elsewhere. C0_lead1_* hands the model a filled
    # value indistinguishable from a reading; this column lets LightGBM route
    # those rows down their own branch instead.
    for col in ["PM10", "CO", "NO2"]:
        out[f"H_rawlead1_{col}"] = rg(col, -1).astype("float32")
        out[f"H_rawlead2_{col}"] = rg(col, -2).astype("float32")
    out["H_rawnow_PM10"] = raw["PM10"].astype("float32")
    out["H_obs_now_PM10"] = raw["PM10"].notna().astype("int8")
    # How many stations actually reported PM10 at the predicted hour (F16):
    # sensors fail in bad weather, so the count may proxy extreme conditions.
    out["H_city_obs_lead1_PM10"] = (
        by_hour(raw["PM10"].notna().astype("float32")).transform("sum")
        .groupby(level=0, observed=True).shift(-1).astype("float32")
    )

    # --- F18 / F4: interactions the trees must otherwise build by splitting -
    pm1 = g("PM10", -1)
    city_pm1 = (by_hour(p["PM10"]).transform("mean")
                .groupby(level=0, observed=True).shift(-1))
    out["H_ix_pm10_x_city"] = (pm1 * city_pm1 / 100.0).astype("float32")
    # When the station and the city disagree at t+1, which one is right?
    out["H_ix_pm10_over_city"] = (pm1 / (city_pm1 + 1.0)).astype("float32")
    # Advection: wind carries the plume. A product a tree gets in one split.
    deg = p["wd"].cat.codes.astype("float32") * 22.5
    deg = deg.where(p["wd"].notna())
    rad = np.deg2rad(deg)
    u1 = (p["WSPM"] * np.sin(rad)).groupby(level=0, observed=True).shift(-1)
    v1 = (p["WSPM"] * np.cos(rad)).groupby(level=0, observed=True).shift(-1)
    out["H_ix_pm10_x_windu"] = (pm1 * u1 / 10.0).astype("float32")
    out["H_ix_pm10_x_windv"] = (pm1 * v1 / 10.0).astype("float32")
    out["H_ix_city_x_wspm"] = (city_pm1 * g("WSPM", -1) / 10.0).astype("float32")

    # --- F6: scavenging happens DURING the predicted hour -----------------
    # B_is_raining is at t. Rain at t+1 is the one that removes the aerosol.
    out["H_rain_lead1"] = (g("RAIN", -1) > 0).astype("int8")
    for w in (6, 24):
        out[f"H_rain_cum{w}"] = (
            g("RAIN", -1).groupby(level=0, observed=True)
             .rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        ).astype("float32")
    # Hours since it last rained. Positions are contiguous hourly within a
    # station because the panel is a complete station x hour grid.
    wet = p["RAIN"] > 0
    idx = pd.Series(np.arange(len(p)), index=p.index)
    last_wet = idx.where(wet).groupby(level=0, observed=True).ffill()
    out["H_rain_hours_since"] = ((idx - last_wet).fillna(999)
                                 .clip(0, 999).astype("float32"))

    # --- F11: vertical stability, which nothing else expresses -------------
    # Nocturnal inversions are the mechanism behind Beijing's worst episodes:
    # warm air aloft caps a shallow, calm boundary layer and pollution
    # accumulates. Proxy it with the temperature change plus low wind.
    dT = (g("TEMP", -1) - g("TEMP", 12)).astype("float32")
    out["H_inv_dtemp12"] = dT
    out["H_inv_stab"] = (dT / (g("WSPM", -1) + 0.5)).astype("float32")
    dpd1 = g("TEMP", -1) - g("DEWP", -1)
    out["H_inv_dpd_delta"] = (dpd1 - (p["TEMP"] - p["DEWP"])).astype("float32")
    # Haze forms as the dew point depression collapses with PM present.
    out["H_inv_dpd_x_pm10"] = (pm1 / dpd1.clip(lower=0.1)).astype("float32")
    return out


_BUILDERS = {
    "raw": _tier_raw, "A": _tier_A, "B": _tier_B, "C0": _tier_C0,
    "C": _tier_C, "D": _tier_D, "F": _tier_F, "G": _tier_G,
}


# --------------------------------------------------------------------------
# Superset build (cached) and tier selection
# --------------------------------------------------------------------------
def build_panel_features(impute: bool = True) -> pd.DataFrame:
    """Build every tier once over the train+test panel. Cached."""
    key = ("panel", impute)
    if key in _CACHE:
        return _CACHE[key]

    from . import data as D
    train, test = D.load_train(), D.load_test()
    panel = raw_panel = _panel(train, test)
    if impute:
        panel = _impute(panel)

    out = {}
    for tier in ["raw", "A", "B", "C0", "C", "D", "F", "G"]:
        out.update(_BUILDERS[tier](panel))
    # Tier H needs the pre-imputation panel as well, to tell a measured lead
    # from a filled one -- so it does not go through _BUILDERS.
    out.update(_tier_H(panel, raw_panel))
    if impute:
        out.update({c: panel[c] for c in panel.columns if c.startswith("E_wasnull_")})

    X = pd.DataFrame(out, index=panel.index)
    _CACHE[key] = X
    return X


def select(X: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    """Columns belonging to the requested tiers."""
    keep = [c for c in X.columns if c.split("_")[0] in set(tiers)]
    return X[keep]


def build_features(df: pd.DataFrame, tiers: list[str] | None = None,
                   impute: bool = True) -> pd.DataFrame:
    """Features for the rows of `df`, aligned to its index.

    `df` must be a normalised train or test frame (see data.load_*).
    """
    tiers = list(tiers or ["raw"])
    X = build_panel_features(impute=impute)
    keys = pd.MultiIndex.from_arrays(
        [df[C.GROUP].astype(str), df[C.TIME]], names=[C.GROUP, C.TIME]
    )
    out = select(X, tiers).reindex(keys)
    out.index = df.index
    return out


def clear_cache() -> None:
    _CACHE.clear()


# --------------------------------------------------------------------------
# Named feature sets
#
# Whole tiers are too coarse: Phase 5 found tier C's lags and diffs help while
# its rolling windows hurt (+0.89 RMSE). A named set is a keep-predicate over
# the built superset, so partial tiers are expressible and reproducible.
# --------------------------------------------------------------------------
def _tiers_only(*tiers):
    s = set(tiers)
    return lambda c: c.split("_")[0] in s


FEATURE_SETS = {
    # Phase 1/4 baseline: measurements as given.
    "baseline_raw": _tiers_only("raw"),

    # Phase 5 winner, fold B 17.692 (4 seeds, sd 0.114), 170 features.
    #   included: raw, C0 leads, D cross-station, B meteorology, C lags+diffs
    #   excluded: A calendar (+4.52), C rolling (+0.89), F ratios (+0.24),
    #             E null flags (+0.11)
    "best_v1": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D"}
        or c.startswith(("B_", "C_lag", "C_diff"))
    ),

    # best_v1 minus every SO2-derived column (D16).
    #
    # Fold B is a statistical tie: 17.691 vs 17.692 over 4 seeds. The tie-break
    # is drift, which fold B cannot measure. SO2 fell 44% between train and test
    # (Phase 2) and is the top adversarial-validation discriminator, so keeping
    # it means carrying a feature whose test-year distribution the model has
    # never seen -- for no measured gain. 155 features.
    "best_v2_no_so2": lambda c: (
        (c.split("_")[0] in {"raw", "C0", "D"}
         or c.startswith(("B_", "C_lag", "C_diff")))
        and "SO2" not in c
    ),

    # best_v1 + extended leads (tier G).
    "best_v3_leadmax": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D", "G"}
        or c.startswith(("B_", "C_lag", "C_diff"))
    ),

    # best_v1 + tier H (centred windows, observation flags, interactions,
    # rain at the target hour, inversion proxy).
    "best_v4_H": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D", "H"}
        or c.startswith(("B_", "C_lag", "C_diff"))
    ),
    # H split in two so a win can be attributed rather than assumed.
    "best_v4_Hcw": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D"}
        or c.startswith(("B_", "C_lag", "C_diff", "H_cw"))
    ),
    "best_v4_Hrest": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D"}
        or c.startswith(("B_", "C_lag", "C_diff",
                         "H_obs", "H_ix", "H_rain", "H_inv"))
    ),
    # ONLY the observation-flag / un-imputed-lead family from tier H (~16
    # columns). The V1 error breakdown showed the 0.84% of rows with a
    # fabricated lead1_PM10 carry 16.6% of all squared error, so this family is
    # aimed at a measured target; the rest of tier H is not. Tested alone
    # because best_v4_H bundles 55 columns and lost.
    "best_v4_Hobs": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D"}
        or c.startswith(("B_", "C_lag", "C_diff",
                         "H_obs", "H_rawlead", "H_rawnow"))
    ),

    # Everything: best_v1 + G + H.
    "best_v5_GH": lambda c: (
        c.split("_")[0] in {"raw", "C0", "D", "G", "H"}
        or c.startswith(("B_", "C_lag", "C_diff"))
    ),

    # Leads only -- the single highest-value tier, kept as a reference point.
    "leads_only": _tiers_only("raw", "C0"),
}


def build_set(df: pd.DataFrame, name: str, impute: bool = True) -> pd.DataFrame:
    """Features for `df` under a named set from FEATURE_SETS."""
    if name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set '{name}'. Have: {sorted(FEATURE_SETS)}")
    pred = FEATURE_SETS[name]

    X = build_panel_features(impute=impute)
    keys = pd.MultiIndex.from_arrays(
        [df[C.GROUP].astype(str), df[C.TIME]], names=[C.GROUP, C.TIME]
    )
    out = X[[c for c in X.columns if pred(c)]].reindex(keys)
    out.index = df.index
    return out
