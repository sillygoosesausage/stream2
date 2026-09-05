"""Feature builders, one function per tier (PLAN.md Phase 5).

Phase 1 ships only the raw passthrough used by the baseline. Tiers A-F land in
Phase 5 and are registered in TIERS so `build_features` can compose any subset
and the experiment log can record exactly which were active.

Every builder must obey the legality rule: computable from the test CSV alone.
Nothing here may touch PM2_5_next_hour.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def tier_raw(df: pd.DataFrame) -> pd.DataFrame:
    """The measurements as given, plus hour/month. The Phase 4 baseline."""
    out = df[C.NUMERIC_RAW + C.CATEGORICAL_RAW].copy()
    out["hour"] = df[C.TIME].dt.hour.astype("int16")
    out["month"] = df[C.TIME].dt.month.astype("int16")
    return out


# Tiers A-F are implemented in Phase 5. Signature for each: take the normalised
# frame, return a DataFrame of features indexed identically.
TIERS = {
    "raw": tier_raw,
}


def build_features(df: pd.DataFrame, tiers: list[str] | None = None) -> pd.DataFrame:
    """Compose the requested tiers into one feature frame."""
    tiers = tiers or ["raw"]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise KeyError(f"Unknown feature tiers: {unknown}. Have: {sorted(TIERS)}")

    parts = [TIERS[t](df) for t in tiers]
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def feature_columns(df: pd.DataFrame, tiers: list[str] | None = None) -> list[str]:
    return list(build_features(df.head(200), tiers).columns)
