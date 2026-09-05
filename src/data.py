"""Data loading, type normalisation, and the leakage guard.

The single most important thing in this module is `leakage_guard`. See PLAN.md
Fact 1: current PM2.5 is reconstructable in train (previous row's target) but
unavailable in test, so any feature touching PM2.5 history validates
beautifully and cannot be computed at submission time.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the type conventions that every downstream module assumes."""
    df[C.TIME] = pd.to_datetime(df[C.TIME])

    # Shared category lists across train and test. If these were inferred
    # per-frame, a category missing from one frame would shift every integer
    # code and the model would silently read the wrong values at test time.
    df["station"] = pd.Categorical(df["station"], categories=C.STATION_CATEGORIES)
    df["wd"] = pd.Categorical(df["wd"], categories=C.WD_CATEGORIES)

    for col in C.NUMERIC_RAW:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df.sort_values([C.GROUP, C.TIME]).reset_index(drop=True)


def load_train() -> pd.DataFrame:
    return _normalise(pd.read_csv(C.TRAIN_CSV))


def load_test() -> pd.DataFrame:
    return _normalise(pd.read_csv(C.TEST_CSV))


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(C.SAMPLE_SUBMISSION_CSV)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_train(), load_test(), load_sample_submission()


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
#: Name fragments that suggest a column was built from PM2.5 history. Matching
#: is on names, which is a heuristic -- `assert_test_computable` below is the
#: stronger, structural check.
_SUSPICIOUS = re.compile(
    r"(pm2[._]?5|pm25)(?!_next_hour$)|next_hour_|_target|^target", re.IGNORECASE
)


class LeakageError(AssertionError):
    """Raised when a feature frame contains something untrue at test time."""


def leakage_guard(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    allow: tuple[str, ...] = (),
) -> None:
    """Assert that `feature_cols` contains nothing derived from the target.

    Call this in every training script, immediately before fitting.

    Parameters
    ----------
    df
        The feature frame (only its columns are inspected here).
    feature_cols
        The columns that will actually be handed to the model.
    allow
        Explicit escape hatch for the Phase 7 recursive experiment, which
        legitimately uses a *predicted* PM2.5 lag. Anything passed here must be
        justified in the experiment log -- it is not a way to silence the guard.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise LeakageError(f"feature_cols not present in frame: {missing}")

    if C.TARGET in feature_cols:
        raise LeakageError(f"{C.TARGET} is in feature_cols -- that is the target.")

    flagged = [
        c for c in feature_cols
        if c not in allow and _SUSPICIOUS.search(c) and c != C.TARGET
    ]
    if flagged:
        raise LeakageError(
            "These features look derived from PM2.5 history, which does not "
            f"exist in the test set: {flagged}. If this is the Phase 7 "
            "recursive model, pass them via allow=(...) and record it in the "
            "experiment log."
        )


def assert_test_computable(
    build_features,
    feature_cols: list[str],
    *,
    allow: tuple[str, ...] = (),
) -> None:
    """Structural check: can these features actually be built from test?

    Runs the feature builder against the real test frame and confirms every
    required column comes out with at least some non-null values. A feature
    that silently becomes all-NaN on test is the failure mode this catches --
    it is invisible in validation and fatal on the leaderboard.
    """
    test = load_test()
    built = build_features(test)

    missing = [c for c in feature_cols if c not in built.columns]
    if missing:
        raise LeakageError(
            f"Features cannot be built from the test set at all: {missing}"
        )

    all_null = [
        c for c in feature_cols
        if c not in allow and built[c].isna().all()
    ]
    if all_null:
        raise LeakageError(
            f"Features are entirely null on the test set: {all_null}. They are "
            "computable in train only -- almost certainly PM2.5 history."
        )


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def raw_checksums() -> dict[str, str]:
    """MD5 of each raw file, recorded with every experiment.

    Confirms the inputs never drifted between an experiment and its rerun.
    """
    out = {}
    for path in (C.TRAIN_CSV, C.TEST_CSV, C.SAMPLE_SUBMISSION_CSV):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[path.name] = h.hexdigest()
    return out


# --------------------------------------------------------------------------
# The reconstructed PM2.5 history -- quarantined on purpose
# --------------------------------------------------------------------------
def reconstruct_pm25_now(train: pd.DataFrame) -> pd.Series:
    """Current PM2.5 for each training row: the previous hour's target.

    ILLEGAL AS A FEATURE. This exists for two legitimate uses only:

    1. Computing the persistence baseline and the "what PM2.5 history would be
       worth" upper bound (PLAN.md Fact 3).
    2. Seeding and training the Phase 7 recursive model, where it is replaced
       by the model's own predictions at inference time.

    Recoverable for ~99.3% of training rows; null where the previous
    observation is not exactly one hour earlier.
    """
    g = train.groupby(C.GROUP, observed=True)
    prev_time = g[C.TIME].shift(1)
    prev_target = g[C.TARGET].shift(1)
    contiguous = (train[C.TIME] - prev_time).dt.total_seconds().eq(3600)
    return prev_target.where(contiguous).astype("float32")
