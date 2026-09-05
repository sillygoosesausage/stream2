"""The validation harness (PLAN.md Phase 3).

Phase 1 provides the fold mechanics and the error breakdowns. Phase 3 extends
this with the leaderboard-correlation tracker and stricter gap assertions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fold_masks(df: pd.DataFrame, fold: str) -> tuple[pd.Series, pd.Series]:
    """Boolean (train, val) masks for a named fold.

    Both are strictly time-based: the validation block sits entirely after the
    training block, mirroring the chronological train/test split.
    """
    spec = C.FOLDS[fold]
    t = df[C.TIME]
    train = t < pd.Timestamp(spec["train_end"])
    val = (t >= pd.Timestamp(spec["val_start"])) & (t < pd.Timestamp(spec["val_end"]))

    # No fold may peek forward.
    assert not (train & val).any(), f"fold {fold}: train and val overlap"
    assert t[train].max() <= t[val].min(), f"fold {fold}: train extends past val start"
    return train, val


@dataclass
class FoldResult:
    fold: str
    rmse: float
    n: int
    best_iteration: int | None
    by_month: pd.Series
    by_station: pd.Series
    by_decile: pd.Series
    predictions: pd.Series


def _breakdowns(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray):
    """Per-month / per-station / per-target-decile RMSE.

    The breakdown is how you diagnose *why* a change helped, rather than just
    that it did -- particularly the decile view, which exposes whether gains
    came from ordinary hours or from the pollution spikes RMSE weights most.
    """
    err = pd.DataFrame({
        "sq": (y_true.to_numpy() - y_pred) ** 2,
        "month": df[C.TIME].dt.month.to_numpy(),
        "station": df[C.GROUP].astype(str).to_numpy(),
        "decile": pd.qcut(y_true, 10, labels=False, duplicates="drop"),
    })
    r = lambda s: np.sqrt(s.mean())
    return (
        err.groupby("month")["sq"].apply(r),
        err.groupby("station")["sq"].apply(r),
        err.groupby("decile")["sq"].apply(r),
    )


def run_fold(
    df: pd.DataFrame,
    fold: str,
    model_name: str,
    model_params: dict | None = None,
    tiers: list[str] | None = None,
    *,
    allow_leaky: tuple[str, ...] = (),
) -> FoldResult:
    train_mask, val_mask = fold_masks(df, fold)

    X = F.build_features(df, tiers)
    feature_cols = list(X.columns)
    D.leakage_guard(X, feature_cols, allow=allow_leaky)

    y = df[C.TARGET]
    model = M.build_model(model_name, model_params)
    model.fit(X[train_mask], y[train_mask], X[val_mask], y[val_mask])

    pred = np.clip(model.predict(X[val_mask]), C.TARGET_MIN, C.TARGET_MAX)
    val_df = df[val_mask]
    by_month, by_station, by_decile = _breakdowns(val_df, y[val_mask], pred)

    return FoldResult(
        fold=fold,
        rmse=rmse(y[val_mask], pred),
        n=int(val_mask.sum()),
        best_iteration=getattr(model, "best_iteration_", None),
        by_month=by_month,
        by_station=by_station,
        by_decile=by_decile,
        predictions=pd.Series(pred, index=val_df.index),
    )


def cross_validate(
    df: pd.DataFrame,
    model_name: str,
    model_params: dict | None = None,
    tiers: list[str] | None = None,
    folds: list[str] | None = None,
    *,
    allow_leaky: tuple[str, ...] = (),
) -> dict:
    folds = folds or list(C.FOLDS)
    results = {
        f: run_fold(df, f, model_name, model_params, tiers, allow_leaky=allow_leaky)
        for f in folds
    }
    scores = {f: r.rmse for f, r in results.items()}
    return {
        "results": results,
        "scores": scores,
        "primary": scores.get(C.PRIMARY_FOLD),
        "mean": float(np.mean(list(scores.values()))),
        "spread": float(np.max(list(scores.values())) - np.min(list(scores.values()))),
    }


def persistence_baseline(df: pd.DataFrame, fold: str) -> float:
    """RMSE of "next hour equals this hour" on a fold.

    ILLEGAL as a submission -- it needs PM2.5 history the test set does not
    have. Reported as the upper bound on what Phase 7 could recover.
    """
    _, val_mask = fold_masks(df, fold)
    pm25_now = D.reconstruct_pm25_now(df)
    m = val_mask & pm25_now.notna()
    return rmse(df.loc[m, C.TARGET], pm25_now[m])
