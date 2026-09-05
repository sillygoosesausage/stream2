"""The validation harness (PLAN.md Phase 3).

Design goal: make running many variants and ranking them cheap, so that
decisions get settled by experiment rather than by argument. See `compare`.

Folds are chronological "seasonal analogues" of the real test block -- a
contiguous Sep-Feb window immediately after the training data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M

#: A feature spec is either a list of tier names or a callable df -> DataFrame.
FeatureSpec = Sequence[str] | Callable[[pd.DataFrame], pd.DataFrame]


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _build(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if callable(spec):
        return spec(df)
    if isinstance(spec, str):          # a named set from features.FEATURE_SETS
        return F.build_set(df, spec)
    return F.build_features(df, list(spec))


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------
def fold_masks(df: pd.DataFrame, fold: str) -> tuple[pd.Series, pd.Series]:
    """Boolean (train, val) masks for a named fold, with peek-forward asserts."""
    spec = C.FOLDS[fold]
    t = df[C.TIME]
    train = t < pd.Timestamp(spec["train_end"])
    val = (t >= pd.Timestamp(spec["val_start"])) & (t < pd.Timestamp(spec["val_end"]))

    assert not (train & val).any(), f"fold {fold}: train and val overlap"
    assert train.any() and val.any(), f"fold {fold}: empty split"
    assert t[train].max() <= t[val].min(), f"fold {fold}: train extends past val start"
    return train, val


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
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
    fit_seconds: float


@dataclass
class CVResult:
    name: str
    scores: dict[str, float]
    results: dict[str, FoldResult] = field(repr=False, default_factory=dict)
    n_features: int = 0
    seconds: float = 0.0

    @property
    def primary(self) -> float:
        return self.scores[C.PRIMARY_FOLD]

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.scores.values())))

    @property
    def spread(self) -> float:
        v = list(self.scores.values())
        return float(max(v) - min(v))

    def oof(self) -> pd.Series:
        """Concatenated validation predictions across folds.

        Fold windows are disjoint, so this is a partial out-of-fold vector
        covering the validated years -- enough to fit blend weights in Phase 9.
        """
        return pd.concat([r.predictions for r in self.results.values()]).sort_index()


def _breakdowns(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray):
    """Per-month / per-station / per-target-decile RMSE.

    The decile view is the important one: it separates gains on ordinary hours
    from gains on the pollution spikes that carry ~50% of total variance.
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


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def run_fold(
    df: pd.DataFrame,
    fold: str,
    model_name: str = "lightgbm",
    model_params: dict | None = None,
    features: FeatureSpec = ("raw",),
    *,
    X: pd.DataFrame | None = None,
    sample_weight: Callable[[pd.DataFrame], np.ndarray] | None = None,
    allow_leaky: tuple[str, ...] = (),
) -> FoldResult:
    """Fit on a fold's training window, score on its validation window.

    `X` lets a caller pass a prebuilt feature frame so that comparing many
    model configs on identical features does not rebuild them each time.
    """
    train_mask, val_mask = fold_masks(df, fold)

    if X is None:
        X = _build(df, features)
    D.leakage_guard(X, list(X.columns), allow=allow_leaky)

    y = df[C.TARGET]
    # Weight functions receive the training slice itself, so they can key off
    # timestamps (recency weighting, period restriction) as well as the target.
    w = sample_weight(df[train_mask]) if sample_weight else None

    t0 = time.time()
    model = M.build_model(model_name, model_params)
    model.fit(X[train_mask], y[train_mask], X[val_mask], y[val_mask], sample_weight=w)
    fit_seconds = time.time() - t0

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
        fit_seconds=fit_seconds,
    )


def cross_validate(
    df: pd.DataFrame,
    model_name: str = "lightgbm",
    model_params: dict | None = None,
    features: FeatureSpec = ("raw",),
    folds: Sequence[str] | None = None,
    *,
    name: str = "unnamed",
    sample_weight=None,
    allow_leaky: tuple[str, ...] = (),
) -> CVResult:
    folds = list(folds or C.DEFAULT_FOLDS)
    t0 = time.time()

    # Build features once and reuse across folds.
    X = _build(df, features)

    results = {}
    for f in folds:
        results[f] = run_fold(
            df, f, model_name, model_params, X=X,
            sample_weight=sample_weight, allow_leaky=allow_leaky,
        )

    return CVResult(
        name=name,
        scores={f: r.rmse for f, r in results.items()},
        results=results,
        n_features=X.shape[1],
        seconds=time.time() - t0,
    )


# --------------------------------------------------------------------------
# Comparing many variants -- the workhorse for settling decisions
# --------------------------------------------------------------------------
@dataclass
class Variant:
    """One thing to try."""
    name: str
    features: FeatureSpec = ("raw",)
    model: str = "lightgbm"
    params: dict = field(default_factory=dict)
    sample_weight: Callable | None = None
    allow_leaky: tuple[str, ...] = ()
    notes: str = ""


#: Measured run-to-run standard deviation of fold B from the seed alone
#: (7 seeds, raw+C0+D features). Deltas below ~2*this are not decidable from
#: a single run -- use `seeds=3` or more.
SEED_NOISE_SD = 0.263


def compare(
    df: pd.DataFrame,
    variants: Sequence[Variant],
    folds: Sequence[str] | None = None,
    *,
    baseline: str | None = None,
    seeds: Sequence[int] | int = 1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, CVResult]]:
    """Run every variant on the same folds and rank them.

    This is the intended way to settle a design question: enumerate the
    options, run them, keep the winner.

    `seeds` repeats each variant over several seeds and reports the mean and
    standard deviation. Single-seed differences below ~0.5 RMSE on fold B are
    indistinguishable from noise (see SEED_NOISE_SD), so pass seeds>=3 whenever
    the expected effect is small.
    """
    folds = list(folds or C.DEFAULT_FOLDS)
    seed_list = list(range(1, seeds + 1)) if isinstance(seeds, int) else list(seeds)
    out, rows = {}, []

    for i, v in enumerate(variants, 1):
        if verbose:
            print(f"[{i}/{len(variants)}] {v.name} ...", end=" ", flush=True)

        runs = []
        for s in seed_list:
            params = {**v.params, "seed": s, "bagging_seed": s,
                      "feature_fraction_seed": s}
            runs.append(cross_validate(
                df, v.model, params, v.features, folds,
                name=v.name, sample_weight=v.sample_weight,
                allow_leaky=v.allow_leaky,
            ))

        cv = runs[0]
        out[v.name] = cv
        primaries = [r.primary for r in runs]
        row = {"variant": v.name}
        row.update({f"fold_{f}": float(np.mean([r.scores[f] for r in runs]))
                    for f in folds})
        row.update({
            "mean": float(np.mean([r.mean for r in runs])),
            "primary": float(np.mean(primaries)),
            "sd": float(np.std(primaries)) if len(runs) > 1 else np.nan,
            "spread": cv.spread,
            "n_feat": cv.n_features,
            "secs": round(sum(r.seconds for r in runs), 1),
            "notes": v.notes,
        })
        rows.append(row)
        if verbose:
            sd = f" +/- {row['sd']:.3f}" if len(runs) > 1 else ""
            print(f"fold {C.PRIMARY_FOLD} = {row['primary']:.3f}{sd}  "
                  f"({row['secs']:.0f}s)")

    tab = pd.DataFrame(rows).sort_values("primary").reset_index(drop=True)

    ref = baseline or tab.iloc[0]["variant"]
    if ref in set(tab["variant"]):
        base = float(tab.loc[tab["variant"] == ref, "primary"].iloc[0])
        tab.insert(tab.columns.get_loc("primary") + 1, "delta_vs_base",
                   (tab["primary"] - base).round(3))

    if verbose:
        print(f"\n{'=' * 78}\nRANKED (by fold {C.PRIMARY_FOLD}; "
              f"delta vs '{ref}', negative = better)\n{'=' * 78}")
        show = [c for c in tab.columns if c != "notes"]
        print(tab[show].round(3).to_string(index=False))
        if "delta_vs_base" in tab:
            thr = 2 * (tab["sd"].mean() if len(seed_list) > 1 else SEED_NOISE_SD)
            undecided = tab.loc[tab["delta_vs_base"].abs() < thr, "variant"].tolist()
            print(f"\nnoise threshold ~{thr:.2f} RMSE; "
                  f"not separable from baseline: {undecided}")
    return tab, out


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def report(cv: CVResult, top_n_stations: int = 3) -> None:
    """Print the error breakdowns for one result."""
    print(f"\n--- {cv.name} ---")
    for f, r in cv.results.items():
        print(f"  fold {f}: {r.rmse:8.3f}  (n={r.n:,}, iter={r.best_iteration}, "
              f"{r.fit_seconds:.0f}s)")
    print(f"  mean {cv.mean:.3f}   spread {cv.spread:.3f}   "
          f"features {cv.n_features}")

    p = cv.results.get(C.PRIMARY_FOLD)
    if p is None:
        return
    print(f"\n  fold {C.PRIMARY_FOLD} RMSE by target decile "
          f"(0=cleanest, 9=worst):")
    print("   " + "  ".join(f"{d}:{v:.1f}" for d, v in p.by_decile.items()))
    print(f"  by month: " +
          "  ".join(f"{m}:{v:.1f}" for m, v in p.by_month.items()))
    worst = p.by_station.sort_values(ascending=False).head(top_n_stations)
    print(f"  worst stations: " +
          "  ".join(f"{s}:{v:.1f}" for s, v in worst.items()))


def persistence_baseline(df: pd.DataFrame, fold: str) -> float:
    """RMSE of "next hour equals this hour". ILLEGAL -- reference only."""
    _, val_mask = fold_masks(df, fold)
    pm25_now = D.reconstruct_pm25_now(df)
    m = val_mask & pm25_now.notna()
    return rmse(df.loc[m, C.TARGET], pm25_now[m])
