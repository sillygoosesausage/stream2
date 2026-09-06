"""Seed averaging, model blending, and the noise floor (PLAN.md Phase 9).

Fits are cached as validation-prediction matrices, one column per seed. Once a
member's predictions exist, every downstream question -- how much does seed
averaging buy, what blend weight is best, is this difference real -- is
arithmetic on saved arrays rather than another round of model fitting.

    python -m src.ensemble fit best_v1 --seeds 8 --fold B
    python -m src.ensemble curve best_v1 --fold B
    python -m src.ensemble blend best_v1 xgb_v1 --fold B
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M
from . import validate as V

PRED_DIR = C.DATA_PROCESSED / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)

#: Registered ensemble members: name -> (feature set, model, params)
MEMBERS = {
    "best_v1":  ("best_v1", "lightgbm", {"learning_rate": 0.05, "num_leaves": 127}),
    "xgb_v1":   ("best_v1", "xgboost",  {"learning_rate": 0.05, "max_depth": 8}),
    "lgb_log":  ("best_v1", "lightgbm", {"learning_rate": 0.05, "num_leaves": 127,
                                         "log_target": True}),
    "lgb_huber": ("best_v1", "lightgbm", {"learning_rate": 0.05, "num_leaves": 127,
                                          "objective": "huber", "alpha": 20.0}),
    "lgb_wspike": ("best_v1", "lightgbm", {"learning_rate": 0.05, "num_leaves": 127,
                                           "spike_weight": 1.0}),
    # Optuna, 30 trials on fold B (experiments/tuning_best.json).
    # Single-seed 17.2368 vs 17.6784 baseline; slower learning rate with
    # heavier regularisation and more aggressive subsampling.
    "tuned_v1": ("best_v1", "lightgbm", {
        "learning_rate": 0.022009077170577436,
        "num_leaves": 242,
        "min_data_in_leaf": 134,
        "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767,
        "bagging_freq": 5,
        "lambda_l1": 4.714235909254678,
        "lambda_l2": 1.393978675022225,
    }),
    "tuned_log": ("best_v1", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "log_target": True,
    }),
    # Tuned params + spike weighting. Individually each beat the baseline
    # (17.149 and 17.103); this tests whether the gains are additive.
    "tuned_wspike": ("best_v1", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    "tuned_wspike2": ("best_v1", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 2.0,
    }),
    # Optuna, 30 trials against mean(fold A, fold B) with spike weight searched
    # (experiments/tuning_best_multifold.json). The fold-B-only config gained
    # 0.90 on B but LOST 0.63 on A; this one is chosen on the metric that
    # actually tracks the leaderboard.
    "tuned_mf": ("best_v1", "lightgbm", {
        "learning_rate": 0.023200077462577844,
        "num_leaves": 188,
        "min_data_in_leaf": 172,
        "feature_fraction": 0.6835850303764879,
        "bagging_fraction": 0.9490925080146037,
        "bagging_freq": 6,
        "lambda_l1": 0.14740587787224071,
        "lambda_l2": 0.049343841032299995,
        "spike_weight": 2.3091811318873208,
    }),
    "lgb_wspike2": ("best_v1", "lightgbm", {"learning_rate": 0.05, "num_leaves": 127,
                                            "spike_weight": 2.5}),
    # Incumbent params on best_v1 + tier G (~90 extra lead/anomaly/city columns).
    # Tier G was written but never wired to a member, so it had never been
    # scored on any fold. Pruning showed the model is not saturated on feature
    # count (all-170 beats every subset), so widening is not an obvious risk.
    "tuned_wspike_G": ("best_v3_leadmax", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    # Tier H: the families with no analogue elsewhere -- centred windows around
    # the predicted hour, pre-impute observation flags, lead interactions,
    # rain at t+1, inversion proxy. Split so a win can be attributed.
    "tuned_wspike_H": ("best_v4_H", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    "tuned_wspike_Hcw": ("best_v4_Hcw", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    "tuned_wspike_Hrest": ("best_v4_Hrest", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    # M13: half the learning rate, and let early stopping find the rounds. The
    # tuner searched lr in a band around 0.022 and never went below it.
    "tuned_wspike_lowlr": ("best_v1", "lightgbm", {
        "learning_rate": 0.011, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    # M6: randomised split thresholds. A genuine change of inductive bias for
    # one parameter -- the cheap version of the decorrelation XGBoost failed to
    # provide (corr 0.9988).
    "tuned_wspike_xt": ("best_v1", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
        "extra_trees": True,
    }),
    "tuned_wspike_Hobs": ("best_v4_Hobs", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
    "tuned_wspike_GH": ("best_v5_GH", "lightgbm", {
        "learning_rate": 0.022009077170577436, "num_leaves": 242,
        "min_data_in_leaf": 134, "feature_fraction": 0.6649900721373205,
        "bagging_fraction": 0.6542533236720767, "bagging_freq": 5,
        "lambda_l1": 4.714235909254678, "lambda_l2": 1.393978675022225,
        "spike_weight": 1.0,
    }),
}


def _path(name: str, fold: str):
    return PRED_DIR / f"{name}__fold{fold}.parquet"


def fit_member(name: str, seeds: int = 5, fold: str = "B",
               force: bool = False) -> pd.DataFrame:
    """Fit `seeds` models and cache their validation predictions."""
    path = _path(name, fold)
    if path.exists() and not force:
        got = pd.read_parquet(path)
        if got.shape[1] >= seeds:
            print(f"using cached {path.name} ({got.shape[1]} seeds)")
            return got

    # NB: copy the params dict. `base_params.pop("spike_weight")` below
    # would otherwise delete the key from MEMBERS for the life of the
    # process, so a second fit of the same member in one session would
    # silently run UNWEIGHTED.
    fset, model_name, base_params = MEMBERS[name]
    base_params = dict(base_params)
    train = D.load_train()
    X = F.build_set(train, fset)
    y = train[C.TARGET]
    trm, vam = V.fold_masks(train, fold)

    spike_w = base_params.pop("spike_weight", None)
    w = None
    if spike_w is not None:
        # Weight rows by target magnitude: RMSE is dominated by the top 5% of
        # hours (Phase 2: ~50% of variance), so trade ordinary-hour accuracy
        # for spike accuracy and see whether it nets out.
        w = (1.0 + spike_w * (y[trm] / y[trm].mean())).to_numpy()

    # Resume from whatever is already cached, and checkpoint after every seed:
    # these runs are long enough that losing completed fits to an interruption
    # is a real cost.
    cols: dict[str, np.ndarray] = {}
    if path.exists() and not force:
        cols = {c: s.to_numpy() for c, s in pd.read_parquet(path).items()}
        if cols:
            print(f"resuming from {path.name} ({len(cols)} seeds done)")

    for s in range(1, seeds + 1):
        key = f"seed{s}"
        if key in cols:
            continue
        t0 = time.time()
        params = {**base_params, "seed": s, "bagging_seed": s,
                  "feature_fraction_seed": s, "random_state": s}
        m = M.build_model(model_name, params)
        m.fit(X[trm], y[trm], X[vam], y[vam], sample_weight=w)
        p = np.clip(m.predict(X[vam]), C.TARGET_MIN, C.TARGET_MAX)
        cols[key] = p
        pd.DataFrame(cols, index=train.index[vam]).to_parquet(path)
        print(f"  {name} seed {s}: rmse {V.rmse(y[vam], p):7.3f} "
              f"({time.time() - t0:.0f}s, checkpointed)", flush=True)

    out = pd.DataFrame(cols, index=train.index[vam])
    out.to_parquet(path)
    print(f"wrote {path.name} ({out.shape[1]} seeds)")
    return out


def _truth(fold: str) -> pd.Series:
    train = D.load_train()
    _, vam = V.fold_masks(train, fold)
    return train.loc[vam, C.TARGET]


def curve(name: str, fold: str = "B") -> pd.DataFrame:
    """How much does averaging k seeds buy, and what is the noise floor at k?

    For each k, averages every combination of k seeds (capped) and reports the
    mean RMSE and the spread across those combinations. The spread at k is the
    run-to-run noise you would face when comparing k-seed models.
    """
    P = pd.read_parquet(_path(name, fold))
    y = _truth(fold).to_numpy()
    seeds = list(P.columns)

    rows = []
    for k in range(1, len(seeds) + 1):
        combos = list(itertools.combinations(seeds, k))[:20]
        scores = [V.rmse(y, P[list(c)].mean(axis=1).to_numpy()) for c in combos]
        rows.append({"k": k, "rmse": float(np.mean(scores)),
                     "sd": float(np.std(scores)),
                     "best": float(np.min(scores)), "n_combos": len(combos)})
    tab = pd.DataFrame(rows)
    print(f"\nseed-averaging curve for '{name}' (fold {fold})")
    print(tab.round(4).to_string(index=False))
    single, full = tab.iloc[0], tab.iloc[-1]
    print(f"\n  1 seed : {single['rmse']:.3f}  (sd {single['sd']:.3f})")
    print(f"  {int(full['k'])} seeds: {full['rmse']:.3f}")
    print(f"  gain from averaging: {full['rmse'] - single['rmse']:+.3f}")
    return tab


def blend(names: list[str], fold: str = "B", step: float = 0.05) -> None:
    """Best convex blend of seed-averaged members, by grid search."""
    y = _truth(fold).to_numpy()
    preds, avail = {}, []
    for n in names:
        p = _path(n, fold)
        if not p.exists():
            print(f"  (skipping {n}: no cached predictions)")
            continue
        preds[n] = pd.read_parquet(p).mean(axis=1).to_numpy()
        avail.append(n)

    print("\nmember scores (seed-averaged):")
    for n in avail:
        print(f"  {n:14s} {V.rmse(y, preds[n]):7.3f}")

    if len(avail) < 2:
        return

    print("\npairwise correlation of member predictions:")
    Pm = pd.DataFrame(preds)
    print(Pm.corr().round(4).to_string())

    best = (None, np.inf)
    if len(avail) == 2:
        for w in np.arange(0, 1 + step, step):
            r = V.rmse(y, w * preds[avail[0]] + (1 - w) * preds[avail[1]])
            if r < best[1]:
                best = ({avail[0]: round(w, 3), avail[1]: round(1 - w, 3)}, r)
    else:
        grid = np.arange(0, 1 + step, step)
        for ws in itertools.product(grid, repeat=len(avail) - 1):
            if sum(ws) > 1:
                continue
            w = np.array(list(ws) + [1 - sum(ws)])
            r = V.rmse(y, sum(w[i] * preds[n] for i, n in enumerate(avail)))
            if r < best[1]:
                best = (dict(zip(avail, w.round(3))), r)

    print(f"\nbest blend: {best[0]}")
    print(f"blend rmse: {best[1]:.3f}   "
          f"(best single member {min(V.rmse(y, preds[n]) for n in avail):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "curve", "blend"])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fold", default="B")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.cmd == "fit":
        for n in a.names:
            fit_member(n, a.seeds, a.fold, a.force)
    elif a.cmd == "curve":
        curve(a.names[0], a.fold)
    else:
        blend(a.names, a.fold)


if __name__ == "__main__":
    main()
