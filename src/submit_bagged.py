"""Feature-bagged submission (the Phase 9b winner).

Each model is fitted on a different random subset of the features, with the
handful that carry most of the gain always retained. Individually the bags are
slightly worse than a full-feature model; averaged they are markedly better,
because their errors are far less correlated.

Measured on fold B, 8 models each, identical config:

    full features, 8 seeds   16.490
    feature-bagged, 8 bags   16.180   (-0.31)

Seed averaging on full features saturates by ~4 models; bagging keeps paying,
so the gain is the subsetting, not the model count.

    python -m src.submit_bagged --bags 15 --frac 0.6
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M
from . import validate as V

FEATURE_SET = "best_v1"
PARAMS = {
    "learning_rate": 0.023200077462577844,
    "num_leaves": 188,
    "min_data_in_leaf": 172,
    "feature_fraction": 0.6835850303764879,
    "bagging_fraction": 0.9490925080146037,
    "bagging_freq": 6,
    "lambda_l1": 0.14740587787224071,
    "lambda_l2": 0.049343841032299995,
}
SPIKE = 2.3091811318873208
IMPORTANCE_CSV = C.EXPERIMENTS / "feature_importance_best_v1.csv"
N_CORE = 8  # features always kept: they carry ~81% of total gain


def spike_weights(y: pd.Series) -> np.ndarray:
    return (1.0 + SPIKE * (y / y.mean())).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", type=int, default=15)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--exp-id", default="exp006_bagged")
    args = ap.parse_args()

    train, test, sample = D.load_all()
    X_tr = F.build_set(train, FEATURE_SET)
    y = train[C.TARGET]
    D.leakage_guard(X_tr, list(X_tr.columns))
    D.assert_test_computable(lambda df: F.build_set(df, FEATURE_SET),
                             list(X_tr.columns))
    X_te = F.build_set(test, FEATURE_SET)[X_tr.columns]

    imp = (pd.read_csv(IMPORTANCE_CSV, index_col=0).iloc[:, 0]
             .sort_values(ascending=False))
    core = list(imp.head(N_CORE).index)
    rest = [c for c in X_tr.columns if c not in core]
    n_sub = int(len(rest) * args.frac)
    print(f"{len(X_tr.columns)} features; each bag sees {N_CORE + n_sub}")

    # Calibrate rounds once on fold B, then scale for the full-data fit.
    trm, vam = V.fold_masks(train, "B")
    probe = M.build_model("lightgbm", {**PARAMS, "seed": 1})
    probe.fit(X_tr[trm], y[trm], X_tr[vam], y[vam],
              sample_weight=spike_weights(y[trm]))
    rounds = int(probe.best_iteration_ * len(train) / trm.sum())
    print(f"fold B stopped at {probe.best_iteration_} -> {rounds} rounds\n")

    w_full = spike_weights(y)
    rng = np.random.default_rng(0)
    preds = []
    for i in range(args.bags):
        sub = core + list(rng.choice(rest, size=n_sub, replace=False))
        t0 = time.time()
        m = M.build_model("lightgbm", {**PARAMS, "seed": i + 1,
                                       "bagging_seed": i + 1,
                                       "feature_fraction_seed": i + 1,
                                       "num_boost_round": rounds})
        m.fit(X_tr[sub], y, sample_weight=w_full)
        preds.append(m.predict(X_te[sub]))
        print(f"  bag {i + 1}/{args.bags} ({time.time() - t0:.0f}s)", flush=True)

    pred = np.clip(np.mean(preds, axis=0), C.TARGET_MIN, C.TARGET_MAX)
    out = pd.DataFrame({C.ID: test[C.ID].to_numpy(), C.TARGET: pred})
    out = sample[[C.ID]].merge(out, on=C.ID, how="left")

    from .predict import validate_submission
    validate_submission(out, sample)

    path = C.SUBMISSIONS / f"{args.exp_id}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
