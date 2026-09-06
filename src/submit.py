"""Final submission: seed-averaged fit on all training data.

`predict.py` handles a single model from a YAML config. This handles the
Phase 9 case -- a registered ensemble member, fitted over several seeds on the
full training set with its sample weights, predictions averaged.

    python -m src.submit tuned_wspike --seeds 5

Round count is taken from a fold-B early-stopping run and scaled for the larger
final fit, since there is no validation split once all data is used.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import ensemble as E
from . import features as F
from . import models as M
from . import validate as V


def spike_weights(y: pd.Series, strength: float) -> np.ndarray:
    return (1.0 + strength * (y / y.mean())).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("member", choices=sorted(E.MEMBERS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--exp-id", default=None)
    args = ap.parse_args()

    fset, model_name, params = E.MEMBERS[args.member]
    params = dict(params)
    spike = params.pop("spike_weight", None)
    exp_id = args.exp_id or f"exp_{args.member}"

    train, test, sample = D.load_all()
    X_tr = F.build_set(train, fset)
    y = train[C.TARGET]

    D.leakage_guard(X_tr, list(X_tr.columns))
    D.assert_test_computable(lambda df: F.build_set(df, fset), list(X_tr.columns))
    print(f"{args.member}: {X_tr.shape[1]} features, {len(train):,} training rows")

    # Calibrate the round count on fold B, where early stopping is available.
    trm, vam = V.fold_masks(train, "B")
    w_fold = spike_weights(y[trm], spike) if spike is not None else None
    probe = M.build_model(model_name, {**params, "seed": 1})
    probe.fit(X_tr[trm], y[trm], X_tr[vam], y[vam], sample_weight=w_fold)
    fold_iter = probe.best_iteration_
    rounds = int(fold_iter * len(train) / trm.sum())
    print(f"fold B early stop at {fold_iter} iters on {trm.sum():,} rows "
          f"-> {rounds} for the full fit")

    w_full = spike_weights(y, spike) if spike is not None else None
    X_te = F.build_set(test, fset)[X_tr.columns]

    preds = []
    for s in range(1, args.seeds + 1):
        t0 = time.time()
        m = M.build_model(model_name, {**params, "seed": s, "bagging_seed": s,
                                       "feature_fraction_seed": s,
                                       "num_boost_round": rounds})
        m.fit(X_tr, y, sample_weight=w_full)
        preds.append(m.predict(X_te))
        print(f"  seed {s} fitted ({time.time() - t0:.0f}s)", flush=True)

    pred = np.clip(np.mean(preds, axis=0), C.TARGET_MIN, C.TARGET_MAX)

    out = pd.DataFrame({C.ID: test[C.ID].to_numpy(), C.TARGET: pred})
    out = sample[[C.ID]].merge(out, on=C.ID, how="left")

    from .predict import validate_submission
    validate_submission(out, sample)

    path = C.SUBMISSIONS / f"{exp_id}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
