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
    ap.add_argument("member", nargs="+",
                    help="one member, or several as name:weight for a blend")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--exp-id", default=None)
    args = ap.parse_args()

    # Parse "name:weight" specs. A blend of members that disagree ACROSS folds
    # is the one form of diversity that paid here: best_v1 is strongest on fold
    # A, the tuned configs on fold B, and the blend beats either on the mean.
    spec = []
    for m in args.member:
        name, _, w = m.partition(":")
        spec.append((name, float(w) if w else 1.0))
    total = sum(w for _, w in spec)
    spec = [(n, w / total) for n, w in spec]
    exp_id = args.exp_id or "exp_" + "_".join(n for n, _ in spec)
    print("blend: " + ", ".join(f"{n}={w:.2f}" for n, w in spec))

    train, test, sample = D.load_all()
    blended = np.zeros(len(test))

    for member_name, weight in spec:
        blended += weight * _fit_member_predict(
            member_name, train, test, args.seeds)

    pred = np.clip(blended, C.TARGET_MIN, C.TARGET_MAX)
    out = pd.DataFrame({C.ID: test[C.ID].to_numpy(), C.TARGET: pred})
    out = sample[[C.ID]].merge(out, on=C.ID, how="left")

    from .predict import validate_submission
    validate_submission(out, sample)

    path = C.SUBMISSIONS / f"{exp_id}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")


def _fit_member_predict(member: str, train, test, seeds: int) -> np.ndarray:
    """Seed-averaged predictions for one member, fitted on all training data."""
    fset, model_name, params = E.MEMBERS[member]
    params = dict(params)
    spike = params.pop("spike_weight", None)
    X_tr = F.build_set(train, fset)
    y = train[C.TARGET]

    D.leakage_guard(X_tr, list(X_tr.columns))
    D.assert_test_computable(lambda df: F.build_set(df, fset), list(X_tr.columns))
    print(f"  {member}: {X_tr.shape[1]} features, {len(train):,} rows")

    # Calibrate rounds on fold B, where early stopping is available.
    trm, vam = V.fold_masks(train, "B")
    w_fold = spike_weights(y[trm], spike) if spike is not None else None
    probe = M.build_model(model_name, {**params, "seed": 1})
    probe.fit(X_tr[trm], y[trm], X_tr[vam], y[vam], sample_weight=w_fold)
    rounds = int(probe.best_iteration_ * len(train) / trm.sum())
    print(f"  {member}: fold B stopped at {probe.best_iteration_} -> {rounds} rounds")

    w_full = spike_weights(y, spike) if spike is not None else None
    X_te = F.build_set(test, fset)[X_tr.columns]

    preds = []
    for s_i in range(1, seeds + 1):
        t0 = time.time()
        m = M.build_model(model_name, {**params, "seed": s_i, "bagging_seed": s_i,
                                       "feature_fraction_seed": s_i,
                                       "num_boost_round": rounds})
        m.fit(X_tr, y, sample_weight=w_full)
        preds.append(m.predict(X_te))
        print(f"    seed {s_i} ({time.time() - t0:.0f}s)", flush=True)
    return np.mean(preds, axis=0)


if __name__ == "__main__":
    main()
