"""Hyperparameter search (PLAN.md Phase 6, Task 6.6).

Searches on fold B with a single seed for speed, then the chosen config must be
re-measured with seed averaging before it is believed -- single-seed
differences below ~0.5 RMSE are noise (validate.SEED_NOISE_SD).

    python -m src.tune --trials 30
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import config as C
from . import data as D
from . import features as F
from . import models as M
from . import validate as V

STUDY_PATH = C.EXPERIMENTS / "tuning_best.json"


def objective_factory(X, y, trm, vam):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "seed": 1, "bagging_seed": 1, "feature_fraction_seed": 1,
        }
        m = M.build_model("lightgbm", params)
        m.fit(X[trm], y[trm], X[vam], y[vam])
        p = np.clip(m.predict(X[vam]), C.TARGET_MIN, C.TARGET_MAX)
        return V.rmse(y[vam], p)
    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--fold", default="B")
    ap.add_argument("--feature-set", default="best_v1")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = D.load_train()
    X = F.build_set(train, args.feature_set)
    y = train[C.TARGET]
    trm, vam = V.fold_masks(train, args.fold)

    baseline = V.rmse(
        y[vam],
        np.clip(
            M.build_model("lightgbm", {"learning_rate": 0.05, "num_leaves": 127,
                                       "seed": 1, "bagging_seed": 1,
                                       "feature_fraction_seed": 1})
            .fit(X[trm], y[trm], X[vam], y[vam]).predict(X[vam]),
            C.TARGET_MIN, C.TARGET_MAX),
    )
    print(f"current params, seed 1, fold {args.fold}: {baseline:.4f}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
    )
    done = {"n": 0}

    def cb(study, trial):
        done["n"] += 1
        print(f"  trial {done['n']:>3}/{args.trials}: {trial.value:.4f}   "
              f"best {study.best_value:.4f}", flush=True)

    study.optimize(objective_factory(X, y, trm, vam), n_trials=args.trials,
                   callbacks=[cb])

    print(f"\nbest single-seed: {study.best_value:.4f} "
          f"(baseline {baseline:.4f}, delta {study.best_value - baseline:+.4f})")
    print("best params:")
    print(json.dumps(study.best_params, indent=2))
    print("\nNOTE: single-seed. Re-measure with seed averaging before trusting.")
    STUDY_PATH.write_text(json.dumps(
        {"best_value": study.best_value, "baseline": baseline,
         "params": study.best_params, "fold": args.fold,
         "feature_set": args.feature_set}, indent=2), encoding="utf-8")
    print(f"wrote {STUDY_PATH.name}")


if __name__ == "__main__":
    main()
