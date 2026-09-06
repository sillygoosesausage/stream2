"""Model wrappers behind one fit/predict interface.

A shared interface is what makes Phase 9 ensembling cheap: every member
produces out-of-fold predictions the same way, so blending is arithmetic
rather than glue code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Model:
    """Base interface. Subclasses implement `_fit` and `_predict`."""

    params: dict = field(default_factory=dict)
    name: str = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
            sample_weight: np.ndarray | None = None) -> "Model":
        self._fit(X, y, X_val, y_val, sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(X)

    def _fit(self, X, y, X_val, y_val, sample_weight):  # pragma: no cover
        raise NotImplementedError

    def _predict(self, X):  # pragma: no cover - interface
        raise NotImplementedError


class MeanModel(Model):
    """Predict the global training mean. The floor everything must beat."""

    name = "mean"

    def _fit(self, X, y, X_val, y_val, sample_weight=None):
        self.value_ = float(np.mean(y))

    def _predict(self, X):
        return np.full(len(X), self.value_)


class LightGBMModel(Model):
    """LightGBM with native categorical and NaN handling.

    NaNs are passed through deliberately rather than imputed -- the brief says
    handling incomplete measurements is part of the challenge, and LightGBM
    learns a default direction per split (PLAN.md D7).
    """

    name = "lightgbm"

    DEFAULTS = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": C.SEED,
        "num_threads": 0,
    }
    NUM_BOOST_ROUND = 3000
    EARLY_STOPPING = 100

    def _fit(self, X, y, X_val, y_val, sample_weight=None):
        import lightgbm as lgb

        params = {**self.DEFAULTS, **self.params}
        rounds = params.pop("num_boost_round", self.NUM_BOOST_ROUND)

        # Train on log1p(target) and invert at predict time. Expected to LOSE
        # on RMSE -- it optimises relative error, under-weighting exactly the
        # large values RMSE cares about -- but its errors are decorrelated from
        # a plain L2 model, which is what an ensemble wants.
        self.log_target_ = bool(params.pop("log_target", False))
        if self.log_target_:
            y = np.log1p(y)
            y_val = np.log1p(y_val) if y_val is not None else None

        # sqrt sits between raw (pure RMSE) and log (relative error). It
        # compresses the tail less aggressively than log, which lost badly.
        self.sqrt_target_ = bool(params.pop("sqrt_target", False))
        if self.sqrt_target_:
            y = np.sqrt(y)
            y_val = np.sqrt(y_val) if y_val is not None else None

        # Predict PM2.5/PM10 at the target hour instead of the level. PM10 at
        # t+1 is observed (it carries ~50% of model gain), so dividing it out
        # leaves a bounded, physically stable ratio to learn and multiplies the
        # known quantity back at predict time.
        self.ratio_col_ = params.pop("ratio_target_col", None)
        if self.ratio_col_:
            self.ratio_clip_ = params.pop("ratio_clip", 3.0)
            den = X[self.ratio_col_].to_numpy()
            y = np.clip(y.to_numpy() / np.maximum(den, 1.0), 0, self.ratio_clip_)
            if y_val is not None:
                dv = X_val[self.ratio_col_].to_numpy()
                y_val = np.clip(y_val.to_numpy() / np.maximum(dv, 1.0),
                                0, self.ratio_clip_)

        dtrain = lgb.Dataset(X, y, weight=sample_weight)
        callbacks = [lgb.log_evaluation(0)]
        valid_sets = []
        if X_val is not None:
            valid_sets = [lgb.Dataset(X_val, y_val, reference=dtrain)]
            callbacks.append(lgb.early_stopping(self.EARLY_STOPPING, verbose=False))

        self.booster_ = lgb.train(
            params, dtrain, num_boost_round=rounds,
            valid_sets=valid_sets, callbacks=callbacks,
        )
        self.best_iteration_ = self.booster_.best_iteration or rounds
        return self

    def _predict(self, X):
        p = self.booster_.predict(X, num_iteration=self.best_iteration_)
        if getattr(self, "log_target_", False):
            return np.expm1(p)
        if getattr(self, "sqrt_target_", False):
            return np.square(np.maximum(p, 0))
        if getattr(self, "ratio_col_", None):
            return p * np.maximum(X[self.ratio_col_].to_numpy(), 1.0)
        return p

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.booster_.feature_importance("gain"),
            index=self.booster_.feature_name(),
        ).sort_values(ascending=False)


class XGBoostModel(Model):
    """XGBoost with native categorical support.

    Kept for ensemble diversity: it splits differently from LightGBM
    (level-wise vs leaf-wise), so its errors decorrelate even at a similar
    solo score, which is what a blend needs.
    """

    name = "xgboost"

    DEFAULTS = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "max_cat_to_onehot": 1,
        "verbosity": 0,
        "nthread": 0,
    }
    NUM_BOOST_ROUND = 3000
    EARLY_STOPPING = 100

    def _fit(self, X, y, X_val, y_val, sample_weight=None):
        import xgboost as xgb

        params = {**self.DEFAULTS, **self.params}
        rounds = params.pop("num_boost_round", self.NUM_BOOST_ROUND)
        params.pop("seed", None)
        params.pop("bagging_seed", None)
        params.pop("feature_fraction_seed", None)

        dtrain = xgb.DMatrix(X, y, weight=sample_weight, enable_categorical=True)
        evals, es = [], None
        if X_val is not None:
            evals = [(xgb.DMatrix(X_val, y_val, enable_categorical=True), "val")]
            es = self.EARLY_STOPPING

        self.booster_ = xgb.train(
            params, dtrain, num_boost_round=rounds, evals=evals,
            early_stopping_rounds=es, verbose_eval=False,
        )
        self.best_iteration_ = getattr(self.booster_, "best_iteration", rounds)
        return self

    def _predict(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X, enable_categorical=True)
        return self.booster_.predict(
            d, iteration_range=(0, self.best_iteration_ + 1)
        )


REGISTRY = {
    "mean": MeanModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}


def build_model(name: str, params: dict | None = None) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Have: {sorted(REGISTRY)}")
    return REGISTRY[name](params=params or {}, name=name)
