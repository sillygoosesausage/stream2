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
        return self.booster_.predict(X, num_iteration=self.best_iteration_)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.booster_.feature_importance("gain"),
            index=self.booster_.feature_name(),
        ).sort_values(ascending=False)


REGISTRY = {
    "mean": MeanModel,
    "lightgbm": LightGBMModel,
}


def build_model(name: str, params: dict | None = None) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Have: {sorted(REGISTRY)}")
    return REGISTRY[name](params=params or {}, name=name)
