"""Test inference, post-processing, and submission-file generation.

    python -m src.predict --config experiments/configs/baseline.yaml

Refits on ALL training data (PLAN.md Phase 11) rather than a fold's training
slice: the split is chronological, so the months closest to the test period are
the most valuable and must not be held out of the final fit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config as C
from . import data as D
from . import features as F
from . import models as M


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Every check that has ever silently ruined a submission."""
    assert list(sub.columns) == [C.ID, C.TARGET], f"bad columns: {list(sub.columns)}"
    assert len(sub) == len(sample), f"{len(sub)} rows, expected {len(sample)}"
    assert sub[C.ID].is_unique, "duplicate ids"
    assert set(sub[C.ID]) == set(sample[C.ID]), "id set differs from sample"
    assert (sub[C.ID].to_numpy() == sample[C.ID].to_numpy()).all(), \
        "ids are not in the sample's order"
    assert sub[C.TARGET].notna().all(), "null predictions"
    assert np.isfinite(sub[C.TARGET]).all(), "non-finite predictions"
    assert (sub[C.TARGET] >= 0).all(), "negative predictions"
    print(f"submission OK: {len(sub):,} rows, "
          f"mean={sub[C.TARGET].mean():.2f}, "
          f"min={sub[C.TARGET].min():.2f}, max={sub[C.TARGET].max():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tiers = cfg.get("tiers", ["raw"])

    train, test, sample = D.load_all()

    X_train = F.build_features(train, tiers)
    feature_cols = list(X_train.columns)
    D.leakage_guard(X_train, feature_cols)
    D.assert_test_computable(lambda df: F.build_features(df, tiers), feature_cols)

    n_rounds = cfg.get("final_num_boost_round")
    params = dict(cfg.get("params", {}))
    if n_rounds:
        params["num_boost_round"] = n_rounds

    model = M.build_model(cfg["model"], params)
    model.fit(X_train, train[C.TARGET])

    X_test = F.build_features(test, tiers)[feature_cols]
    pred = np.clip(model.predict(X_test), C.TARGET_MIN, C.TARGET_MAX)

    # Reindex to the sample's id order -- test rows were sorted by station/time
    # on load, so their order no longer matches the submission template.
    out = pd.DataFrame({C.ID: test[C.ID].to_numpy(), C.TARGET: pred})
    out = sample[[C.ID]].merge(out, on=C.ID, how="left")
    validate_submission(out, sample)

    path = args.out or C.SUBMISSIONS / f"{cfg['exp_id']}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
