"""Two-stage prediction feedback (PLAN.md Phase 7, reframed).

For a row at hour t the target is PM2.5 at t+1. So the prediction made for row
t-1 is an estimate of PM2.5 at hour t -- i.e. of `pm25_now`, the feature the
test set does not contain. Stage 2 consumes that estimate.

Two passes, not a recursion: every row's feature comes from one independent
stage-1 prediction, and the entire test set is predicted before stage 2 runs.
Nothing compounds.

Measured ceiling (fold B, on top of best_v1 at 17.678):
    true pm25_now (illegal)                     11.629   -6.05
    pm25_now degraded to our own accuracy       14.501   -3.18

CRITICAL: stage-2's training feature must come from OUT-OF-FOLD stage-1
predictions. In-sample predictions are far more accurate than anything a test
row will get, and stage 2 would learn to trust a feature that does not exist.

    python -m src.stage2 oof      # expanding-window OOF stage-1 (slow, cached)
    python -m src.stage2 test     # stage-1 predictions for the test set
    python -m src.stage2 eval     # score stage 2 against best_v1
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import models as M
from . import validate as V

OOF_PATH = C.DATA_PROCESSED / "stage1_oof.parquet"
TEST_PATH = C.DATA_PROCESSED / "stage1_test.parquet"

#: Expanding-window block boundaries. Each block is predicted by a model fit on
#: everything strictly before it. Rows before the first boundary get no stage-1
#: prediction and are dropped from stage-2 training.
OOF_BLOCKS = [
    "2013-09-01", "2014-03-01", "2014-09-01",
    "2015-03-01", "2015-09-01", "2016-03-01", "2016-09-01",
]

STAGE1_SET = "best_v1"
STAGE1_PARAMS = {"learning_rate": 0.05, "num_leaves": 127}
STAGE1_ROUNDS = 500  # no early stopping available without a validation split


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------
def generate_oof(blocks=None, rounds: int = STAGE1_ROUNDS,
                 force: bool = False) -> pd.DataFrame:
    """Expanding-window out-of-fold stage-1 predictions over the train period."""
    blocks = blocks or OOF_BLOCKS
    if OOF_PATH.exists() and not force:
        print(f"using cached {OOF_PATH.name} (pass --force to rebuild)")
        return pd.read_parquet(OOF_PATH)

    train = D.load_train()
    X = F.build_set(train, STAGE1_SET)
    y = train[C.TARGET]
    t = train[C.TIME]

    preds = pd.Series(np.nan, index=train.index, dtype="float64")
    for i in range(len(blocks) - 1):
        start, end = pd.Timestamp(blocks[i]), pd.Timestamp(blocks[i + 1])
        fit_mask = t < start
        blk_mask = (t >= start) & (t < end)
        if not blk_mask.any() or fit_mask.sum() < 5000:
            continue

        t0 = time.time()
        model = M.build_model("lightgbm", {**STAGE1_PARAMS,
                                           "num_boost_round": rounds})
        model.fit(X[fit_mask], y[fit_mask])
        preds[blk_mask] = model.predict(X[blk_mask])
        print(f"  block {start.date()} -> {end.date()}: "
              f"fit on {fit_mask.sum():,} rows, predicted {blk_mask.sum():,} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame({
        C.GROUP: train[C.GROUP].astype(str),
        C.TIME: train[C.TIME],
        "stage1": np.clip(preds, C.TARGET_MIN, C.TARGET_MAX),
    })
    out.to_parquet(OOF_PATH)
    print(f"wrote {OOF_PATH.name}: {out['stage1'].notna().sum():,} predictions")
    return out


def generate_test(rounds: int = 760, force: bool = False) -> pd.DataFrame:
    """Stage-1 predictions for the test set, from a fit on all training data."""
    if TEST_PATH.exists() and not force:
        print(f"using cached {TEST_PATH.name}")
        return pd.read_parquet(TEST_PATH)

    train, test = D.load_train(), D.load_test()
    X_tr = F.build_set(train, STAGE1_SET)
    model = M.build_model("lightgbm", {**STAGE1_PARAMS, "num_boost_round": rounds})
    model.fit(X_tr, train[C.TARGET])

    pred = np.clip(model.predict(F.build_set(test, STAGE1_SET)[X_tr.columns]),
                   C.TARGET_MIN, C.TARGET_MAX)
    out = pd.DataFrame({
        C.GROUP: test[C.GROUP].astype(str),
        C.TIME: test[C.TIME],
        "stage1": pred,
    })
    out.to_parquet(TEST_PATH)
    print(f"wrote {TEST_PATH.name}: {len(out):,} predictions")
    return out


# --------------------------------------------------------------------------
# Stage 2 features
# --------------------------------------------------------------------------
def _hat_features(stage1: pd.DataFrame, true_tail: pd.Series | None = None
                  ) -> pd.DataFrame:
    """Turn a stage-1 prediction series into pm25_now estimates.

    The prediction for row t-1 estimates PM2.5 at hour t, so shifting the
    prediction series forward by one hour gives `pm25_now` for each row.
    """
    s = (stage1.set_index([C.GROUP, C.TIME])["stage1"]
               .sort_index())
    g = s.groupby(level=0, observed=True)

    out = pd.DataFrame(index=s.index)
    out["S_pm25_now_hat"] = g.shift(1)      # prediction for t-1 == PM2.5 at t
    out["S_pm25_lag1_hat"] = g.shift(2)
    out["S_pm25_lag2_hat"] = g.shift(3)
    out["S_pm25_lag5_hat"] = g.shift(6)

    # Momentum: is the episode building or clearing?
    out["S_pm25_trend1"] = out["S_pm25_now_hat"] - out["S_pm25_lag1_hat"]
    out["S_pm25_trend3"] = out["S_pm25_now_hat"] - out["S_pm25_lag2_hat"]
    out["S_pm25_roll6"] = (g.shift(1).groupby(level=0, observed=True)
                            .rolling(6, min_periods=2).mean()
                            .reset_index(level=0, drop=True))
    out["S_pm25_vs_roll6"] = out["S_pm25_now_hat"] - out["S_pm25_roll6"]

    # The stage-1 prediction for THIS row is itself a strong summary feature.
    out["S_stage1"] = s

    if true_tail is not None:
        # The first test hour per station has a genuinely known pm25_now: the
        # last training row's target. Tiny (12 rows) but free and exact.
        out["S_pm25_now_hat"] = out["S_pm25_now_hat"].fillna(true_tail)

    return out.astype("float32")


def stage2_builder(oof: pd.DataFrame, test_pred: pd.DataFrame | None = None):
    """Feature builder: best_v1 plus the stage-1-derived pm25_now estimates."""
    frames = [oof] if test_pred is None else [oof, test_pred]
    hats = _hat_features(pd.concat(frames, ignore_index=True))

    def build(df: pd.DataFrame) -> pd.DataFrame:
        X = F.build_set(df, STAGE1_SET)
        keys = pd.MultiIndex.from_arrays(
            [df[C.GROUP].astype(str), df[C.TIME]], names=[C.GROUP, C.TIME]
        )
        h = hats.reindex(keys)
        h.index = df.index
        return pd.concat([X, h], axis=1)

    return build


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(seeds: int = 3, folds=("B",)) -> None:
    train = D.load_train()
    oof = generate_oof()
    build = stage2_builder(oof)

    n_hat = build(train)["S_pm25_now_hat"].notna().mean() * 100
    print(f"pm25_now_hat available for {n_hat:.1f}% of training rows\n")

    vs = [
        V.Variant("stage 1 only (best_v1)", features=STAGE1_SET),
        V.Variant("stage 2 (+pm25 hat)", features=build,
                  allow_leaky=tuple(c for c in build(train.head(50)).columns
                                    if c.startswith("S_pm25"))),
    ]
    V.compare(train, vs, baseline="stage 1 only (best_v1)",
              folds=list(folds), seeds=seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oof", "test", "eval", "smoke"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "oof":
        generate_oof(force=args.force)
    elif args.cmd == "test":
        generate_test(force=args.force)
    elif args.cmd == "eval":
        evaluate(seeds=args.seeds)
    elif args.cmd == "smoke":
        # One block, few rounds: verifies the plumbing before the real run.
        print("SMOKE: 2 blocks, 60 rounds -- correctness check only")
        oof = generate_oof(blocks=["2015-03-01", "2015-09-01", "2016-03-01"],
                           rounds=60, force=True)
        cov = oof["stage1"].notna()
        print(f"predictions produced for {cov.sum():,} rows")
        hats = _hat_features(oof)
        print(f"hat features: {list(hats.columns)}")
        print(f"pm25_now_hat non-null: {hats['S_pm25_now_hat'].notna().sum():,}")
        tr = D.load_train()
        pm = D.reconstruct_pm25_now(tr)
        k = pd.MultiIndex.from_arrays([tr[C.GROUP].astype(str), tr[C.TIME]])
        h = hats["S_pm25_now_hat"].reindex(k).to_numpy()
        m = ~np.isnan(h) & pm.notna().to_numpy()
        print(f"\nCHECK: hat vs TRUE pm25_now on {m.sum():,} overlapping rows")
        print(f"  corr = {np.corrcoef(h[m], pm.to_numpy()[m])[0, 1]:.4f}")
        print(f"  rmse = {np.sqrt(np.mean((h[m] - pm.to_numpy()[m]) ** 2)):.3f}")
        OOF_PATH.unlink(missing_ok=True)
        print("\nsmoke cache removed; run 'oof' for the real thing")


if __name__ == "__main__":
    main()
