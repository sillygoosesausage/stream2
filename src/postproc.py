"""Post-processing sweeps on cached out-of-fold predictions (IDEAS.md P4/P8/P9).

The last untouched stage. Everything here is arithmetic on saved prediction
matrices, so a whole sweep costs seconds rather than a refit.

The decision rule is borrowed from the global-bias-correction failure: an
adjustment is only kept if its sign is the SAME on fold A and fold B. That
result found the optimal multiplier to be 0.95 on A and 1.04 on B -- opposite
directions -- which is how we learned that a correction fitted on one fold is
describing that year's pollution level, not a property of the model.

    python -m src.postproc tuned_wspike
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import validate as V

PRED_DIR = C.DATA_PROCESSED / "preds"


def _frame(name: str, fold: str) -> pd.DataFrame:
    """Seed-averaged predictions with station/time attached, sorted."""
    P = pd.read_parquet(PRED_DIR / f"{name}__fold{fold}.parquet")
    train = D.load_train()
    _, vam = V.fold_masks(train, fold)
    df = train.loc[vam, [C.GROUP, C.TIME, C.TARGET]].copy()
    df["pred"] = P.mean(axis=1).to_numpy()
    return df.sort_values([C.GROUP, C.TIME]).reset_index(drop=True)


def smooth(df: pd.DataFrame, window: int, w: float) -> np.ndarray:
    """Blend predictions toward a centred moving average within each station.

    The true series is smooth hour to hour; independent per-row model noise is
    not. Centred rather than trailing: at inference every hour's prediction
    already exists, so there is nothing causal to respect here.
    """
    s = (df.groupby(C.GROUP, observed=True)["pred"]
           .transform(lambda x: x.rolling(window, center=True, min_periods=1).mean()))
    return ((1 - w) * df["pred"] + w * s).to_numpy()


def stretch(df: pd.DataFrame, q: float, k: float) -> np.ndarray:
    """Monotone expansion above a quantile -- the P8 spike-expansion idea."""
    p = df["pred"].to_numpy().copy()
    thr = np.quantile(p, q)
    hi = p > thr
    p[hi] = p[hi] * (1 + k * (p[hi] - thr) / thr)
    return p


def run(name: str, folds=("A", "B")) -> pd.DataFrame:
    data = {f: _frame(name, f) for f in folds}
    base = {f: V.rmse(d[C.TARGET], d["pred"]) for f, d in data.items()}
    print(f"baseline  " + "  ".join(f"fold {f} {base[f]:.4f}" for f in folds))

    rows = []

    def add(label, fn):
        r = {"variant": label}
        for f, d in data.items():
            r[f"fold_{f}"] = V.rmse(d[C.TARGET], fn(d)) - base[f]
        rows.append(r)

    for win in (3, 5):
        for w in (0.1, 0.2, 0.3, 0.5):
            add(f"smooth w{win} blend{w}", lambda d, win=win, w=w: smooth(d, win, w))
    for q in (0.90, 0.95):
        for k in (0.02, 0.05, 0.10):
            add(f"stretch q{q} k{k}", lambda d, q=q, k=k: stretch(d, q, k))
    for m in (1.02, 1.04, 0.98):
        add(f"scale x{m}", lambda d, m=m: d["pred"].to_numpy() * m)
    for lo in (0.0, 2.0, 5.0):
        add(f"clip lo={lo}", lambda d, lo=lo: np.clip(d["pred"].to_numpy(), lo, 999))

    tab = pd.DataFrame(rows)
    cols = [f"fold_{f}" for f in folds]
    # Consistent = helps on BOTH folds. Anything else is fitting one year.
    tab["consistent"] = (tab[cols] < 0).all(axis=1)
    tab["worst"] = tab[cols].max(axis=1)
    tab = tab.sort_values("worst").reset_index(drop=True)
    print(f"\ndelta vs baseline (negative = better), member '{name}':")
    print(tab.round(4).to_string(index=False))
    keep = tab[tab["consistent"]]
    print("\nconsistent on both folds: "
          + (", ".join(keep["variant"]) if len(keep) else "NONE"))
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("member", default="tuned_wspike", nargs="?")
    a = ap.parse_args()
    run(a.member).to_csv(C.EXPERIMENTS / f"postproc_{a.member}.csv", index=False)


if __name__ == "__main__":
    main()
