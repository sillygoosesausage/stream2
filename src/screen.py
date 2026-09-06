"""Screen a candidate on the two properties that decide pool membership.

Section 9.7 of `reports/experiment_record.md` established that a blend member
must be decorrelated from the pool **and** within roughly 10% of the incumbent's
quality: the three most decorrelated members available (`lgb_log` 0.896,
`extra_trees` 0.935, `xgb_v1` 0.949) were the three worst additions, because at
equal weights a member contributes its own bias in proportion to its share.

Section 9.8 then established that fold B cannot rank blends at all -- not
composition, and not nested addition either. So this script deliberately does
NOT predict a blend score, and it does NOT gate on a threshold.

That restraint is forced by the evidence. Any error-correlation cutoff tight
enough to exclude the members that lost (`xgb_v1` 0.949, `extra_trees` 0.935)
also excludes tier G at 0.985 and tier Hcw at 0.993 -- the two additions that
actually gained. No local rule proposed in this project has survived contact
with the leaderboard, and four have now been falsified: nested addition (9.6,
killed by 9.8), decorrelation alone (9.5, narrowed by 9.7), solo quality
ordering (9.7, killed by 9.8), and the dilution hypothesis (9.8, killed by
exp026/exp027).

So the output places a candidate among members whose leaderboard effect is
*known*, and the decision is made by submitting. That is the only instrument
that has worked.

    python -m src.screen tuned_wspike_ratio catboost_v1
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import validate as V

PRED_DIR = C.DATA_PROCESSED / "preds"

#: Fold-B analogues of the five members of the standing entry (exp022). The
#: bagged member is proxied by its cached fold-B array, as in phase 14.
POOL = ["tuned_wspike", "best_v1", "tuned_wspike_G", "tuned_wspike_Hcw"]
POOL_NPY = [C.DATA_PROCESSED / "bag_frac35_foldB.npy"]

INCUMBENT = "tuned_wspike"

#: Members whose effect on the standing pool has been MEASURED on the
#: leaderboard. A candidate is judged by where it lands among these, not by a
#: threshold. The value is the LB delta the member produced when added.
KNOWN: dict[str, tuple[float, str]] = {
    "tuned_wspike_G":   (-0.058, "gained (exp021)"),
    "tuned_wspike_Hcw": (-0.011, "neutral (exp022)"),
    "tuned_mf":         (+0.097, "lost (exp024)"),
}


def _load(name: str, fold: str = "B") -> np.ndarray:
    path = PRED_DIR / f"{name}__fold{fold}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no cached predictions: {path}")
    return pd.read_parquet(path).mean(axis=1).to_numpy()


def _pool_prediction(fold: str = "B") -> np.ndarray:
    members = [_load(n, fold) for n in POOL]
    for p in POOL_NPY:
        a = np.load(p)
        members.append(a.mean(axis=1) if a.ndim > 1 else a)
    return np.mean(members, axis=0)


def screen(names: list[str], fold: str = "B") -> pd.DataFrame:
    train = D.load_train()
    _, vam = V.fold_masks(train, fold)
    y = train.loc[vam, C.TARGET].to_numpy()

    e_pool = _pool_prediction(fold) - y
    incumbent = V.rmse(y, _load(INCUMBENT, fold))

    rows = []
    for name in names:
        p = _load(name, fold)
        r = V.rmse(y, p)
        corr = float(np.corrcoef(p - y, e_pool)[0, 1])
        gap = (r - incumbent) / incumbent
        rows.append({
            "member": name,
            "fold_B": r,
            "gap_vs_incumbent": gap,
            "err_corr_vs_pool": corr,
            "known_LB_effect": KNOWN.get(name, (np.nan, "--"))[1],
        })

    tab = pd.DataFrame(rows).sort_values("err_corr_vs_pool")
    print(f"\nincumbent ({INCUMBENT}) fold {fold}: {incumbent:.4f}")
    print("no threshold is applied -- see the module docstring. Place the "
          "candidate among the members whose LB effect is known.\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*",
                    help="member names; default = every cached member")
    ap.add_argument("--fold", default="B")
    a = ap.parse_args()

    names = a.names
    if not names:
        names = sorted(
            os.path.basename(f).replace(f"__fold{a.fold}.parquet", "")
            for f in glob.glob(str(PRED_DIR / f"*__fold{a.fold}.parquet"))
        )
    screen(names, a.fold)


if __name__ == "__main__":
    main()
