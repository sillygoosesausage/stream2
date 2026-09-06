"""Paired comparison of two cached prediction matrices (IDEAS.md V9).

The noise floor quoted everywhere in this project -- "deltas under ~0.4 on fold
B are undecidable" -- comes from comparing two *independent* means. That is the
wrong test. Both models are scored on the identical validation rows, and when
they share seeds they also share the draw of the bagging/feature subsample. The
shared variance cancels if you difference per row instead of differencing two
RMSEs, which resolves effects several times smaller.

Test statistic: mean over rows of (se_candidate - se_incumbent), where se is the
squared error of the seed-averaged prediction. Its sign is the sign of the RMSE
difference (RMSE is monotone in MSE), and a bootstrap over rows gives an honest
interval. Rows are resampled in contiguous day-long blocks because consecutive
hours are strongly autocorrelated -- an iid bootstrap would understate the
interval badly.

    python -m src.paired tuned_wspike_G tuned_wspike --fold B
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import validate as V

PRED_DIR = C.DATA_PROCESSED / "preds"


def _load(name: str, fold: str, seeds: list[str] | None) -> tuple[np.ndarray, list[str]]:
    P = pd.read_parquet(PRED_DIR / f"{name}__fold{fold}.parquet")
    if seeds is not None:
        P = P[seeds]
    return P.mean(axis=1).to_numpy(), list(P.columns)


def compare(cand: str, base: str, fold: str = "B", n_boot: int = 2000,
            block_hours: int = 24, match_seeds: bool = True) -> dict:
    train = D.load_train()
    _, vam = V.fold_masks(train, fold)
    y = train.loc[vam, C.TARGET].to_numpy()
    ts = train.loc[vam, C.TIME]

    cols_c = list(pd.read_parquet(PRED_DIR / f"{cand}__fold{fold}.parquet").columns)
    cols_b = list(pd.read_parquet(PRED_DIR / f"{base}__fold{fold}.parquet").columns)
    shared = [c for c in cols_c if c in cols_b] if match_seeds else None
    if match_seeds and not shared:
        raise SystemExit(f"no shared seeds between {cand} and {base}")

    pc, sc = _load(cand, fold, shared)
    pb, sb = _load(base, fold, shared)

    rc, rb = V.rmse(y, pc), V.rmse(y, pb)
    d = (pc - y) ** 2 - (pb - y) ** 2          # per-row paired difference in SE

    # Block bootstrap over day-long blocks of the validation window.
    block = ts.dt.floor(f"{block_hours}h").to_numpy()
    keys, inv = np.unique(block, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(keys)))
    ends = np.append(starts[1:], len(order))
    groups = [order[s:e] for s, e in zip(starts, ends)]
    sums = np.array([d[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])

    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean_d = float(d.mean())
    # Convert the MSE difference back to the RMSE scale everything is quoted in.
    to_rmse = lambda dm: float(np.sqrt(max(rb ** 2 + dm, 0.0)) - rb)

    out = {
        "candidate": cand, "baseline": base, "fold": fold,
        "seeds": shared if shared else f"{len(sc)}v{len(sb)}",
        "rmse_cand": rc, "rmse_base": rb, "rmse_delta": rc - rb,
        "paired_delta_rmse": to_rmse(mean_d),
        "ci_lo": to_rmse(lo), "ci_hi": to_rmse(hi),
        "p_worse": float((boot > 0).mean()),
        "n_rows": len(y), "n_blocks": len(groups),
    }

    print(f"\n{cand}  vs  {base}   (fold {fold}, seeds {out['seeds']})")
    print(f"  {base:22s} {rb:8.4f}")
    print(f"  {cand:22s} {rc:8.4f}   ({rc - rb:+.4f})")
    print(f"  paired delta        {out['paired_delta_rmse']:+8.4f} RMSE"
          f"   95% CI [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]"
          f"   ({len(groups)} day-blocks)")
    verdict = ("CANDIDATE BETTER" if hi < 0 else
               "CANDIDATE WORSE" if lo > 0 else "not separable")
    print(f"  P(candidate worse) = {out['p_worse']:.3f}   -> {verdict}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("baseline")
    ap.add_argument("--fold", default="B")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-match-seeds", action="store_true")
    a = ap.parse_args()
    compare(a.candidate, a.baseline, a.fold, a.boot,
            match_seeds=not a.no_match_seeds)


if __name__ == "__main__":
    main()
