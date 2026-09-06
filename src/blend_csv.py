"""Blend existing submission CSVs (IDEAS.md P5/X7). No refitting.

Two submitted models that score similarly but disagree row-to-row average to
something better than either, because their errors partly cancel. This is the
one variance-reduction move that costs nothing but a submission slot.

    python -m src.blend_csv exp004_tuned_wspike:0.5 exp006_bagged15:0.5 --out exp007_blend
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="exp_id:weight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", action="store_true",
                    help="rank-average instead of value-average (P7)")
    a = ap.parse_args()

    parts = []
    for spec in a.specs:
        name, _, w = spec.partition(":")
        parts.append((name, float(w) if w else 1.0))
    tot = sum(w for _, w in parts)
    parts = [(n, w / tot) for n, w in parts]

    frames = {}
    for name, _ in parts:
        df = pd.read_csv(C.SUBMISSIONS / f"{name}.csv").sort_values(C.ID)
        frames[name] = df.reset_index(drop=True)

    ids = frames[parts[0][0]][C.ID]
    for n, f in frames.items():
        assert f[C.ID].equals(ids), f"{n} has different ids"

    M = pd.DataFrame({n: f[C.TARGET] for n, f in frames.items()})
    print("pairwise correlation:")
    print(M.corr().round(5).to_string())
    print("\nmeans: " + "  ".join(f"{n}={M[n].mean():.3f}" for n in M))

    if a.rank:
        R = M.rank(pct=True)
        blended = sum(w * R[n] for n, w in parts)
        # map the blended rank back onto the leading member's value distribution
        ref = np.sort(M[parts[0][0]].to_numpy())
        pred = np.interp(blended, np.linspace(0, 1, len(ref)), ref)
    else:
        pred = sum(w * M[n].to_numpy() for n, w in parts)

    pred = np.clip(pred, C.TARGET_MIN, C.TARGET_MAX)
    out = pd.DataFrame({C.ID: ids.to_numpy(), C.TARGET: pred})

    sample = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)
    out = sample[[C.ID]].merge(out, on=C.ID, how="left")
    assert out[C.TARGET].notna().all(), "blend left null predictions"

    path = C.SUBMISSIONS / f"{a.out}.csv"
    out.to_csv(path, index=False)
    print(f"\nblend mean {out[C.TARGET].mean():.3f}   wrote {path}")


if __name__ == "__main__":
    main()
