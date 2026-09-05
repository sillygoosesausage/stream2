"""Local-vs-leaderboard correlation tracker (PLAN.md Phase 3, Task 3.3).

Local validation is only useful if it moves with the leaderboard. This records
both for every submission and reports the relationship, so that a decision to
trust or distrust fold B rests on evidence.

    python -m src.tracker add exp003_leads --lb 21.84
    python -m src.tracker show
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

COLUMNS = ["exp_id", "date", "fold_A", "fold_B", "local", "leaderboard", "notes"]


def _load() -> pd.DataFrame:
    if C.TRACKER_CSV.exists():
        return pd.read_csv(C.TRACKER_CSV)
    return pd.DataFrame(columns=COLUMNS)


def add(exp_id: str, leaderboard: float, notes: str = "") -> None:
    """Record a leaderboard result against the experiment's logged local score."""
    log = pd.read_csv(C.EXPERIMENT_LOG) if C.EXPERIMENT_LOG.exists() else pd.DataFrame()
    row = log[log["exp_id"] == exp_id] if len(log) else pd.DataFrame()
    if row.empty:
        raise SystemExit(
            f"'{exp_id}' is not in {C.EXPERIMENT_LOG.name}. Run it first so the "
            "local score is recorded, then add the leaderboard result."
        )
    r = row.iloc[-1]

    df = _load()
    df = df[df["exp_id"] != exp_id].dropna(axis=1, how="all")
    df = pd.concat([df, pd.DataFrame([{
        "exp_id": exp_id,
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "fold_A": r.get("fold_A_rmse"),
        "fold_B": r.get("fold_B_rmse"),
        "local": r.get("fold_B_rmse"),
        "leaderboard": leaderboard,
        "notes": notes,
    }])], ignore_index=True)
    df.to_csv(C.TRACKER_CSV, index=False)
    print(f"recorded {exp_id}: local {r.get('fold_B_rmse')} -> LB {leaderboard}")
    show()


def show() -> None:
    df = _load()
    if df.empty:
        print("No submissions tracked yet.")
        print("After scoring a submission:  python -m src.tracker add <exp_id> --lb <score>")
        return

    df = df.copy()
    df["offset"] = (df["leaderboard"] - df["local"]).round(3)
    print(df[["exp_id", "fold_A", "fold_B", "local", "leaderboard", "offset"]]
          .round(3).to_string(index=False))

    n = df["leaderboard"].notna().sum()
    print(f"\nn = {n} scored submission(s)")
    if n < 3:
        print("Need ~3+ before the relationship means anything.")
        return

    x = df["local"].to_numpy(dtype=float)
    y = df["leaderboard"].to_numpy(dtype=float)
    r = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)

    print(f"\ncorrelation local vs leaderboard : r = {r:.4f}")
    print(f"mean offset (LB - local)         : {np.mean(y - x):+.3f}")
    print(f"sd of offset                     : {np.std(y - x):.3f}")
    print(f"fit: leaderboard ~ {slope:.3f} * local + {intercept:.3f}")

    # Rank agreement matters more than absolute offset: a constant bias is
    # harmless, but if local ranking disagrees with leaderboard ranking then
    # local validation cannot be used to choose between models.
    if n >= 3:
        rank_r = pd.Series(x).corr(pd.Series(y), method="spearman")
        print(f"rank correlation (Spearman)      : {rank_r:.4f}")
        if rank_r > 0.9:
            print("\n=> Local validation tracks the leaderboard. Trust it.")
        elif rank_r > 0.6:
            print("\n=> Partial agreement. Prefer local, but verify big changes.")
        else:
            print("\n=> Local validation DISAGREES with the leaderboard. "
                  "Stop tuning on it and re-examine the fold design "
                  "(PLAN.md Appendix C).")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("exp_id")
    a.add_argument("--lb", type=float, required=True)
    a.add_argument("--notes", default="")
    sub.add_parser("show")

    args = ap.parse_args()
    if args.cmd == "add":
        add(args.exp_id, args.lb, args.notes)
    else:
        show()


if __name__ == "__main__":
    main()
