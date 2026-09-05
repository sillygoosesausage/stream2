"""Run one experiment end to end and append it to the experiment log.

    python -m src.train --config experiments/configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from . import config as C
from . import data as D
from . import validate as V

LOG_FIELDS = [
    "exp_id", "date", "git_commit", "description", "feature_tiers", "model",
    "key_hyperparams", "fold_A_rmse", "fold_B_rmse", "mean_rmse",
    "leaderboard_rmse", "runtime_min", "submitted", "notes",
]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=C.ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def append_log(row: dict) -> None:
    new = not C.EXPERIMENT_LOG.exists()
    with open(C.EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--no-log", action="store_true",
                    help="run without appending to experiments/log.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    started = time.time()

    print(f"=== {cfg['exp_id']}: {cfg.get('description', '')} ===")
    print(f"raw checksums: {json.dumps(D.raw_checksums(), indent=None)}")

    train = D.load_train()
    print(f"loaded {len(train):,} rows, "
          f"{train[C.TIME].min()} -> {train[C.TIME].max()}")

    cv = V.cross_validate(
        train,
        model_name=cfg["model"],
        model_params=cfg.get("params", {}),
        features=cfg.get("feature_set") or cfg.get("tiers", ["raw"]),
        folds=cfg.get("folds"),
        name=cfg["exp_id"],
    )
    V.report(cv)

    # Keep the out-of-fold predictions for Phase 9 ensembling.
    oof_path = C.OOF_DIR / f"{cfg['exp_id']}.parquet"
    cv.oof().rename("pred").to_frame().to_parquet(oof_path)
    print(f"\noof predictions -> {oof_path.name}")

    runtime_min = (time.time() - started) / 60
    print(f"\nruntime: {runtime_min:.1f} min")

    if not args.no_log:
        append_log({
            "exp_id": cfg["exp_id"],
            "date": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "description": cfg.get("description", ""),
            "feature_tiers": cfg.get("feature_set") or "+".join(cfg.get("tiers", ["raw"])),
            "model": cfg["model"],
            "key_hyperparams": json.dumps(cfg.get("params", {})),
            "fold_A_rmse": round(cv.scores.get("A", float("nan")), 4),
            "fold_B_rmse": round(cv.scores.get("B", float("nan")), 4),
            "mean_rmse": round(cv.mean, 4),
            "leaderboard_rmse": "",
            "runtime_min": round(runtime_min, 2),
            "submitted": "no",
            "notes": cfg.get("notes", ""),
        })
        print(f"logged to {C.EXPERIMENT_LOG}")


if __name__ == "__main__":
    main()
