# Resume here — state as of 2026-09-06 (validation recalibrated)

## SUBMIT THIS NEXT

**`submissions/exp005_blend_A40_mf60.csv`** — blend of `best_v1` (0.40) and
`tuned_mf` (0.60), 5 seeds each on all training data.
Mean(A,B) **19.513**, projected LB **18.63** vs current best 18.788.

    python -m src.submit best_v1:0.4 tuned_mf:0.6 --seeds 5 --exp-id exp005_blend_A40_mf60

## THE KEY FINDING: use mean(fold A, fold B), never fold B alone

Fold B alone is overfitted and its gains do not transfer. Fold A acts as the brake.

| Metric | best_v1 -> tuned_wspike | Actual LB delta | Transfer |
|---|---|---|---|
| Fold B | -0.902 | -0.237 | **26%** |
| **Mean(A,B)** | -0.138 | -0.237 | ~100% |

**`LB ~= mean(A,B) - 0.88`, accurate to +/-0.05 on both known points:**

| | Projected | Actual | Error |
|---|---|---|---|
| best_v1 | 18.977 | 19.025 | 0.048 |
| tuned_wspike | 18.839 | 18.788 | 0.051 |

**You can now evaluate ideas without spending submissions.** Always report
mean(A,B) and the projection; never decide on fold B alone.

## Member scoreboard (seed-averaged: fold B k=5, fold A k=3)

| Member | Fold A | Fold B | Mean | Proj LB |
|---|---|---|---|---|
| **blend best_v1:0.4 + tuned_mf:0.6** | 22.276 | 16.749 | **19.513** | **18.63** |
| tuned_mf | 22.815 | 16.509 | 19.662 | 18.78 |
| tuned_wspike | 22.839 | 16.599 | 19.719 | 18.84 |
| best_v1 | **22.212** | 17.502 | 19.857 | 18.98 |
| tuned_v1 | 22.658 | 17.149 | 19.903 | 19.02 |
| lgb_wspike | 22.725 | 17.103 | 19.914 | 19.03 |

`best_v1` is BEST on fold A; the tuned configs win fold B. They fail in
different places, and that cross-fold disagreement is the only diversity that
has paid. Same-fold diversity (XGBoost, log-target) all correlated >0.996 and
blended to nothing.

## Leaderboard history

| Submission | Fold B | Mean(A,B) | Leaderboard |
|---|---|---|---|
| exp001_baseline_raw | 32.960 | — | — |
| exp002_best_v1 | 17.535 | 19.857 | 19.025 |
| exp003_best_v2_no_so2 | 17.748 | — | 19.641 |
| exp004_tuned_wspike | 16.599 | 19.719 | **18.788** |
| **exp005_blend_A40_mf60** | 16.749 | 19.513 | *pending* |

    python -m src.tracker add exp005_blend_A40_mf60 --lb <score>

## What is exhausted

- **Hyperparameter tuning.** Two 30-trial searches. The second, on mean(A,B),
  beat the first by 0.006 projected. Fold A sits at 22.8-23.2 across every
  trial regardless of configuration -- hyperparameters cannot move it.
- **Model diversity within a fold.** XGBoost corr 0.9988, blend weight 0.00.
  Log-target lost outright (21.095) and still correlated 0.9962.
- **Two-stage prediction feedback** (`src/stage2.py`): -0.127, inside noise.

## Where the remaining upside is

1. **Re-check the Phase 5 feature decisions on mean(A,B).** Every tier call was
   made on fold B alone, so some are likely wrong in the same way the tuning
   was. Cheapest real lever: cached panel, `validate.compare(..., folds=['A','B'],
   seeds=3)`. Especially re-test what was rejected as noise.
2. **More folds.** Two windows is thin, and fold A is the only thing holding
   fold B honest. A rolling-origin scheme with 4-5 Sep-Feb windows would cut
   decision variance further.
3. **Why is fold A stuck at ~22.2?** It has 155k training rows vs fold B's
   258k. If the gap is data volume, nothing fixes it; if it is something about
   the 2014-15 winter, that is worth knowing.
4. **Per-station / per-month bias correction** (PLAN Phase 10 Task 10.2), now
   measurable offline thanks to the calibration.

## Cached artifacts (do not rebuild)

| File | What |
|---|---|
| `data/processed/stage1_oof.parquet` | Expanding-window OOF stage-1 predictions, 309,029 rows (~72s to rebuild) |
| `data/processed/stage1_test.parquet` | Stage-1 predictions for the test set, 51,063 rows |
| `data/processed/oof/*.parquet` | Per-experiment OOF predictions |
| `data/processed/preds/*__fold{A,B}.parquet` | Per-seed validation predictions, 10 members. All blending/averaging is arithmetic on these |
| `experiments/tuning_best_multifold.json` | Optuna result on mean(A,B) |
| `submissions/*.csv` | Five generated submissions |
| `experiments/log.csv` | Experiment log |
| `experiments/leaderboard_tracker.csv` | Local vs leaderboard, 3 entries |
| `experiments/phase5_*.csv` | Feature-tier comparison tables |

The feature panel itself is cached **in-process only** (~5s to build). Adding a
disk cache for it is a small, worthwhile task.

---

## Settled decisions

| ID | Outcome | Evidence |
|---|---|---|
| D1 | Scripts, not notebooks | — |
| D2 | **User does all git operations** | Never run git/gh commands |
| D3 | All training data, uniform weights | Every restriction lost; last-1-year +3.77 |
| D7 | Interpolate ≤6h → cross-station fill → NaN | Worth ~3.1 RMSE (32.96 → 29.88) |
| D16 | **KEEP SO2** | Dropping it cost +0.62 on the leaderboard despite a local tie |

---

## Key findings

1. **Fact 4 — lead features.** For a row at *t* the target is PM2.5 at *t+1*,
   and the *t+1* row is in the test file with observed covariates (99.25% of
   rows). The task is a nowcast, not a forecast. Worth **−9.15 RMSE**.
2. **Fact 1 — the trap.** PM2.5 history is reconstructable in train (previous
   row's target) but absent from test. `leakage_guard` and
   `assert_test_computable` enforce this on every fit.
3. **Calendar features are toxic** (+4.52), isolated to `A_doy` (+3.74) and
   `A_month` (+1.94) — memorisation of which specific days were dirty.
4. **Seed noise is large.** Fold B sd 0.263, range 0.882 on a single seed.
   Deltas under ~0.5 are undecidable; use `compare(..., seeds=4)`.
5. **Two-stage prediction feedback failed** (−0.127, inside noise). The ceiling
   probe promised −3.18 but used *Gaussian* noise; real stage-1 error is a
   function of the same features, so the estimate re-encodes information the
   model already has. Keep as a documented negative result — `src/stage2.py`
   and its caches remain.

---

## Working preferences

- User does all git commits. Recommend commit points; never run git.
- Cheap decisions with few options: build all, score them, report the winner.
  Ask only when expensive or when validation cannot separate them.
- Warn before any run over ~5 min, with an ETA. Background it, print progress
  with `python -u`, cache intermediate results.
- Smoke-test on a small slice before a long run.

---

## Suggested commit

    Validation recalibrated: mean(A,B) predicts LB to +/-0.05; fold B alone
    transfers only 26%. Blend best_v1+tuned_mf projected 18.63.

Untracked/changed: `src/{ensemble,tune,stage2,eda}.py`, `RESUME.md`,
`reports/phase2_findings.md`, `experiments/`, `submissions/`, updated
`PLAN.md` / `README.md` / `src/{features,models,validate,config,train,predict}.py`.
