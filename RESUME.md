# Resume here — state as of 2026-09-06 (Phase 9 complete)

## Current best — SUBMIT THIS

**`submissions/exp004_tuned_wspike.csv`** — fold B **16.599**, projected LB ~17.9.

Model: feature set `best_v1` (170 features), LightGBM with the Optuna config
plus target-magnitude sample weighting, averaged over 5 seeds, 1847 rounds,
fitted on all 360,954 training rows. Reproduce with:

    python -m src.submit tuned_wspike --seeds 5 --exp-id exp004_tuned_wspike

## Leaderboard history

| Submission | Fold B | Leaderboard |
|---|---|---|
| exp002_best_v1 | 17.535 | 19.025 |
| exp003_best_v2_no_so2 | 17.748 | 19.641 |
| **exp004_tuned_wspike** | **16.599** | *pending* |
| exp001_baseline_raw | 32.960 | — |

Local fold B runs ~1.5 optimistic but ranks correctly. Record the next score:

    python -m src.tracker add exp004_tuned_wspike --lb <score>

## Phase 9 scorecard (all measured at k=5, noise floor 0.054)

| Step | Fold B | Delta | Separable? |
|---|---|---|---|
| baseline (1 seed) | 17.724 | — | — |
| Seed averaging | 17.502 | -0.222 | **yes** |
| XGBoost blend | 17.502 | 0.000 | **no** (corr 0.9988) |
| Tuning | 17.149 | -0.353 | **yes** |
| Spike weighting | 17.103 | -0.399 | **yes** |
| Tuning + spike (additive) | **16.599** | **-1.125** | **yes** |
| Model blending | 16.561 | -0.038 | **no** (inside noise) |

Cached member predictions: `data/processed/preds/*__foldB.parquet` (9 members,
5-8 seeds each). All blending/averaging questions are arithmetic on these.

## What did NOT work — keep for the write-up

1. **Two-stage prediction feedback** (`src/stage2.py`): -0.127, inside noise.
   The ceiling probe promised -3.18 but injected *Gaussian* noise; real stage-1
   error is a function of the same features, so the estimate re-encodes what
   the model already has. True pm25_now is worth -6.05 and is unreachable.
2. **XGBoost / model diversity**: predictions correlate 0.9988 with LightGBM.
   Optimal blend weight was 0.00. Features dominate, not the algorithm.
3. **Log-target** (21.095) and **Huber** (17.580): both lost. Log-target even
   correlated 0.9962 — a different *loss* decorrelates no better than a
   different algorithm here.
4. **Calendar features**: +4.52, driven by `A_doy` (+3.74) and `A_month`
   (+1.94) memorising which specific days were dirty in past years.
5. **Rolling windows** (+0.89), **interaction ratios** (+0.24).
6. **Dropping SO2** (D16): local tie, but +0.62 on the leaderboard.

## Remaining ideas, untested

- Fold A is 22.6 vs fold B 16.6. That 6-point gap is unexplained and worth
  understanding — it may indicate fold B is an easy window.
- Spike-weight strength between 1.0 and 2.5 barely moved the score; a two-stage
  high/normal regime split (PLAN Phase 8 Task 8.4) was never tried.
- Per-station or per-month bias correction (Phase 10 Task 10.2).
- CatBoost (never installed) — though given finding 2, expect little.

## Cached artifacts (do not rebuild)

| File | What |
|---|---|
| `data/processed/stage1_oof.parquet` | Expanding-window OOF stage-1 predictions, 309,029 rows (~72s to rebuild) |
| `data/processed/stage1_test.parquet` | Stage-1 predictions for the test set, 51,063 rows |
| `data/processed/oof/*.parquet` | Per-experiment OOF predictions for ensembling |
| `submissions/*.csv` | Three generated submissions |
| `experiments/log.csv` | Experiment log |
| `experiments/leaderboard_tracker.csv` | Local vs leaderboard, 2 entries |
| `experiments/phase5_*.csv` | Feature-tier comparison tables |

The feature panel itself is cached **in-process only** (~5s to build). Adding a
disk cache for it is a small, worthwhile task.

---

## Next steps, in order

The user asked for all four, seed averaging first so the noise floor drops
before the rest are measured, each reported as a seeded variant against
`best_v1` with a clear separable / not-separable verdict.

1. ~~**Seed averaging**~~ — **DONE. −0.222, separable.** Fold B 17.724 → 17.502.
   Predictions cached. Use k=5 for everything below (noise floor sd 0.054).
2. **XGBoost + blend** — `python -m src.ensemble fit xgb_v1 --seeds 5 --fold B`
   then `python -m src.ensemble blend best_v1 xgb_v1 --fold B`.
   Expected −0.2 to −0.5. ~8 min. (`XGBoostModel` is written and untested.)
3. **Hyperparameter tuning** — `python -m src.tune --trials 30`. optuna 4.9.0 is
   installed. Single-seed search, so the winner **must** be re-measured with
   seed averaging. Expected −0.3 to −1.0. ~20 min.
4. **Spike handling** — members `lgb_log`, `lgb_huber`, `lgb_wspike` are
   registered in `ensemble.MEMBERS` but never run. Top 5% of hours still carry
   ~50% of total variance. Expected gain unknown. ~15 min.

Realistic landing zone: **18.0–18.5 on the leaderboard**. The structural win
(lead features) is already banked; what remains is grinding.

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

    Phase 5-9 WIP: best_v1 at LB 19.02; stage2 negative result; ensemble scaffolding

Untracked/changed: `src/{ensemble,tune,stage2,eda}.py`, `RESUME.md`,
`reports/phase2_findings.md`, `experiments/`, `submissions/`, updated
`PLAN.md` / `README.md` / `src/{features,models,validate,config,train,predict}.py`.
