# Beijing Multi-Site Air Quality — PM2.5 Next-Hour Forecast

Inter-Uni Datathon, Stream 2. Predict `PM2_5_next_hour` for 51,063 held-out
station-hours. Metric: **RMSE**.

Full method and roadmap: **[PLAN.md](PLAN.md)**.

---

## Quick start

```bash
pip install -r requirements.txt

python -m src.train   --config experiments/configs/baseline.yaml   # validate
python -m src.predict --config experiments/configs/baseline.yaml   # submit
```

`train.py` prints per-fold RMSE with per-month / per-station / per-decile
breakdowns and appends a row to `experiments/log.csv`.
`predict.py` refits on all training data and writes
`submissions/<exp_id>.csv`, running every submission sanity check first.

## Layout

```
data/raw/          the three provided CSVs, unmodified (test(1).csv -> test.csv)
data/processed/    cached feature frames (gitignored)
src/config.py      paths, seed, column groups, fold definitions
src/data.py        loading, shared categoricals, LEAKAGE GUARD
src/features.py    feature tiers (Phase 5)
src/validate.py    fold mechanics, RMSE breakdowns, persistence bound
src/models.py      model wrappers behind one fit/predict interface
src/train.py       run an experiment, log it
src/predict.py     final fit -> submission file
experiments/       log.csv (one row per experiment) + configs/
submissions/       every submission, named by exp_id
```

## The critical constraint

The competition overview states each row contains "current PM2.5
concentration". **It does not.** There is no PM2.5 column — only the target.

In **train**, current PM2.5 is reconstructable: the previous hour's target is
this hour's PM2.5 (99.31% of rows). In **test** it is not — the test set is a
contiguous hourly block, so the preceding row is itself a test row with a
hidden target.

**A feature is legal only if it is computable from `data/raw/test.csv` alone.**

`src/data.py` enforces this two ways, and both are called before every fit:

- `leakage_guard()` — rejects feature names derived from PM2.5 history.
- `assert_test_computable()` — builds the features against the real test frame
  and fails if any come out entirely null.

`reconstruct_pm25_now()` exists but is quarantined: it is only for the
persistence upper bound and the Phase 7 recursive experiment.

## Validation

Chronological "seasonal analogue" folds, mirroring the real Sep–Feb test block
(`src/config.py:FOLDS`):

| Fold | Train | Validate | Rows |
|---|---|---|---|
| A | → 2014-09 | 2014-09 → 2015-03 | 50,799 |
| B *(primary)* | → 2015-09 | 2015-09 → 2016-03 | 51,349 |

Fold B is primary: most training data, same months as test, and almost exactly
the test set's size. Random CV would be badly optimistic here.

## Current results

| Experiment | Fold A | Fold B | Notes |
|---|---|---|---|
| Global mean | 87.61 | 102.04 | floor |
| LightGBM, raw features | 28.63 | **32.96** | honest baseline to beat |
| *Persistence (illegal)* | — | *22.25* | upper bound if PM2.5 history existed |

Runs are deterministic (seed 42, verified identical across repeat runs).

## Reproducibility

- Global seed: `src/config.py:SEED = 42`.
- Raw-file MD5s are printed on every run and recorded per experiment:
  `train.csv 31062b0e…`, `test.csv e4cdef2d…`, `sample_submission.csv ca98dae9…`.
- Every experiment logs its git commit, feature tiers, hyperparameters and
  fold scores to `experiments/log.csv`.

## Disclosure

- **External data:** none. No station coordinates or outside sources are used
  (see PLAN.md D6).
- **AI tools:** Claude (Claude Code) used for data exploration, plan authoring,
  and code scaffolding. See PLAN.md D15 for final disclosure wording.
- **Not yet installed:** `catboost` (Phase 6), `optuna` (Phase 6 tuning).
