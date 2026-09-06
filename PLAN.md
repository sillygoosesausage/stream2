# Beijing PM2.5 Next-Hour Forecast — Full Project Plan

Target metric: **RMSE** on 51,063 hidden test rows. Lower is better.

---

## 0. Read this first — three facts that shape the entire project

I checked these against the actual data before writing anything below. They are not assumptions.

### Fact 1 — The "current PM2.5" trap (most important thing in this competition)

The overview says each row contains "Current PM2.5 concentration". **It does not.** There is no
PM2.5 column. There is only `PM2_5_next_hour`, the target.

In **train**, you can rebuild current PM2.5: sort by station and time, and the previous row's
target *is* this row's current PM2.5. This works for 99.31% of training rows.

In **test**, you cannot. Test is a contiguous hourly block (2016-08-31 23:00 → 2017-02-28 22:00,
~4,255 hours × 12 stations). The row before any test row is *also* a test row, and its target is
hidden. So current PM2.5 is unavailable for every test row except the very first hour of each
station (whose predecessor is the last training row).

**Consequence:** any feature built from lagged PM2.5 will look fantastic in local validation and
be impossible to compute at test time. If you build a model on it and then have to fill those
features with a constant or a guess at test time, your leaderboard score will collapse.

**This is the single easiest way to lose this competition, and probably how most teams will lose it.**

The rule for the whole project: **a feature is legal only if it can be computed from the test CSV
alone.** Everything in `test(1).csv` — PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM,
station, timestamp — is available at every test hour, so lags, rolling windows and cross-station
aggregates of *those* are all fair game. Only PM2.5 history is off-limits.

There is one legitimate way to use PM2.5 history: **recursive forecasting**, where your own
prediction for hour *t* becomes the "current PM2.5" input for hour *t+1*. That is Phase 7. It is
genuinely promising here but it is an advanced move with real failure modes, so it comes after a
solid non-recursive model exists.

### Fact 2 — The test period is Beijing's dirty season

| | Train | Test |
|---|---|---|
| Period | 2013-03-01 → 2016-08-31 | 2016-08-31 → 2017-02-28 |
| Length | 3.5 years | 6 months |
| Months | all | Sep, Oct, Nov, Dec, Jan, Feb |

Sep–Feb is the heating season. Monthly mean target in train: Aug 53, Sep 64, Oct 94, Nov 92,
Dec 96, Jan 87, Feb 94. The test window is roughly the worst half of the year, dominated by the
severe pollution episodes that RMSE punishes hardest.

**Consequence:** validating on a random split, or on a summer block, will mislead you. Your
validation must be a Sep–Feb block. Details in Phase 3.

### Fact 3 — Measured anchors (I ran these; they are real numbers, not estimates)

Validation = Sep 2015 – Feb 2016 held out, trained on everything before it.

| Approach | Val RMSE |
|---|---|
| Predict the global mean | 77.7 |
| **LightGBM, raw columns only, no PM2.5 history (legal)** | **33.08** |
| LightGBM + reconstructed current PM2.5 (illegal at test time) | 25.18 |
| Pure persistence: predict "same as current PM2.5" (illegal) | 22.25 |

Read this table carefully. **33.08 is your honest starting line.** The 22–25 range shows what
PM2.5 history would be worth if you could get it — that ~11-point gap is the prize Phase 7 chases.
Anyone reporting a local score near 20 has almost certainly leaked.

Feature correlations with the target (legal features only): PM10 **0.85**, CO **0.76**, NO2 0.64,
SO2 0.50, WSPM −0.28, O3 −0.13, TEMP −0.12, DEWP 0.12. PM10 and CO do most of the work — PM2.5 is
a component of PM10, so PM10 is effectively a noisy observation of the thing you're predicting.

Other data facts worth knowing:
- 12 stations, identical set in train and test.
- Missingness is mild: CO worst at 4.4% (train) / 1.5% (test), weather ~0.05–0.4%, `wd` 0.2%/2.0%.
- Target range 2–999, median 55, 99th pct 354. The 999 cap is a real ceiling in the source data.
- `wd` is 16 compass points plus NaN.
- CO is recorded in coarse steps (300, 500, 600...), so it is chunkier than it looks.
- Hourly series are near-continuous; gaps of 2–5 hours exist but are rare.

### Fact 4 — Lead features: the mirror image of Fact 1 (found in Phase 2)

Fact 1 says PM2.5 *history* is unavailable at test time. The reverse is also true and is a large
opportunity: **every other covariate is available for the hour you are predicting.**

For a row at hour *t* you predict PM2.5 at *t+1*. The test set is contiguous, so the row at *t+1*
is also in the test file — and it carries observed PM10, CO, NO2, SO2, O3 and weather **for hour
*t+1***, concurrent with the target. Measured: **99.25%** of test rows have their *t+1* row present
(the 383 that don't are the last hour of each station plus a few gaps).

This reframes the task. It is not really "forecast one hour ahead" — it is **"estimate PM2.5 at
hour *t+1* from everything except PM2.5 that was measured at hour *t+1*."** A nowcast, not a forecast.

Correlation with the target, at *t* versus at *t+1*:

| | at *t* | at *t+1* (lead) |
|---|---|---|
| PM10 | 0.848 | **0.866** |
| CO | 0.762 | **0.778** |
| SO2 | 0.499 | 0.504 |
| NO2 | 0.643 | 0.642 |

Measured effect on fold RMSE (LightGBM, otherwise identical):

| Features | Fold A | Fold B |
|---|---|---|
| raw baseline | 28.97 | 32.86 |
| + lead-1 covariates | 23.11 | 24.95 |
| + city-wide aggregates and their leads | 23.27 | **23.71** |

**A 9.15 RMSE improvement on fold B**, from features that were sitting in plain sight. This already
beats the illegal `pm25_now` model (25.18) and closes most of the distance to illegal persistence
(22.25) — which means Phase 7's recursive experiment now has far less headroom to chase, and its
priority drops accordingly.

Lead features are legal under the Fact 1 rule: they are built from the test CSV alone and touch no
PM2.5 value. `leakage_guard` permits them by design; `assert_test_computable` confirms they are
non-null on the real test frame.

---

## How to use this plan

Phases are ordered by dependency. Do not skip ahead — Phase 3 (validation) genuinely must exist
before Phase 5 (features), or you will have no way to tell whether a feature helped.

Two markers appear throughout:

- **DECISION Dn** — a design choice. **Default: settle it by experiment.** Where the options are
  few and each is cheap to build, enumerate them, run them through `validate.compare()`, log the
  table, and keep the winner — do not ask. Escalate to the user only when an option is expensive to
  build or when validation genuinely cannot separate the alternatives; in that case say why, and
  give a recommendation. Record every outcome in the Decision Register at the bottom.
- **Task** — heavy lifting. Hand these to me.

Each phase ends with an **Exit check**: what must be true before you move on.

---

## Phase 1 — Repo setup

Goal: a structure that makes the required submission materials fall out naturally instead of being
reconstructed at the end.

```
stream2/
  data/raw/            # the 3 given CSVs, never modified
  data/processed/      # cached feature frames (.parquet)
  src/
    config.py          # paths, seeds, feature lists, constants
    data.py            # loading, type fixing, timestamp handling
    features.py        # all feature builders, one function per tier
    validate.py        # the CV harness (Phase 3)
    models.py          # model wrappers with a shared fit/predict interface
    train.py           # runs one experiment end to end
    predict.py         # test inference -> submission.csv
  experiments/
    log.csv            # one row per experiment (see Appendix A)
    configs/           # one config file per experiment, for reproducibility
  submissions/         # every submission file, named by experiment id
  notebooks/
    01_eda.ipynb
    99_final_report.ipynb
  README.md
  requirements.txt
```

**Task 1.1** — Create this skeleton, move the CSVs into `data/raw/`, write `config.py` with
paths and a global seed, and generate `requirements.txt`. Note: `catboost` is not installed;
`lightgbm 4.6.0`, `xgboost 3.2.0`, `sklearn 1.8.0`, `pandas 2.3.3`, `numpy 2.3.5` are.

**Task 1.2** — Write `data.py`: load train/test, parse timestamps, cast `station` and `wd` to
category with a **shared category list across train and test** (a classic silent bug — if the
categories differ, the integer codes differ and your model reads garbage at test time).

**Task 1.3** — Write a `leakage_guard()` function that takes a feature frame and asserts no
column is derived from `PM2_5_next_hour`. Call it in every training script. Given Fact 1, this
guard is worth more than any single feature you'll build.

**DECISION D1 — Do you want notebooks or scripts as the primary workflow?**
- *Scripts + a thin notebook for the report* (recommended): reproducible, diffable, no hidden state.
  The submission explicitly gets checked for reproducibility.
- *Notebook-first*: faster to poke at, but out-of-order execution has sunk many datathon submissions.

**DECISION D2 — Git.** The repo currently has no commits and you're on `master`. Do you want me
to commit at the end of each phase? I'd recommend yes — being able to reproduce the exact code that
made a given submission is a stated review criterion.

**Exit check:** you can run `python -m src.train --config experiments/configs/baseline.yaml` and it
does something, even if that something is trivial.

---

## Phase 2 — Data audit and EDA

Goal: understand the data well enough to design features on purpose rather than by reflex. Also
produces figures you'll reuse in the final report.

**Task 2.1 — Missingness structure.** Not just the rate, the *pattern*: are missing values
isolated hours, or multi-hour outages? Do stations fail together (suggesting a central system) or
independently? Is missingness correlated with high pollution (instruments saturating)? This decides
whether interpolation is safe.

**Task 2.2 — Target distribution.** Histogram raw and log1p. Quantify the tail: what fraction of
total squared error would come from the top 1% of hours? Under RMSE this number will be large, and
it justifies Phase 8.

**Task 2.3 — Temporal structure.** Autocorrelation of PM2.5 by lag (1–72h). Hour-of-day profile.
Day-of-week profile. Month profile. Year-over-year trend — Beijing air improved measurably from
2013 to 2017, which is a distribution shift you'll have to handle.

**Task 2.4 — Cross-station structure.** Correlation matrix of stations on the same hour. My
expectation is that it's very high (pollution is regional, not local), which would make
cross-station features in Tier D one of the biggest available wins. Verify it.

**Task 2.5 — Meteorology.** PM2.5 vs wind speed and wind direction, as a polar plot. In Beijing,
north-westerly wind clears the air (mountains) and southerly wind brings it in (industrial plain).
If that shows up, wind direction deserves careful encoding, not a plain one-hot.

**Task 2.6 — Train vs test covariate drift.** Compare distributions of every feature between
train and the test CSV. Anything that shifted hard is a feature to be suspicious of. Pay specific
attention to whether pollutant levels dropped in the test period (policy changes took effect
around then) — if they did, your model may be systematically biased high.

**Task 2.7 — The 999 ceiling and other artefacts.** How many target values sit at exactly 999?
Are there frozen/repeated values suggesting sensor faults? Are there impossible values (negative
pollutants, pressure outliers)?

**DECISION D3 — After seeing Task 2.6, do you want to restrict training data by period?**
Options: use all 3.5 years / use the last 2 years only / weight recent data more heavily. There is
a real trade-off — more data vs. data that resembles the test period. This is testable in Phase 6,
but you should decide whether it's a priority.

**Exit check:** you can state, in one sentence each, what drives PM2.5 in this dataset and where
train and test differ.

---

## Phase 3 — The validation harness (build this before any modelling)

This is the highest-leverage phase in the project. A trustworthy local score lets you run 50
experiments and keep the right ones. An untrustworthy one means you're guessing.

The test set is a **single contiguous 6-month Sep–Feb block, immediately after train**. Your
validation should imitate that exactly.

**Proposed scheme — "seasonal analogue" folds:**

**Measured fold characteristics (Phase 3):**

| Fold | Train rows | Val window | Val rows | Val mean y | Baseline RMSE | Use |
|---|---|---|---|---|---|---|
| A | 155,190 | 2014-09 → 2015-02 | 50,799 | 86.8 | 28.63 | stability check |
| **B** | 257,927 | 2015-09 → 2016-02 | 51,349 | 83.2 | **32.96** | **primary** |
| C | 51,925 | 2013-09 → 2014-02 | 51,442 | 93.5 | 49.81 | **do not use** |
| R | 309,276 | 2016-03 → 2016-08 | 51,678 | 64.2 | 23.11 | off-season check only |

*(Test set, for comparison: 51,063 rows, Sep 2016 → Feb 2017.)*

Fold B is the closest analogue to the real task — same months, same "one year later" relationship,
most training data, and a validation window within 1% of the test set's size. Treat B as primary
and A as a stability check; if a change helps on B but hurts on A, be suspicious.

**Fold C is unusable and was retired.** With only 183 days of training data its baseline RMSE is
49.81 against fold B's 32.96 — so starved that it ranks changes differently. It stays defined in
`config.py` with that warning recorded, but is excluded from `DEFAULT_FOLDS`.

**Fold R is the wrong season** (Mar–Aug, mean target 64 vs 83) and is simply an easier problem.
Use it only to confirm a change is not winter-specific, never as a score.

Verified in Phase 3: every fold has a clean +1h gap between train end and val start with zero
overlap, malformed folds are rejected by assertion, and repeated runs are bit-identical.

**Task 3.1** — Implement `validate.py` exposing one function that takes a feature-building
function and a model config, runs both folds, and returns per-fold RMSE, mean, spread, plus RMSE
broken down by month, by station, and by target decile. That breakdown is how you'll diagnose *why*
something helped.

**Task 3.2** — Enforce a strict time gap so no fold can peek forward. Since all features are
backward-looking this is mostly automatic, but assert it.

**Task 3.3** — Build a **leaderboard-correlation tracker**: a small table logging local fold-B
RMSE next to the actual leaderboard score for every submission you make. After ~5 submissions you
will know whether your local score is trustworthy and what the offset is. This is worth more than
any feature.

**Task 3.4** — Fix seeds everywhere and verify a repeated run gives an identical number.

**DECISION D4 — How many leaderboard submissions per day do you get, and how do you want to spend
them?** This determines whether you can afford to probe the leaderboard or must trust local
validation. Please check the competition page and tell me. My default recommendation: spend the
first 2–3 submissions calibrating local-vs-leaderboard offset with deliberately different models,
then trust local validation for the rest.

**Exit check:** running the harness twice on the same config gives the same number, and you have a
per-month/per-station error breakdown to read.

---

## Phase 4 — Baselines

Every one of these must be beaten by anything more complicated. Record all in the experiment log.

1. Global mean → ~77.7 (measured).
2. Per-station × per-month × per-hour mean.
3. **Linear regression on PM10 + CO alone.** PM2.5 is physically part of PM10, so this simple
   model should already be decent. If a GBM can't beat it by a wide margin, something is wrong.
4. **LightGBM on raw columns, no feature engineering → 33.08 (measured).** This is the number to beat.

**Task 4.1** — Implement all four, log them, and confirm my 33.08 reproduces inside your harness.
If it doesn't, the harness differs from what I ran and that must be resolved before continuing.

**Exit check:** a populated `experiments/log.csv` with four honest rows.

---

## Phase 5 — Feature engineering

Build in tiers. **Evaluate after each tier** so you know what each is worth. Resist building all of
them and then measuring once — that tells you nothing about which to keep.

Remember the legality rule from Fact 1: computable from `test(1).csv` alone.

### Tier A — Calendar and cyclical
- hour, day-of-week, day-of-year, month, is_weekend
- sin/cos encodings of hour (period 24) and day-of-year (period 365)
- **is_heating_season** (Nov 15 – Mar 15 in Beijing; central heating switches on and PM2.5 steps up).
  This is a domain fact, not external data.
- Hours since start of series (a linear trend term) — but see the extrapolation warning in D5.

### Tier B — Meteorology, done properly
- **Wind as vectors**: `u = WSPM·sin(θ)`, `v = WSPM·cos(θ)` where θ is `wd` in radians. This
  encodes speed and direction jointly and is far better than one-hot `wd`.
- Raw `wd` as a category as well — let the model use both.
- **Relative humidity** from TEMP and DEWP (Magnus formula). Humidity drives particle growth and
  is one of the strongest known meteorological predictors of PM2.5. Derived from given columns, so legal.
- **Dew point depression** (TEMP − DEWP) — a proxy for fog/haze conditions.
- Pressure tendency: PRES change over 3h, 6h, 24h. Falling pressure precedes stagnation.
- `is_raining` flag and hours-since-rain (rain scavenges particles).
- Temperature inversion proxy: TEMP now vs TEMP 12/24h ago.
- **Ventilation index**: WSPM × (some mixing proxy). Worth trying WSPM × hour-of-day interaction.

### Tier C0 — Lead features (do this first; Fact 4)
**Highest-value tier in the plan — worth ~9 RMSE on its own.** Covariates observed at *t+1*, the
hour being predicted.
- lead-1 of every one of PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
- lead-2 and lead-3 as well (the *t+1* row's own future is also in the test file)
- **deltas across the prediction boundary**: value at *t+1* minus value at *t*. This is the
  direction the atmosphere is moving during the hour being predicted, and it is observed, not guessed.
- city-wide aggregates at *t+1* (see Tier D)
- Handle the ~0.75% of rows with no *t+1* row by leaving the lead NaN — do not drop them, they
  appear in the test set and must be predicted.

### Tier C — Lags and rolling windows of the *legal* pollutants
For each of PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM, per station:
- lags at 1, 2, 3, 6, 12, 24 hours
- rolling mean and std over 3, 6, 12, 24, 72 hours
- deltas: value now minus value 1h/3h/24h ago (trend/momentum)
- rate of change of PM10 specifically — this is your best available proxy for whether PM2.5 is
  currently rising or falling, which is exactly what "next hour" depends on

Because PM2.5 history is banned, **PM10 history is the substitute and deserves the most attention.**
Consider also PM10 ratios: PM10 now / PM10 24h-mean.

Careful with time gaps: use a reindexed-to-hourly frame per station so a "lag 1" is genuinely one
hour ago, not "previous available row". Build lags on a complete hourly index, then join back.

### Tier D — Cross-station / regional features (likely a big win)
All 12 stations appear at every test hour, so city-wide aggregates are fully legal.
- City-wide mean/median/std/max of PM10, CO, NO2, SO2, O3 at the same hour
- This station's value minus the city mean (how anomalous is this site right now)
- City-wide values lagged 1–6h, and city-wide trend
- **Upwind/downwind construction**: given the wind vector, the concentration at stations in the
  upwind direction is a physical leading indicator. This needs relative station geometry — see D6.
- Spatial gradient: max minus min across stations (is a front moving through the city?)

### Tier E — Station identity
- station as a native categorical (LightGBM handles this well)
- per-station target statistics computed **only on data strictly before the fold's validation start**
  (out-of-fold target encoding — done carelessly this leaks and is a common failure)
- per-station × hour-of-day mean, same discipline

### Tier F — Interactions and ratios
- PM10 × humidity, PM10 / WSPM (stagnation-weighted load)
- CO / NO2 (combustion source signature)
- SO2 / NO2 (coal vs traffic signature — informative in Beijing winter)

**Task 5.1–5.6** — Implement each tier as an isolated function, evaluate cumulatively, log the
delta each tier buys.

**DECISION D5 — How do you want to handle `year` and any linear time trend?**
Trees cannot extrapolate. Test includes 2017, which never appears in training, so a `year` feature
puts all 2017 rows into the "2016" leaf — harmless but useless — while a `days_since_start` feature
does the same thing more subtly and may actively mislead. Options: drop time-trend features
entirely (safest) / keep them and rely on validation / detrend the target by year first. I lean
toward dropping raw year and testing a detrend variant as an experiment.

**DECISION D6 — Station coordinates.** The upwind/downwind idea in Tier D needs to know where the
stations are relative to each other. The rules say no external data and require disclosure of
anything used beyond the competition files. Looking up the 12 stations' latitudes and longitudes
would be external data. **Your options:**
- *Don't use coordinates* (safest, and what I'd recommend) — you can still use city-wide aggregates,
  which capture most of the value.
- *Learn relative geometry from the data itself* — e.g. estimate which stations lead which by
  cross-correlating their pollutant series at various lags under different wind directions. This is
  derived entirely from the provided data and is legal. More work, and genuinely interesting.
- *Use published coordinates and disclose them.* Possibly permitted since the disclosure section
  anticipates external data, but it risks being read as violating "must not use external copies of
  the data". I would not.

This is a rules-interpretation call and it's yours. If you want, ask the organisers directly.

**DECISION D7 — Missing value strategy.**
- *Leave NaN and let LightGBM/XGBoost route them* (recommended default — the brief explicitly says
  handling incomplete measurements is part of the challenge, and native NaN handling is principled)
- *Time-interpolate within station* for short gaps only, keeping NaN for long outages
- *Impute + add "was missing" indicator columns*

Recommendation: start with native NaN, then test adding missingness-indicator columns as a separate
experiment — sometimes the fact that a sensor failed is itself informative.

**Exit check:** a table showing val RMSE after each tier, so you know what everything cost and bought.

---

## Phase 6 — Model development

Only start once features are stable. Use identical features across models so comparisons are fair.

**Task 6.1 — LightGBM.** Primary workhorse. Tune in this order (roughly decreasing importance):
`num_leaves`, `min_data_in_leaf`, `learning_rate` + `n_estimators` (jointly), `feature_fraction`,
`bagging_fraction` + `bagging_freq`, then `lambda_l1`/`lambda_l2`. Use early stopping on fold B.

**Task 6.2 — XGBoost.** `hist` tree method. Different enough from LightGBM to be useful in an
ensemble even at slightly worse solo score.

**Task 6.3 — CatBoost.** Not installed; would need `pip install catboost`. Often the best of the
three on data with meaningful categoricals (station, wd) and it handles them natively. Worth it.

**Task 6.4 — A linear/ridge model** on a well-chosen feature subset. It will lose badly alone but
its errors are decorrelated from the trees, which is exactly what an ensemble wants.

**Task 6.5 — A neural model.** Two candidates:
- An MLP on the same tabular features (cheap, adds diversity)
- **A sequence model (GRU/LSTM/temporal CNN) over the past 24–72 hours of legal covariates per
  station.** This is the one architecture that can learn temporal dynamics the lag features only
  approximate. Higher effort, plausible payoff, and a strong differentiator in the write-up.

**Task 6.6 — Hyperparameter search.** Optuna over fold B with a modest budget, then confirm the
chosen config on fold A. Don't over-tune on a single fold.

**DECISION D8 — How much compute and wall-clock do you want to spend on tuning?**
Rough guide: 30 minutes of Optuna typically gets most of the available gain; multi-hour searches buy
maybe another 0.3–0.8 RMSE. Tell me your budget and I'll size the search.

**DECISION D9 — Do you want to pursue the sequence model (Task 6.5)?**
It's the biggest single time investment in the plan. It's also the most likely source of a result
that beats what everyone else is doing with plain GBMs, and it makes for a much stronger methodology
report. Yes / no / "only if the GBM plateaus".

**Exit check:** ≥3 model families trained on identical features, all logged, with fold A and B scores.

---

## Phase 7 — Two-stage prediction feedback (REFRAMED — now the top priority)

**Reframed after Phase 5.** The original plan here was sequential recursion, rolling predictions
forward hour by hour for 4,255 steps. That is unnecessary and fragile. There is a far safer
formulation that gets the same signal in **two passes with no sequential dependency at all**:

> Stage 1 predicts every row. For row *t*, the prediction made for row *t−1* is an estimate of
> PM2.5 at hour *t* — that is, of `pm25_now`. Feed it to stage 2 as a feature.

No chaining, no compounding drift: every row's feature comes from one independent stage-1
prediction, and the whole test set is predicted in a single pass before stage 2 runs.

**Measured ceiling (fold B, on top of `best_v1` at 17.678):**

| Added feature | Fold B | Δ |
|---|---|---|
| true `pm25_now` (illegal — upper bound) | 11.629 | −6.05 |
| `pm25_now` + noise at our current accuracy (19 RMSE) | **14.501** | **−3.18** |
| nothing (best_v1) | 17.678 | — |

So even a *noisy* estimate of PM2.5 history, at exactly the quality our own model currently
achieves, is worth **−3.18 RMSE**. This is the largest remaining lever by a wide margin.

**The one thing that must be right:** the stage-2 training feature has to come from
**out-of-fold** stage-1 predictions, generated by expanding-window time-series CV across the whole
training period. In-sample stage-1 predictions would be far too accurate, and stage 2 would learn to
trust a feature that is much worse at test time. This is the standard stacking failure and it is the
only real risk in the design.

Note the OOF-vs-test asymmetry is in our favour: OOF predictions on early training blocks come from
models fit on less data, so they are *noisier* than the stage-1 predictions test rows will get.
Stage 2 therefore learns to handle a worse input than it is actually given.

**Why it might work anyway:** unlike a pure autoregressive forecast, every step is re-anchored by
*genuinely observed* covariates — PM10 (corr 0.85), CO (0.76), NO2 (0.64) are all measured at every
test hour. The model can't drift far from reality because reality keeps being fed in. This is
"forecasting with exogenous inputs", not free-running simulation, and that's a much better position.

**Task 7.1 — Build an honest simulator.** On validation fold B, run the exact recursive procedure:
seed with the last true value before the fold, then roll forward all ~4,300 hours using only
predictions for the PM2.5 lag. Compare against the non-recursive model on the same fold. **This
number is the whole experiment.** If recursion doesn't beat the non-recursive model in this
simulation, abandon it — no amount of theory outweighs the measurement.

**Task 7.2 — Drift diagnostics.** Plot recursive RMSE as a function of hours-since-seed. If error
grows without bound, it's dead. If it rises then plateaus, it's viable and the plateau level is what
matters.

**Task 7.3 — Train with scheduled sampling / noise injection.** The classic fix for exposure bias:
during training, replace the true `pm25_now` with a noisy or model-predicted version some fraction
of the time, so the model learns to be robust to imperfect inputs instead of trusting them completely.

**Task 7.4 — Blend rather than switch.** Even if recursion is unstable alone, a weighted average
of recursive and non-recursive predictions may beat both. Tune the weight on fold B.

**Task 7.5 — Damped variant.** Shrink the recursive input toward the local climatological mean at
each step, with a tunable damping factor. This trades responsiveness for stability and often
stabilises exactly this kind of chain.

**DECISION D10 — Do you want to run Phase 7 at all?**
Honest framing: it is the largest identified upside in the project (the gap between 33 and ~25 is
enormous in leaderboard terms), and it is also the only part of the plan that could produce a
submission that scores *worse* than a boring model while looking fine locally if the simulation is
built wrong. Task 7.1 is the safeguard and I'd build it very carefully.

My recommendation: yes, but strictly gated on Task 7.1 showing a clear win on fold B, and submit it
only after you've submitted a safe non-recursive model and seen its leaderboard score.

**Exit check:** a definitive fold-B number for recursive vs non-recursive, and a drift plot.

---

## Phase 8 — The RMSE spike problem

RMSE is dominated by the largest errors. In this dataset, that means severe pollution episodes —
exactly the hours the brief says matter most for public warnings. Underestimating a 500 µg/m³ event
costs enormously more than being 5 off on a clean day.

**Task 8.1 — Quantify it.** What share of total squared error comes from the worst 1% and 5% of
hours? Break the fold-B error down by target decile. This tells you how much is available here.

**Task 8.2 — Sample weighting.** Train with weights increasing in target magnitude. This trades
accuracy on ordinary hours for accuracy on spikes; whether that's net positive under RMSE is an
empirical question. Test several weighting curves.

**Task 8.3 — Objective experiments.** Compare plain L2 against Huber and against training on
log1p(target) with predictions back-transformed. **Expect log-target to *hurt* RMSE** — it optimises
relative error, which under-weights exactly the large values RMSE cares about — but it will be a
strong, decorrelated ensemble member precisely because its errors differ.

**Task 8.4 — Two-stage / specialist model.** Stage 1 classifies "is the next hour a high-pollution
event"; stage 2 uses separate regressors for the high and normal regimes. Or simply train a
dedicated model on high-target rows and blend it in where stage 1 fires.

**Task 8.5 — Quantile check.** Train models at several quantiles and inspect where the conditional
distribution is skewed. Under RMSE you want the conditional *mean*, and a skewed distribution means
the median-ish prediction of a poorly-specified model is biased low on spikes.

**DECISION D11 — How aggressively do you want to chase spikes?**
This can consume a lot of time for a modest RMSE gain, but it is also the most defensible part of the
methodology report given the competition's stated public-health framing. Judges may value it beyond
its score contribution.

**Exit check:** you know what fraction of your error is spikes, and whether any of 8.2–8.4 moved it.

---

## Phase 9 — Ensembling

**Task 9.1** — Correlation matrix of out-of-fold predictions across all your models. Members with
correlation below ~0.95 are the valuable ones. Highly correlated members add nothing.

**Task 9.2** — Weighted blend, weights optimised on fold A and *verified* on fold B (never fit
weights on the fold you report). Constrain weights to be non-negative and sum to 1.

**Task 9.3** — Stacking: train a ridge or shallow LightGBM meta-model on out-of-fold predictions
plus a few key raw features (station, hour, PM10). Usually beats a flat blend, but needs careful
fold discipline or it leaks badly.

**Task 9.4** — Seed averaging: retrain the best config over 5–10 seeds and average. Reliably worth
a few tenths of RMSE, costs nothing but compute, and carries almost no risk.

**DECISION D12 — Simple blend or stacked ensemble?**
Blends are robust and trivially explainable in the report. Stacks usually score a little better and
are easier to get subtly wrong. If the gap between them on fold B is under ~0.3 RMSE, I'd take the
blend for robustness — but that's your call once you see the numbers.

---

## Phase 10 — Post-processing

**Task 10.1 — Clip to `[2, 999]`.** The observed target range. Free, strictly non-negative gain.
Check whether a tighter lower clip (e.g. to the observed per-station minimum) helps.

**Task 10.2 — Bias correction.** Check mean predicted vs mean actual on fold B, overall and per
month and per station. If there's a consistent offset — plausible given the 2013→2017 downward
trend in Beijing pollution — a small additive or multiplicative correction may help. **Be careful:**
this is fitted on validation and may not transfer. Only apply if it's stable across both folds.

**Task 10.3 — Smoothness check.** Since the true series is smooth in time, wildly jumpy
hour-to-hour predictions indicate model noise. Light temporal smoothing of predictions within a
station occasionally helps RMSE. Test it; don't assume it.

**Task 10.4 — Sanity checks on the submission.** Exactly 51,063 rows; ids match
`sample_submission.csv` exactly and in the same order; no NaN; no negatives; distribution broadly
resembles the Sep–Feb training distribution. Automate this as `validate_submission.py` and run it
every single time.

**DECISION D13 — Any post-processing must be disclosed** ("any manual modification or
post-processing of predictions"). Everything above is principled and easy to justify, but decide now
that you'll document each step rather than remember later.

---

## Phase 11 — Final submission run

**Task 11.1** — Refit the chosen configuration on **all** training data (both folds' training and
validation periods). More data, and the most recent months matter most for a chronological split.
Use the iteration count found by early stopping on fold B, scaled up proportionally to the extra data.

**Task 11.2** — Generate predictions, run post-processing, run `validate_submission.py`.

**Task 11.3** — Save the exact model artefacts, the exact config, the git commit hash, and the
submission file together under one experiment id. The rules require the *exact* file you submitted,
not a recreation.

**Task 11.4** — Record the leaderboard score against local fold B in the correlation tracker.

**DECISION D14 — Which submission is your final answer?** If local validation and leaderboard
disagree, you have to choose which to trust. Rule of thumb: with 51k test rows the leaderboard is
statistically stable, so a large disagreement means a genuine train/test difference — but the
leaderboard may itself be a public subset. Check whether it is.

---

## Phase 12 — Write-up and submission materials

Map directly to the required-materials checklist. Draft as you go; do not leave to the end.

| Required | Where it comes from |
|---|---|
| 1. Methodology report | `notebooks/99_final_report.ipynb`, built from phase exit checks |
| 2. Complete source code | `src/` — must run end to end from raw CSVs |
| 3. Final prediction file | `submissions/<exp_id>.csv`, the exact submitted file |
| 4. Processed datasets | State they regenerate from `src/features.py`, and name the step |
| 5. README / reproduction | Execution order, dependencies, seeds, which script makes the final file |
| 6. Final model info | Model family, full hyperparameters, ensemble weights |
| Disclosure | External data (none, pending D6), external code, pretrained models, **AI tools used**, post-processing |

**Task 12.1** — The report should open with the Fact 1 finding. "We identified that current PM2.5
is reconstructable in training but unavailable at test time, and built our validation to prevent
that leak" is the single most impressive thing you can say to a technical reviewer, and it directly
addresses the brief's warning that "random cross-validation may produce overly optimistic results".

**Task 12.2** — Include the experiment log as a results table: what you tried, what it scored,
what you kept. Reviewers explicitly ask for "experiment/model comparison results".

**Task 12.3** — Write the limitations section honestly: distribution shift from 2013–2016 to
2016–2017, the missing PM2.5 history, spike underprediction, single-city generalisation.

**DECISION D15 — AI tool disclosure.** You are required to disclose "any AI tools or coding agents
used". I'm being used substantially here. Decide how you want to describe that — my suggestion is a
plain, specific statement (which tool, for what: exploration, feature implementation, code
scaffolding, plan authoring) rather than something vague. Specific disclosure reads as confidence;
vagueness invites questions.

---

## Suggested order of attack

If you want a single sequence to follow:

1. Phases 1–3 (setup, EDA, validation). **Do not compress this.**
2. Phase 4 — reproduce 33.08 in your own harness.
3. Phase 5 Tiers A–C — expect the largest single jump here.
4. First leaderboard submission. Start the local↔LB correlation tracker.
5. Phase 5 Tiers D–F.
6. Phase 6 — LightGBM tuning, then a second and third model family.
7. Phase 7 Task 7.1 only — the recursive simulation. This decides the shape of the rest of the project.
8. Phase 8 if spikes prove to be a large error share.
9. Phase 9 ensembling, Phase 10 post-processing.
10. Phases 11–12.

---

## Appendix A — Experiment log schema (`experiments/log.csv`)

`exp_id, date, git_commit, description, feature_tiers, model, key_hyperparams, fold_A_rmse,
fold_B_rmse, mean_rmse, leaderboard_rmse, runtime_min, submitted, notes`

One row per experiment, no exceptions. This file becomes the comparison table in your report.

---

## Appendix B — Decision register

Fill this in as you go; it becomes part of the methodology report.

| ID | Decision | Options | Your choice | Date | Rationale |
|---|---|---|---|---|---|
| D1 | Scripts vs notebooks | | | | |
| D2 | Git commit discipline | | | | |
| D3 | Restrict training period? | all / last-N / decay | **all data, uniform** (settled by experiment, Phase 3) | 2026-09-06 | Every restriction lost: last-2y +0.04, decay-180d +0.10, decay-365d +0.12, last-1y +3.77 |
| D4 | Submission budget strategy | | | | |
| D5 | Year / time-trend features | | | | |
| D6 | Station coordinates (rules risk) | | | | |
| D7 | Missing value strategy | | | | |
| D8 | Tuning compute budget | | | | |
| D9 | Sequence model yes/no | | | | |
| D10 | Run the recursive experiment? | | | | |
| D11 | How hard to chase spikes | | | | |
| D12 | Blend vs stack | | | | |
| D13 | Post-processing disclosure | | | | |
| D14 | Final submission choice | | | | |
| D15 | AI tool disclosure wording | | | | |
| D7 | Missing value strategy | NaN / interpolate / hybrid | **hybrid: interpolate ≤6h, then cross-station fill, else NaN** | 2026-09-06 | User-approved; worth ~3.1 RMSE (raw baseline 32.96 → 29.88) |
| D16 | SO2 collapse (−44% in test) | drop / normalise / leave | **KEEP SO2** (reverted) | 2026-09-06 | Fold B was a tie so the drift argument was used to drop it — **the leaderboard disproved that**: keeping SO2 scored 19.025 vs 19.641 without. SO2's signal transfers despite the level shift. Drift magnitude is not by itself a reason to drop a feature |

---

## Appendix C — Things that would invalidate this plan

Be alert for these; if one occurs, the plan needs revising rather than following.

- **Fold B and the leaderboard diverge badly.** Then the 2016–17 test period differs from
  2015–16 in some way you haven't modelled, and validation design needs rethinking first.
- **The recursive simulation looks great but the leaderboard disagrees.** Most likely cause: the
  simulation seeded or gap-handled differently from real test conditions. Re-audit 7.1, don't tune.
- **A feature gives an implausibly large gain.** Above ~2 RMSE from one feature, suspect leakage
  before celebrating. Re-run `leakage_guard()` and trace the feature's dependency chain by hand.
