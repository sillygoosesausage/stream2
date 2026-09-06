# Experiment record — Beijing PM2.5 next-hour forecast

Complete record of what was tried, what it scored, and what was kept.
Metric: RMSE on 51,063 hidden test rows.

**Best submission: `exp004_tuned_wspike.csv` — leaderboard 18.78847.**
Three subsequent attempts failed to beat it.

---

## 1. Submission history

| # | Submission | Fold A | Fold B | Leaderboard | Verdict |
|---|---|---|---|---|---|
| 001 | baseline_raw | 28.63 | 32.96 | not submitted | reference |
| 002 | best_v1 | 22.21 | 17.502 | 19.02468 | |
| 003 | best_v2_no_so2 | 22.81 | 17.748 | 19.64126 | worse — SO2 kept |
| **004** | **tuned_wspike** | 22.84 | 16.599 | **18.78847** | **BEST** |
| 005 | blend_A40_mf60 | 22.28 | 16.749 | 18.97675 | worse |
| 006 | bagged15 | — | 16.180 | 18.84847 | worse |

Local fold B went 32.96 → 16.18 (−51%). The leaderboard went 19.02 → 18.79
(−1.2%) and then stopped moving. Everything after exp004 improved locally and
did not improve on the leaderboard.

---

## 2. The two structural findings that produced all the real gain

### 2.1 The "current PM2.5" trap

The competition overview states each row contains "Current PM2.5 concentration".
It does not — there is no PM2.5 column, only the target `PM2_5_next_hour`.

In **train** it is reconstructable: sort by station and time, and the previous
row's target is this row's current PM2.5 (99.31% of rows). In **test** it is
not — test is a contiguous hourly block, so the preceding row is itself a test
row with a hidden target.

Any feature built on PM2.5 history validates beautifully and is uncomputable at
submission time. Two guards in `src/data.py` enforce this on every fit:
`leakage_guard` (rejects PM2.5-derived column names) and
`assert_test_computable` (builds features against the real test frame and fails
if any come out entirely null).

Reference values, fold B: illegal persistence 22.25; illegal model with true
`pm25_now` 11.63. The final legal model reaches 16.18, i.e. **better than
persistence** without ever seeing a PM2.5 value.

### 2.2 Lead features — the task is a nowcast, not a forecast

The mirror image of the trap. For a row at hour *t* the target is PM2.5 at
*t+1*, and **the row at *t+1* is also in the test file**, carrying observed
PM10/CO/NO2/SO2/O3 and weather for the hour being predicted (99.25% of test
rows have their *t+1* row present).

So the problem is not "forecast one hour ahead" — it is "estimate PM2.5 at hour
*t+1* from every other measurement taken at hour *t+1*."

| Feature set | Fold A | Fold B |
|---|---|---|
| raw baseline | 28.97 | 32.86 |
| + lead-1 covariates | 23.11 | 24.95 |
| + city-wide aggregates and leads | 23.27 | **23.71** |

**−9.15 RMSE.** This single insight is most of the project's gain. Correlation
with the target: PM10 at *t* is 0.848, PM10 at *t+1* is 0.866.

---

## 3. Feature engineering

Built as prefixed tiers on a station × hour panel spanning train and test
together (so lags mean real hours, and the first test hours get proper history).
Each tier measured separately.

| Tier | Contents | Fold B effect | Kept |
|---|---|---|---|
| raw | measurements as given | baseline | yes |
| **C0** | **lead-1/2/3 covariates, boundary deltas** | **−10.66** | **yes** |
| **D** | city-wide mean/median/std/max, anomalies, city leads | **−1.09** | **yes** |
| B | wind vectors, relative humidity, dew point depression | −0.08 (noise) | yes |
| C (lags) | lags 1–24h of pollutants | −0.59 | yes |
| C (diffs) | 1/3/24h changes | −0.63 | yes |
| C (rolling) | rolling mean/std 3–72h | **+0.89** | no |
| A | calendar, cyclical, heating season | **+4.52** | no |
| F | interaction and source-signature ratios | +0.24 | no |
| E | missingness indicator flags | +0.11 | no |
| G | extended leads t+4..t+6, acceleration, more city leads | +0.26 | no |

Final set `best_v1`: **170 features** = raw + C0 + D + B + C(lag,diff).

### Calendar features are actively harmful

Tier A cost **+4.52 RMSE**, isolated to `A_doy` (+3.74) and `A_month` (+1.94).
Day-of-year lets the model memorise which specific days were polluted in past
years; with three years of training that is noise, since pollution episodes are
synoptic weather events that do not recur on calendar dates. Even hour-of-day
hurt (+0.33) — NO2 and CO encode the diurnal cycle more informatively than a
clock does.

### Missing-value handling (D7)

Phase 2 showed gaps are bimodal (median 1–2h, max 1517h) and stations fail
semi-independently (missingness correlation ~0.25). Strategy: interpolate gaps
≤6h, then fill from the city median scaled by the station's long-run ratio, then
leave NaN. Worth **~3.1 RMSE** (raw baseline 32.96 → 29.88).

### Feature importance is extremely concentrated

| Feature | Share of gain |
|---|---|
| `C0_lead1_PM10` | 49.99% |
| `raw_PM10` | 19.50% |
| `D_city_lead1_PM10` | 3.30% |
| `D_city_med_PM10` | 3.13% |
| `D_city_mean_PM10` | 2.89% |
| `D_city_lead1_CO` | 2.27% |

Top 6 carry **81%**; 129 of 170 features contribute under 0.1% each. This is a
two-variable problem — PM10 at the predicted hour and PM10 now — wearing a
170-feature costume. It explains why nothing else worked: there is no
information left to add, and every later idea merely rearranged what the model
already had.

**Pruning fails too.** Top-120 16.666, top-90 16.949, top-60 17.060, top-40
17.274, top-25 17.776, top-15 19.276, against all-170 at 16.637. The
individually negligible features matter collectively. The model cannot be added
to and cannot be cut down.

---

## 4. Everything tried

### Worked

| Change | Fold B | Transferred to LB? |
|---|---|---|
| Lead features (C0) | −10.66 | yes — the core of the model |
| Cross-station features (D) | −1.09 | yes |
| D7 imputation | ~−3.1 | yes |
| Lags + diffs (C) | −0.59 | yes |
| Seed averaging (k=5) | −0.222 | yes |
| Hyperparameter tuning | −0.353 | partially |
| Spike sample weighting | −0.399 | partially |
| Tuning + spike combined | −1.125 | **exp004, best submission** |

### Did not work

| Attempt | Fold B | Note |
|---|---|---|
| Calendar features | +4.52 | memorisation |
| Rolling windows | +0.89 | redundant with lags |
| Extended leads (tier G) | +0.26 | no new information |
| Interaction ratios | +0.24 | |
| Missingness flags | +0.11 | |
| Feature pruning | +0.03 to +2.64 | all subsets worse |
| Dropping SO2 | local tie | **+0.62 on leaderboard** |
| XGBoost | +1.18 solo | corr 0.9988, blend weight 0.00 |
| Log-target | +3.59 | corr 0.9962 |
| sqrt target | +0.55 | |
| Huber (α=20) | +0.08 | |
| Ratio target (PM2.5/PM10) | +0.65 to +1.29 | |
| Weighting curves (quadratic, sqrt, threshold, w=4, w=6) | all inside noise | mechanism insensitive |
| Two-stage prediction feedback | −0.127 (noise) | see below |
| Global bias correction | fold-specific | see below |
| Per-station / per-month bias | +5 to +11 | |
| Cross-fold blend (exp005) | −0.15 locally | **+0.19 on leaderboard** |
| Feature-subset bagging (exp006) | −0.42 locally | **+0.06 on leaderboard** |
| Training-period restriction / recency weighting | +0.04 to +3.77 | all data, uniform, wins |

### Three negatives worth explaining

**Two-stage prediction feedback.** For a row at *t*, the prediction made for row
*t−1* estimates PM2.5 at *t* — the missing `pm25_now`. A ceiling probe that
injected Gaussian noise at our own accuracy promised −3.18. Actual result:
−0.127, inside noise. The probe was wrong because Gaussian noise is independent
of the features, whereas real model error is a *function* of them — so the
estimate re-encodes information the model already has. True `pm25_now` is worth
−6.05 and is unreachable: it is a physical measurement, not something derivable
from covariates. This is the general reason nothing else worked either.

**Global bias correction.** Scaling predictions by ×1.04 improves fold B by
−0.75. But the optimal multiplier is **0.95 on fold A and 1.04 on fold B** —
opposite directions. Cross-applying costs +2.8. The bias is year-to-year
pollution variation, not a model property, and the test year's direction is
unknowable.

**Feature-subset bagging.** 15 models, each on a random 60% of features with the
top-8 always kept. Controlled comparison at 8 models each: full-feature seed
averaging 16.490, feature-bagged **16.180** — a genuine −0.31 from the
subsetting rather than model count (seed averaging saturates by ~4 models;
bagging does not). It was the only mechanism that produced real variance
reduction, and it still **lost 0.06 on the leaderboard**.

---

## 5. Validation — the hardest part of this project

The fold design mimics the test set: a contiguous Sep–Feb block immediately
after the training data.

| Fold | Train rows | Validate | Baseline RMSE | Status |
|---|---|---|---|---|
| A | 155,190 | 2014-09 → 2015-03 | 28.63 | unreliable, see below |
| **B** | 257,927 | 2015-09 → 2016-03 | **32.96** | **primary** |
| C | 51,925 | 2013-09 → 2014-03 | 49.81 | retired — too little training data |
| R | 309,276 | 2016-03 → 2016-09 | 23.11 | wrong season |

### Seed noise had to be measured before anything could be decided

Single-seed fold B has **sd 0.263, range 0.882**. Deltas under ~0.5 are
undecidable from one run. This was not obvious at first and it **reversed a
conclusion**: tier C lags measured +0.673 (harmful) on one seed and −0.593
(helpful) over four. `validate.compare(..., seeds=n)` reports the noise
threshold and flags variants that are not separable from baseline.

Seed averaging drops the floor: sd 0.177 (k=1) → 0.080 (k=3) → 0.054 (k=5).

### The metric question was got wrong twice

1. Decisions were made on fold B alone. After three submissions the transfer
   slope was 0.59 and the offset was growing — the signature of overfitting the
   validation fold.
2. Switching to mean(A,B) looked well calibrated on two points (±0.05) and
   predicted exp005 would score 18.63. **It scored 18.98.** A two-point fit
   cannot support a slope.
3. With five points: fold B Spearman **0.90**, mean(A,B) worse, fold A
   **negatively** correlated (−0.40). Fold A trains on 155k rows against fold
   B's 258k and the final fit's 361k, so its difficulty is an artifact of data
   volume rather than a second opinion about the test period. exp005 was built
   to improve fold A and lost.

### Current transfer statistics (5 submissions)

- fold B ↔ LB Spearman **0.80–0.90**
- transfer slope **0.42** — less than half of each local gain arrives
- offset growing at −0.58 per unit of local improvement

**Deltas from exp004, the standing best:**

| | Fold B | Leaderboard |
|---|---|---|
| exp005 | +0.150 | +0.188 |
| exp006 | **−0.419** | **+0.060** |

exp006 improved fold B by 0.42 and still lost on the leaderboard. Local
validation can no longer resolve differences at this scale.

---

## 6. Where things stand

The leaderboard has been flat at **18.79–18.98 across the last three
submissions** while fold B improved by 0.57. Roughly 25 decisions have now been
made against fold B, and it has been consumed as a selection set.

What this most likely means:

- The **structural** gains (lead features, cross-station, imputation) transferred
  cleanly and are banked in exp004.
- The **marginal** gains since (bagging, blending, bias correction) are variance
  reduction specific to fold B's particular noise, and do not describe the test
  period.
- Further optimisation against fold B has close to zero expected value.

### If work continues

1. **Fresh validation.** Rolling-origin folds with 4–5 Sep–Feb windows, never
   used for any past decision. Costly but it is the only way to regain a signal.
2. **More bags (25–30) plus row subsampling.** Bagging is the one mechanism that
   produced genuine variance reduction; it lost by only 0.06 and may be worth
   one more attempt at greater strength.
3. **Accept exp004.** Three consecutive attempts to beat it have failed. That is
   itself evidence that the plateau is real rather than a run of bad luck.

---

## 7. Reproduction

```bash
pip install -r requirements.txt          # + optuna, catboost not used

# best submission (exp004, LB 18.78847)
python -m src.submit tuned_wspike --seeds 5 --exp-id exp004_tuned_wspike

# validation
python -m src.train --config experiments/configs/best_v1.yaml
python -m src.tracker show               # local vs leaderboard transfer
python -m src.eda all                    # Phase 2 analysis
```

**Final model (exp004):** feature set `best_v1` (170 features), LightGBM,
5 seeds averaged, 1847 rounds on all 360,954 training rows.
Hyperparameters: `learning_rate` 0.0220, `num_leaves` 242,
`min_data_in_leaf` 134, `feature_fraction` 0.665, `bagging_fraction` 0.654,
`bagging_freq` 5, `lambda_l1` 4.714, `lambda_l2` 1.394; sample weights
`1 + 1.0 × y/mean(y)`; predictions clipped to [2, 999]. Seed 42.

**Files:** `src/config.py` (paths, folds), `src/data.py` (loading, leakage
guards), `src/features.py` (tiers, feature sets), `src/validate.py` (folds,
`compare`), `src/models.py`, `src/ensemble.py` (seed/bag machinery),
`src/tune.py` (Optuna), `src/stage2.py` (two-stage, negative),
`src/submit.py` / `src/submit_bagged.py`, `src/tracker.py`.

**Disclosure:** no external data (station coordinates deliberately not used);
no pretrained models; AI coding agent (Claude) used for exploration, feature
implementation, and code scaffolding; post-processing limited to clipping
predictions to the observed target range [2, 999].

---

# Part II — Session of 2026-09-06 afternoon (phases 13–14)

Written after the record above closed with "accept exp004." That conclusion was
correct about *features* and wrong about *ensembling*. The leaderboard moved
from 18.78847 to **18.61153**, and the mechanism was not a better model.

## 8. What was tested and what it cost

### 8.1 Two bugs and a measurement tool

**`ensemble.fit_member` mutated the member registry.** `base_params` was bound
by reference and `base_params.pop("spike_weight")` therefore deleted the key
from `MEMBERS` for the life of the process — so fitting the same member twice in
one session silently ran the second fit *unweighted*. Fixed by copying the dict
(`src/ensemble.py`). Audited for damage: the cached `tuned_wspike` predictions
differ from `tuned_v1` by −0.55 RMSE, which they could not if the spike weights
had been dropped, so no past result was corrupted. `submit.py` already copied,
so no submission was ever affected.

**Paired significance testing** (`src/paired.py`). Every "noise floor" figure in
Part I came from comparing two *independent* means, which is the wrong test:
both models are scored on identical rows and, when they share seeds, on
identical bagging draws. Differencing per row and bootstrapping over day-long
blocks removes the shared variance.

The gain is large. Comparing `tuned_wspike_Hcw` against the incumbent — two
models correlated at 0.99 — gives a 95% CI of **±0.10**, against the ±0.4
independent-means floor quoted throughout Part I. Effects a quarter the size are
now decidable. Blocks are day-long rather than iid because consecutive hours are
strongly autocorrelated; an iid bootstrap understates the interval badly.

### 8.2 Feature engineering — four families, all negative

| Family | What | Fold B Δ | 95% CI | Verdict |
|---|---|---|---|---|
| Tier G | extended leads t+4..t+6, acceleration, city leads for all pollutants, anomalies, spreads (237 feat) | +0.18 | [−0.28, +0.73] | loses |
| Tier H `cw` | mean/std/min/max over a window **centred on t+1** (195 feat) | +0.017 | [−0.075, +0.129] | neutral |
| Tier H `rest` | pre-impute observation flags, un-imputed leads, lead×wind and station×city interactions, rain at t+1, inversion proxy (192 feat) | +0.60 | [+0.31, +0.94] | conclusively worse |
| Tier H full | all of the above (225 feat) | +0.49 | [+0.25, +0.77] | conclusively worse |

Tier G had been written but wired to no ensemble member, so it had never been
scored on any fold. It was the top-ranked item on the improvement backlog,
described there as "the single most likely source of the remaining 0.57." It
loses.

Tier H's centred windows deserve a note: trailing windows cost +0.89 in Part I,
and centring them recovers all of that but adds nothing — exactly +0.017, with a
tight interval around zero. The window is not the problem and the window is not
the answer; there is simply no information in it that the leads do not already
carry.

### 8.3 Post-processing — every variant is fold-specific

Sweeps on cached out-of-fold predictions, costing no fits at all:

| Variant | Fold A | Fold B |
|---|---|---|
| centred 3h smoothing, w=0.1 | −0.032 | +0.080 |
| centred 5h smoothing, w=0.2 | −0.047 | +0.452 |
| spike stretch q0.95 k0.02 | +0.127 | −0.193 |
| spike stretch q0.90 k0.05 | +0.858 | −0.393 |
| scale ×1.04 | +2.443 | −0.750 |
| scale ×0.98 | −0.858 | +0.913 |
| **clip lower bound 5.0** | **−0.002** | **−0.000** |

Every adjustment that helps one fold hurts the other, and the two are almost
perfectly anti-symmetric. This reproduces the global-bias-correction finding of
Part I on a much wider class of transforms: smoothing, monotone stretching and
rescaling are all describing a particular year's pollution level, not a property
of the model. The only consistent adjustment is worth 0.0004.

### 8.4 The error breakdown that should have been run on day one

Nobody had looked at *where* the squared error is. On fold B, seed-averaged:

| Slice | n | Bias (pred − actual) | RMSE | Share of total SE |
|---|---|---|---|---|
| **`lead1_PM10` imputed** | **430** | −4.08 | **73.96** | **16.6%** |
| `lead1_PM10` observed | 50,919 | −1.62 | 15.22 | 83.4% |
| rising fast (Δ > +20) | 3,150 | **−20.01** | 40.14 | 35.9% |
| clearing fast (Δ < −20) | 3,497 | +0.73 | 25.48 | 16.0% |
| top target decile | 5,125 | −19.23 | 41.51 | **62.4%** |
| bottom target decile | 6,249 | +4.22 | 9.20 | 3.7% |

Three things fall out. **0.84% of rows carry 16.6% of all squared error** — the
rows where D7 imputation fabricated the model's dominant feature and handed it
over indistinguishable from a real measurement. **Onsets are under-predicted by
20 µg/m³** while clearances are not, so the error is asymmetric in time. And the
model over-predicts clean air by +4.2 while under-predicting the top decile by
−19.2, which is precisely why no global rescaling can help — the required
correction has opposite signs at the two ends.

The first of these was attacked directly: `H_rawlead1_*` supplies the un-imputed
lead with NaN preserved, so LightGBM can route those rows down their own branch,
alongside honest pre-imputation observation flags. It is part of `Hrest`, which
lost by 0.60. The idea is not thereby refuted — it was tested inside a block of
22 columns — but it was not rescued either, and it remains the largest
identified and unexploited concentration of error in the model.

Two audits came out clean. `C0_has_lead` is confirmed dead: mean 0.99998 in
train and 0.99976 in test, because it is computed *after* imputation. And the
imputation is symmetric across periods — the lead is fabricated for 0.74% of
train rows and 0.76% of test rows, so train and test are filled at the same
rate.

## 9. What actually worked: blending submitted predictions

### 9.1 The result

| Submission | Composition | LB |
|---|---|---|
| exp004 | tuned_wspike, 5 seeds | 18.78847 |
| exp006 | feature-bagged, 15 bags | 18.84847 |
| exp002 | best_v1 | 19.02468 |
| exp007 | exp004 + exp006, equal | 18.69287 |
| **exp015** | **exp004 + exp006 + exp002, equal thirds** | **18.61153** |

The three-way blend beats every one of its own components — including by 0.41
the weakest of them — and improves on the standing best by **0.177**. It has no
fitted parameters: three models, equal weights.

### 9.2 Diversity beats component quality

The weight sweeps make the mechanism explicit:

| Blend | LB |
|---|---|
| exp015: 004 + 006 + 002 equal | **18.61153** |
| exp017: 004:2 + 006:2 + 002:1 | 18.62090 |
| exp019: 004:2 + 006:2 + 002:2 + 005:1 | 18.65110 |
| exp020: rank-average of 004/006/005/002 | 18.67002 |
| exp011: 004 + 006 + 005 + 002 equal | 18.68357 |
| exp007: 004 + 006 equal | 18.69287 |
| exp016: 004 + 006 + 002 + 003 equal | 18.70816 |
| exp018: five-way equal | 18.74176 |
| exp010: 004 + 006 + 005 equal | 18.74589 |

`best_v1` (exp002) is the *worst* standalone model in the pool at 19.02, and it
wants a full third of the weight — giving it only a fifth (exp017) is worse.
Meanwhile `exp003` (no-SO2) and `exp005` both *hurt* when added. The distinction
is that exp002 is a genuinely different model, whereas exp003 is a **degraded**
one (it is best_v1 with SO2 removed, which D16 already established costs 0.62 on
the leaderboard) and exp005 is **redundant** (it is itself a blend of best_v1 and
tuned_mf, so it adds no new direction). Diversity pays; damage and duplication
do not.

Rank-averaging lost to value-averaging by 0.06.

### 9.3 Local validation cannot select blend weights

Twelve submitted models and blends, each with a fold-B analogue reconstructed
from cached predictions:

| Submission | Fold B | LB |
|---|---|---|
| exp015 4+6+2 | 16.5278 | **18.6115** |
| exp017 4:2,6:2,2:1 | 16.4050 | 18.6209 |
| exp019 | 16.5500 | 18.6511 |
| exp011 | 16.5687 | 18.6836 |
| exp007 4+6 | 16.2689 | 18.6929 |
| exp009 | 16.3422 | 18.6952 |
| exp008 4:.35/6 | **16.2183** | 18.7133 |
| exp010 | 16.3774 | 18.7459 |
| exp004 | 16.5993 | 18.7885 |
| exp006 | **16.1894** | 18.8485 |
| exp005 | 16.7492 | 18.9768 |
| exp002 | 17.5018 | 19.0247 |

**Spearman = +0.21, p = 0.51.** There is no relationship. The Pearson of +0.65
is produced entirely by the two solo outliers at the bottom; among the blends
the ranking is close to inverted. The leaderboard-best blend has one of the
*worst* fold-B scores, and the fold-B-best blend is fourth-worst on the
leaderboard.

This is the sharpest statement of the project's central difficulty. Fold B ranks
*single models* acceptably (Part I: Spearman 0.80–0.90) and ranks *blends* not
at all. The likely reason is that blending's benefit is variance reduction, and
fold B's 51,349 rows are a single fixed sample — averaging more models reduces
the variance that would appear across *resampled* validation sets, which a fixed
fold cannot observe. It measures bias; blending buys variance.

A practical consequence, stated plainly: these blend weights were chosen against
the *public* leaderboard, with the private split unseen. That is a genuine
overfitting risk. It is why the standing entry is the **unfitted equal-thirds**
blend rather than the marginally better tuned one — equal weights across three
models have no free parameters to overfit with.

### 9.4 Leaderboard noise is small

The weight sweep 0.35 / 0.50 / 0.65 gives 18.71327 / 18.69287 / 18.69516 — a
smooth curve with a shallow minimum. Differences of 0.09 are therefore signal,
though differences of 0.01 are not.

## 10. Revised conclusion

Part I concluded that the plateau was real. On features that was right, and this
session added four more negative families to the pile. What was wrong was the
assumption that the plateau applied to the *pipeline* as a whole. There was
0.177 sitting in an ensembling step that local validation was structurally
incapable of detecting, and it was found by submitting rather than by
validating.

The generalisable lesson is not "blend more." It is that **a validation scheme
should be audited for what class of improvement it can see.** Fold B was
consumed by roughly 25 feature decisions and it served those adequately; it was
then used, implicitly, to rule on ensembling, which it cannot measure at all.
