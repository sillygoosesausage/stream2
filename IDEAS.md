# Improvement Backlog — Beijing PM2.5 Next-Hour

**Current state:** exp004_tuned_wspike = **18.788 LB** (fold B 16.599). Leader ≈ 18.22. Gap ≈ 0.57.

> ## ⏰ DEADLINE IS TODAY
> `Expected-submission-requirements.txt`: **"End of Sunday, 6 September 2026."**
> It is Sunday 6 September 2026. Roughly **9 working hours remain.**
>
> Two consequences, both binding:
>
> 1. **Nothing above time class `L` should be started after ~19:00.** Anything `XL` is
>    dead on arrival — it is listed below only so the report can say it was considered.
> 2. **Phase 12 deliverables do not exist and are REQUIRED.** There is no methodology
>    report, no reproduction README, no model-information sheet, no AI-tool disclosure.
>    The rules say a submission whose materials "cannot plausibly explain or reproduce
>    the leaderboard result" may be "deemed ineligible for finalist selection."
>    **A 0.3 RMSE gain is worth nothing if the entry is ineligible.**
>    Reserve the last 90 minutes for `W1`–`W6` in section 9. That is not optional work.
>
> Suggested split of the remaining day: **6 h experiments → 1 h final submission run →
> 1.5 h write-up → 0.5 h buffer.**

---

## How to use this file

A menu, not a plan. Work the **Shortlist** top-down; drop into the themed sections when
the shortlist is exhausted or a result suggests a direction. Exercise judgement — skip
anything already done, and stop pulling from a section once two consecutive ideas land
inside noise.

Log every attempt in `experiments/log.csv`, win or lose. Negative results are a required
deliverable of the report, not waste.

**Ratings.** Impact = expected *leaderboard* movement after transfer, not fold-B delta.
Time = wall clock including fits, on this machine.

- Impact: **H** > 0.3 · **M** 0.1–0.3 · **L** < 0.1 · **?** unmeasured, could be either
- Time: **S** < 10 min · **M** 10–30 min · **L** 30–90 min · **XL** > 90 min

**Transfer rate.** Four LB points say fold B ranks correctly but compresses: roughly
**a quarter to a third of a fold-B gain reaches the leaderboard.** To close 0.57 on the
LB you need ≈ **1.7–2.3 on fold B**. Nothing in this file is that big alone. The realistic
target is *a stack of three or four 0.3–0.6 fold-B wins*, which is exactly why the
shortlist is ordered for throughput rather than for elegance.

---

## ⚠️ Corrections to the previous version of this file

The previous backlog proposed as "most likely missing" a set of features that **are already
built** in `src/features.py::_tier_G`. They were never tested only because **no ensemble
member uses tier G at all.** Do not re-implement these:

| Previously proposed | Reality |
|---|---|
| "Cross-station aggregates at t+1" (was F2, rated H/S) | **Exists.** `D_city_lead1_{PM10,CO,NO2}` is already *in* `best_v1`; `G_city_lead1_*` covers all five pollutants + WSPM/TEMP |
| "Leads of derived weather" (was F3) | **Exists.** `G_lead1_rh`, `G_lead1_dpd`, `G_rh_delta` |
| "Station-relative leads" (was F7) | **Exists.** `G_anom_lead1_{PM10,SO2,NO2,CO,O3,WSPM,TEMP}` |
| "Second-order lead differences" (was F11) | **Exists.** `G_delta2_*`, `G_delta3_*`, `G_accel_*` |
| "Per-station rolling ratio" (was F13) | **Exists.** `G_pm10_lead1_vs_roll24`, `G_co_lead1_vs_roll24` |
| "Humidity × PM10 at t+1" (was F14) | **Exists.** `G_lead1_pm10_x_rh` |
| "Spatial gradient at t+1" (was F18) | **Exists.** `G_city_range_lead1_{PM10,CO}` |
| "Audit Tier G's status" (was F1) | Answered: G was **never scored on any fold and never submitted**. `best_v3_leadmax` is defined in `FEATURE_SETS` and referenced nowhere else in the repo |

**So the single highest-value action in this file is not to write a feature. It is to add
one line to `ensemble.MEMBERS` and run it.** See `A1`.

Also corrected: "clipping does nothing (one row at 999)" — the clip is
`np.clip(pred, 2.0, 999.0)` and the *lower* bound at 2.0 is the one that binds, not the
upper. And "Fold C retired" is right, but **fold R (Mar–Aug 2016) exists in `config.FOLDS`,
is the most recent data in the file, and has never been used** — see `V2`.

---

## Already tried — do not repeat

| Thing | Result |
|---|---|
| Stage-2 prediction feedback | −0.127, inside noise. Structural: the estimate re-encodes existing features |
| XGBoost solo + blend | corr 0.9988 with LGBM, blend weight 0.00 — **but see `A4`, the seeds may have been fake** |
| Log-target | corr 0.9962, lost outright at 21.095 |
| Tier A calendar features | +4.52 (harmful); `A_doy` +3.74, `A_month` +1.94 |
| Recency weighting / period restriction | all four variants lost; last-1-year +3.77 |
| Fold C | retired, data-starved (52k rows, baseline 49.81) |
| Dropping SO2 (D16) | local tie (17.691 vs 17.692) but **+0.62 on LB** — reverted, keep SO2 |
| Blend A40/M60 (exp005) | 18.98 vs 18.79, lost. Built to improve fold A; fold A is anti-correlated |
| Missingness null flags (tier E) | +0.113, noise |
| Tier F ratios | +0.24, mild loss on fold B |
| Tier C rolling windows | +0.89, clear loss |
| Pressure tendency | corr −0.041, dud |
| Lags beyond 24h | dud |
| Hyperparameter tuning (2 × 30 trials) | fold A immovable 22.8–23.2 regardless of config |
| Feature pruning | **more is better**: all 170 = 16.637, top 120 = 16.666, top 60 = 17.060. Do not prune |

The pruning result is important and under-used: **the model is not saturated on feature
count.** Adding ~90 more columns (tier G) is not obviously a regularisation risk.

---

# ★ THE SHORTLIST — work this order

Ranked by (expected LB gain) ÷ (time). The first four are all under 30 minutes and three
of them are one-line changes.

| Rank | ID | Idea | Impact | Time |
|---|---|---|---|---|
| 1 | `A1` | **Turn Tier G on.** Register `tuned_wspike_G` = tuned_wspike params + `best_v3_leadmax`, fit fold B, 5 seeds | **H** | S |
| 2 | `A2` | **Audit the `MEMBERS` mutation bug** — cached spike-weighted preds may be unweighted | **H** | S |
| 3 | `P1` | **Bias correction** — never run, and every LB score has come in *above* local | **H** | S |
| 4 | `A3` | **Verify test features are built on the joint panel** (they are — confirm and move on) | **H** | S |
| 5 | `V1` | **Mine the error breakdown** on saved OOF: which spikes, which stations, onsets vs clearances | H (indirect) | S |
| 6 | `M1` | **sqrt-target member** — implemented in `models.py`, wired to nothing | M | S |
| 7 | `F1` | **PM10-fraction framing** — predict PM2.5/PM10 at t+1, multiply back | **H** | M |
| 8 | `M4` | **Round-count scaling audit** — the ×1.4 multiplier touches every submission | M | S |
| 9 | `A5` | **Fix `C0_has_lead`** — computed post-imputation, so it is ~always 1 and carries nothing | M | S |
| 10 | `F2` | **Centred windows around t+1** — the one genuinely missing feature family | **H** | M |
| 11 | `M2` | **Feature-subsample ensembling** — forced diversity that XGB failed to provide | M | M |
| 12 | `M3` | **Seeds 5 → 15** — boring, near-certain, lowers the floor for everything else | M | M |
| 13 | `P2` | **Multiplicative vs additive correction** — ratios transfer better across level shifts | M | S |
| 14 | `X1` | **Pseudo-labelling the test set** — the biggest *conceptual* swing left | ? | L |
| 15 | `M5` | **CatBoost** — the only untried algorithm with a real shot at decorrelation | M | M |

**Stop-rule.** If ranks 1–9 all land inside noise, stop experimenting, run the best
available submission, and spend the rest of the day on section 9. That is the
higher-expected-value branch at that point.

---

## 0. Audits — cheap, and two of them are possible bugs

These are not tuning choices. If any of them is broken, every measurement above it is
wrong, which makes them worth doing before more experiments.

| # | Idea | Impact | Time |
|---|---|---|---|
| `A1` | **Tier G has never been scored.** `FEATURE_SETS["best_v3_leadmax"]` = `best_v1` + G exists; no member, no log row, no submission. Tier G contains ~90 features implementing eight of the previous backlog's "most promising" ideas. Add to `ensemble.MEMBERS`: `"tuned_wspike_G": ("best_v3_leadmax", "lightgbm", {…tuned_wspike params…})` and run `python -m src.ensemble fit tuned_wspike_G --seeds 5 --fold B`. Compare against the cached `tuned_wspike__foldB.parquet`. **This is the single most likely source of the remaining 0.57.** | **H** | S |
| `A2` | **`ensemble.fit_member` mutates the registry.** Line ~110 `fset, model_name, base_params = MEMBERS[name]` binds a *reference*, then line ~116 `base_params.pop("spike_weight", None)` **deletes the key from `MEMBERS` for the life of the process.** Fit the same member twice in one session — or fit then blend — and the second fit silently runs *unweighted*. Check whether any cached `preds/*.parquet` was produced this way; if so its score is not what the scoreboard claims. Fix: `base_params = dict(MEMBERS[name][2])`. `submit.py` does copy (`params = dict(params)`), so submissions are safe — validation may not be. | **H** | S |
| `A3` | **Confirm test features use the joint panel.** `_panel()` concatenates train+test before building, so lags/rollings at the train→test boundary are correct. Verify once with an assert on a known test row's `C_lag24_PM10`, then record it in the report as a checked property — reviewers will look for exactly this. | H (insurance) | S |
| `A4` | **XGBoost seed-averaging may have been a no-op.** `XGBoostModel._fit` does `params.pop("seed", None)`, and `ensemble` passes `random_state` — which the *native* `xgb.train` API ignores (it is the sklearn-wrapper name; the native param is `seed`). If so, every "seed" of `xgb_v1` fitted an identical model, its cached spread is fake, and the corr-0.9988 verdict that killed model diversity rests on a broken comparison. Cheap to check: are the columns of `xgb_v1__foldB.parquet` identical? | M | S |
| `A5` | **`C0_has_lead` is computed after imputation.** `_tier_C0` runs on the imputed panel, so `_shift_by_station(p,"PM10",-1).notna()` is ~always true and the flag is near-constant — exactly the "is my key feature real or filled in?" signal the model most needs, and it is dead. Build it from the *pre-impute* panel instead. Cheap, and plausibly worth real RMSE on the rows where the lead is fabricated. | M | S |
| `A6` | **Audit lead imputation symmetry.** `C0_lead1_PM10` is the top feature by an order of magnitude (gain 8.3e10 vs 3.2e10 for `raw_PM10`). Where it lands on an *imputed* value, is the imputation identical in train and test? The cross-station fill uses a per-station ratio computed over the **whole panel including the test period** — legal (covariates only) but it means train and test rows are filled from different denominators. Quantify: what fraction of test rows have an imputed `lead1_PM10` vs train rows? | **H** | S |
| `A7` | **Verify the leakage guard fires.** Plant a deliberately leaky column, confirm `leakage_guard` raises. Test the test. Also a report-worthy paragraph. | L (insurance) | S |
| `A8` | Assert feature column order and dtypes match between fit and predict frames. `submit.py` does `F.build_set(test, fset)[X_tr.columns]` — already correct, so this is confirmation, not a fix. | L (insurance) | S |
| `A9` | Confirm the category codes for `station`/`wd` are identical across train and test (they are — `config.STATION_CATEGORIES` / `WD_CATEGORIES` are fixed lists). Confirm, log, move on. | L | S |
| `A10` | Check for duplicate `(station, timestamp)` pairs surviving the panel reindex — a duplicate would silently corrupt every groupby-shift. | L | S |

---

## 1. Feature generation

The only stage that has ever produced a real gain (−13 RMSE total). But note the ceiling
evidence: XGB corr 0.9988 and log-target corr 0.9962 say *features determine predictions*.
That cuts both ways — it means features are also the only lever left.

### Tier 1 — genuinely missing, do these

| # | Idea | Impact | Time |
|---|---|---|---|
| `F1` | **PM10-fraction framing.** PM2.5 is physically a subset of PM10, and `C0_lead1_PM10` already dominates every importance table. So the model is really being asked "what fraction of next hour's PM10 is fine?" Make that explicit two ways: (a) feature — `PM10_lead1 × trailing PM2.5:PM10 ratio` from *training-observable* history only (careful: the ratio needs past PM2.5, which is legal as a **lagged** value in train but absent in test — so build it from the station's long-run ratio estimated on train and applied as a static per-station/per-month constant); (b) target — regress `PM2.5(t+1) / PM10(t+1)` and multiply back. (b) is the cleaner test and reframes what the model learns without touching the feature set. | **H** | M |
| `F2` | **Centred rolling windows around t+1.** Every window in tier C is trailing. A window spanning t−1…t+3 is legal (all covariates present in test) and is the one feature family with no analogue anywhere in the codebase. Centred mean/std/min/max on PM10, CO, NO2, WSPM. Caveat: tier C's *trailing* rollings lost +0.89 — so centre them and drop the trailing ones rather than adding both. | **H** | M |
| `F3` | **Lead depth sweep.** Tier C0 goes to t+3, tier G to t+6. Nobody has found where the curve flattens. Score t+1 / t+1–3 / t+1–6 / t+1–12 as four variants. Directly extends the biggest known win. | M | M |
| `F4` | **Wind × lead pollutant interactions.** Phase 2 found wind direction only matters when it is windy, and tier G only has `G_lead1_pm10_per_wspm`. Add explicit `C0_lead1_wind_u × C0_lead1_PM10`, same for `v`, and `WSPM_lead1 × city_lead1_PM10`. Advection, expressed as a product a tree can use in one split. | M | S |
| `F5` | **Retest `A_heating` alone.** Tier A died as a block at +4.52, but that was isolated to `A_doy` (+3.74) and `A_month` (+1.94) — pure day-memorisation. `A_heating` is a genuine step change in coal burning and may have been guilty by association. One feature, one run. | M | S |
| `F6` | **Rain scavenging at the target hour.** `B_is_raining` is at t. Scavenging happens *during* the hour predicted. Add `is_raining(t+1)`, `RAIN(t+1)`, hours-since-rain, and cumulative rain over 6/24 h. RAIN is in `NUMERIC_RAW` so `C0_lead1_RAIN` exists — but the derived binary and the accumulators do not. | M | S |

### Tier 2 — plausible

| # | Idea | Impact | Time |
|---|---|---|---|
| `F7` | **Station-cluster aggregates.** Currently there are only two spatial scales: this station, and all twelve. K-means the stations on their pollution profile (or just on their pairwise correlation matrix) into 3–4 groups, then cluster-mean/lead aggregates as a middle layer. Beijing's sites split cleanly urban/suburban/background. | M | M |
| `F8` | **Learned upwind neighbour.** Under a given wind sector, which station leads which? Cross-correlate station series at lags conditioned on wind sector — learned from data, so it stays inside the no-external-data rule (do not import coordinates). Then feed the upwind station's PM10 at t and t+1. Physically the *right* feature; realistically too slow for today. | H | XL |
| `F9` | **Per-station × hour-of-day target encoding**, strictly out-of-fold. Note the calendar-feature toxicity result — target encoding on a time key is the same failure mode wearing a different hat. Guard with OOF folds and expect it to lose. | ? | M |
| `F10` | **Delta-target framing.** Predict `PM2.5(t+1) − anchor`, anchor = a simple PM10-derived estimate, add back. Cousin of `F1`; if `F1(b)` wins, try this too. | ? | M |
| `F11` | **Boundary-layer / inversion proxy.** `TEMP(t+1) − TEMP(t−12)` combined with low WSPM. Nocturnal inversions are the mechanism behind Beijing's worst episodes, and nothing in the feature set expresses vertical stability. | M | M |
| `F12` | **Dew-point depression at t+1** exists (`G_lead1_dpd`) — but its *change* across the boundary and its interaction with PM10 do not. Fog/haze forms when DPD collapses. | M | S |
| `F13` | **Ozone as a photochemistry proxy:** `O3_lead1 × A_hour`. Secondary aerosol formation is daylight-driven. Note `A_hour` alone was not individually toxic — only `A_doy`/`A_month` were — so this is safe to try. | L | S |
| `F14` | **Trailing volatility:** rolling std of PM10 over 6/12 h. `C_rstd6/24_PM10` exist but sit in the rolling block that was excluded from `best_v1`. Try re-admitting *only* the std columns, not the means. | M | S |
| `F15` | **CO quantisation.** CO is recorded in coarse ~100-unit steps, and it is the #2 pollutant by importance. Try `log(CO)` and a rank transform to see whether the chunking is costing split resolution. | L | S |
| `F16` | **Missing-pattern count:** how many stations are offline this hour. May proxy for extreme conditions (sensors fail in bad weather). Tier E's per-column flags were noise, but a city-level count is a different signal. | L | S |
| `F17` | **Pressure-tendency retest at the lead.** Raw pressure tendency was a dud at corr −0.041, but `PRES(t+3) − PRES(t)` spans the *forecast* window and is a frontal-passage indicator. `G_delta3_PRES` exists but has never been scored. Covered by `A1`. | L | S |
| `F18` | **Interaction between the two biggest features:** `C0_lead1_PM10 × D_city_lead1_PM10`, and their ratio. When the station and the city disagree at t+1, which wins? Trees need many splits to express a ratio; hand it over directly. | M | S |
| `F19` | **Sub-hourly persistence proxy:** `C0_delta1_PM10` exists, but not its sign × magnitude decomposition, nor `|delta|`. Direction and volatility may matter separately. | L | S |

### Tier 3 — abstract, speculative, cheap to think about

| # | Idea | Impact | Time |
|---|---|---|---|
| `F20` | **Ask what the test period's weather looked like.** The test covariates are fully observed. Cluster test hours against training hours; if a regime is over-represented in test and rare in train, weight training rows *by similarity to the test distribution* (importance weighting, not recency weighting — recency already lost, but similarity is a different axis). | ? | L |
| `F21` | **Adversarial-validation-driven feature selection.** SO2 was already flagged as the top train/test discriminator. Run the full adversarial model and look at ranks 2–10 — features that separate train from test are features whose learned relationship may not transfer. Do not drop them blindly (D16 taught that lesson) but do look. | M | M |
| `F22` | **Encode the target's own dynamics without leaking.** PM2.5 history is banned in test, but its *proxy* — PM10 — is not. Build the "PM2.5-like" series: `PM10 × per-station long-run PM2.5:PM10 ratio`, then take its lags/rollings. A reconstructed pseudo-PM2.5 history, fully computable at test time. | **H** | M |
| `F23` | **Hour-of-day × station interaction as a categorical**, not as separate columns — LightGBM splits categoricals differently from numeric pairs. | L | S |
| `F24` | **Regime tagging.** Cluster hours into 3–5 meteorological regimes (stagnant / windy-clean / humid-haze / …) and pass the cluster id as a categorical. Gives trees a cheap top-level split. | M | M |
| `F25` | **Reverse the frame: model the change, not the level.** The target minus `raw_PM10 × ratio` is a much smaller-variance quantity. RMSE on the level is dominated by the level; the model may spend most of its capacity re-deriving persistence. | ? | M |

---

## 2. Post-processing

Nearly untouched, and it contains the cheapest H-rated idea in the file.

| # | Idea | Impact | Time |
|---|---|---|---|
| `P1` | **Bias correction — never run.** Every single LB score has come in **above** local (19.02 vs 17.54; 19.64 vs 17.75; 18.79 vs 16.60). Check mean predicted vs mean actual on both folds, overall / per month / per station / per decile. The test period is dirtier than fold B (PM10 +16.9%), so systematic under-prediction is physically plausible. Apply only if the direction is consistent across A **and** B. | **H** | S |
| `P2` | **Multiplicative vs additive.** For right-skewed concentration data a ratio transfers across level shifts better than an offset. Test both; prefer the ratio if they tie. | M | S |
| `P3` | **Decile-conditional / isotonic recalibration.** If the model under-predicts spikes and over-predicts calm hours, a monotone curve fitted on OOF beats a flat offset. Risky — fit on fold A, verify on fold B, never fit on both. | ? | M |
| `P4` | **Temporal smoothing of predictions.** The true series is smooth hour-to-hour; jumpy predictions are noise. 3-hour centred moving average within station, blend weight tunable — `pred_final = (1−w)·pred + w·smoothed`. Sweep w ∈ {0, 0.1, 0.2, 0.3}. Pure post-processing, evaluable on cached OOF in seconds. | M | S |
| `P5` | **Blend the incumbent submission CSV with a new candidate 50/50.** No refitting — arithmetic on two CSVs. Cheap variance reduction, and a reasonable hedge when a candidate is close but unproven. | M | S |
| `P6` | **Per-station floors/ceilings.** Lower-clip to each station's observed minimum rather than a global 2.0; upper-clip to the 99.9th training percentile rather than 999. | L | S |
| `P7` | **Rank-average multiple submissions** rather than value-average — robust to one member's scale being off. | L | S |
| `P8` | **Spike expansion.** If `V1` shows spikes are systematically under-predicted, apply a monotone stretch above a threshold: `pred > q → pred × (1 + k(pred − q)/q)`. RMSE rewards getting the top decile right more than anything else. Tune k on OOF. | ? | S |
| `P9` | **Check the lower clip at 2.0.** In a clean-air hour the true value can be below 2; the clip may be adding bias on exactly the rows where errors are small anyway. Measure how many predictions the clip touches before assuming it is free. | L | S |

---

## 3. Preprocessing

The D7 scheme earned −3.1 RMSE. What is left are refinements, plus one real question.

| # | Idea | Impact | Time |
|---|---|---|---|
| `R1` | **Aggressively fill the target-hour covariates specifically.** A NaN in `lead1_PM10` wastes the single most valuable feature for that row. Currently the lead is filled only as a side effect of filling the panel. Consider a dedicated, more permissive fill for t+1 columns — and pair it with a *correct* `C0_has_lead` (see `A5`) so the model knows which is which. | **H** | M |
| `R2` | **Correlation-weighted neighbour fill.** Step 2 uses the city median scaled by a station ratio. Weight instead by each station's historical correlation with the target station, or use the 3 nearest-behaving. More principled, and stations are known to fail semi-independently. | M | M |
| `R3` | **Log-scale interpolation for pollutants.** Concentrations are multiplicative and right-skewed; linear interpolation across a 6-hour gap in a rising episode systematically under-fills. | M | S |
| `R4` | **Interpolation limit sweep.** The 6 h cap is arbitrary. Test 3 / 6 / 12 / 24. | L | S |
| `R5` | **Multiplicative vs additive station offset** in the cross-station fill. Currently multiplicative (`city × ratio`) — confirm that beats additive rather than assuming it. | L | S |
| `R6` | **Iterative fill:** a second pass that may use already-imputed neighbours, vs the current strict one-pass-from-observed. | L | M |
| `R7` | **Kalman / state-space smoothing** per station per pollutant instead of linear interpolation. Principled; slow. | ? | L |
| `R8` | **Do not impute at all for the tree.** LightGBM routes NaN natively and D7 was justified on the *lag* features needing continuity. Test the ablation: impute for lag construction, then restore NaN on the raw columns. Separates "gaps break my windows" from "gaps break my splits". | ? | M |

---

## 4. Validation

Does not move RMSE directly, but it currently cannot resolve anything under ~0.4 — which
is the size of every remaining idea. On a deadline day, spend here only if an experiment
comes back ambiguous.

| # | Idea | Impact | Time |
|---|---|---|---|
| `V1` | **Mine the error breakdown.** `FoldResult` already carries `by_month`, `by_station`, `by_decile`, and OOF predictions are cached. Half the squared error is the top decile and *nobody has looked at which spikes are missed* — onset hours vs clearance hours, which stations, which months, and whether the error is signed. A 10-minute query on data you already have, and the most likely source of a real feature idea in this whole file. | H (indirect) | S |
| `V2` | **Use fold R as a drift check.** `FOLDS["R"]` (Mar–Aug 2016) is the **most recent data in the file** and has never been used. Wrong season, so never a primary score — but it is the closest thing to "does this change survive into 2016?" that exists. Run the top two candidates on it as a tie-break. | M | S |
| `V3` | **The unvalidated tail.** Fold B validates on Sep 2015–Mar 2016, but the final fit trains through Aug 2016 — roughly 100k rows that enter every submission and appear in no validation. If anything about that period is unusual, no fold can see it. At minimum, characterise it: mean target, mean PM10, missingness rate vs fold B's training window. | M | S |
| `V4` | **Rolling-origin folds.** Four to five Sep–Feb windows with expanding origins would cut decision variance for everything downstream. Correct answer, wrong day — the data only supports three Sep–Feb windows and one is starved. Note it in the report as a known limitation. | H (indirect) | L |
| `V5` | **Match fold training size.** Fold A's weakness may be pure data volume (155k vs 258k vs 361k). Build a fold-B variant trained on only 155k rows: if it degrades to ~22, fold A's difficulty is an artifact and the A-vs-B argument is settled by measurement instead of another reversal. | M | M |
| `V6` | **Make `tune.py` optimise fold B.** It reportedly optimises mean(A,B) while the settled conclusion is that fold B ranks the LB perfectly (Spearman +1.000) and fold A is *anti*-correlated (−0.400). Pick one and make the code match. Cheap, and prevents a future search from being steered by the wrong metric. | M | S |
| `V7` | **Track transfer rate as a first-class column** in `leaderboard_tracker.csv`: LB delta ÷ fold-B delta, per submitted pair. Three points so far suggest ~0.25–0.35. If it is stable you can forecast; if not, you cannot, and should stop trying. | M | S |
| `V8` | **Bootstrap CIs on fold RMSE** rather than seed spread alone — separates model noise from data noise, and gives the report an honest error bar. | L | M |
| `V9` | **Paired seed comparison.** Compare candidate and incumbent on *the same seeds*, and test the per-row squared-error difference rather than the difference of RMSEs. Massively more statistical power than comparing two noisy means — this alone might resolve the 0.1–0.3 differences that currently read as noise. | **H** (indirect) | S |

`V9` deserves emphasis: the "noise floor 0.054" figure comes from comparing independent
means. Paired differencing on identical seeds and identical validation rows removes the
shared variance and can resolve deltas several times smaller. If you do one validation
item today, do this one.

---

## 5. Model training

Near-exhausted for *architecture*, but three specific configurations were never tried.

| # | Idea | Impact | Time |
|---|---|---|---|
| `M1` | **sqrt-target member.** `LightGBMModel` already implements `sqrt_target` (`models.py` ~line 96) — "sits between raw and log" — and **no member in `MEMBERS` uses it.** Log lost badly (21.095) because it under-weights the tail RMSE cares about; sqrt compresses far less. Written, tested by nobody. One registry line. | M | S |
| `M2` | **Feature-subsample ensembling.** Train the same LightGBM on random 70% column subsets, 8–10 times, average. Forces the disagreement that a different *library* failed to provide (XGB corr 0.9988). Different features → genuinely different errors, unlike different split algorithms on identical features. | M | M |
| `M3` | **Seeds 5 → 15–20.** Boring, near-certain, small. Also lowers the noise floor for every comparison after it. Run it in the background while doing something else. | M | M |
| `M4` | **Round-count scaling audit.** `submit.py` computes `rounds = best_iteration × len(train)/trm.sum()` — a ×1.40 multiplier applied to a fold-B early-stopping result. That guess sits underneath **every submission ever made.** Test ×1.0, ×1.2, ×1.4, ×1.7 by scoring each on fold A (the only held-out data once fold B is used for stopping). Cheap, and it is systematic error, not noise. | M | S |
| `M5` | **CatBoost.** Never installed. Ordered target statistics handle `station`/`wd` by a genuinely different mechanism — the only remaining algorithm with a plausible shot at decorrelated predictions. Budget install time. | M | M |
| `M6` | **`extra_trees: true` in LightGBM.** One parameter. Randomised split thresholds — a real change in inductive bias, far cheaper than installing a new library. Same for `boosting: dart`. | L | S |
| `M7` | **Two-stage regime split.** Classifier for "is t+1 a high-pollution hour", then separate regressors either side. Spike weighting already showed that biasing toward spikes helps (`tuned_wspike` is the incumbent), so the direction is evidenced; this is its sharper form. | ? | M |
| `M8` | **Spike-weight sweep.** `tuned_mf` searched it to 2.31, `tuned_wspike` uses 1.0, `lgb_wspike2` uses 2.5. Nobody has swept it *at fixed tuned hyperparameters* on fold B. 0 / 0.5 / 1 / 2 / 3 / 4. Given the incumbent is a spike-weighted model, this axis is proven live. | M | M |
| `M9` | **Tweedie or Gamma objective.** Right-skewed positive target — a natural fit, and never tried. `objective: tweedie, tweedie_variance_power: 1.1–1.5`. | ? | S |
| `M10` | **Huber / fair loss retest** under the current feature set (previously tested under a different one). | L | S |
| `M11` | **Monotonic constraint on `C0_lead1_PM10`.** Physically PM2.5 must increase with PM10. One constraint on the single dominant feature may regularise usefully and improve transfer to a dirtier test year. | ? | M |
| `M12` | **Per-station models** (12 fits) vs one model with station categorical. Usually loses; occasionally wins when sites genuinely differ. Data per station is 30k rows — probably too thin. | ? | L |
| `M13` | **Lower learning rate, more rounds.** Tuning found 0.022; nobody has tried 0.01 with 4000+ rounds. The classic last-day free win when you have compute to spare. Run it in the background. | M | L |
| `M14` | **Sequence model (GRU / temporal CNN)** over 24–72 h of legal covariates. Days of work; evidence says the ceiling is information, not capacity. Listed for the report's "considered and rejected" section only. **Do not start.** | L | XL |

---

## 6. Test inference

Well-plumbed. One real question, the rest is confirmation.

| # | Idea | Impact | Time |
|---|---|---|---|
| `T1` | **Confirm the default submission path is not blending.** exp005's A40/M60 blend lost 0.19. `submit.py` blends only when given `name:weight` specs, so a single-member call is clean — verify and move on. | M | S |
| `T2` | **The train→test boundary row.** 99.25% of test rows have their t+1 neighbour, but the first test hour per station takes its lags from the last training hours. `_panel` concatenates train+test so this works — confirm with an actual value check on one station. | M | S |
| `T3` | **Predict with several round counts and average** rather than committing to one scaled guess. Turns `M4`'s risk into a variance reduction. | L | M |
| `T4` | **Where `has_lead` is false, fall back.** For the ~0.75% of rows with no t+1 covariates, the dominant feature is imputed or NaN. Consider a separate simpler model, or a persistence-style fallback, for exactly those rows. Small row count, but they are likely the worst errors in the set. | M | M |
| `T5` | **Re-run the incumbent with a different seed set and submit it.** Measures leaderboard-side noise, which is currently *completely unknown* — you have been treating LB deltas of 0.19 as signal with no idea what the LB's own variance is. Costs one slot; informs every future decision. | M (indirect) | M |

---

## 7. Data loading

Little to try. Plumbing that either works or is broken, and it works. Covered by `A7`–`A10`.

---

## 8. Wildcards — abstract, high-variance, listed because you asked for them

| # | Idea | Impact | Time |
|---|---|---|---|
| `X1` | **Pseudo-labelling.** Predict the test set, take the most confident rows, add them to training as labelled data, refit. The test period is a *different year* with a dirtier distribution; pseudo-labels are the only mechanism that lets the model adapt to it. Risk: it amplifies its own bias. Mitigate by using only high-agreement rows (low variance across seeds). Genuinely the biggest conceptual swing available. | ? | L |
| `X2` | **Exploit the test set's own structure.** Test rows are a contiguous panel. For a test row at time t, the *test* row at t+1 exists — and its `raw_PM10` is that hour's observed PM10. You already use this (tier C0). But you can go further: the model's own prediction for row t is an estimate of PM2.5 at t+1, which is the *target hour* of row t and the *observation hour* of row t+1. Consistency between consecutive predictions is checkable and enforceable. Stage-2 tried a version of this and failed at −0.127 — but it fed back a prediction as a *feature*; this is instead a smoothness *constraint* (see `P4`). | ? | M |
| `X3` | **Fit on train + fold-B-validated hyperparameters, but early-stop against fold R.** Uses the most recent data as the stopping signal, which is closest in time to test. Sidesteps `M4`'s scaling guess entirely. | M | M |
| `X4` | **Quantile ensemble.** Fit q=0.4, 0.5, 0.6 and blend toward the mean. RMSE wants the conditional mean, but an average of nearby quantiles is a robust estimator of it and may transfer better under distribution shift. | ? | M |
| `X5` | **Train on the target's rank, predict, map back through the training quantile function.** Distribution-free, immune to the level shift between train and test years. Almost certainly loses on RMSE; interesting if it does not. | L | M |
| `X6` | **Look at what the top of the leaderboard implies.** 18.22 vs your 18.79 is a 3% relative gap. Given the lead features are worth −9.15 and everything since has been worth <1 combined, the leader almost certainly has the same core trick plus better execution — more seeds, better calibration, a longer lead horizon. That is an argument for `A1`, `M3`, and `P1` rather than for a new idea. | — | — |
| `X7` | **Accept the plateau and buy insurance instead.** Two submissions: the best candidate, and a 50/50 blend of the best candidate with the current incumbent (`P5`). If the candidate is real you keep most of the gain; if it is noise you lose half as much. On the last day with unknown LB variance, this is defensible risk management, not timidity. | M | S |
| `X8` | **Re-read `reports/phase2_findings.md` for unexploited EDA.** Phase 2 produced correlations, missingness structure, and inter-station relationships. Several findings (stagnation, wind sectors, the 0.88 inter-station correlation) were turned into features; check the list for ones that were not. | ? | S |

---

## 9. ⚑ Required deliverables — NOT OPTIONAL, reserve 90 minutes

From `Expected-submission-requirements.txt`. None of this exists yet. Missing docs alone
will not disqualify, but "if the submitted materials cannot plausibly explain or reproduce
the leaderboard result, the submission may be held for technical review or **deemed
ineligible for finalist selection**."

| # | Deliverable | Status |
|---|---|---|
| `W1` | **Methodology report** — approach, cleaning, feature engineering, validation strategy, models tested, final selection, ensembling/post-processing, key results, observations, **limitations** | ❌ missing |
| `W2` | **README / reproduction instructions** — required files, execution order, dependencies, seeds, and *which script generates the final prediction file* | ⚠️ `README.md` exists; verify it covers all five points |
| `W3` | **Final model information** — the exact member, its hyperparameters, seed count, round count, and any blend weights | ❌ missing |
| `W4` | **Required disclosure** — external data (none), external code consulted, pretrained models (none), **AI tools / coding agents used** (Claude Code — say so plainly), manual post-processing | ❌ missing |
| `W5` | **The exact submission CSV** corresponding to the LB score being considered, unmodified | ✅ `submissions/` — make sure the right one is identified |
| `W6` | **Local validation scores + LB scores + experiment comparison** — "strongly recommended" and you already have them in `experiments/`; just point at them | ✅ mostly done |

**The report writes itself from the logs, and it is unusually strong material.** Most teams
will not have: the lead-feature discovery (−9.15, reframing a forecast as a nowcast), the
seed-noise finding that reversed two tier conclusions, the SO2 reversal where a local tie
cost 0.62 on the leaderboard, the stage-2 negative result *with a correct causal
explanation*, and the validation saga where the team discovered its own metric was
misleading it. Write the negative results up properly — they are the most credible thing
in the package.

---

## Submission batching

Local validation cannot resolve differences under ~0.4 (before `V9`), so the leaderboard
has to do the tie-breaking. Produce candidates in batches, not one at a time.

**Design each batch for information, not just score.** A good batch = one incumbent re-run
(measures LB noise), one high-confidence variant, and two or three testing *different*
hypotheses. Five variants of one idea teach you almost nothing.

### Batch 1 — run these now, in this order

Each is a single command. Times assume ~2–3 min per seed on the full 361k fit.

```bash
# 1. TIER G — the headline experiment. Requires the one-line MEMBERS addition from A1.
#    Validate on fold B FIRST (5 min) and only submit if it beats 16.599.
python -m src.ensemble fit tuned_wspike_G --seeds 5 --fold B
python -m src.submit tuned_wspike_G --seeds 5 --exp-id exp006_tierG

# 2. LB-NOISE PROBE — the incumbent, different seeds. Tells you what a 0.19 delta means.
python -m src.submit tuned_wspike --seeds 5 --exp-id exp007_seedprobe

# 3. SQRT TARGET — M1. Register the member first, then:
python -m src.ensemble fit tuned_wsqrt --seeds 5 --fold B
python -m src.submit tuned_wsqrt --seeds 5 --exp-id exp008_sqrt

# 4. MORE SEEDS — M3. Same model as the incumbent, 15 seeds. Near-certain small gain.
python -m src.submit tuned_wspike --seeds 15 --exp-id exp009_seeds15

# 5. BIAS-CORRECTED — P1. Post-process an existing CSV; no refit needed.
#    Derive the offset from fold A + fold B OOF, apply to exp004's predictions.
```

That batch answers four independent questions in one round: **is tier G real, how noisy is
the leaderboard, does a milder target transform beat both raw and log, and does bias
correction transfer.**

### Batch 2 — conditional on batch 1

- If tier G wins → `F3` lead-depth sweep and `F2` centred windows, both built on top of G
- If tier G loses → `F1` PM10-fraction framing and `F22` reconstructed pseudo-PM2.5 history
- If the seed probe shows LB noise > 0.15 → **stop chasing sub-0.3 differences entirely**
  and spend the day on section 9
- Always: `P5` blend-with-incumbent as the final hedge (`X7`)

### Manifest

For every CSV written, append a row to `experiments/log.csv` with: feature recipe, model
config, seed count, round count, post-processing applied, fold A and fold B RMSE, and a
one-line hypothesis. Then `python -m src.tracker add <exp_id> --lb <score>` the moment a
score comes back. The tracker is a required deliverable (`W6`), so this is not bookkeeping
for its own sake.

---

## If two rounds produce nothing

That is a result, not a failure. The lead features were the structural win and they are
banked at 18.788. At that point every remaining mark is in section 9, and section 9 is
worth more than 0.1 RMSE.
