# Resume here — state as of 2026-09-06, 16:10

## Standing entry

**`submissions/exp022_b5_GHcw.csv` — leaderboard 18.54180. Currently 2nd place.**

Equal fifths of `exp004_tuned_wspike` + `exp006_bagged15` + `exp002_best_v1`
+ `exp012_tierG` + `exp013_tierHcw`. No fitted weights.

    python -m src.blend_csv exp004_tuned_wspike:1 exp006_bagged15:1 \n        exp002_best_v1:1 exp012_tierG:1 exp013_tierHcw:1 --out exp022_b5_GHcw

| # | Team | Score |
|---|---|---|
| 1 | Tavieh | 18.21598 |
| **2** | **tennis net sitters** | **18.54180** |
| 3 | Victoria Team | 18.69571 |

Gap to first: **0.326**. Started the session 3rd at 18.78847 — **−0.247**.

Full model card, reproduction and disclosure: **[FINAL_SUBMISSION.md](FINAL_SUBMISSION.md)**.

---

## READ THIS BEFORE SPENDING A SUBMISSION

**Fold B cannot be trusted on any blending question.** Not composition
(Spearman −0.564, p=0.090 over ten blends — actively anti-correlated), and not
nesting either. Mid-session I thought nesting was safe because fold B ranked
three nested blends correctly; I used that to pick `tuned_mf` as the best of
eleven possible pool additions (fold B said −0.024) and it **lost 0.097** on the
leaderboard (exp024). The rule was induced from n=3 and died on its first real
test. Full account: `reports/experiment_record.md` §9.8.

There is no cheap local substitute. Every blend decision that transferred was
either confirmed by a submission or was an unfitted equal-weight construction.

---

## The one thing that worked today

**Blending submitted CSVs.** Not a better model — an average of three existing
ones. Components score 18.788 / 18.848 / 19.025 standalone; the equal-weight
blend scores 18.612, beating all of them.

Diversity is what pays, not component quality. `best_v1` is the *worst*
component and wants a full third of the weight. Adding `exp003` (no-SO2) or
`exp005` *hurts* — the first is a degraded model, the second is already a blend
and adds no new direction.

Two models that LOST on fold B (`tierG` +0.18, `tierHcw` +0.017) both improved
the blend. But the ordering of pool candidates tracks **solo quality**, not
decorrelation: the three most decorrelated members available (`lgb_log` 0.896,
`extra_trees` 0.935, `xgb_v1` 0.949) are the three worst additions. A member
must be decorrelated **and** within ~10% of the incumbent's quality.

---

## Kaggle CLI is wired up

    kaggle competitions submissions -c inter-uni-datathon-stream-2-beijing-multi-site-air-quality -v
    kaggle competitions leaderboard  -c inter-uni-datathon-stream-2-beijing-multi-site-air-quality -s
    kaggle competitions submit -c inter-uni-datathon-stream-2-beijing-multi-site-air-quality -f submissions/X.csv -m "msg"

**60 submissions remaining today** (14 used this session). Budget them: the public/private split is
unseen, and weight-tuning against the public half is a real overfitting risk.
LB noise is small (< ~0.02), so 0.09 differences are signal and 0.01 are not.

---

## What was tested today and lost

| Idea | Fold B Δ | 95% CI | Verdict |
|---|---|---|---|
| Tier G — extended leads (237 feat) | +0.18 | [−0.28, +0.73] | loses |
| Tier H `cw` — centred windows (195 feat) | +0.017 | [−0.075, +0.129] | neutral |
| Tier H `rest` — obs flags, interactions, rain, inversion | +0.60 | [+0.31, +0.94] | **worse** |
| Tier H full (225 feat) | +0.49 | [+0.25, +0.77] | **worse** |
| Smoothing / stretch / rescale (all variants) | — | — | fold-specific, opposite signs on A and B |

Tier G was the top item on the improvement backlog and had never been scored on
any fold. It loses. **Every feature family tried today failed.** The model is
information-saturated: PM10 at t+1 plus PM10 now is the whole problem.

---

## New tooling worth keeping

- **`src/paired.py`** — paired per-row squared-error comparison on identical
  seeds, day-block bootstrap. CI of **±0.10** on correlated models against the
  old ±0.4 independent-means floor. Use this for every future comparison.
- **`src/postproc.py`** — post-processing sweeps on cached OOF, no fits.
- **`src/blend_csv.py`** — value or rank blending of submission CSVs.
- **Fixed:** `ensemble.fit_member` mutated `MEMBERS` via
  `base_params.pop("spike_weight")`. Audited — no past result was corrupted.

---

## The largest unexploited finding

From the fold-B error breakdown (`reports/experiment_record.md` §8.4):

| Slice | n | Bias | RMSE | Share of squared error |
|---|---|---|---|---|
| **`lead1_PM10` imputed** | **430** | −4.08 | **73.96** | **16.6%** |
| rising fast | 3,150 | **−20.01** | 40.14 | 35.9% |
| top decile | 5,125 | −19.23 | 41.51 | 62.4% |

**0.84% of rows carry 16.6% of all squared error** — the ones where D7
imputation fabricated the dominant feature and handed it over looking like a
real reading. Halving their error would be worth roughly −1.0 on fold B.

Attacked directly and **it backfired.** `tuned_wspike_Hobs` gives the model the
un-imputed lead (NaN preserved) plus honest observation flags, in isolation.
Those 430 rows got **worse**: 73.96 -> 76.42 RMSE, while the other 50,919 rows
were untouched (15.2208 -> 15.2229).

The D7 city-median fill is a *better* input than an honest NaN. Those rows are
hard because the sensor was down, and no test-time covariate recovers what was
never measured. Same wall as the stage-2 result. **Right diagnosis, wrong
treatment — and the concentration of error appears to be irreducible.**

---

## Deliverables status

| # | Required material | Status |
|---|---|---|
| 1 | Methodology report | ✅ `reports/experiment_record.md` (735 lines, Parts I + II) |
| 2 | Complete source code | ✅ `src/` |
| 3 | Final prediction file | ✅ `submissions/exp022_b5_GHcw.csv` |
| 4 | Processed datasets | ✅ regenerable — `FINAL_SUBMISSION.md` §4 |
| 5 | README / reproduction | ✅ `FINAL_SUBMISSION.md` §3 |
| 6 | Final model information | ✅ `FINAL_SUBMISSION.md` §2 |
| — | Required disclosure | ✅ `FINAL_SUBMISSION.md` §5 |

---

## Settled decisions (unchanged)

| ID | Outcome |
|---|---|
| D1 | Scripts, not notebooks |
| D2 | **User does all git operations** — never run git/gh |
| D3 | All training data, uniform weights |
| D7 | Interpolate ≤6h → cross-station fill → NaN (~−3.1 RMSE) |
| D16 | **KEEP SO2** — dropping it cost +0.62 on the leaderboard |

## Working preferences

- User does all git commits. Recommend commit points; never run git.
- Warn before any run over ~5 min, with an ETA. Background it, `python -u`.
- Cheap decisions with few options: build all, score them, report the winner.

## Suggested commit

    Blending beats modelling: 5-member equal-weight CSV blend scores 18.54180
    (from 18.78847), 2nd place. Every feature family tried lost; the whole
    -0.247 came from ensembling. Fold B is anti-correlated with the LB on
    blends (Spearman -0.564) and its nested-addition rule also failed (exp024,
    +0.097). Adds src/paired.py (paired testing, CI +/-0.10), src/postproc.py,
    src/blend_csv.py; fixes the MEMBERS mutation bug in ensemble.py.
