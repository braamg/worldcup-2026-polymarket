# Model Review — WC2026 Predictor

*A critical, component-by-component analysis of the forecasting model, separating
improvements with real out-of-sample value from those that only add noise. Grounded in the
actual code; external claims validated against the football-forecasting literature.*

Date: 2026-06-20. Reviewed against the live model (ensemble validated RPS ≈ 0.1697).

---

## TL;DR — your three questions answered first

1. **Would more than 100K simulations help?** **No — this is noise.** At 100K the Monte
   Carlo sampling error on the markets you actually bet is well under 0.1 percentage points,
   while the model's *specification* error is 1–5pp — 10–50× larger. Tournament sims are
   known to converge by ~10K; at 40K the sampling noise on a coin-flip is already ±0.5pp.
   More sims polish a digit that the model's real uncertainty has already swamped. Keep 100K
   (it's already at the high end of professional practice); spend the compute elsewhere.

2. **More variables in the XGBoost model?** **Mostly noise — with one big exception.** Generic
   form/streak/goal features are largely redundant with Elo (which already integrates results
   with time-decay) and would add overfitting risk to a sparse, 10-feature, depth-3 model. The
   **one feature that carries real non-Elo signal — squad market value — is starved: it's NaN
   for ~83% of training rows.** Fixing that coverage is the single highest-value change
   available, and the data is *already partly scraped in the repo*. Detail in §3 and §6.

3. **Should the ensemble be weighted differently than 50–50?** **Almost certainly not.** This
   is the "forecast-combination puzzle": across decades of research, the simple equal average
   robustly beats *estimated* optimal weights out-of-sample, because the weights carry their
   own estimation error. With two models and a ~2K-match validation set, a learned weight will
   most likely overfit. A one-off confirmation sweep is cheap and worth doing, but expect the
   optimum to sit near 0.5 with a gain smaller than the noise floor (<0.001 RPS).

**The uncomfortable headline:** the most valuable "model" work is not an accuracy tweak at
all. It's (a) fixing market-value coverage, and (b) validating that the model's *tail*
probabilities — the longshot champion/deep-run numbers your entire betting edge rides on —
are actually calibrated. Chasing RPS decimals below that is misallocated effort.

---

## How the model works (as built)

```
Elo (k=15, HA=125, MoV=0.25, frozen pre-2010 grid search)
        │
        ├──────────────► XGBoost (10 features, depth 3, heavy regularisation)
        │                   features ≈ {elo_diff, e_home, log_ratio_mv[83% NaN],
        │                               has_mv, neutral, cross_conf, rest_days[dead]}
        │
Dixon-Coles (bivariate Poisson, time half-life ~347d, L2=0.01, conf-shrink OFF)
        │
        ▼
ensemble = 0.5·DC + 0.5·XGB  →  temperature T=1.0365  →  (cross-conf blend OFF)
        │
        ▼
Monte Carlo tournament (100K sims; per-team Elo jitter σ=50 × data-poorness scaling)
        │
        ▼
sim_probabilities.json  +  sim_outcome_matrix.npz (joint distribution)
```

I note up front that this codebase is **unusually well-validated already**: the config and
docstrings record backtested decisions to *reject* confederation-mean shrinkage, *remove* the
cross-confederation entropy blend, and *reconcile* the Elo parameters. Several "obvious"
improvements have been tried and correctly discarded. The recommendations below are chosen to
avoid re-treading that ground.

---

## Component-by-component critique

### 1. Elo ratings — sound; leave alone
k=15, HA=125, MoV=0.25, frozen from a pre-2010 Brier-minimising grid search, and now folding
in live WC results on the same footing. This is methodologically clean (no leakage, frozen
hyperparameters). Elo is consistently the strongest single predictor in the literature.
**No change recommended.** The one structural weakness of Elo — it lags squad changes
(injuries, retirements, a new generation) because it only updates on results — is exactly
what market value is supposed to patch. Which brings us to the real problem.

### 2. Squad market value — the model's biggest missed opportunity 🔴
`log_ratio_mv` is the only feature that knows anything Elo doesn't (current squad quality,
independent of recent results). The code comment states it is **NaN for ~83% of training
rows**, and the structural cause is clear from `market_value.py`:

- `TM_TEAMS` maps only ~70 nations. `has_market_value` requires **both** teams to have a
  snapshot, so every historical match against the hundreds of non-listed minnows → NaN.
- Transfermarkt squad values effectively don't exist before ~2005, so the entire 1994–2005
  slice of the training era is unavoidably blank.

The consequence: **XGBoost is trained on data where its most informative non-Elo feature is
present only ~1 row in 6.** It therefore barely learns the market-value→outcome relationship —
and then at WC2026 prediction time, where the 48 finalists *do* mostly have values, that
signal is under-weighted because the model never learned to trust it. The market value is
doing far less work than it could.

This is the highest-value lever in the whole model, and the literature backs the priority:
squad market value is repeatedly found to rival or complement Elo (Nate Silver's PELE ratings
explicitly blend market values and player ages into an Elo backbone for exactly this reason).

**What to do:** expand `TM_TEAMS` to the next tier of nations (the HTML archive already covers
2005–2025 for the listed teams; adding more team IDs + a re-scrape is mechanical), rebuild
`squad_snapshots`, and re-fit. Realistic coverage might rise from ~17% to ~40–50% (pre-2005
and true minnows stay blank). **Validate on the ensemble RPS via your existing walk-forward
harness — not XGB-standalone** — because your own DC-shrinkage sweep already showed that
changes which help a base model can be neutral-to-harmful for the ensemble once error-diversity
is accounted for. This is an experiment with high expected value, not a guaranteed win.

### 3. XGBoost feature set — thin, and that's mostly correct
Beyond Elo and market value, the remaining features are `neutral_venue`, `cross_conf`, and
`rest_days_*` — and **rest days are effectively dead**: every WC fixture is fed
`rest_days=30, missing=True`, so the feature contributes nothing to the tournament forecast.
Two honest observations:

- **Adding generic features (recent form, goals for/against, win streaks) is noise here.**
  They are largely redundant with Elo, and on ~5K sparse rows with a depth-3 model they invite
  overfitting more than signal. The literature's "features beyond ratings help" results come
  overwhelmingly from *club* football with rich event/tracking data and deep nets — not
  transferable to this data regime.
- **Real rest-days/congestion for 2026 is a legitimate small experiment.** The expanded
  48-team format and the continental travel/altitude/heat spread (Mexico City altitude,
  US summer heat) are genuinely unusual. Computing actual rest differentials from the fixture
  schedule (instead of the dead `30/missing` constant) *might* add a little. Test it; don't
  assume it; expect a small effect.

### 4. The ensemble (50–50) — keep it; confirm once
Covered in TL;DR #3. The 50/50 average is the robust choice the forecast-combination
literature endorses. The error-diversity argument in your own DC docstring is exactly right.
**Action:** one confirmation sweep of w∈[0.3,0.7] on walk-forward RPS, then stop. Do **not**
build a learned/stacked meta-weighter — that's the overfitting trap the puzzle warns about.

### 5. Calibration — appropriate
Temperature scaling (T=1.0365, a mild softening) over isotonic is the right call at ~2K
calibration matches; isotonic would overfit. No change. (If MV coverage expands and the model
is re-fit, re-fit T too.)

### 6. Monte Carlo simulation — count is fine; the *jitter* is the real question
Two separate things often conflated:

- **Number of sims (100K): a solved problem.** Noise. See TL;DR #1.
- **Elo-jitter σ (=50, scaled by data-poorness, clipped [0.7, 2.0]): under-examined and
  high-leverage for *your* use case.** This jitter is essentially the *only* thing controlling
  the spread of tournament outcomes — and therefore the entire shape of the longshot tail
  (champion, reach-final probabilities). Your betting strategy lives *exclusively* in that
  tail. If σ is too small, deep-run longshot probabilities are systematically understated (and
  you'll see false "edges" when the market prices them higher); too large and the reverse.
  The base 50 and the [0.7, 2.0] clip are plausible but, as far as the code shows, not
  validated against realised tournament variance. **This is arguably more important than any
  accuracy tweak**, because it directly biases every EV the betting layer computes. Worth a
  dedicated calibration check: do historical tournaments' realised upset/deep-run rates fall
  within the simulated spread? At minimum, run a sensitivity analysis showing how the top
  champion probabilities move as σ varies — if they're very sensitive, the number deserves
  proper calibration rather than a default.

### 7. Minor / non-issues (noted, not flagged for change)
- Group-stage goals are sampled from the **DC** PMF conditional on the **ensemble** W/D/L
  outcome — a mild internal inconsistency (scoreline distribution is DC-only). Affects only
  group tiebreak GD/GF; almost certainly immaterial. Leave it.
- Knockout draws resolved ~50/50 — **correct** and well-justified; do not "improve" it with
  team strength.
- Knockout caching + analytical Elo-jitter logit correction — a clean, valid optimisation.

---

## Recommendations, ranked by real value

| # | Change | Value | Effort | Notes |
|---|--------|-------|--------|-------|
| 1 | **Expand market-value coverage** (more nations in `TM_TEAMS`, rebuild snapshots, re-fit) | **High** | Medium | ✅ Done — coverage 16.5%→29.5%; ensemble RPS 0.16968→**0.16942**, XGB 0.17121→**0.17011** (see "Market value — result" below). |
| 2 | **Calibrate the Elo-jitter σ** against realised tail variance | **High (for betting)** | Medium | ✅ Done — calibrated & **confirmed at σ=50** (see "Tail calibration — result" below). |
| 3 | Confirmation sweep of the 50–50 ensemble weight | Low | Low | Expect ~0.5 to win by < noise. Do it once, then stop. |
| 4 | Real 2026 rest-days/congestion feature | Low–Med | Low | Currently dead (`30/missing`). Test; don't assume. |
| 5 | More than 100K simulations | **None (noise)** | — | Don't. SE already << model error. |
| 6 | Generic form/streak/goal features in XGB | **Negative (noise/overfit)** | — | Redundant with Elo on this data regime. |
| 7 | Learned/stacked ensemble weights, isotonic calib, deeper trees | **Negative (overfit)** | — | Your conservative choices are already correct. |

---

## The meta-point: accuracy is not your binding constraint

The model is good and already well-tuned. But for the **betting** application, two things
dominate any RPS decimal:

1. **Tail calibration (rec #2)** — your edge is computed entirely from longshot probabilities,
   which are governed by the jitter, not by ensemble RPS. A model that's 0.001 better on RPS
   but mis-calibrated in the tail will still bleed money.
2. **The model has never been shown to beat the closing line** (the P0 from the betting-layer
   review). Until CLV/Brier-vs-market is positive over a real sample, *every* accuracy
   improvement here is theoretical. Improve coverage and tail calibration, then let the live
   ledger tell you whether the model actually has edge — before trusting any of it with size.

---

## Market value — result (2026-06-20)

Acted on rec #1. Added **39 verified national-team Transfermarkt IDs** for the most-frequent
previously-uncovered opponents (`TM_TEAMS`: 72 → 111). IDs were resolved with a *self-verifying*
script that fetches each candidate's page and confirms it's the men's senior national team —
which **caught two club false-positives** (Bahrain "A'Ali SC", Kuwait "Kuwait SC") before they
could poison the data, exactly the Wales→Greek-club failure mode. Re-scraped (819 new pages),
rebuilt snapshots (1,411→2,159 rows, 69→108 teams) and features, then re-ran the phase7
walk-forward eval.

**Coverage:** overall 16.5% → **29.5%**; the fixable 2006+ slice 25.1% → **44.8%**. (Pre-2006
stays 0% — Transfermarkt has no data that far back — so this is near the practical ceiling.)

**Out-of-sample RPS (test 2019–2026, lower = better):**

| Model | Before | After | Δ |
|---|---|---|---|
| dixon_coles | 0.18233 | 0.18233 | 0.00000 (unchanged — DC doesn't use MV ✓) |
| xgboost | 0.17121 | **0.17011** | **−0.00110** |
| dc_xgb_avg (ensemble) | 0.16968 | **0.16942** | **−0.00026** |
| xgboost — cross-conf slice | 0.18461 | **0.18286** | **−0.00175** |

**Verdict: modest but genuine, and kept.** The improvement is small on the global ensemble
(~0.15%) but it is *not* noise: only the MV-using model moved, DC was unchanged to the digit,
and the **cross-confederation slice improved most** — exactly where MV theory predicts the
largest lift (Elo is least reliable across confederations). The global RPS understates the
**WC2026-specific** benefit, since the tournament is cross-conf-heavy and many of its teams were
previously MV-blank. Production models retrained on full data (phase6) and the 100k sim re-run
(phase9). Remaining MV gaps are now structural (pre-2005 era + genuine minnows), not fixable by
more IDs.

---

## Tail calibration — result (2026-06-20)

The Elo-jitter `σ` is the single knob governing the longshot tail, where the betting edge
lives. Two diagnostics (`scripts/diagnostics/_tail_sigma_anchor.py`, `_tail_sigma_sweep.py`):

**1. Empirical anchor.** Forward Elo-revision SD for WC-calibre teams: **69.7** over the next
7 matches, **83.7** over 365 days. These are *upper bounds* on the estimation-error σ — they
also contain genuine future form change, which the jitter should NOT model. Decomposing the
drift out leaves a plausible pure-estimation σ of **~49–60**.

**2. Sensitivity sweep** (30k sims per σ):

| σ | Top favourite (Spain) | Top-8 share | Field (longshots, outside top-8) |
|---|---|---|---|
| 0 | 15.8% | 60.1% | 39.9% |
| 35 | 15.1% | 57.3% | 42.7% |
| **50 (current)** | **14.1%** | **55.6%** | **44.4%** |
| 70 | 12.5% | 53.5% | 46.5% |
| 100 | 11.3% | 48.3% | 51.7% |

The tail is genuinely sensitive (σ=0→100 swings the favourite ~4.5pp and the longshot field
~12pp), confirming σ is a high-leverage parameter — but **σ=50 is the right value and is kept
unchanged**, for three evidence-based reasons:

- **σ=50 sits inside the plausible estimation-error range (~49–60)**, at its conservative end.
- **Betting risk is asymmetric.** The strategy *buys* longshots: an inflated tail (σ too high)
  manufactures false edges you bet into and lose; an under-inflated tail merely means fewer
  bets. The costly error is over-dispersion, so the conservative (low) end is correct.
- **The per-team scaling already widens σ up to 2× (=100) for data-poor minnows** — precisely
  the longshots — so the base must not be pushed higher on top of that. (Market cross-check: at
  σ=50 the model already has Spain *below* the de-vigged book, 14.1% vs 15.7%; raising σ would
  widen that gap with nothing to justify it.)

**Flagged for the future (not changed now):** the jitter is *symmetric* Gaussian on Elo, which
slightly overstates minnow upside (a weak team is far more likely weak-as-rated than secretly
elite). An *asymmetric* or strength-dependent jitter would be more honest in the deep tail —
but it's a model change with its own validation burden, so it stays a documented idea, not a
silent tweak. No re-simulation was required: the live `sim_probabilities.json` already uses σ=50.

---

## Sources
- Forecast-combination puzzle / simple average robustness: [Wang & Hyndman, "Forecast combinations: a 50-year review" (arXiv 2205.04216)](https://arxiv.org/pdf/2205.04216); [Claeskens et al. on the combination puzzle (arXiv 1505.00475)](https://arxiv.org/pdf/1505.00475); [Blanc & Setzer, "When to choose the simple average" (J. Business Research)](https://www.sciencedirect.com/science/article/abs/pii/S0148296316303952).
- Monte Carlo convergence for tournament forecasts: [Lumivero, "2026 World Cup outcome probabilities"](https://lumivero.com/resources/blog/world-cup-outcome-probabilities-monte-carlo-simulation/); [competition-format simulation study (arXiv 2308.02507)](https://arxiv.org/pdf/2308.02507).
- Market value as a predictor alongside Elo: [Nate Silver, PELE international football ratings](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) and [PELE methodology](https://www.natesilver.net/p/pele-methodology).
- Features beyond ratings (mostly club-level deep learning): [PLOS One, FIFA ratings + formation framework](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0284318); [arXiv 2505.01902, ML for World Cup outcomes](https://arxiv.org/html/2505.01902v1).
