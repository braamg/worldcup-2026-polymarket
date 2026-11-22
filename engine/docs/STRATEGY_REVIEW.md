# V2 High-Yield Strategy — Validity Review

*A critical, honest evaluation of the betting approach behind this dashboard.*

*Public version: money amounts are shown as a percentage of the bankroll, and references to internal notes have been removed.*

---

## TL;DR

The **dashboard is well-built**; the **strategy is unvalidated**. Every number on the
screen — EV, Kelly, "edge", expected profit — is downstream of one unproven assumption:

> *Our Monte Carlo model estimates outcome probabilities more accurately than the
> Polymarket price does.*

Nothing in this project tests that assumption. Worse, the strategy deliberately fishes
in **longshot contracts (<40¢)** — the exact zone where betting markets are *known* to
misprice in a direction that manufactures fake "value". Until the model is validated
against the market, the EV column is not a measure of edge; it is a measure of *how much
the model disagrees with the market*, which is not the same thing.

This document separates what is sound, what is questionable, and what to fix — in priority
order.

---

## STATUS UPDATE (2026-06-28) — what's been built since this review

The original critique below stands, but its central gap ("no calibration evidence anywhere in
the repo") has been **directly addressed**. What changed:

- **The verdict system (`utils/verdict.py`, 📋 Verdict tab) now answers the central question.**
  Q1 tests market efficiency on real results+prices *today*; Q2 measures whether the model beats
  the market (Brier skill, CLV, calibration) **forward-only** via `forecast_log.csv` — no
  look-ahead. Pre-registered tests, colour-coded verdict, confidence by sample size.
- **A daily forecast snapshot** (on-load + scheduled task) logs model vs market for every market,
  so we accumulate ~170+ clean data points rather than a handful of biased bets. **Complete answer
  expected ~2026-07-09.**
- **Shrinkage is now first-class and calibrated, not hidden.** `model_trust` (0.35) +
  selection-haircut keep only ~26% of the raw edge; `calibrate_trust()` tunes it from clean resolved
  data (and refuses to over-react to a flat curve — see the trust-calibration trap in the main README).
- **Sizing was the bigger lever than the model.** Correlation-aware `optimize_blend()` Pareto-beats
  naive weighting; the once-off "size to survive" logic (hit-rate floor, fractional Kelly, bet cap)
  replaced the original flat-stakes plan.

### The early verdict (n=48 resolved, 2026-06-30): this review was RIGHT to be sceptical.

The forward scoreboard now has enough data for a read, and it is **NO EDGE so far**:

- **Brier skill −0.032** — the market is *sharper* than the model on resolved forecasts.
- **CLV +0.06¢, p=0.69** — closing-line value is statistically indistinguishable from zero.
- **Paper-trade −80%** — of 20 "backable" picks (model > market by ≥5pp), only **1 won**.

In other words, exactly the failure mode this review warned about: the EV column was measuring *how
much the model disagrees with the market*, and on the evidence so far the **market was more often
right**. The draw "edge" is also fading, not confirming (gap +7pp→+4.3pp, p≈0.30→0.44 as n grew).
This is not a final verdict — knockouts are still resolving and it could move — but the discipline
of pre-registered, significance-gated tests is doing its job: it is stopping us from telling a
winners' story the data doesn't support. (A verdict-logic bug that had counted a near-zero CLV as
"PROMISING" was fixed on 2026-06-30; the honest reading is NO EDGE, Medium confidence.)

The consolidated lessons are in the repository's main README.

---

## 1. The central problem: you are betting that the model beats the market

EV and Kelly are both computed as functions of `model_prob` vs `implied_prob`
([data_engine.py](utils/data_engine.py)). They are only meaningful if `model_prob` is the
*better* estimate. But:

- Polymarket prices aggregate the capital-weighted opinions of many participants. For a
  liquid market the rational prior is that the price is **well-calibrated**, and the burden
  of proof is on the model to show it is sharper.
- A single offline Monte Carlo simulation, refreshed manually, is a weak challenger to a
  live market that continuously incorporates injuries, suspensions, draw news and form.
- **There is no calibration evidence anywhere in the repo.** No Brier score, no log-loss,
  no reliability curve, no backtest, no closing-line comparison.

**Implication:** the entire pipeline currently dresses up an *untested opinion* in the
language of expected value. The math is correct; the inputs are unaudited.

> This is fixable and is improvement **#1 below**. Everything else is secondary to it.

---

## 2. The strategy targets the worst zone for false positives (favourite–longshot bias)

One of the most replicated findings in sports-betting and prediction-market research is the
**favourite–longshot bias**: longshots are systematically *over*priced (punters overpay for
lottery-ticket payoffs) and favourites are mildly *under*priced.

This strategy filters to contracts **under 40¢** and the mock set goes as low as 3–7¢
(`champion` longshots). That is precisely the region where the market's own pricing error
tends to make a longshot *look* underpriced relative to a naïve model, even when it is
fairly priced or overpriced.

So the strategy must overcome a known structural bias that runs *against* it. A model that
merely "disagrees with the market on cheap contracts" will generate a stream of
plausible-looking positive-EV picks that are, in aggregate, losers.

---

## 3. Selection bias / winner's curse (under-appreciated, large effect)

The engine scans ~60 contracts and surfaces the **top 5 by Kelly**
([`process_opportunities`](utils/data_engine.py)). By construction, the picks are the
contracts where the model *most* disagrees with the market.

The contracts where a noisy model most disagrees with the market are disproportionately the
ones where **the model is most wrong**, not where the market is most wrong. This is
regression to the mean: conditional on "largest measured edge", the *true* edge is
systematically smaller than measured — often negative. Ranking-and-selecting amplifies the
problem the more contracts you scan.

**Consequence:** even if the model were unbiased on average, the *selected* bets would still
underperform their advertised EV. The displayed +75% EV is an overestimate of the realised
edge of the picks that survive selection.

---

## 4. EV ignores variance — Kelly is the right instinct, but the implementation is unsafe

You already added Kelly, which is the correct direction (your earlier intuition was right:
two bets with identical EV can have wildly different win frequencies and ruin risk). But:

- The code uses **full Kelly**. Full Kelly assumes the probability is *known exactly*. With
  an unvalidated, likely-overconfident model, full Kelly massively overbets and is a fast
  route to ruin. Standard practice is **fractional Kelly (¼–½)** plus shrinking the model
  probability toward the market.
- Kelly is currently used only as a *ranking* signal (stakes are flat because the small bankroll forces
  sub-minimum Kelly stakes). That is fine — but the dashboard should *say* "fractional
  Kelly, shrunk" so the ranking itself isn't driven by overconfident point estimates.

---

## 5. "Price = implied probability" is an approximation that costs you edge

`implied_prob = polymarket_cents / 100` assumes the quoted price is a clean probability.
In reality:

- **You execute at the ask, not the mid/last.** Real entry cost ≥ quoted price, so real EV
  is **lower** than computed. Thin longshot order books make this gap material.
- In a multi-outcome market (e.g. 48-team "World Cup Winner"), the outcome prices **sum to
  more than 1** (the overround / vig). The de-vigged probability the market actually
  believes is *lower* than the headline price. The dashboard should at least surface the
  book sum so the user knows how much vig is baked in.
- ZAR↔USD conversion + deposit/withdrawal friction further erodes a thin edge on a small
  bankroll.

None of this is fatal, but the current EV is an **optimistic upper bound**, not the number
you will actually realise.

---

## 6. The portfolio simulator multiplies marginals — this is mathematically wrong here

[views.py](components/views.py) computes "P(all win)" as the **product of marginal
probabilities**, i.e. it assumes the selected bets are independent. Tournament outcomes are
**strongly correlated**:

- *Nested* outcomes on one team are near-perfectly correlated: "Morocco to win" ⊂ "Morocco
  to reach final" ⊂ "Morocco to reach SF". Betting these together is not diversification —
  it is one leveraged view on Morocco, and multiplying their marginals badly understates the
  true joint probability.
- Across teams, outcomes share structure (only one team can win; group rivals are mutually
  exclusive to advance).

**The good news — the fix is already sitting in your data generator.** The Monte Carlo
simulation produces full tournament *paths*. The exposed `sim_probabilities.json` keeps only
the **marginals** (per-stage probabilities per team — confirmed by inspection). The **joint**
distribution exists upstream. If the simulator also emitted joint counts (or the raw paths),
the dashboard could compute the *true* probability of a bundle by counting the fraction of
simulated tournaments in which all selected legs occur — exactly right, correlation included.

---

## What is actually sound (so this isn't all criticism)

- **Flat staking at this bankroll size is correct.** You correctly identified that Kelly-optimal stakes
  fall below Polymarket's ~$1 minimum, so equal splitting is the honest choice at this scale.
- **Read-only / decision-support posture is prudent** — no auto-execution.
- **The mispricing framework is legitimate** *if and only if* the model is validated. The
  skeleton is right; the missing piece is evidence.
- **Adding Kelly over raw EV** was a genuine improvement.
- **The dashboard's transparency** (showing model %, market %, gap, payout side by side)
  is good practice and makes the assumptions inspectable.

---

## Recommended improvements — prioritised

### P0 — Without these, the strategy has no demonstrated validity

1. **Validate the model against the market.** Pull historical Polymarket (or bookmaker)
   closing prices and tournament outcomes; compute the model's **Brier score and log-loss
   vs. the market's**. If the model cannot beat the closing line, there is no edge — full
   stop. Show a reliability diagram on the System Overview tab.
2. **Track Closing-Line Value (CLV) going forward.** For every logged bet, record the price
   you got vs. the price at resolution/kickoff. Beating the closing line over many bets is
   the single most reliable evidence of real edge. Make CLV the headline KPI, above P&L.

### P1 — Corrects systematically optimistic numbers

3. **Shrink the edge for selection bias.** Apply a haircut to the measured edge of selected
   picks (or raise the minimum-edge threshold well above 0) to counter the winner's curse
   from scanning-and-ranking.
4. **Fractional Kelly + probability shrinkage.** Rank and (eventually) size on **¼ Kelly**
   using a probability shrunk toward the market: `p* = w·model + (1−w)·market`, with `w`
   reflecting how much you trust the model (start low, e.g. 0.3).
5. **Use executable prices and de-vig.** Compute EV against the **ask**, and display the
   market **book sum / overround** so the vig is visible. Add a transaction-cost buffer to
   the minimum-edge filter.

### P2 — Makes the portfolio tool correct and the strategy self-aware

6. **Compute true joint bundle probabilities from the simulation.** ✅ **DONE.** The
   simulator already persists `sim_outcome_matrix.npz` + `sim_outcome_matrix_ids.json`
   (the raw `n_sims × 288` binary joint matrix) — no phase9 change was needed. `utils/joint_matrix.py`
   now consumes it and the portfolio simulator AND-s the bundle's columns for the exact joint
   frequency (true cross-team correlation), falling back to the intra-team approximation only
   when the matrix is absent. Nested same-team legs are flagged separately.
7. **Confront the favourite–longshot bias explicitly.** Add a calibration check *within the
   <40¢ bucket specifically*. If the model is poorly calibrated on longshots (likely), either
   recalibrate (e.g. isotonic/Platt) or widen the price band away from the extreme-longshot
   zone.
8. **Distinguish simulation error from model error.** With 10k runs, the Monte Carlo
   *sampling* SE on a probability `p` is `sqrt(p(1−p)/10000)` — small (≈0.2pp at p=0.05). Show
   it, but make clear the dominant uncertainty is **model specification**, not sampling. Don't
   let a tight MC interval masquerade as confidence in the estimate.

### P3 — Hygiene

9. **Automate / timestamp the model refresh.** A manually-refreshed model vs. a live market
   guarantees that some "disagreement" is just staleness. At minimum, display the model's
   age and warn when it is stale.
10. **Report the median outcome, not just the mean.** On a tiny non-replenished bankroll of
    longshots, the *expected* return can be positive while the *median* outcome is near-total
    loss. Show the full outcome distribution of a bundle, not only "if ALL win".

---

## Pre-flight reality checks (run 2026-06-20, before placing real money)

A sanity sweep of the live pipeline. **Two blockers must be fixed before betting real money;**
the engine math itself is correct.

### ✅ BLOCKER 1 — RESOLVED: app now uses live data with a mock fallback
`app.py` loads model probabilities (`load_model_probs`) and live Polymarket prices
(`fetch_futures_prices`, cached 5 min), merges them on `market_id` into the engine schema, and
only falls back to the mock set if either source is unavailable. A 🟢 banner reports the live
match count; a 🚨 banner warns loudly ("**do not place real bets**") whenever mock data is in
use. A *Refresh live prices* button clears the cache.

### ✅ BLOCKER 2 — RESOLVED: `model_loader` finds the local file
`_resolve_path()` now checks `SIM_PROBS_PATH`, then `<v2>/sim_probabilities.json`, then the
data/processed locations — returning the first that exists. Verified: it loads the local file
and returns all 288 markets across the 6 market types. (Team-name and `advance`↔`reach_r32`
aliasing is handled in the joint-matrix loader.)

### 🟢 PASSED — engine math is exact
Hand-recomputed shrinkage, raw EV, adjusted EV, and fractional Kelly for a known contract;
all matched to 4 dp. The joint-matrix loader maps real markets correctly (100k×288 matrix
loads, aliases resolve). The probability/EV/Kelly plumbing is trustworthy — it's the **inputs**
(mock data) and **validation** (no CLV/Brier yet, P0) that are not.

### 🟡 Standing caveats when you do go live
- Model still **unvalidated** (P0) — betting now is betting on faith, not demonstrated edge.
- `<40¢` targets the **favourite–longshot bias** zone (Section 2).
- Deploying **100% of bankroll** across 5 longshots carries a real chance of total loss
  (≈14% on the test bundle, per the *model* — likely optimistic). True Kelly would deploy
  less than the full bankroll; we deploy all of it because min-stake constraints force it.
- `ZAR_USD = 18.5` is hardcoded; conversion + Polymarket fees are not modelled and eat thin edges.

---

## Staking: equal split vs. weighted — decision & logic

**Question:** with a small fixed bankroll over the 5 most-mismatched contracts, is there value in weighting the
stake unequally (to reduce loss), or is an equal split best?

**Simulation** (5 real legs, exact joint matrix, 100k tournaments, full bankroll deployed; profit as % of bankroll):

| Scheme | Mean | Median | P(profit) | P(total loss) | Std dev | p90 |
|---|---|---|---|---|---|---|
| Equal (1/N) | +55.5% | +26.5% | 61.5% | **13.7%** | 124.8% | +230% |
| Kelly-proportional | +49.8% | **+49.3%** | 61.5% | **13.7%** | **102.3%** | +190.5% |
| Safe (prob-weighted) | +47.0% | +21.8% | 61.5% | **13.7%** | 95.8% | +185% |

**What this shows:**

1. **Weighting cannot change `P(total loss)`** — it is identical (13.7%) across all schemes,
   because the chance every leg loses depends only on *which* legs you pick, not how you split
   the stake. To cut ruin probability you must change the **selection** (fewer legs, or include a
   higher-probability one), not the weights. `P(profit)` was also near-identical here.
2. **Weighting reshapes the win-scenario payoffs.** Equal weighting unwittingly tilts money
   *toward the extreme longshots* (they're an equal slice despite tiny win chances), giving the
   **highest mean but the lowest median and fattest variance** — a higher jackpot, a worse
   typical outcome. **Kelly-proportional weighting tilts toward the higher-probability legs**
   (Kelly inherently down-weights extreme longshots), **nearly doubling the median (+49% vs
   +27% of bankroll) and cutting volatility ~18%, for only ~10% less mean.**

**Decision — tilt toward Kelly-proportional, but bounded.** For a small, non-replenishable
bankroll, the **median and downside matter more than the mean**, so the equal split is
quietly the *riskiest* of the three. Recommended default:

```
w_i  ∝  blend(equal, kelly_frac_i)        e.g.  0.5·(1/N) + 0.5·(kelly_i / Σkelly)
```

A 50/50 blend of equal and Kelly weights captures most of Kelly's variance reduction while
**capping concentration** so a single overstated edge can't dominate the book (important given
the winner's-curse haircut already tells us the largest edges are the least trustworthy). Cap
any single leg at ~35% and floor each stake at Polymarket's ~$1 minimum.

**Caveat:** all of this assumes the probabilities are roughly right. If you don't trust the
edge estimates at all, pure equal (1/N) is the maximally robust choice (the classic result that
naive 1/N often beats optimized weights out-of-sample under estimation error). Since the model
is still unvalidated, **start near equal and shift toward the Kelly blend only as CLV/Brier skill
is demonstrated.**

**✅ IMPLEMENTED.** `portfolio.bounded_kelly_weights()` computes
`w = (1−tilt)·(1/N) + tilt·(kelly_i / Σkelly)`, then iteratively caps each leg at 35% and
renormalises. A **"Kelly tilt" sidebar slider** (default **0.5**, the bounded blend) drives the
per-leg stakes in the simulator and the placement table; `simulate_bundle` accepts the per-leg
stake vector so the outcome distribution reflects the actual split. Tilt 0 = equal, tilt 1 =
fully Kelly-proportional (still capped). A warning fires if any stake falls below the ~$1 minimum.

---

## One-line verdict

> The engineering is solid and the recent Kelly addition shows good instincts, but the
> strategy is **assuming its conclusion**: it presents model-vs-market disagreement as
> profit. Prove the model beats the closing line first (P0). Until then, treat every EV
> figure on this dashboard as a hypothesis to be tested, not an edge to be banked.
