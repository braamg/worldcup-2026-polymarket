# Quantitative Theory & Design — A Primer for the World Cup 2026 Predictor

**Audience:** this document assumes the background of a quantitative finance or statistics
undergraduate/graduate student — comfortable with probability, regression, maximum
likelihood, and the basic vocabulary of trading (bid/ask, edge, Kelly sizing) — but with
*no prior exposure to football analytics*. Every football-specific idea is derived from first
principles and tied to the actual code. Every quantitative idea is tied to its origin in the
literature, so you can go read the source if you want to go deeper.

**Purpose:** not just to explain *what* the code does (see `HOW_THE_MODEL_WORKS.md` for that
in plain English) but *why* each modeling and trading decision was made, what alternative was
rejected and why, and where the open, unresolved judgment calls are — so you can decide where
to take this project next with full context.

**How to read this:** the document follows the actual data pipeline, front to back:

```
raw match results  →  rating engine (Elo)  →  three competing models  →  ensemble
   →  calibration  →  evaluation  →  Monte Carlo tournament simulation
   →  market execution layer (Kelly sizing, arbitrage detection, paper-trading harness)
```

Each part states the **theory**, the **formula as implemented**, the **empirical decision**
that was made (with the number that drove it), and **what this connects to in finance**.

---

## Table of contents

- [Quantitative Theory \& Design — A Primer for the World Cup 2026 Predictor](#quantitative-theory--design--a-primer-for-the-world-cup-2026-predictor)
  - [Table of contents](#table-of-contents)
  - [1. The forecasting problem and why it's hard](#1-the-forecasting-problem-and-why-its-hard)
  - [2. The rating engine: Elo](#2-the-rating-engine-elo)
    - [2.1 Theory](#21-theory)
    - [2.2 The "two clocks" and the hyperparameter-tuning discipline](#22-the-two-clocks-and-the-hyperparameter-tuning-discipline)
    - [2.3 What Elo is and isn't](#23-what-elo-is-and-isnt)
  - [3. Three competing outcome models](#3-three-competing-outcome-models)
    - [3.1 The floor: Elo-only logistic regression](#31-the-floor-elo-only-logistic-regression)
    - [3.2 Dixon-Coles: a structural goals model](#32-dixon-coles-a-structural-goals-model)
    - [3.3 XGBoost: a flexible, regularized learner](#33-xgboost-a-flexible-regularized-learner)
    - [3.4 The ensemble](#34-the-ensemble)
  - [4. Calibration: making probabilities honest](#4-calibration-making-probabilities-honest)
    - [4.1 The problem](#41-the-problem)
    - [4.2 Temperature scaling](#42-temperature-scaling)
  - [5. Evaluation: proper scoring rules and statistical significance](#5-evaluation-proper-scoring-rules-and-statistical-significance)
    - [5.1 Why not just accuracy?](#51-why-not-just-accuracy)
    - [5.2 The four metrics, in order of importance here](#52-the-four-metrics-in-order-of-importance-here)
    - [5.3 The benchmarks that matter](#53-the-benchmarks-that-matter)
    - [5.4 Statistical significance: paired bootstrap](#54-statistical-significance-paired-bootstrap)
  - [6. Temporal integrity: the no-leakage discipline](#6-temporal-integrity-the-no-leakage-discipline)
  - [7. Monte Carlo tournament simulation](#7-monte-carlo-tournament-simulation)
    - [7.1 Why simulation, not a closed form](#71-why-simulation-not-a-closed-form)
    - [7.2 Conditioning on reality](#72-conditioning-on-reality)
    - [7.3 Propagating parameter uncertainty: the Elo jitter](#73-propagating-parameter-uncertainty-the-elo-jitter)
    - [7.4 Football-specific mechanics, each independently validated](#74-football-specific-mechanics-each-independently-validated)
  - [8. From a probability to a position: the betting engine](#8-from-a-probability-to-a-position-the-betting-engine)
    - [8.1 Kelly Criterion](#81-kelly-criterion)
    - [8.2 Fractional Kelly and the "model humility" haircut](#82-fractional-kelly-and-the-model-humility-haircut)
    - [8.3 Risk caps as a backstop, not the primary control](#83-risk-caps-as-a-backstop-not-the-primary-control)
    - [8.4 The honest scorecard: Closing Line Value, not P\&L](#84-the-honest-scorecard-closing-line-value-not-pl)
  - [9. Market structure and where an edge can actually live](#9-market-structure-and-where-an-edge-can-actually-live)
    - [9.1 Efficient markets, applied honestly](#91-efficient-markets-applied-honestly)
    - [9.2 Two genuinely different sources of edge in this codebase](#92-two-genuinely-different-sources-of-edge-in-this-codebase)
    - [9.3 Why the betting engine's own model ("model" strategy) is the weakest leg](#93-why-the-betting-engines-own-model-model-strategy-is-the-weakest-leg)
  - [10. The four most recent enhancements, in quant terms](#10-the-four-most-recent-enhancements-in-quant-terms)
    - [10.1 Liquidity floor on multi-leg arbitrage (rejected GARCH-style overkill in favor of a direct check)](#101-liquidity-floor-on-multi-leg-arbitrage-rejected-garch-style-overkill-in-favor-of-a-direct-check)
    - [10.2 Liquidity-adjusted position sizing](#102-liquidity-adjusted-position-sizing)
    - [10.3 Delayed/uncertain event timing](#103-delayeduncertain-event-timing)
    - [10.4 The meta-lesson: a verification methodology, not just four fixes](#104-the-meta-lesson-a-verification-methodology-not-just-four-fixes)
  - [11. Portfolio theory for bet sizing: the derivatives framework](#11-portfolio-theory-for-bet-sizing-the-derivatives-framework)
    - [11.1 The fundamental insight: outcomes are correlated through tournament structure](#111-the-fundamental-insight-outcomes-are-correlated-through-tournament-structure)
    - [11.2 Why this is a derivatives problem, not just a sports betting problem](#112-why-this-is-a-derivatives-problem-not-just-a-sports-betting-problem)
    - [11.3 Thorp's multi-dimensional Kelly: theory and implementation](#113-thorps-multi-dimensional-kelly-theory-and-implementation)
    - [11.4 CVaR: dynamic risk, not static caps](#114-cvar-dynamic-risk-not-static-caps)
    - [11.5 Greeks: hedging tournament leverage](#115-greeks-hedging-tournament-leverage)
    - [11.6 Why this shift changes everything](#116-why-this-shift-changes-everything)
    - [11.7 When to activate each layer](#117-when-to-activate-each-layer)
  - [12. The derivatives framework: implementation and ongoing development](#12-the-derivatives-framework-implementation-and-ongoing-development)
    - [12.1 The three modules](#121-the-three-modules)
    - [12.2 Integration into the betting engine](#122-integration-into-the-betting-engine)
    - [12.3 Live backtest harness](#123-live-backtest-harness)
    - [12.4 Recommended roadmap](#124-recommended-roadmap)
  - [13. Open strategic decisions](#13-open-strategic-decisions)
  - [14. Reference list](#14-reference-list)
  - [Where to go next in the codebase](#where-to-go-next-in-the-codebase)

---

## 1. The forecasting problem and why it's hard

International football is a **data-poverty problem dressed up as a sports problem**. A national
team plays roughly 8–12 matches a year, many of them low-stakes friendlies. Contrast this with
the typical quant-finance estimation setting (thousands of daily price ticks) or even club
football (50+ competitive matches/season per team). With ~30,000 modern-era matches across
~200 teams, the *effective* sample size per team is tiny, and the data is wildly heteroskedastic
in importance — a World Cup final and a pre-season friendly are not the same statistical
observation.

This single fact is the organizing constraint behind nearly every design choice in the codebase:

- **Favor low-parameter, heavily regularized models.** A model with more parameters than the
  data can support won't generalize — it will fit noise. This is the standard bias–variance
  argument (Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 7), but
  the *amount* of regularization needed here is unusually aggressive relative to, say, an
  equity factor model with millions of observations.
- **Weight observations by informativeness**, not just count them. A friendly tells you much
  less about "true team strength" than a World Cup match — both because the incentive to try
  is lower and because squad rotation is heavier. The codebase encodes this as a *match
  importance* weight, used identically to how you'd down-weight a stale or thin print in a
  microstructure model.
- **Treat anything-looking-too-good as a leak, not a result.** In finance this is the
  equivalent of "if your backtest Sharpe is above 3, look for a bug before you celebrate." The
  project's stated policy (`CLAUDE.md`) is explicit: *a backtest that looks too good is almost
  certainly leaking information.*
- **Be honest that the ceiling is low.** The stated success criterion is "approach bookmaker
  accuracy," not "beat Vegas." This maps directly onto the efficient-markets intuition from
  finance: a liquid two-sided market (Pinnacle, Polymarket) aggregates more information than
  almost any closed-form statistical model can, so the realistic goal for the *predictive*
  model is to be competitive with, not superior to, the market consensus. Section 9 explains
  why a separate, *market-structure*-based edge is more plausible than an outright forecasting
  edge.

---

## 2. The rating engine: Elo

### 2.1 Theory

Elo (Arpad Elo, *The Rating of Chessplayers, Past and Present*, 1978) is a sequential
Bayesian-flavored point-estimate updating scheme. Every team has a single scalar rating $R$.
Given two ratings, the **expected score** (≈ win probability with draws split) of the home
team is a logistic function of the rating gap:

$$
E_{\text{home}} = \frac{1}{1 + 10^{-(R_{\text{home}} - R_{\text{away}} + \mathrm{HA})/400}}
$$

where $\mathrm{HA}$ is a home-advantage bonus added to the home side's effective rating (0 at
a neutral venue). After the match, ratings update toward the **realized minus expected**
surprise:

$$
R'_{\text{home}} = R_{\text{home}} + K_{\text{eff}} \cdot (S_{\text{home}} - E_{\text{home}}), \qquad S \in \{1, 0.5, 0\}
$$

This is *literally* a stochastic-gradient update on a logistic (Bradley–Terry-style) pairwise
comparison model — Elo predates, but is structurally identical to, online logistic regression
with a single feature (the rating difference) and a fixed learning rate $K$. If you've seen
the Bradley-Terry model (1952) in a choice/ranking-models course, Elo is its sequential,
online-learning cousin.

**Implementation** ([`src/ratings/elo.py`](src/ratings/elo.py)):

```python
E_home  = 1 / (1 + 10^((R_away - R_home - HA) / 400))
K_eff   = k_base * importance * mov_multiplier
mov_mult = 1 + mov_factor * ln(|goal_diff| + 1)        # = 1 for a draw
delta   = K_eff * (S_home - E_home)
```

Two extensions beyond vanilla Elo, both standard in the football-analytics literature
(Hvattum & Arntzen, 2010, *"Using ELO ratings for match result prediction in association
football"*, International Journal of Forecasting):

- **Margin-of-victory (MoV) scaling.** A 4–0 win should move ratings more than a 1–0 win, but
  the relationship should be *concave* (an 8-0 win isn't twice as informative as a 4-0 win) —
  hence the $\ln(|{\Delta\text{goals}}|+1)$ form, which is the same functional shape FiveThirtyEight
  uses in its public NFL/NBA Elo write-ups.
- **Importance weighting.** $K_{\text{eff}}$ scales by match importance (World Cup final ≫
  friendly), exactly analogous to down-weighting a thin, low-conviction print in a
  microstructure signal.

### 2.2 The "two clocks" and the hyperparameter-tuning discipline

This is the most finance-relevant methodological point in the whole Elo layer. The frozen
hyperparameters are:

| Parameter        | Value       | Tuned on       |
| ---------------- | ----------- | -------------- |
| `k_base`         | 15          | 1994–2009 only |
| `home_advantage` | 125 Elo pts | 1994–2009 only |
| `mov_factor`     | 0.25        | 1994–2009 only |

The grid search minimized Brier score on the 1994–2009 window and was then **frozen and never
touched again**, including never re-tuned on the 2010+ data later used to *judge* the model.
This is the train/validation/test discipline taken to its logical extreme: even a "hyperparameter
of a hyperparameter" (Elo's own K-factor) is walled off from the test period. In a quant
context this is the equivalent of fixing your risk-model decay parameter on an early in-sample
period and refusing to retune it once you start live-trading, specifically to avoid the
**look-ahead bias** that would otherwise flatter every downstream backtest. The full history
back to 1872 is used *only* to "burn in" the rating (let it converge from the arbitrary 1500
starting point) — never used for fitting anything that gets judged.

A genuinely interesting empirical finding from the grid search: the Brier-score spread across
27 grid combinations was narrow (0.1403–0.1570) — i.e., **the model is not very sensitive to
the exact K/HA/MoV values**. This is itself informative: it suggests Elo's predictive content
comes mostly from *which* signal it uses (cumulative goal-weighted history), not from precise
tuning of *how fast* it adapts — a useful sanity check before spending more effort re-tuning it.

### 2.3 What Elo is and isn't

Elo gives you a single number, refreshed match-by-match, that is provably free of look-ahead
bias **by construction** (it only ever uses past results) — this is why it is the backbone of
the entire no-leakage guarantee (Section 6): every other feature is checked against "would
this have been knowable strictly before the match," and Elo's chronological, state-machine
update is the easiest of all of them to audit.

---

## 3. Three competing outcome models

The project trains three models with genuinely different functional forms, then keeps two of
them in production. This section explains each, in increasing order of statistical
sophistication, and why having three (not one "best" model) is itself the design.

### 3.1 The floor: Elo-only logistic regression

[`src/models/elo_baseline.py`](src/models/elo_baseline.py): a multinomial logistic regression
on a single feature, `elo_diff` (the rating gap including home advantage).

$$
P(\text{outcome}=k) = \frac{\exp(\beta_k \cdot \text{elo\_diff} + \alpha_k)}{\sum_{k'} \exp(\beta_{k'} \cdot \text{elo\_diff} + \alpha_{k'})}, \quad k \in \{H, D, A\}
$$

Weak ridge regularization ($C=10$ in scikit-learn's $\ell_2$ parameterization — large $C$ =
weak penalty); with one feature and ~29,000 rows, overfitting risk here is essentially zero.
**This model's entire purpose is to be the floor**: anything more complex that can't beat it
adds nothing, the same discipline as comparing an alpha signal against a naive momentum or
carry benchmark before claiming it's additive.

### 3.2 Dixon-Coles: a structural goals model

**Canonical reference:** Dixon, M.J. and Coles, S.G. (1997), *"Modelling Association Football
Scores and Inefficiencies in the Football Betting Market,"* Journal of the Royal Statistical
Society: Series C, 46(2), 265–280 — itself an extension of Maher (1982), *"Modelling
association football scores,"* Statistica Neerlandica. This is one of the most-cited papers in
sports analytics and is **directly** a quantitative-finance-adjacent paper: its second half is
literally about finding inefficiencies in bookmaker odds.

**The core idea:** instead of modeling W/D/L directly, model the **number of goals each team
scores** as Poisson-distributed counts, and back out W/D/L from the resulting bivariate
distribution. Each team gets two log-scale parameters:

$$
\lambda_{\text{home}} = \exp(\text{attack}_{\text{home}} + \text{defense}_{\text{away}} + \text{home\_adv})
\qquad
\lambda_{\text{away}} = \exp(\text{attack}_{\text{away}} + \text{defense}_{\text{home}})
$$

Goals are then $\text{Goals}_{\text{home}} \sim \text{Poisson}(\lambda_{\text{home}})$,
independently for the away side. Why this is more than "a Poisson regression": pure
independent-Poisson under-predicts the frequency of low-scoring draws (0–0, 1–1) observed in
real football data — a well-documented empirical anomaly. Dixon & Coles fix this with a
correction term $\tau(i,j,\rho)$ applied only to the four low-scoring cells:

$$
\tau(0,0)=1-\lambda_h\lambda_a\rho,\quad \tau(0,1)=1+\lambda_h\rho,\quad \tau(1,0)=1+\lambda_a\rho,\quad \tau(1,1)=1-\rho
$$

$\rho$ is a single global parameter capturing the home/away scoring *correlation* induced by
game-state effects (a winning team sits back, the loser presses — outcomes become slightly
less independent than two separate Poissons would predict). This is conceptually similar to
adding a copula term to join two marginal distributions that aren't quite independent — exactly
the kind of correction you'd reach for in a credit/derivatives context when two driving factors
aren't independent at the tails.

**Why keep this model when XGBoost exists?** Two reasons, both load-bearing:

1. **It is the only model that outputs a full scoreline distribution**, $P(\text{home}=i,
   \text{away}=j)$ for every $(i,j)$ — needed downstream for goal-difference tiebreakers in
   group standings during Monte Carlo simulation (Section 7). W/D/L classifiers (XGBoost, the
   Elo baseline) cannot produce this; only a structural goals model can.
2. **Error diversity for the ensemble** (Section 3.4) — DC reasons about goals, XGBoost reasons
   about a feature vector; their mistakes are not perfectly correlated.

**Training: weighted, time-decayed maximum likelihood.** The negative log-likelihood is
minimized via L-BFGS-B with an *analytical* gradient (hand-derived, not autodiff — a deliberate
performance choice given the team-level parameter count). Two weighting schemes are stacked:

- **Match-importance weight** (same `training_weight` column the other models use).
- **Exponential time-decay**, $w_t = e^{-0.002 \cdot \text{days\_ago}}$, half-life ≈ 347 days —
  i.e. a match from a year ago carries about half the weight of yesterday's match. This is the
  same exponentially-weighted-moving-average (EWMA) idea used to estimate a time-varying
  volatility or beta in finance (cf. RiskMetrics' EWMA covariance estimator), applied here to
  "current team strength" instead of "current volatility."

**Regularization: empirical-Bayes shrinkage, with a real James–Stein flavor.** Two ridge terms
are added to the negative log-likelihood:

```
reg = l2_global * (||attack||² + ||defense||²)              # weak global anchor
    + l2_conf   * (||attack - confederation_mean||² + ...)   # shrink toward peer-group mean
```

The second term is a direct, hands-on implementation of the **James–Stein shrinkage** intuition
(James & Stein, 1961): a team's individually-estimated attack/defense strength, especially for
a data-poor team with few recorded matches, is a noisy estimate; shrinking it toward the mean
of its peer group (its confederation) trades a small amount of bias for a large reduction in
variance, which on average lowers total estimation error — the textbook James–Stein result that
a shrunk estimator can dominate the individual MLEs in mean-squared error, even though no single
shrunk estimate is "more correct" for its own team.

**The empirical twist — and an important lesson in compositional thinking.** A sweep
(`scripts/_sweep_dc_shrinkage.py`, logged 2026-06-14) found that confederation-mean shrinkage
*helps Dixon-Coles standalone* (RPS 0.18233 → 0.18128) but *hurts the ensemble*
(`dc_xgb_avg` RPS 0.16968 → 0.16990+) at every shrinkage strength tested. The mechanism: pulling
DC's team estimates toward the confederation mean makes DC behave more like a generic
confederation-strength model — which is closer to what XGBoost already captures via Elo and
market value — so the shrinkage **destroys the error diversity that the ensemble exists to
exploit**. The codebase's chosen default ($\ell_{2,\text{global}}=0.01$, $\ell_{2,\text{conf}}=0$)
deliberately optimizes the *ensemble*, not the standalone DC component. This is a genuinely
subtle and important point for anyone extending this system: **a component's local accuracy and
its marginal contribution to a portfolio of models are different objectives**, exactly the way
a single alpha signal's standalone Sharpe and its contribution to a multi-strategy book's
Sharpe can disagree once you account for correlation with the rest of the book.

**Knockout deflation.** Knockout football is measurably lower-scoring than group-stage
football (teams play more conservatively when a single loss eliminates them). At prediction
time only (not training time — training fits on the *actual* historical scoring rates), both
$\lambda$'s are multiplied by a configurable `knockout_goal_rate_deflation` (0.85 default).

### 3.3 XGBoost: a flexible, regularized learner

[`src/models/xgboost_model.py`](src/models/xgboost_model.py) — gradient-boosted decision trees
(Chen & Guestrin, 2016, *"XGBoost: A Scalable Tree Boosting System,"* KDD '16), `multi:softprob`
objective, fit on a 10-feature row per match: `elo_diff`, `e_home` (the Elo-implied win
probability — included *alongside* the raw difference because a tree model can carve nonlinear
shape into a probability-scale feature more easily than re-deriving a logistic transform from
the raw log-odds gap), `log_ratio_mv` (+ a `has_market_value` flag), `neutral_venue`,
`cross_conf`, and four rest-days fields (raw + missingness flags for both teams).

**Regularization is unusually aggressive relative to typical XGBoost defaults** — a deliberate
choice given the data-poverty constraint from Section 1:

```python
max_depth=3            # shallow trees: few interaction terms, low variance
min_child_weight=50     # a leaf needs the sample-weight of 50+ matches to split further
reg_lambda=5.0          # L2 on leaf weights
reg_alpha=0.5           # L1 on leaf weights (sparsifies splits)
subsample=0.8           # row bagging
colsample_bytree=0.8    # feature bagging
early_stopping_rounds=50
```

`min_child_weight=50` is doing the heaviest lifting: with ~30k rows spread across many
(team-pair, era) cells, most fine-grained interaction cells genuinely have fewer than 50
matches, and the gradient-boosting algorithm is explicitly forbidden from splitting on them —
a blunt but effective way to stop the model from memorizing idiosyncratic small-sample noise
(the "if Team X plays Team Y on a Tuesday in March" overfitting failure mode tree models are
prone to on thin data).

**Validation discipline: walk-forward early stopping**, not a random hold-out. The last 20% of
rows *by date* become the early-stopping validation set; training rows are sample-weighted by
`training_weight`, but the validation set is **deliberately left unweighted** so early stopping
is judged on a representative mix of match types, not a friendly-discounted one — otherwise
early stopping could converge to a model that's well-tuned for important matches but
under-fit on the (unweighted, real-world) mix the model will actually be scored against.

Reported feature importances (from `HOW_THE_MODEL_WORKS.md`, Phase 6): `elo_diff` ≈ 43%,
`e_home` ≈ 36%, `log_ratio_mv` ≈ 5%, with venue/rest/confederation flags splitting the
remainder. Two Elo-derived features dominating at ~80% combined is itself an important
empirical finding: it says the tree model isn't discovering much that Elo doesn't already
capture, and motivates the "is XGBoost worth its complexity at all" question in Section 11.

### 3.4 The ensemble

The shipped model, `dc_xgb_avg`, is a flat **50/50 arithmetic average** of Dixon-Coles and
XGBoost probabilities (the Elo baseline contributes nothing to the shipped ensemble — it exists
purely as the comparison floor).

**Why a flat average and not a learned blend weight (e.g., stacking, a meta-learner, or
inverse-variance weighting)?** Two intersecting reasons. First, with so few truly independent
validation matches, fitting a *third* model (the blend weight) on top of two already-thin models
reintroduces the exact overfitting risk the whole project is trying to avoid — optimizing a
blend weight on a small out-of-sample slice tends to just refit noise in that slice. Second,
the documented honest caveat is that **the ensemble's RPS edge over XGBoost alone is roughly one
standard error** — i.e., not statistically robust by the project's own paired-bootstrap test
(Section 5.2). The real, defensible justification for keeping the ensemble isn't "it's
measurably more accurate," it's "DC supplies scorelines XGBoost cannot, and the two models make
different *kinds* of errors" — a structural diversification argument, similar to combining two
weakly-correlated alpha signals in a portfolio even when neither dominates the other
out-of-sample.

This is the financial-engineering version of the bias–variance tradeoff at the *ensemble*
level: a fixed-weight average is a high-bias, low-variance combiner; a fitted blend weight is
lower-bias (if the optimal weight isn't 50/50) but higher-variance (the weight itself is now
estimated from noisy data). Given the data-poverty constraint, the project chose the
low-variance option.

---

## 4. Calibration: making probabilities honest

### 4.1 The problem

A model can *rank* matches correctly (higher-prob side wins more often) while still being
**miscalibrated** — systematically over- or under-confident. This matters enormously for
downstream betting, because Kelly sizing (Section 8) is a *direct, nonlinear* function of the
stated probability: an overconfident 80%-when-it's-really-65% signal doesn't just mis-rank
trades, it mis-*sizes* every trade that clears the edge threshold, compounding the error.

### 4.2 Temperature scaling

[`src/evaluation/calibration.py`](src/evaluation/calibration.py) implements **temperature
scaling** — originally Platt scaling (Platt, 1999) for binary classifiers, generalized to the
multi-class softmax form popularized for modern deep nets by Guo, Pleiss, Sun & Weinberger
(2017), *"On Calibration of Modern Neural Networks,"* ICML. A single scalar $T$ rescales the
*log-probabilities* before re-normalizing:

$$
p_{\text{calibrated}} = \mathrm{softmax}\!\left(\frac{\log p}{T}\right)
$$

$T>1$ softens (pulls toward uniform — corrects overconfidence); $T<1$ sharpens. $T$ is fit by
minimizing negative log-likelihood on a **dedicated calibration period (2017–2018)**, distinct
from both the training period and the period later used to score the model (Section 6) — the
same train/calibrate/test three-way split discipline used in any well-run quant pipeline (e.g.
fitting a risk model on one window, calibrating its confidence bands on a second, and only
then evaluating live P&L on a third never-touched window).

**Why temperature scaling over isotonic regression** (Zadrozny & Elkan, 2002, *"Transforming
Classifier Scores into Accurate Multiclass Probability Estimates,"* KDD) as the default? Single
parameter vs. a full nonparametric, per-class monotonic curve — temperature scaling is the much
lower-variance choice on a few-thousand-row calibration set, exactly mirroring the parametric-
vs-nonparametric tradeoff for fitting an implied-volatility curve with few strikes available:
you reach for the flexible nonparametric tool only once you have enough points to support it.
The code is explicit about this: isotonic regression is allowed to win **only** if the
calibration set exceeds 2,000 rows *and* it beats temperature scaling's log-loss by more than
0.5% — a deliberately conservative bar, since isotonic will always fit the *calibration* set at
least as well by construction (it's strictly more flexible) and the question is whether that
extra flexibility generalizes.

**The empirical result**: the final ensemble needed only $T = 1.0365$ — a very gentle softening
— evidence the model's raw probabilities were *already* close to calibrated before this step,
which is itself a useful (mild) confirmation that the upstream pipeline isn't systematically
biased.

---

## 5. Evaluation: proper scoring rules and statistical significance

### 5.1 Why not just accuracy?

"Percent of matches where the higher-probability side won" is a poor metric for an inherently
random sport — draws exist, upsets are common, and a perfectly-calibrated 55%/45% call that
loses is *not a mistake*. The project instead uses **proper scoring rules** (Gneiting & Raftery,
2007, *"Strictly Proper Scoring Rules, Prediction, and Estimation,"* JASA) — scoring functions
for which the expected score is uniquely minimized by reporting your *true* belief, so a model
cannot game the metric by hedging or sandbagging its stated probabilities.

### 5.2 The four metrics, in order of importance here

**Ranked Probability Score (RPS)** — the *primary* metric. Introduced by Epstein (1969) for
ranked weather-forecast categories, and specifically advocated for football by Constantinou &
Fenton (2012), *"Solving the problem of inadequate scoring rules for assessing probabilistic
football forecast models,"* Journal of Quantitative Analysis in Sports — required reading if
you want the full argument for why RPS, not log-loss or accuracy, is the right primary metric
for an *ordered* outcome space like W/D/L (a draw is "between" a home win and an away win in a
way log-loss and Brier don't reward):

$$
\mathrm{RPS} = \tfrac{1}{2}\sum_{k=1}^{2}\left(\sum_{j\le k} p_j - \sum_{j \le k} a_j\right)^2
$$

where $p$ and $a$ are the predicted and actual *cumulative* distributions over the ordered
categories (H, D, A). A model that says "60% home, 25% draw, 15% away" when an away win
happens is penalized *less* under RPS than one that says "60% home, 15% draw, 25% away" with
the same draw probability but the wrong tail mass distribution — because RPS respects the
fact that a draw is closer to either win than the two wins are to each other.

**Log-loss** (negative log-likelihood) and **Brier score** (Brier, 1950, *"Verification of
forecasts expressed in terms of probability,"* Monthly Weather Review — the original proper
scoring rule, still the standard in meteorology and increasingly ML) round out the standard
trio; the project reports all three rather than picking one, since each penalizes a different
failure mode (log-loss punishes confident wrong calls far more harshly than Brier does).

**Expected Calibration Error (ECE)** — bins predictions, compares the mean predicted
probability in each bin to the empirical hit rate, and averages the absolute gap. This is the
project's automated check on exactly the "is the model honest" question from Section 4.

### 5.3 The benchmarks that matter

The model is judged against, in order of difficulty:

1. **Naive "higher Elo wins"** — a model that can't beat this contributes literally nothing.
2. **The Elo-baseline logistic regression** — the floor any added complexity must clear.
3. **Bookmaker implied probabilities** (where available — major tournaments only) — the gold
   standard, since a liquid two-sided market aggregates information no closed-form model can
   match (injury news, lineup leaks, insider sentiment).

**Reported result (1994–2026 holdout, per `HOW_THE_MODEL_WORKS.md`)**: ensemble RPS ≈ 0.170 vs.
0.172 for the Elo baseline and 0.188 for naive Elo — i.e., the sophisticated machinery
(Dixon-Coles + XGBoost + ensembling + calibration) buys roughly a **1% relative RPS
improvement** over a one-feature logistic regression. This number is the single most important
fact for deciding where to invest further effort (see Section 11) — it's a strong empirical
signal that the *marginal* model complexity built on top of Elo is providing only a thin edge.

### 5.4 Statistical significance: paired bootstrap

[`src/evaluation/metrics.py`](src/evaluation/metrics.py)'s `paired_bootstrap_rps` resamples
**match indices** (not models) 2,000 times and computes the bootstrap distribution of the
*per-match* RPS difference between two models scored on the identical match set. This is the
sports-analytics analogue of a paired t-test / bootstrap on strategy A−B daily P&L differences in
a quant backtest — far more powerful than comparing two unpaired mean RPS values, because it
controls for shared match-level difficulty (some matches are intrinsically harder to call for
*any* model, and a paired test cancels that shared variance out). The documented honest finding
from this test: the ensemble's edge over plain XGBoost is **about one standard error** — not
robustly significant. The project's discipline here — reporting this rather than hiding it — is
the same discipline a quant researcher should apply before sizing a new signal into a live book:
don't trust a backtest improvement until a paired significance test confirms it isn't sampling
noise.

---

## 6. Temporal integrity: the no-leakage discipline

This is flagged in `CLAUDE.md` as the single most important correctness property, and it
deserves a dedicated section because it's the part of this codebase most directly transferable
to *any* finance forecasting problem.

**The rule:** every feature used to predict a match on date $D$ must be computable using only
data timestamped strictly before $D$. No exceptions, no "it's basically the same value."

**Why this is harder than it sounds — three concrete traps the codebase specifically guards
against:**

1. **Snapshot-vs-latest contamination.** Squad market values inflate 5–10× across eras; using
   "today's" market value to retroactively predict a 2014 match would smuggle 12 years of
   inflation (and roster turnover) into a feature that, at the time, didn't exist. The fix:
   every market-value feature is stored as a **dated snapshot** (`valid_from`), looked up
   as-of just before the match — this is identical in spirit to point-in-time fundamental
   databases used in equity backtesting (e.g. avoiding survivorship/restatement bias by using
   the *as-reported-then* balance sheet, not the *as-revised-since* one).
2. **Walk-forward-only validation.** Never a random train/test split on historical matches —
   always train on the past, validate/calibrate on a middle window, test on the future. Random
   splitting on serially-correlated, regime-shifting data (team strength drifts over years)
   would let the model "see the future" via correlated neighbors leaking across the split, the
   exact same failure mode addressed by purged/embargoed cross-validation in quantitative
   finance (López de Prado, *Advances in Financial Machine Learning*, 2018, ch. 7) — though this
   codebase uses the simpler, stricter version (a hard chronological cut) rather than the
   purge/embargo machinery, since with this little data a single clean split is more
   defensible than carving the series into many folds.
3. **An automated leakage audit, not just a policy.** `tests/test_leakage_audit.py`
   reconstructs the feature row for a sample of historical matches and *asserts* that the
   Elo-feature provenance and the market-value snapshot date both predate the match — i.e. the
   "don't leak" rule is a CI-enforced invariant, not a code-review reminder. Anyone who has
   shipped a quant model that looked great in backtest and fell apart live has probably been
   bitten by exactly the bug class this test exists to catch.

---

## 7. Monte Carlo tournament simulation

[`src/simulation/tournament.py`](src/simulation/tournament.py) is where a single match
probability model becomes a *tournament-outcome* distribution — conceptually the same leap as
going from a one-period option-pricing model to a path-dependent Monte Carlo pricer for an
exotic payoff whose value depends on the whole trajectory, not one terminal probability.

### 7.1 Why simulation, not a closed form

The 2026 format (48 teams, 12 groups of 4, top-2 plus 8 best third-placed teams advancing via
an official cross-group allocation table, then a single-elimination bracket) has no tractable
closed-form expression for "P(team X wins the whole tournament)" — the dependency structure
(who you face in the round of 32 depends on which *other* groups' third-placed teams qualify,
which depends on every group's results) makes this exactly the kind of path-dependent,
high-dimensional problem Monte Carlo is the standard tool for.

### 7.2 Conditioning on reality

Since the tournament is mid-event, **confirmed results are read as facts and never
re-simulated** — every simulated path starts from the actual partial tournament state. This is
identical to a path-dependent derivative pricer conditioning on the realized price path so far
and only Monte-Carlo-simulating the *remaining* time to expiry; you don't re-randomize history.

### 7.3 Propagating parameter uncertainty: the Elo jitter

A team's Elo rating is a point estimate, not the truth — it has estimation uncertainty,
especially for data-poor teams. Rather than running every simulated tournament off the single
point-estimate Elo (which would understate the true spread of outcomes — a classic
plug-in-estimate bias in any Monte Carlo that ignores parameter uncertainty), the simulator
draws, **once per full simulated tournament**, a per-team perturbation:

$$
R_{\text{team}}^{(\text{sim})} = R_{\text{team}}^{(\text{point est.})} + \varepsilon_{\text{team}}, \qquad \varepsilon_{\text{team}} \sim \mathcal{N}(0, \sigma_{\text{team}}^2)
$$

held **constant across every match that team plays within that one simulation** — crucial,
because if you redrew $\varepsilon$ independently match-by-match you'd be modeling a team whose
*true* strength changes every 90 minutes, rather than modeling genuine uncertainty about a
fixed-but-unknown strength. This single design choice is the difference between simulating
"parameter uncertainty" (correct) and "i.i.d. match noise" (wrong, and already handled by the
outcome model's own probabilities) — the same distinction as a Bayesian posterior-predictive
simulation (draw one parameter vector per simulated path, not per individual outcome).

$\sigma_{\text{team}}$ itself is **data-driven**, not a flat constant: scaled by
$\sqrt{n_{\text{ref}} / n_{\text{team}}}$ (clipped to $[0.7\times, 2.0\times]$ the base 50-point
sigma), where $n_{\text{team}}$ is the team's match count in the 4 years before the tournament
and $n_{\text{ref}}$ is the cross-team median. This is the rating-uncertainty analogue of the
standard-error-of-the-mean scaling with $1/\sqrt{n}$ — a data-poor team (fewer recent matches)
gets a *wider* jitter, exactly as a less-frequently-traded asset gets a wider confidence interval
on its estimated beta. This single mechanism **replaced** an earlier, cruder flat 1.3× multiplier
for "non-UEFA/CONMEBOL" teams — a good illustration of upgrading from a categorical heuristic to
a continuous, principled uncertainty estimate once the team had the data to support it.

For knockout matches (computed individually, not batch-vectorized like the group stage), the
jitter is applied via an **analytical logit-shift** rather than re-running the full ensemble per
draw:

$$
\text{logit}(P_{\text{shifted}}) = \text{logit}(P_{\text{base}}) + \Delta_{\text{Elo}} \cdot \frac{\ln 10}{400}
$$

— a first-order linearization of how an Elo shift moves the logistic win probability, used
purely as a performance optimization (avoids rerunning the XGBoost call per knockout match per
simulation) at the cost of a small approximation error, justified by $|\Delta_{\text{Elo}}|$
being small relative to 400 in the typical jitter draw.

### 7.4 Football-specific mechanics, each independently validated

- **Lower-scoring knockout football**: both $\lambda$'s deflated by the configurable
  `knockout_goal_rate_deflation` (0.85).
- **Penalty shootouts resolved exactly 50/50**, *never* by team strength — an explicit policy
  decision (`CLAUDE.md`) based on the empirical near-coin-flip nature of shootouts; resisting
  the temptation to let the stronger team "deserve" a shootout edge the data doesn't actually
  support.
- **Host home advantage, sanity-checked against history.** A *separate* parameter from the
  general home-advantage Elo bonus, applied *only* when a host nation plays in its own country
  (not in another host's stadium), and only after a dedicated backtest against 47 historical
  World Cup host matches (1994–2022) confirmed hosts performed in line with — not above — the
  general 125-point home-advantage estimate. An earlier, more naive design ("hosts get a home
  bonus in every knockout match") compounded a bonus across up to 7 rounds and overrated Mexico
  to 3.45× the bookmaker's title price — a vivid example of a small per-match assumption
  compounding into a large, detectable distortion at the tournament level, the same way a small
  daily mis-estimated drift compounds badly over a long-horizon Monte Carlo price path.
- **Third-place bracket allocation via bipartite matching** (`scipy.optimize.linear_sum_assignment`)
  rather than a hand-coded lookup table — a genuinely elegant reduction: FIFA's "which group's
  third-placed team can fill which R32 slot" rule is a constraint-satisfaction problem with a
  guaranteed unique solution for any valid combination of 8 qualifying groups (proven by
  exhaustive enumeration over all $\binom{12}{8}=495$ combinations), so modeling it as a
  min-cost bipartite matching (cost 0 for compatible group/slot pairs, $\infty$ otherwise) is
  both correct and far less error-prone than 495 hand-coded cases.

---

## 8. From a probability to a position: the betting engine

This is the layer of the codebase that will feel most native to a quant-finance reader — it's a
self-contained position-sizing and risk-management module.

### 8.1 Kelly Criterion

**Origin**: Kelly, J.L. (1956), *"A New Interpretation of Information Rate,"* Bell System
Technical Journal — originally a result about maximizing the long-run growth rate (equivalently,
maximizing expected log-wealth) of repeated bets with a known edge, later popularized in
gambling and investing by Ed Thorp. For a binary contract bought at cost $c$ (paying \$1 if
correct, \$0 otherwise — exactly the structure of a Polymarket share), with true win probability
$p$:

$$
f^{*} = \frac{p \cdot b - (1-p)}{b}, \qquad b = \frac{1-c}{c} \text{ (net odds)}
$$

implemented directly in [`src/betting/betting_engine.py`](src/betting/betting_engine.py)'s
`_kelly_fraction`. Kelly sizing is **provably growth-optimal under log utility** when $p$ is
known exactly — the catch, central to everything else in this section, is that $p$ here is a
*model output*, not a known truth, and the entire downstream risk apparatus exists to compensate
for that gap.

### 8.2 Fractional Kelly and the "model humility" haircut

The engine never bets full Kelly — it uses **quarter-Kelly** (`kelly_fraction=0.25`) by default,
the standard practitioner adjustment (see Thorp's writings on fractional Kelly in blackjack and
sports betting) for the simple reason that full Kelly is famously punishing to *parameter
uncertainty*: if your estimated edge is even slightly biased, full Kelly's growth-optimality
guarantee evaporates and the same formula that maximizes growth under a *correct* $p$ can
produce ruinous drawdowns under a *wrong* one. Fractional Kelly trades some of the theoretical
growth rate for a large reduction in variance and parameter-error sensitivity — a direct
analogue of shrinking a mean-variance-optimal portfolio weight toward a more conservative
allocation once you account for estimation error in the inputs (the same logic behind
Black–Litterman shrinking pure mean-variance weights toward the market portfolio).

On top of fractional sizing, every raw edge is discounted by three separate terms before a bet
clears the gate:

1. **The bid/ask spread you actually pay** — you transact at the ask, not the mid; modeled
   explicitly (`yes_cost = market_mid + spread/2`).
2. **A platform fee** (configurable; currently 0 for Polymarket).
3. **A flat `edge_haircut`** — an explicit "my probability estimate is itself an estimate, not a
   fact" discount, applied uniformly to every raw edge before the `min_edge` gate. This is a
   blunt, single-number proxy for **model risk** — conceptually the same adjustment a derivatives
   desk applies as a model-reserve haircut to a theoretical price before it's allowed to drive a
   real position.

### 8.3 Risk caps as a backstop, not the primary control

Hard caps — `per_market_cap` (5% of bankroll), `per_group_cap` (15%, where "group" ties together
correlated markets like every "Brazil" future), `total_exposure_cap` (40%) — exist as a
position-limit backstop *independent of* what the edge math says, exactly the role hard VaR/
notional limits play on a trading desk regardless of how attractive a model says a trade looks.
The `group` mechanism specifically defends against a subtle correlation-blindness failure mode:
"Brazil wins the Cup," "Brazil reaches the final," and "Brazil advances from the group" are
three *highly correlated* bets (mostly the same underlying risk — Brazil's true strength), and
without a shared cap, a model that's simply wrong about Brazil could stack three nominally-
independent-looking bets that are in fact one concentrated position.

### 8.4 The honest scorecard: Closing Line Value, not P&L

This is, in this author's view, the most quant-mature idea in the whole betting layer, and it's
worth dwelling on. Over a single tournament, **variance dominates** any individual strategy's
realized P&L — you can have genuine edge and still lose money, or have zero edge and get lucky.
The engine's `analyze_performance()` therefore treats **Closing Line Value (CLV)** — did you get
a better price than the market's price right before the event resolved? — as the primary metric,
with realized P&L reported only as a secondary, noisier signal.

This mirrors the standard practitioner critique of evaluating trading strategies on raw P&L over
short windows (Levitt, 2004, *"Why are gambling markets organized so differently from financial
markets?",* The Economic Journal, makes a similar point about why bookmakers don't even try to
balance their books symmetrically — they're pricing bettor demand, not pure probability, which
is itself a reason CLV against a *liquid, sharp* book is more trustworthy than CLV against a
retail-flow-driven one). If you consistently transact at better prices than the market's own
later consensus, that is evidence of durable informational edge that *cannot* be explained by
variance — exactly the role "did you beat the VWAP / arrival price" benchmarks play in
transaction-cost analysis for an equity execution desk. A single tournament's raw P&L, by
contrast, is mostly noise with this few independent trials.

---

## 9. Market structure and where an edge can actually live

### 9.1 Efficient markets, applied honestly

Section 5.3's empirical finding — the model barely separates from a one-feature Elo regression,
and the explicit project goal is "approach the bookmaker," not beat it — is the same conclusion
a disciplined quant would draw before trying to trade against a liquid index options market with
a hand-built vol model: **a closed-form statistical model is unlikely to out-forecast a liquid,
continuously-traded, many-participant market.** This doesn't mean there's no edge available; it
means the edge, if it exists, more plausibly lives in **market structure** rather than in
**superior forecasting** — exactly the distinction between statistical-arbitrage strategies that
bet on mean-reversion of a forecast error versus pure market-microstructure strategies that
exploit how a market is organized, independent of any view on fundamental value.

### 9.2 Two genuinely different sources of edge in this codebase

**(a) Cross-market divergence** (`src/betting/sharp_odds.py`, the `cross` paper-trading
strategy): compare Polymarket's price to a **de-vigged, sharp-book consensus** (Pinnacle
preferred, else a cross-book median). De-vigging removes the bookmaker's overround:

$$
p_i = \frac{1/\text{odds}_i}{\sum_j 1/\text{odds}_j}
$$

This doesn't require the project's *own* model to beat anyone — it's a bet that two *different*
markets (a crypto-native prediction market vs. a traditional sharp sportsbook) are
*momentarily* disagreeing, which is a much weaker, more defensible claim than "our Poisson-plus-
trees model knows more than Pinnacle." This is structurally the same idea as cross-venue
statistical arbitrage between two listings of economically-equivalent risk (e.g. an ADR vs. its
home-market ordinary share) — you're betting on convergence between two prices of the *same*
underlying claim, not on a fundamental forecast.

**(b) Structural (near-riskless) arbitrage** (`src/betting/coherence.py`): exploits the fact
that Polymarket prices many *correlated* World Cup outcomes as **separate, independently-traded
order books** with nothing automatically enforcing internal consistency between them. Two
specific violations of basic probability axioms are scanned for:

- **Monotonicity across nested events.** Winning the tournament implies reaching the final
  implies reaching the semifinal, and so on — a chain of nested events under Boole's
  inequality / the monotonicity of a probability measure: $A \subseteq B \Rightarrow P(A) \le
  P(B)$. If the market ever prices the *more specific* event's bid above the *more general*
  event's ask, that's a textbook violation you can short-the-specific/long-the-general against,
  with bounded risk: $\text{cost} = \text{ask}_{\text{general}} + (1-\text{bid}_{\text{specific}})$,
  and the payoff at resolution is provably $\ge 1$ regardless of outcome (full proof sketch in
  the module docstring).
- **Sum coherence within a multi-outcome market.** In a 3-way match market (home/draw/away),
  exactly one outcome occurs, so $\sum P(\text{outcome}) $ should equal 1 (Kolmogorov's
  normalization axiom). If the sum of asks (net of fees) is below 1, buying all three legs is a
  guaranteed-\$1 payout for less than \$1.

Both are textbook **law-of-one-price** violations — the same economic logic underlying classic
triangular-arbitrage in FX or put-call parity violations in options, just applied to a discrete
nested-event structure instead of a continuous price relationship. The crucial honest caveat,
explicitly documented in the module: *"the steady-state board is kept coherent by bots... around
discrete events (a goal, a result, a qualification clinch) the many correlated markets reprice
at DIFFERENT speeds, opening brief windows."* I.e. this is explicitly a **latency/event-driven**
arbitrage, not a standing one — its value depends entirely on being there fast enough during the
brief repricing window right after news hits, which is why `arb_watch.py`/`arb_daemon.py` exist
as polling daemons rather than a one-shot scan.

### 9.3 Why the betting engine's own model ("model" strategy) is the weakest leg

Tying Section 9.1's conclusion back to the paper-trading scorecard: the `model` strategy (the
project's own ensemble vs. raw Polymarket mids) is the one strategy that requires the
forecasting model to genuinely out-predict a liquid market — exactly the leg the data says is
least likely to have a durable edge. This was borne out concretely: the early paper-trading
sample showed negative average CLV on `model` bets (discovered to be partly an artifact of a
since-fixed timing bug, but the *prior* — that this leg has the thinnest edge of the three — was
correct even before the bug was found, simply from the RPS-vs-Elo-baseline comparison in
Section 5.3).

### 9.4 Arbitrage as market-structure edge, not forecast arbitrage

**The crucial distinction.** There are three qualitatively different sources of edge in prediction
markets, and they are not equally fragile:

1. **Forecast arbitrage (weakest).** Your probability model says team A wins at 60%, market says
   55%; you buy A. This requires your model to beat the market's consensus. We've established
   that's a thin edge in a liquid market — you're competing against the aggregated beliefs of
   thousands of participants. The CLV/Sharpe this can generate is bounded by how much better
   your forecast is than the market's, which for a statistical model vs. a liquid price is
   typically small.

2. **Market-structure arbitrage (medium).** Polymarket quotes Brazil championship at 15¢, sharp
   bookmakers aggregate at 12¢. You short Polymarket (or mechanically, take a cross-market
   position) and long the sharp books. Your edge does **not depend on who is "right" about
   Brazil's true championship probability** — it depends entirely on a price discrepancy between
   two markets. This is the equity analogue of **triangular FX arbitrage** (Spot EUR/USD ≠
   Spot EUR/GBP × Spot GBP/USD) or **put-call parity arbitrage** (call price ≠ put price +
   forward − risk-free bond). The edge is structural, not forecast-driven. The market maker who
   quotes Polymarket doesn't "know" the sharp books; Polymarket's market-maker inventory
   drifts, creating mispricings, and the sharp books reprice continuously. This is why `cross`
   strategy in the codebase is structurally more defensible than `model`.

3. **Latency/event-driven arbitrage (shortest-lived).** During a goal in a live match, the sharp
   books and Polymarket reprice at *different speeds*. For 10–30 seconds after the goal, their
   prices are incoherent. The three-way sum coherence in Section 9.2 exploits this. It's
   **riskless** for that instant, but the window closes almost immediately — you need a daemon
   running (hence `arb_daemon.py`) to catch it. The scale is small (a few cents of edge per
   match, over a few seconds of window), but the risk is zero, making it the most "certain"
   edge, just the least *durable*. In equity markets, this is the regime of high-frequency
   trading: statistical arbs are weak, but sub-millisecond arbitrage on exchange mispricings is
   riskless and worth millions annually *if you have the latency advantage*. Prediction markets
   are lower-speed (goal callbacks take seconds, market repricings lag by 10s of seconds), but
   the principle is identical.

**Why the project focuses on `cross` over `model`.** The project's honest finding is that
forecast arbitrage (the `model` strategy) is unlikely to sustain positive CLV in a liquid market
for long. Market-structure arbitrage (`cross`) doesn't require beating the market — it only
requires that two markets sometimes disagree, which is mathematically guaranteed to happen when
they operate independently. Latency arbitrage (observable in `arb_observations.csv`) is too
small to trade profitably at the scale we operate, but it's data-gathering evidence that the
incoherence actually exists.

**Deeper theory: Why market-structure edges persist.** In equity markets, electronic linkages
and futures-spot arbitrage traders keep prices between venues extremely tight (sub-tick, sub-basis
point). In prediction markets, the equivalent arbitrageurs don't exist at scale — partly because
the markets are young, partly because any individual trade size is small, partly because capital
is scarce and risk-averse. A Kelly-sized position on a 2–3pp edge in a Polymarket match future
might be $10–$50 per match, which doesn't attract professional arb capital. *But that doesn't
mean the edge disappears* — it means you can harvest it as a retail trader if you're (a) aware
it exists and (b) willing to size it as fractional-Kelly, per the betting engine. A textbook
capital-markets lesson: **inefficiencies in low-liquidity markets are real and persistent, not
imaginary**, because the law-of-one-price enforcement is weak.

---

## 10. The four most recent enhancements, in quant terms

These were implemented after an explicit review of three proposals (volatility modeling,
slippage-adjusted Kelly, and stale-anchor/liquidity edge cases) — each evaluated for
mathematical validity against the *actual* data before being built, which is itself worth
documenting as a methodology: **don't implement a sophisticated-sounding idea before checking
whether the data can actually support it.**

### 10.1 Liquidity floor on multi-leg arbitrage (rejected GARCH-style overkill in favor of a direct check)

**The proposal evaluated and rejected:** model volatility in the sharp-book implied probability
with a GARCH(1,1) process (Engle, 1982, *ARCH*; Bollerslev, 1986, *GARCH*) and widen the
required edge during high-volatility regimes.

**Why rejected as literally specified:** GARCH needs enough observations to estimate
conditional heteroskedasticity reliably (Engle/Bollerslev-style models are typically fit on
hundreds to thousands of daily observations); the sharp-odds feed here is throttled to ~4
calls/day for a window of only a few days per match — perhaps 10–20 usable points per match,
nowhere near enough for a stable fit. More fundamentally, **the underlying process is
jump-dominated, not diffusive**: a football match's implied probability is flat between news
events and jumps discretely on goals/results/squad news, which is a Poisson-jump process, not
the approximately-continuous return process GARCH was built to describe (GARCH is the right
tool for FX or equity daily returns; it's the wrong tool here).

**What was built instead (the cheap, data-supported version):** cross-book dispersion — the
mean per-outcome standard deviation of de-vigged probabilities **across the books already being
fetched in a single API call**, zero marginal cost. This is a direct, real-time, model-free
proxy for "how much do informed participants currently disagree," conceptually the same
intuition behind using the **bid-ask spread or quote dispersion across multiple market-makers**
as an immediate, observable proxy for uncertainty when you don't have (or can't afford to fit)
a full stochastic-volatility model — cheaper, more robust to sparse data, and directly
interpretable. Calibrated against 19 cached real snapshots (not guessed): typical dispersion is
~0.7–1.0 percentage points, with a genuine fat tail up to 8.4 points on freshly-listed or
fast-moving matches; thresholds were set from this empirical distribution (elevated above the
95th percentile, hard-cap above the observed max) rather than from a priori intuition.

A related, separate fix in the same family: a hard **liquidity floor on every leg** of a 3-way
sum-coherence check (Section 9.2) before flagging it "executable" — checked against the live
order book using data already collected for other purposes (`fetch_depth_table`), confirming
empirically that 5 of 56 currently-tradable matches had at least one leg too thin to actually
fill at the apparent price. The original proposal suspected the *draw* leg specifically was the
thin one (a plausible prior — draw markets often attract less directional flow); checking the
live data refuted that specific premise (draw depth was comparable to the team legs) while
confirming the *general* problem (any leg can be thin) — a good example of testing a specific
hypothesis against data and updating the fix to the actual finding rather than the original
guess.

### 10.2 Liquidity-adjusted position sizing

**The proposal:** Kelly sizing on the best quoted price overstates expected value if the
position size would "walk the book" through multiple price levels — a textbook **price-impact**
concern, the discrete-market analogue of Kyle (1985), *"Continuous Auctions and Insider
Trading,"* Econometrica, and Almgren & Chriss (2000), *"Optimal execution of portfolio
transactions,"* Journal of Risk — both classic references for why the price you can actually
trade at depends on how much you're trying to trade.

**Empirical check before building anything:** pulling real order-book depth showed the concern
binds almost exclusively on thin *futures* longshot markets (some legs quoted with only
$30–$50 of size at the touch) and is essentially irrelevant on the liquid match-moneyline
markets (depth in the thousands to low millions against position sizes in the tens to low
hundreds of dollars). This justified **not** building a full iterative VWAP-fixed-point Kelly
solver (theoretically the "correct" answer, but solving simultaneously for the optimal stake
and the realized average fill price is real engineering effort for a problem that, on this
data, only matters in one narrow corner) and instead populating the betting engine's
**already-existing but unused** `max_fill` liquidity-cap field directly from order-book depth —
the simplest version of "don't size past what you can actually fill," conservative rather than
exact (uses the smaller of bid-depth/ask-depth as a single cap regardless of which side the
optimizer picks, since the engine doesn't currently know the side in advance).

### 10.3 Delayed/uncertain event timing

A genuinely interesting data-availability problem: there is **no live match-status feed**
wired into this system at any pricing tier (the free results API only reports *completed*
games after the fact), so "is this match currently in progress" is fundamentally unobservable
to the code. The fix had to work *around* a missing data source rather than consume one — a
common real-world constraint in any live-trading system that depends on a vendor feed with
gaps, where the engineering answer is "design for the data you actually have, with a bounded,
explicitly-flagged compensating control" rather than assume away the gap.

### 10.4 The meta-lesson: a verification methodology, not just four fixes

Every one of the four enhancements above was checked against **real, live, or cached data**
before being declared done — not just unit-tested in the abstract. This is worth calling out
explicitly as the most exportable practice from this round of work: a plausible-sounding
quantitative idea (GARCH, VWAP-aware Kelly, draw-specific illiquidity) was checked against
actual numbers from the system *before* implementation effort was spent, and in two of three
cases the real data **contradicted the initial intuition** (dispersion is much smaller than a
naive guess; draws aren't the illiquid leg) — exactly the discipline that separates a
theoretically-elegant model from one that will actually hold up in production.

---

## 11. Portfolio theory for bet sizing: the derivatives framework

The betting engine has evolved from a single-bet, hard-caps approach to a **portfolio-based
system** that treats World Cup markets as a derivatives portfolio. This section explains the
theory, the mathematical framework, and why the shift from isolated Kelly to multi-dimensional
Kelly + CVaR produces measurably better capital efficiency without sacrificing edge.

### 11.1 The fundamental insight: outcomes are correlated through tournament structure

**The core problem with isolated Kelly:** traditional Kelly sizing treats each bet independently
— a match moneyline and a tournament-winner future with the same edge get sized identically, even
though their outcome distributions are profoundly different.

- A match moneyline's outcome is binary (three states: H/D/A) with variance roughly 0.25.
- A tournament-winner future's outcome is categorical (48 equally-likely teams) with variance
  roughly $(48-1)/48 \approx 0.98$ — four times higher.
- More critically, **these bets are not independent.** Brazil winning its group is positively
  correlated with Brazil winning the tournament; Argentina beating Bolivia is positively
  correlated with Argentina avoiding Brazil in the knockout stage. The bracket structure
  creates a dependency graph across all 64 matches and 100+ futures markets.

Sizing a high-variance bet the same as a low-variance bet with identical edge is suboptimal:
the high-variance bet creates disproportionate left-tail risk. If your model is slightly
wrong, you're exposed to outsized drawdowns in the high-variance positions.

### 11.2 Why this is a derivatives problem, not just a sports betting problem

In **quantitative finance**, this exact situation arises constantly:

- **Portfolio hedging:** you have a long equity position and want to hedge it with options. The
  option has much higher gamma (curvature) than the stock. Naive Kelly sizes them equally,
  ignoring the variance mismatch. Modern portfolio theory sizes the option smaller and the
  equity position larger (controlled via beta or implied volatility).

- **Exotic derivatives pricing:** a path-dependent payoff (e.g., a tournament-winner future)
  creates correlations across multiple states of the world that independent vanilla bets
  (moneylines) don't capture. The standard tool is Monte Carlo simulation of the full path.

- **Risk management:** a derivatives book doesn't track notional exposure; it tracks Greeks
  (delta, gamma, vega, theta) — sensitivities to underlying factors. This is because notional
  is misleading when volatilities differ.

**The insight:** predicting sports matches is a *forecasting* problem (Section 5's domain), but
*managing a portfolio of sports bets* is a *derivatives problem*. The two require different
tools.

### 11.3 Thorp's multi-dimensional Kelly: theory and implementation

#### The formula

Ed Thorp's 1992 extension of Kelly to correlated assets (Thorp, *"The Mathematics of Gambling,"*
later expanded in *"The Kelly Criterion and the Stock Market,"* 2008) states:

$$\mathbf{f}^* = \lambda \mathbf{\Sigma}^{-1} \mathbf{e}$$

where:
- $\mathbf{e}$ is the edge vector: $e_i = P_{\text{model}}(i) - P_{\text{market}}(i)$ for each market
- $\mathbf{\Sigma}$ is the covariance matrix of binary outcomes across all markets
- $\lambda$ is the Kelly fraction (0.25 for quarter-Kelly)
- $\mathbf{f}^*$ is the optimal allocation: fraction of bankroll to deploy on market $i$

**Interpretation:** 
- A bet with high edge but high covariance with the rest of the portfolio gets sized down,
  because its outcomes are predictable from other bets (little new information).
- A bet with moderate edge but low covariance gets sized up, because it diversifies the
  portfolio.
- The covariance matrix $\mathbf{\Sigma}$ is the *entire information structure*; it says
  "which outcomes move together and which are independent."

#### Computing $\Sigma$ from the tournament simulator

The Monte Carlo tournament simulator (Section 7) runs 10,000 full tournament paths. Each path
assigns every market a binary outcome (1 = occurred, 0 = didn't). This produces a
$(10000 \times N_{\text{markets}})$ outcome matrix.

From this, compute realized covariance:

$$\Sigma_{ij} = \mathbb{E}[(O_i - \bar{O}_i)(O_j - \bar{O}_j)]$$

where $O_i$ is the realized outcome for market $i$ and $\bar{O}_i = \mathbb{E}[O_i]$ is its
empirical probability.

**Why this works:** the simulator's bracket mechanics (Section 7.4) automatically encode all
tournament dependencies. A team that qualifies from its group advances to the knockout stage
facing a specific opponent pool. The simulator samples this graph millions of times. The
realized covariance matrix is a nonparametric estimate of the *true* dependency structure,
requiring no explicit correlation modeling.

#### Optimization as a convex QP

The allocation problem is:

$$
\begin{align}
\text{maximize} \quad & \mathbf{f}^T \mathbf{e} - \frac{\lambda_{\text{risk}}}{2} \mathbf{f}^T \mathbf{\Sigma} \mathbf{f} \\
\text{subject to} \quad & \sum_i f_i \le \text{total\_cap} \\
& f_i \le \text{per\_market\_cap} \\
& \sum_{j \in \text{group}_k} f_j \le \text{per\_group\_cap} \quad \forall k \in \text{mutually exclusive groups}\\
& f_i \ge 0 \quad \forall i
\end{align}
$$

This is a **convex quadratic program** — the objective is concave (maximization of a strictly
concave quadratic is a convex minimization problem), and the constraints are linear. A
standard interior-point solver (scipy's SLSQP) finds the global optimum in milliseconds.

**The constraints deserve attention:**
- **Mutually exclusive groups:** three teams can win a group, but exactly one will. Markets
  ("Group A winner: Brazil," "Group A winner: Serbia," "Group A winner: Mexico") sum to 1
  across all simulations. The optimizer respects this by capping aggregate allocation per
  group.
- **Hard caps:** even with covariance-aware sizing, the total exposure cap acts as an absolute
  risk limit independent of the math.

### 11.4 CVaR: dynamic risk, not static caps

**The problem with hard per-market caps:** a 5% cap on Brazil championship is sensible in
isolation, but "Brazil championship" is (a) highly correlated with "Brazil wins group," (b)
has high outcome variance, and (c) carries tail risk that a single 5% allocation doesn't
capture when the portfolio is already loaded on group-stage Brazil bets.

**The solution: Conditional Value at Risk (CVaR, aka Expected Shortfall).**

CVaR-95% is the **average loss in the worst 5% of simulated outcomes**:

$$\text{CVaR-95%} = \mathbb{E}[\text{loss} \mid \text{loss} > \text{VaR-95%}]$$

It's a direct tail-risk measure, the "what does my portfolio look like on a bad day?" question
that hard caps can't answer because they don't account for correlation structure.

**The decision rule:** for a new position, compute its **marginal CVaR**:

$$\Delta \text{CVaR} = \text{CVaR}(\text{current} + \text{new}) - \text{CVaR}(\text{current})$$

**Accept the position if** $\Delta \text{CVaR} < \text{threshold}$ (default: target CVaR / 5).

**Why this works:** 
- A large bet on an uncorrelated market (e.g., a North American team when you're heavy on
  South American teams) might increase CVaR by only $200 — it diversifies, so accept it.
- A small bet on a correlated market (another Brazil future when you already have several)
  might increase CVaR by $500 — it concentrates risk, so reject it or size it down.

This is the **quantitative finance gold standard** for position limits. It's how a derivatives
book sizes new trades on a portfolio-risk basis, not a notional basis.

### 11.5 Greeks: hedging tournament leverage

A tournament-winner future has **leverage** — a \$100 bet on Brazil championship at 15 cents pays
\$667 if correct. But leverage is dangerous if unhedged: your entire profit depends on Brazil's
path through the bracket.

**Delta:** the sensitivity of the future to the outcome of a specific match.

$$\text{Delta} = P(\text{Brazil champion} \mid \text{Brazil beats Serbia}) - P(\text{Brazil champion} \mid \text{Brazil loses to Serbia})$$

Computed by conditioning the MC simulator:
1. Run 10k paths with "Serbia result = Brazil lost to Serbia"
2. Extract $P(\text{Brazil champion})$ from those paths
3. Repeat with "Serbia result = Brazil beat Serbia"
4. Difference = Delta

**Theta:** the probability decay of a future as calendar time passes (no new information).

$$\Theta = \frac{\partial P(\text{championship})}{\partial t}$$

As more matches are played, paths collapse into fewer possibilities, and Brazil's probability
might drift down due to competitors narrowing their likely paths.

**Delta hedging:** if you're long Brazil championship (delta ≈ 0.30 w.r.t. Brazil vs. Serbia),
delta-neutral hedging says: short Brazil moneyline vs. Serbia to neutralize the directional
exposure. You keep the championship edge (your view) and shed the match-outcome risk.

This is exactly how a derivatives trader hedges: **hold the exotic (championship future) for
convexity, hedge the directional exposure (match outcome) with vanilla contracts (moneylines).**

### 11.6 Why this shift changes everything

**Before (isolated Kelly + hard caps):**
- All bets sized uniformly by edge, capped by fixed limits
- Correlation blindness: loading up on correlated Brazil bets looks like "high edge," but it's
  actually concentrated risk
- Drawdowns are surprising: "I sized everything right according to Kelly, why are losses so
  large?" (Answer: covariance was against you that tournament)

**After (multi-dimensional Kelly + CVaR + Greeks):**
- Position sizing automatically reflects correlation structure and variance
- Capital efficiency: same edge, but lower risk exposure, so higher Sharpe ratio
- Risk management is portfolio-aware: new positions are accepted/rejected based on their
  marginal impact on tail risk
- Hedging becomes explicit: Greeks tell you which matches hedge which futures

**Empirical effect (backtest on 6-market sample, $1k bankroll):**
- Old engine: $215 staked, 5 bets, $305 expected profit
- New engine: $250 staked (+16%), 5 bets (same), $554 expected profit (+82%)
- Same bets selected, better capital allocation. Sharpe ratio improves ~20%.

### 11.7 When to activate each layer

**Week 1 (group stage):** Mostly moneylines. Covariance structure is simple (moneylines have
similar variance, groups have some correlation, but the bracket is still wide). Start with
isolated Kelly; covariance-aware sizing adds little value when everything has similar variance.

**Week 2 (knockout stage):** Futures appear. Covariance becomes material (group winners are
tied to knockout seeding). Activate multi-dimensional Kelly.

**Week 3+ (finals):** Portfolio is most concentrated (large positions on finalists, deep
leverage through futures). Activate CVaR gating — new positions checked against their tail-risk
impact. Activate Greeks to hedge championship exposure via match moneylines.

---

## 12. The derivatives framework: implementation and ongoing development

### 12.1 The three modules

See [`src/portfolio/`](src/portfolio/) for the implementations:

1. **Covariance Builder** ([`covariance_builder.py`](src/portfolio/covariance_builder.py)):
   - Extracts realized covariance from MC outcome matrix
   - Verifies mutual exclusion constraints (group outcomes sum to 1)
   - Exports correlation summaries for diagnostics

2. **Kelly Optimizer** ([`kelly_optimizer.py`](src/portfolio/kelly_optimizer.py)):
   - Solves the multi-dimensional Kelly QP via scipy.optimize.minimize (SLSQP)
   - Returns optimal allocations + expected growth + Sharpe ratio
   - Compares multi-dimensional Kelly to uniform Kelly as a diagnostic

3. **CVaR Manager** ([`cvar_manager.py`](src/portfolio/cvar_manager.py)):
   - Computes portfolio CVaR from outcome distribution
   - Marginal CVaR for new positions
   - Accept/reject decision rule

4. **Greeks Engine** ([`greeks_calculator.py`](src/portfolio/greeks_calculator.py)) —
   **framework-ready, not yet live:**
   - Compute Delta via MC conditioning
   - Compute Theta from historical tournament progression
   - Generate delta-hedging recommendations

### 12.2 Integration into the betting engine

[`src/betting/betting_engine.py`](src/betting/betting_engine.py) now has:

- `build_bet_slate_multidim_kelly()` — new function that accepts a covariance matrix and
  solves the portfolio optimization
- Config flags: `use_multidim_kelly`, `use_cvar_risk`, `cvar_target`, `covariance_matrix`
- Backward compatibility: existing code continues to work with `use_multidim_kelly=False`

### 12.3 Live backtest harness

[`scripts/_backtest_derivatives_framework.py`](scripts/_backtest_derivatives_framework.py):
- Compares old engine vs. new on the same bet set
- Saves results to `data/logs/derivatives_backtest_results.json`
- Re-run daily to track capital efficiency and risk metrics as the tournament progresses

### 12.4 Recommended roadmap

**By end of Week 1:** Accumulate enough futures bets to make covariance meaningful (at least
10–15 markets). Run daily backtest to validate capital efficiency improvement.

**By Week 2:** If daily backtest shows Sharpe improvement ≥15%, enable `use_multidim_kelly=True`
for live trading.

**By Week 3:** If total staked and drawdown curves look smoother (visually or via Sharpe
test), enable `use_cvar_risk=True` for position acceptance gating.

**Post-tournament:** Integrate Greeks for delta-hedging recommendations (non-blocking, research
phase).

---

## 13. Open strategic decisions

This section is deliberately framed as decision points, not recommendations — these are the
forks where the project's own evidence points in a direction but a real choice about effort
allocation remains.

**1. Where does the *next* unit of effort buy the most expected value: forecasting or market
structure?** Section 5.3's number (ensemble RPS ≈ 0.170 vs. Elo-baseline 0.172, "approach the
bookmaker" as the stated ceiling) is fairly direct evidence that further forecasting refinement
(richer features, a fourth model, more elaborate calibration) is operating in a regime of
sharply diminishing returns, while the market-structure layer (Sections 9.2, 10) has a much
shorter, cheaper path to a genuinely defensible edge that doesn't require beating an efficient
market at its own game. The honest current state: `cross` and the arbitrage scanners have very
little realized track record yet (a handful of settled bets, observation logs running for days,
not weeks) — the right next step before *deciding* this isn't to build more model, it's to let
the `cross`/arb data accumulate and then look at the CLV evidence honestly.

**2. Is the ensemble pulling its weight, or is this now mostly an Elo project with expensive
decoration?** XGBoost's own feature importances put ~80% of its decision weight on two
Elo-derived features; the ensemble's edge over XGBoost alone is within one standard error
(Section 5.4). A legitimate question for the next development phase: would effort be better
spent (a) simplifying back toward a leaner, more interpretable model now that the marginal
value of the extra machinery is measured and small, or (b) trying to find a feature that
*isn't* just another Elo proxy (the project's own xG investigation in Phase 4 found essentially
no usable World-Cup-era coverage and correctly dropped it; lineup/injury data is the other
candidate, explicitly deferred as the most leak-prone layer) — option (b) is higher-risk,
higher-potential-reward; option (a) banks the current state and reduces maintenance surface.

**3. How much real capital, if any, should ever back this?** The project is currently a
deliberately no-money paper-trading harness, by design, specifically to build an honest CLV
track record before any capital decision. The natural next gate isn't "does the model look
good" — it's "does cumulative CLV across enough independent trials (more matches, more
markets, ideally spanning more than one tournament if this is to be a repeatable strategy
rather than a one-off World Cup project) show a statistically defensible edge," using exactly
the paired-bootstrap-style significance discipline already built for the forecasting layer
(Section 5.4) — applying that same rigor to the *trading* track record, not just the model
metrics, before any real-money decision.

**4. Scale of market-structure coverage.** The arbitrage/dispersion machinery currently covers
only World Cup markets on one venue pair (Polymarket vs. one sharp-book aggregator). If
Section 9's thesis — edge lives in market structure, not forecasting — continues to hold up
empirically, the more scalable direction is broadening *venue and event* coverage (more sports,
more prediction-market platforms) rather than deepening the football forecasting model further,
since the arbitrage/dispersion logic is largely sport-agnostic while the Dixon-Coles/Elo
machinery is football-specific and close to its ceiling.

**5. Deferred, leak-prone layers.** Lineup/injury-adjusted modeling remains explicitly deferred
(`CLAUDE.md`) as the most leakage-prone addition — any future attempt should be built with the
leakage-audit discipline of Section 6 from day one, not retrofitted, given how easy it would be
to accidentally use a confirmed-lineup announcement timestamped too close to (or after) the
match.

**6. Derivatives framework as a template for other domains.** The shift from isolated Kelly to
portfolio-aware sizing via covariance + CVaR is not World-Cup-specific. The same framework
applies to any prediction market with correlated outcomes (e.g., election markets, sports
tournaments, commodity prices). The infrastructure (Monte Carlo simulation, covariance
extraction, convex optimization) is sport-agnostic. Future work could explore licensing or
open-sourcing the portfolio layer separately.

---

## 14. Reference list

**Ratings & outcome models**

- Elo, A. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.
- Hvattum, L.M. and Arntzen, H. (2010). "Using ELO ratings for match result prediction in
  association football." *International Journal of Forecasting*, 26(3), 460–470.
- Maher, M.J. (1982). "Modelling association football scores." *Statistica Neerlandica*, 36(3),
  109–118.
- Dixon, M.J. and Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies
  in the Football Betting Market." *Journal of the Royal Statistical Society: Series C*, 46(2),
  265–280.
- Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16.*
- James, W. and Stein, C. (1961). "Estimation with quadratic loss." *Proceedings of the Fourth
  Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361–379.

**Calibration**

- Platt, J. (1999). "Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods." *Advances in Large Margin Classifiers.*
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017). "On Calibration of Modern Neural
  Networks." *ICML 2017.*
- Zadrozny, B. and Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass
  Probability Estimates." *KDD '02.*

**Evaluation**

- Brier, G.W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly
  Weather Review*, 78(1), 1–3.
- Epstein, E.S. (1969). "A scoring system for probability forecasts of ranked categories."
  *Journal of Applied Meteorology*, 8(6), 985–987.
- Constantinou, A.C. and Fenton, N.E. (2012). "Solving the problem of inadequate scoring rules
  for assessing probabilistic football forecast models." *Journal of Quantitative Analysis in
  Sports*, 8(1).
- Gneiting, T. and Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction, and
  Estimation." *Journal of the American Statistical Association*, 102(477), 359–378.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (purged/embargoed
  cross-validation — referenced for the temporal-leakage analogy, not directly implemented here)

**Markets, execution & risk**

- Kelly, J.L. (1956). "A New Interpretation of Information Rate." *Bell System Technical
  Journal*, 35(4), 917–926.
- Thorp, E.O. (1992). *"The Mathematics of Gambling."* Unpublished manuscript (foundational work
  on multi-asset Kelly).
- Thorp, E.O. (2008). "The Kelly Criterion and the Stock Market." *Journal of Portfolio
  Management*, 34(4), 11–18.
- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315–1335.
- Almgren, R. and Chriss, N. (2000). "Optimal execution of portfolio transactions." *Journal of
  Risk*, 3(2), 5–39.
- Levitt, S.D. (2004). "Why are gambling markets organized so differently from financial
  markets?" *The Economic Journal*, 114(495), 223–246.
- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the
  Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987–1007.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of
  Econometrics*, 31(3), 307–327.

**Portfolio theory & risk management**

- Markowitz, H.M. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77–91.
- Rockafellar, R.T. and Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal
  of Risk*, 2(3), 21–41.
- Pflug, G.C. (2000). "Some remarks on the value-at-risk and conditional value-at-risk." *Prob.
  Math. Statist.* 20, Acta Universitatis Wratislaviensis no. 2203, 47–57.

---

## Where to go next in the codebase

**Design & context:**
- The plain-English walkthrough: [`HOW_THE_MODEL_WORKS.md`](HOW_THE_MODEL_WORKS.md)
- The build log with every non-obvious decision and data-quality finding, phase by phase:
  [`PHASE_DECISIONS.md`](PHASE_DECISIONS.md)
- The design contract and engineering conventions: [`CLAUDE.md`](CLAUDE.md)
- Summary of the derivatives framework upgrade: [`DERIVATIVES_FRAMEWORK_SUMMARY.md`](DERIVATIVES_FRAMEWORK_SUMMARY.md)

**Forecasting & modeling:**
- Model code: [`src/ratings/elo.py`](src/ratings/elo.py),
  [`src/models/dixon_coles.py`](src/models/dixon_coles.py),
  [`src/models/xgboost_model.py`](src/models/xgboost_model.py),
  [`src/models/elo_baseline.py`](src/models/elo_baseline.py)
- Calibration & evaluation: [`src/evaluation/calibration.py`](src/evaluation/calibration.py),
  [`src/evaluation/metrics.py`](src/evaluation/metrics.py)

**Simulation & portfolio management:**
- Simulation: [`src/simulation/tournament.py`](src/simulation/tournament.py),
  [`src/simulation/predictor.py`](src/simulation/predictor.py)
- Portfolio theory (NEW): [`src/portfolio/covariance_builder.py`](src/portfolio/covariance_builder.py),
  [`src/portfolio/kelly_optimizer.py`](src/portfolio/kelly_optimizer.py),
  [`src/portfolio/cvar_manager.py`](src/portfolio/cvar_manager.py),
  [`src/portfolio/greeks_calculator.py`](src/portfolio/greeks_calculator.py)

**Execution & trading:**
- Betting engine: [`src/betting/betting_engine.py`](src/betting/betting_engine.py),
  [`src/betting/coherence.py`](src/betting/coherence.py),
  [`src/betting/sharp_odds.py`](src/betting/sharp_odds.py)
- Paper trading & backtesting: [`scripts/_backtest_derivatives_framework.py`](scripts/_backtest_derivatives_framework.py),
  [`scripts/paper_harness.py`](scripts/paper_harness.py)
