"""
betting_engine.py
=================

A staking / bankroll-management layer that sits on top of your World Cup 2026
predictor. It takes your model's calibrated probabilities and current market
prices (e.g. Polymarket Yes-share prices) and decides *whether* and *how much*
to bet, then logs everything so you can measure whether you actually have an edge.

Design philosophy (read this before using it):
----------------------------------------------
1. Edge comes from DISAGREEMENT with the market, not from your prediction alone.
   The market price is itself a strong probability estimate. You only bet where
   your model disagrees enough to clear costs AND survive your own uncertainty.

2. Your model is not the truth. Every raw edge is discounted by:
     - the bid/ask spread you actually pay (you don't fill at the mid),
     - any platform fee,
     - a fixed "model humility" haircut for the fact that your probabilities
       are estimates, not facts.
   Only edge that survives all three is treated as real.

3. Sizing matters more than selection. Stakes are set by FRACTIONAL Kelly
   (default quarter-Kelly), then clamped by hard caps: per market, per
   correlation group (e.g. all "Brazil" markets), and total exposure. A
   bankroll stop-loss can halt betting entirely.

4. The real scorecard is CLOSING LINE VALUE, not whether you're up or down.
   Over a single tournament variance dominates. If you consistently get a
   better price than the market's closing price, you have a durable edge.
   analyze_performance() computes this.

This module has NO platform integration and places NO orders. It only produces a
recommended bet slate and keeps the log. You place bets yourself.

Dependencies: pandas, numpy (already in your project).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger
log = get_logger(__name__)

# Portfolio management imports (optional, for multi-dimensional Kelly + CVaR)
try:
    from src.portfolio.covariance_builder import MCCovarianceExtractor, MarketDefinition
    from src.portfolio.kelly_optimizer import MultiDimensionalKellyOptimizer
    from src.portfolio.cvar_manager import CVaRPortfolioManager
    PORTFOLIO_MODULES_AVAILABLE = True
except ImportError:
    PORTFOLIO_MODULES_AVAILABLE = False


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class BettingConfig:
    """All knobs in one place. Defaults are deliberately conservative."""

    # --- Sizing ---
    kelly_fraction: float = 0.25      # fraction of full Kelly (0.25 = quarter-Kelly)
    use_variance_adjustment: bool = False  # scale Kelly inversely to outcome variance
    variance_adjustment: dict = None  # {market_category: variance_factor}
                                      # e.g. {'moneyline': 1.0, 'future': 3.5}
                                      # Kelly gets scaled by 1/variance_factor

    # --- Multi-dimensional Kelly + CVaR (derivatives framework) ---
    # use_multidim_kelly defaults True: the covariance matrix comes from the
    # 100k-path MC simulation (sim_outcome_matrix.npz), NOT from active trade
    # count. Even 2 correlated positions benefit from covariance-aware sizing.
    # Falls back to isolated Kelly gracefully when covariance_matrix is None.
    use_multidim_kelly: bool = True
    use_cvar_risk: bool = False       # enable CVaR pre-filter acceptance gate
    cvar_target: float = 5000.0       # portfolio CVaR budget in dollars
    covariance_matrix: Optional[np.ndarray] = None  # injected by caller from sim_covariance

    # --- Edge gating (all in probability terms, i.e. 0.04 = 4 percentage points) ---
    min_edge: float = 0.04            # required edge AFTER all costs to place a bet
    edge_haircut: float = 0.02        # flat "model humility" discount on every raw edge

    # --- Cost model ---
    default_spread: float = 0.02      # assumed bid/ask spread if none supplied
    fee_rate: float = 0.0             # proportional trading fee at execution. Polymarket's
                                      # real fee schedule (sports markets, since 2026-03-30) is
                                      # 0% for MAKER (resting limit) fills, which also earn a
                                      # rebate, vs. up to ~1.8% for TAKER fills at 50/50 odds.
                                      # fee_rate=0 is the correct default ONLY if order_style
                                      # is "maker"; a taker-style config should set this >0.
    order_style: str = "taker"        # "taker": pay the spread (cost = mid +/- half-spread,
                                      # i.e. cross to the opposite side's price) — the
                                      # pessimistic/simple default. "maker": rest a limit order
                                      # at the mid with no spread cost paid (and, per the fee
                                      # schedule above, no fee) — use this ONLY for a strategy
                                      # that will actually execute as resting GTC limit orders,
                                      # since it does not model fill risk (a maker order may
                                      # simply never fill if price never reaches it).

    # --- Risk limits ---
    # per_market_cap: single-position blowup floor — retained even with CVaR
    # active because CVaR silently ignores positions whose market_id is not in
    # the MC outcome matrix (cross-strategy bets, structural arb legs). Without
    # this floor, an unmodelled position gets marginal_cvar=0 and could be
    # sized by fractional Kelly alone with no upper bound.
    # Raised from 0.05 to 0.15: CVaR is now the primary sizing control for
    # correlated futures; this cap only fires as a single-position backstop.
    per_market_cap: float = 0.15
    # per_group_cap removed: replaced by CVaRPortfolioManager which correctly
    # computes the marginal tail-risk of adding to a correlated group. A hard
    # group cap double-counts risk that CVaR already prices properly.
    total_exposure_cap: float = 0.40  # aggregate bankroll fraction deployed

    # --- Price sanity bounds: avoid extreme longshots/locks where the model is least reliable
    min_price: float = 0.03
    max_price: float = 0.97

    # --- Bankroll stop-loss: if bankroll <= this absolute value, place no bets ---
    stop_loss_bankroll: float = 0.0

    def validate(self) -> None:
        assert 0 < self.kelly_fraction <= 1, "kelly_fraction must be in (0, 1]"
        assert 0 <= self.min_edge < 1
        assert 0 <= self.edge_haircut < 1
        assert 0 <= self.default_spread < 1
        assert 0 <= self.fee_rate < 1
        assert self.order_style in ("taker", "maker")
        for cap in (self.per_market_cap, self.total_exposure_cap):
            assert 0 < cap <= 1
        assert 0 <= self.min_price < self.max_price <= 1


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
@dataclass
class Opportunity:
    """One market you could bet on.

    model_prob : your simulator's CALIBRATED probability that 'Yes' resolves true.
    market_mid : current Yes-share mid price in (0, 1), i.e. the market's implied prob.
    spread     : bid/ask spread (optional; falls back to config default).
    group      : correlation tag so related bets share a cap. e.g. "BRA" for every
                 Brazil market, so 'Brazil wins group' + 'Brazil wins cup' don't stack.
    max_fill   : optional liquidity ceiling in $ you can realistically get filled for.
    market_category : market type for variance adjustment (e.g. 'moneyline', 'future').
    """

    market_id: str
    description: str
    model_prob: float
    market_mid: float
    spread: Optional[float] = None
    group: Optional[str] = None
    max_fill: Optional[float] = None
    market_category: Optional[str] = None


# ----------------------------------------------------------------------------
# Core math
# ----------------------------------------------------------------------------
def _net_odds(cost: float) -> float:
    """Profit per unit staked if a share bought at `cost` resolves true."""
    return (1.0 - cost) / cost


def _kelly_fraction(p_win: float, cost: float) -> float:
    """Full-Kelly fraction of bankroll for a binary contract bought at `cost`.

    Buying one share costs `cost` and returns 1 if it resolves true.
    b = net odds = (1-cost)/cost. Kelly: f* = (p*b - (1-p)) / b.
    Returns may be negative (meaning: don't bet this side).
    """
    if not (0.0 < cost < 1.0):
        return 0.0
    b = _net_odds(cost)
    return (p_win * b - (1.0 - p_win)) / b


def evaluate_opportunity(opp: Opportunity, cfg: BettingConfig) -> dict:
    """Evaluate both sides (back Yes / back No) and return the better one.

    Accounts for the spread (you pay the ask), the fee, and the humility haircut.
    """
    spread = opp.spread if opp.spread is not None else cfg.default_spread
    # Taker: you cross the spread, paying the ask (or its NO-side mirror). Maker: you rest a
    # limit order at the mid and pay no spread cost — but the order may never fill (not modeled
    # here; see BettingConfig.order_style docstring).
    half = 0.0 if cfg.order_style == "maker" else spread / 2.0

    # Clamp to a sane open interval.
    yes_cost = min(max(opp.market_mid + half, 1e-4), 1 - 1e-4)
    no_cost = min(max((1.0 - opp.market_mid) + half, 1e-4), 1 - 1e-4)

    p = opp.model_prob
    sides = [
        # side, cost paid, model prob that side wins, raw edge vs the price paid
        ("YES", yes_cost, p, p - yes_cost),
        ("NO", no_cost, 1.0 - p, (1.0 - p) - no_cost),
    ]

    evaluated = []
    for side, cost, p_win, raw_edge in sides:
        edge_after = raw_edge - cfg.edge_haircut - cfg.fee_rate
        evaluated.append(
            {
                "side": side,
                "cost": cost,
                "p_win": p_win,
                "raw_edge": raw_edge,
                "edge_after_costs": edge_after,
                "full_kelly": _kelly_fraction(p_win, cost),
            }
        )

    return max(evaluated, key=lambda r: r["edge_after_costs"])


# ----------------------------------------------------------------------------
# Slate construction (selection + sizing + caps)
# ----------------------------------------------------------------------------
def build_bet_slate(
    opportunities: list[Opportunity],
    bankroll: float,
    cfg: Optional[BettingConfig] = None,
    cvar_manager=None,
    existing_positions: Optional[list] = None,
) -> pd.DataFrame:
    """Turn a list of opportunities into a sized, risk-capped bet slate.

    Sizing hierarchy (applied in order):
      1. Edge gate: must clear min_edge after all costs.
      2. CVaR pre-filter (if cfg.use_cvar_risk and cvar_manager provided):
         marginal_cvar of adding this position < cvar_target / 5.
         Note: CVaR only models positions whose market_id is in the MC outcome
         matrix. Cross-strategy bets not in the simulation are NOT filtered by
         CVaR — they fall through to step 3.
      3. Multi-dimensional Kelly sizing (if cfg.use_multidim_kelly and
         cfg.covariance_matrix provided) or isolated fractional Kelly.
      4. per_market_cap: single-position blowup floor (0.15). Fires only as a
         backstop for positions the CVaR manager cannot see.
      5. total_exposure_cap: aggregate bankroll ceiling.

    per_group_cap has been removed. Cross-correlation between positions in the
    same team group is now handled by the covariance matrix in the Kelly
    optimizer. CVaR acceptance further limits concentration in correlated groups.
    """
    cfg = cfg or BettingConfig()
    cfg.validate()

    COLS = [
        "market_id", "description", "group", "side", "cost",
        "model_prob", "market_mid", "raw_edge", "edge_after_costs",
        "full_kelly", "stake_fraction", "stake", "n_shares", "max_profit",
        "market_category", "cvar_checked",
    ]
    empty = pd.DataFrame(columns=COLS)

    if bankroll <= cfg.stop_loss_bankroll:
        return empty

    # ── 1. Evaluate and edge-gate every opportunity ───────────────────────────
    candidates = []
    for opp in opportunities:
        best = evaluate_opportunity(opp, cfg)
        if best["edge_after_costs"] < cfg.min_edge:
            continue
        if not (cfg.min_price <= best["cost"] <= cfg.max_price):
            continue
        if best["full_kelly"] <= 0:
            continue

        target_frac = min(best["full_kelly"] * cfg.kelly_fraction, cfg.per_market_cap)

        if cfg.use_variance_adjustment and cfg.variance_adjustment and opp.market_category:
            var_factor = cfg.variance_adjustment.get(opp.market_category, 1.0)
            target_frac = target_frac / var_factor

        candidates.append({
            "opp": opp,
            "market_id": opp.market_id,
            "description": opp.description,
            "group": opp.group or opp.market_id,
            "side": best["side"],
            "cost": best["cost"],
            "model_prob": opp.model_prob,
            "market_mid": opp.market_mid,
            "raw_edge": best["raw_edge"],
            "edge_after_costs": best["edge_after_costs"],
            "full_kelly": best["full_kelly"],
            "target_frac": target_frac,
            "max_fill": opp.max_fill,
            "market_category": opp.market_category,
            "cvar_checked": False,
        })

    if not candidates:
        return empty

    # ── 2. CVaR pre-filter ────────────────────────────────────────────────────
    # Reject positions whose marginal tail-risk impact exceeds the budget.
    # Positions not modelled in the outcome matrix (market_id not in id_to_col)
    # are silently passed through — they are governed by per_market_cap instead.
    if cfg.use_cvar_risk and cvar_manager is not None:
        from src.portfolio.cvar_manager import Position as CVaRPos
        filtered = []
        for c in candidates:
            opp = c["opp"]
            cost = c["cost"]
            stake_estimate = c["target_frac"] * bankroll

            # Only apply CVaR filter if this market is in the outcome matrix
            in_matrix = opp.market_id in cvar_manager.id_to_col

            if in_matrix:
                test_pos = CVaRPos(
                    market_id=opp.market_id,
                    side=c["side"],
                    stake=stake_estimate,
                    cost=cost,
                    model_prob=opp.model_prob,
                    group=c["group"],
                )
                accepted, marginal = cvar_manager.accept_position(test_pos, cfg.cvar_target)
                c["cvar_checked"] = True
                if not accepted:
                    log.debug(
                        "CVaR rejected %s: marginal=%.2f > threshold=%.2f",
                        opp.market_id, marginal, cfg.cvar_target / 5,
                    )
                    continue
            # Not in matrix: pass through to per_market_cap backstop below
            filtered.append(c)
        candidates = filtered

    if not candidates:
        return empty

    # ── 3 & 4. Size via multi-dim Kelly or isolated Kelly + per_market_cap ────
    candidates.sort(key=lambda c: c["edge_after_costs"], reverse=True)

    if cfg.use_multidim_kelly and cfg.covariance_matrix is not None and len(candidates) > 1:
        # Delegate sizing to the SLSQP optimizer for the full candidate set.
        # The optimizer respects per_market_cap and total_exposure_cap as
        # bounds/constraints; per_group_cap is intentionally omitted.
        try:
            edges = np.array([c["edge_after_costs"] for c in candidates])
            n = len(candidates)
            cov = cfg.covariance_matrix
            if cov.shape[0] >= n:
                cov_sub = cov[:n, :n]
            else:
                cov_sub = np.diag(np.full(n, 0.25))

            from src.portfolio.kelly_optimizer import MultiDimensionalKellyOptimizer
            opt = MultiDimensionalKellyOptimizer(
                edges=edges,
                covariance=cov_sub,
                kelly_fraction=cfg.kelly_fraction,
                min_edge_threshold=cfg.min_edge,
            )
            alloc = opt.solve(
                bankroll=bankroll,
                max_total_exposure_fraction=cfg.total_exposure_cap,
                per_market_cap=cfg.per_market_cap,
            )
            fracs = alloc.allocations
        except Exception as exc:
            log.warning("Multi-dim Kelly solver failed (%s); falling back to isolated Kelly.", exc)
            fracs = np.array([c["target_frac"] for c in candidates])
    else:
        fracs = np.array([c["target_frac"] for c in candidates])

    # ── 5. Apply total_exposure_cap and liquidity cap, emit rows ─────────────
    total_used = 0.0
    rows = []

    for c, frac in zip(candidates, fracs):
        frac = min(float(frac), cfg.per_market_cap)            # backstop
        frac = min(frac, cfg.total_exposure_cap - total_used)  # aggregate cap
        if frac <= 0:
            continue

        stake = frac * bankroll
        if c["max_fill"] is not None:
            stake = min(stake, c["max_fill"])
            frac = stake / bankroll if bankroll else 0.0
        if stake <= 0:
            continue

        n_shares = stake / c["cost"]
        max_profit = n_shares * (1.0 - c["cost"])
        total_used += frac

        rows.append({
            "market_id": c["market_id"],
            "description": c["description"],
            "group": c["group"],
            "side": c["side"],
            "cost": round(c["cost"], 4),
            "model_prob": round(c["model_prob"], 4),
            "market_mid": round(c["market_mid"], 4),
            "raw_edge": round(c["raw_edge"], 4),
            "edge_after_costs": round(c["edge_after_costs"], 4),
            "full_kelly": round(c["full_kelly"], 4),
            "stake_fraction": round(frac, 4),
            "stake": round(stake, 2),
            "n_shares": round(n_shares, 2),
            "max_profit": round(max_profit, 2),
            "market_category": c.get("market_category"),
            "cvar_checked": c.get("cvar_checked", False),
        })

    return pd.DataFrame(rows) if rows else empty


# ----------------------------------------------------------------------------
# Multi-dimensional Kelly with CVaR (derivatives framework)
# ----------------------------------------------------------------------------
def build_bet_slate_multidim_kelly(
    opportunities: list[Opportunity],
    bankroll: float,
    covariance_matrix: Optional[np.ndarray] = None,
    cfg: Optional[BettingConfig] = None,
) -> pd.DataFrame:
    """Build a bet slate using multi-dimensional Kelly (covariance-aware) sizing.

    This is the derivatives framework approach: every position's size depends on
    its correlation with the rest of the portfolio, not just its isolated edge.

    Args:
        opportunities: list of Opportunity objects
        bankroll: current bankroll
        covariance_matrix: shape (n_opps, n_opps), pre-computed covariance
        cfg: BettingConfig with use_multidim_kelly=True

    Returns:
        DataFrame with recommended bets, or empty if covariance unavailable
    """
    if not PORTFOLIO_MODULES_AVAILABLE:
        raise ImportError("Portfolio modules not available. Check imports.")

    cfg = cfg or BettingConfig()
    cfg.validate()

    empty = pd.DataFrame(
        columns=[
            "market_id", "description", "group", "side", "cost",
            "model_prob", "market_mid", "raw_edge", "edge_after_costs",
            "full_kelly", "stake_fraction", "stake", "n_shares", "max_profit", "market_category",
        ]
    )

    if bankroll <= cfg.stop_loss_bankroll:
        return empty

    # 1) Evaluate each opportunity to pick best side (YES/NO)
    candidates = []
    for opp in opportunities:
        best = evaluate_opportunity(opp, cfg)
        if best["edge_after_costs"] < cfg.min_edge:
            continue
        if not (cfg.min_price <= best["cost"] <= cfg.max_price):
            continue
        if best["full_kelly"] <= 0:
            continue

        candidates.append(
            {
                "opp": opp,
                "market_id": opp.market_id,
                "description": opp.description,
                "group": opp.group or opp.market_id,
                "side": best["side"],
                "cost": best["cost"],
                "model_prob": opp.model_prob,
                "market_mid": opp.market_mid,
                "raw_edge": best["raw_edge"],
                "edge_after_costs": best["edge_after_costs"],
                "full_kelly": best["full_kelly"],
                "market_category": opp.market_category,
            }
        )

    if not candidates:
        return empty

    # 2) If no covariance provided, fall back to isolated Kelly
    if covariance_matrix is None:
        # Fall back to standard approach
        return build_bet_slate(opportunities, bankroll, cfg)

    # 3) Extract relevant covariance submatrix for candidates only
    n_candidates = len(candidates)
    edges = np.array([c["edge_after_costs"] for c in candidates])

    # Match the covariance matrix dimensions to the number of candidates
    # If covariance is larger, take the first n_candidates x n_candidates block
    # (assumes covariance was built in same order as opportunities)
    if covariance_matrix.shape[0] >= n_candidates and covariance_matrix.shape[1] >= n_candidates:
        cov_sub = covariance_matrix[:n_candidates, :n_candidates]
    else:
        # Fallback: use diagonal covariance (assume independence)
        log.warning("Covariance matrix too small; using diagonal approximation")
        cov_sub = np.diag(np.ones(n_candidates) * 0.25)  # ~0.25 variance for binary 50/50

    optimizer = MultiDimensionalKellyOptimizer(
        edges=edges,
        covariance=cov_sub,
        kelly_fraction=cfg.kelly_fraction,
        mutual_exclusion_groups={},  # TODO: extract from group field if needed
        min_edge_threshold=cfg.min_edge,
    )

    kelly_allocation = optimizer.solve(
        bankroll=bankroll,
        max_total_exposure_fraction=cfg.total_exposure_cap,
        per_group_cap=cfg.per_group_cap,
        per_market_cap=cfg.per_market_cap,
    )

    # 4) Build slate from Kelly allocations
    rows = []
    for i, c in enumerate(candidates):
        frac = kelly_allocation.allocations[i]
        if frac <= 0:
            continue

        stake = frac * bankroll
        cost = c["cost"]
        n_shares = stake / cost if cost > 0 else 0
        max_profit = n_shares * (1.0 - cost) if cost > 0 else 0

        rows.append(
            {
                "market_id": c["market_id"],
                "description": c["description"],
                "group": c["group"],
                "side": c["side"],
                "cost": round(cost, 4),
                "model_prob": round(c["model_prob"], 4),
                "market_mid": round(c["market_mid"], 4),
                "raw_edge": round(c["raw_edge"], 4),
                "edge_after_costs": round(c["edge_after_costs"], 4),
                "full_kelly": round(c["full_kelly"], 4),
                "stake_fraction": round(frac, 4),
                "stake": round(stake, 2),
                "n_shares": round(n_shares, 2),
                "max_profit": round(max_profit, 2),
                "market_category": c.get("market_category"),
            }
        )

    return pd.DataFrame(rows) if rows else empty


# ----------------------------------------------------------------------------
# Bet log (persistence + settlement + CLV)
# ----------------------------------------------------------------------------
class BetLog:
    """Append-only log of placed bets, persisted to CSV.

    Workflow:
      log = BetLog("data/bets.csv")
      bet_id = log.record(slate_row, bankroll_at_bet=1000)
      ... later, when the market closes / settles ...
      log.settle(bet_id, outcome="win", closing_yes_mid=0.55)
      perf = analyze_performance(log.load())
    """

    COLUMNS = [
        "bet_id", "timestamp", "market_id", "description", "group", "side",
        "entry_cost", "model_prob", "market_mid_at_bet", "edge_after_costs",
        "stake", "n_shares", "bankroll_at_bet",
        "outcome", "closing_yes_mid", "pnl",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.path, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def record(self, slate_row: dict | pd.Series, bankroll_at_bet: float) -> str:
        r = dict(slate_row)
        df = self.load()
        bet_id = f"bet_{len(df) + 1:05d}"
        entry = {
            "bet_id": bet_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": r.get("market_id"),
            "description": r.get("description"),
            "group": r.get("group"),
            "side": r.get("side"),
            "entry_cost": r.get("cost"),
            "model_prob": r.get("model_prob"),
            "market_mid_at_bet": r.get("market_mid"),
            "edge_after_costs": r.get("edge_after_costs"),
            "stake": r.get("stake"),
            "n_shares": r.get("n_shares"),
            "bankroll_at_bet": bankroll_at_bet,
            "outcome": np.nan,
            "closing_yes_mid": np.nan,
            "pnl": np.nan,
        }
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.path, index=False)
        return bet_id

    def settle(self, bet_id: str, outcome: str, closing_yes_mid: Optional[float] = None) -> None:
        """outcome: 'win' or 'loss' (from the perspective of the side actually bet)."""
        if outcome not in ("win", "loss"):
            raise ValueError("outcome must be 'win' or 'loss'")
        df = self.load()
        mask = df["bet_id"] == bet_id
        if not mask.any():
            raise KeyError(f"unknown bet_id {bet_id}")
        row = df.loc[mask].iloc[0]
        stake = float(row["stake"])
        cost = float(row["entry_cost"])
        # PnL: win -> stake*(1-cost)/cost ; loss -> -stake
        pnl = stake * (1.0 - cost) / cost if outcome == "win" else -stake
        # Ensure the outcome column can hold strings (it loads as float when all-NaN).
        df["outcome"] = df["outcome"].astype(object)
        df.loc[mask, "outcome"] = outcome
        df.loc[mask, "pnl"] = round(pnl, 2)
        if closing_yes_mid is not None:
            df.loc[mask, "closing_yes_mid"] = closing_yes_mid
        df.to_csv(self.path, index=False)


# ----------------------------------------------------------------------------
# Performance analysis (the real scorecard)
# ----------------------------------------------------------------------------
def analyze_performance(log_df: pd.DataFrame) -> dict:
    """Compute the metrics that actually tell you whether you have an edge.

    Returns a dict with:
      - realized PnL, ROI, win rate, record
      - closing line value (CLV): did you beat the market's closing price?
      - calibration table: do your model probabilities match reality?
    CLV is the headline number. Positive average CLV across many bets is the
    strongest available sign of a durable edge, independent of variance.
    """
    out: dict = {}
    settled = log_df[log_df["outcome"].notna()].copy()
    out["n_bets_placed"] = int(len(log_df))
    out["n_bets_settled"] = int(len(settled))

    if settled.empty:
        out["note"] = "No settled bets yet."
        return out

    # --- Realized performance ---
    total_stake = settled["stake"].sum()
    total_pnl = settled["pnl"].sum()
    wins = (settled["outcome"] == "win").sum()
    out["total_staked"] = round(float(total_stake), 2)
    out["total_pnl"] = round(float(total_pnl), 2)
    out["roi"] = round(float(total_pnl / total_stake), 4) if total_stake else None
    out["record"] = f"{int(wins)}-{int(len(settled) - wins)}"
    out["win_rate"] = round(float(wins / len(settled)), 4)

    # --- Closing line value ---
    # For each bet, compare the cost you paid to what that same side cost at close.
    # Beating the close means you paid LESS than the closing cost for your side.
    clv_rows = settled[settled["closing_yes_mid"].notna()].copy()
    if not clv_rows.empty:
        def _closing_cost(row):
            yes_close = float(row["closing_yes_mid"])
            return yes_close if row["side"] == "YES" else (1.0 - yes_close)

        clv_rows["closing_cost"] = clv_rows.apply(_closing_cost, axis=1)
        # Positive = you got a cheaper (better) price than the close.
        clv_rows["clv"] = clv_rows["closing_cost"] - clv_rows["entry_cost"]
        out["avg_clv"] = round(float(clv_rows["clv"].mean()), 4)
        out["pct_beating_close"] = round(float((clv_rows["clv"] > 0).mean()), 4)
        out["n_bets_with_clv"] = int(len(clv_rows))
    else:
        out["avg_clv"] = None
        out["note_clv"] = "Record closing_yes_mid at settlement to get CLV."

    # --- Calibration: bucket model win-prob vs actual hit rate ---
    # model_prob is P(Yes); the bet's win prob depends on the side.
    settled["bet_win_prob"] = np.where(
        settled["side"] == "YES", settled["model_prob"], 1.0 - settled["model_prob"]
    )
    settled["won"] = (settled["outcome"] == "win").astype(int)
    bins = np.linspace(0, 1, 6)  # 0-0.2, ... 0.8-1.0
    settled["bucket"] = pd.cut(settled["bet_win_prob"], bins, include_lowest=True)
    calib = (
        settled.groupby("bucket", observed=True)
        .agg(n=("won", "size"), predicted=("bet_win_prob", "mean"), actual=("won", "mean"))
        .reset_index()
    )
    calib["predicted"] = calib["predicted"].round(3)
    calib["actual"] = calib["actual"].round(3)
    out["calibration"] = calib

    return out


# ----------------------------------------------------------------------------
# Convenience: load opportunities straight from a DataFrame
# ----------------------------------------------------------------------------
def opportunities_from_dataframe(df: pd.DataFrame) -> list[Opportunity]:
    """Build Opportunity objects from a DataFrame.

    Expected columns: market_id, description, model_prob, market_mid.
    Optional columns: spread, group, max_fill, market_category.
    This is the join point between your simulator output + a market price table.
    """
    required = {"market_id", "description", "model_prob", "market_mid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    opps = []
    for _, r in df.iterrows():
        opps.append(
            Opportunity(
                market_id=str(r["market_id"]),
                description=str(r["description"]),
                model_prob=float(r["model_prob"]),
                market_mid=float(r["market_mid"]),
                spread=float(r["spread"]) if "spread" in df.columns and pd.notna(r.get("spread")) else None,
                group=str(r["group"]) if "group" in df.columns and pd.notna(r.get("group")) else None,
                max_fill=float(r["max_fill"]) if "max_fill" in df.columns and pd.notna(r.get("max_fill")) else None,
                market_category=str(r["market_category"]) if "market_category" in df.columns and pd.notna(r.get("market_category")) else None,
            )
        )
    return opps


# ----------------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    cfg = BettingConfig()

    # Pretend these came from your simulator (model_prob) + a market snapshot (market_mid).
    sample = pd.DataFrame(
        [
            # Clear value: model 0.24 vs market 0.12 -> strong YES edge.
            dict(market_id="wc_winner_bra", description="Brazil win World Cup",
                 model_prob=0.24, market_mid=0.12, spread=0.02, group="BRA"),
            # Correlated to the above (same group cap applies) -> also strong, but
            # the group cap should curtail how much extra gets deployed on Brazil.
            dict(market_id="wc_bra_final", description="Brazil reach final",
                 model_prob=0.44, market_mid=0.27, spread=0.03, group="BRA"),
            # Model AGREES with market -> no bet.
            dict(market_id="wc_winner_fra", description="France win World Cup",
                 model_prob=0.14, market_mid=0.145, spread=0.02, group="FRA"),
            # Model LOWER than market -> edge is on the NO side.
            dict(market_id="wc_winner_eng", description="England win World Cup",
                 model_prob=0.05, market_mid=0.16, spread=0.02, group="ENG"),
            # Thin longshot outside price bounds -> filtered.
            dict(market_id="wc_winner_xyz", description="Minnow win World Cup",
                 model_prob=0.03, market_mid=0.01, spread=0.01, group="XYZ"),
        ]
    )

    bankroll = 1000.0
    opps = opportunities_from_dataframe(sample)
    slate = build_bet_slate(opps, bankroll, cfg)

    print("=== Recommended bet slate (bankroll $%.0f) ===" % bankroll)
    if slate.empty:
        print("No bets clear the thresholds.")
    else:
        print(slate.to_string(index=False))
        print("\nTotal staked: $%.2f (%.1f%% of bankroll)"
              % (slate["stake"].sum(), 100 * slate["stake"].sum() / bankroll))

    # Log, settle, and analyze (simulated outcomes for the smoke test).
    log = BetLog(Path(tempfile.gettempdir()) / "bets_demo.csv")
    for _, row in slate.iterrows():
        log.record(row, bankroll_at_bet=bankroll)

    df = log.load()
    if len(df):
        # Fake-settle: first bet wins, rest lose; record closing prices.
        log.settle(df.iloc[0]["bet_id"], "win", closing_yes_mid=0.16)
        for bid in df["bet_id"].iloc[1:]:
            log.settle(bid, "loss", closing_yes_mid=0.10)

    print("\n=== Performance ===")
    perf = analyze_performance(log.load())
    for k, v in perf.items():
        if k == "calibration":
            print("calibration:\n", v.to_string(index=False))
        else:
            print(f"{k}: {v}")
