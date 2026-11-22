"""
components/views.py
===================
Streamlit UI components for the V2 High-Yield dashboard.

Surfaces both the naive metrics (raw EV) and the risk-adjusted ones (shrunk probability,
fractional Kelly) so the user can see the gap. The portfolio simulator models intra-team
correlation and reports the full outcome distribution. The System Overview tab is the
validation scoreboard: ledger, CLV, and model-vs-market calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_engine import MAX_BET_FRAC
from utils.portfolio import (
    simulate_bundle, find_redundant_legs, clv_cents, clv_summary, bounded_kelly_weights,
    simulate_blend,
)
from utils.optimizer import optimize_blend
from utils.calibration import calibration_report, calibrate_trust, model_vs_market_positioning
from utils.verdict import (
    load_results, load_price_history, market_efficiency_report, model_edge_report,
    final_verdict, compute_standings, resolved_observations,
)
from utils.forecast_log import load_forecast_log, FORECAST_LOG_PATH
from utils.joint_matrix import bundle_win_matrix, is_available as joint_matrix_available
from utils.schedule import resolves_label, resolution_date, days_until
from utils.ledger_store import load_ledger, save_ledger, ledger_path

ZAR_USD = 18.5

_MARKET_TYPE_LABELS = {
    "Champion market":        "World Cup Winner",
    "Reach Final":            "Nation to Reach the Final",
    "Reach Semi-Finals":      "Nation to Reach the Semi-Finals",
    "Reach Quarter-Finals":   "Nation to Reach the Quarter-Finals",
    "Reach Round of 16":      "Nation to Reach the Round of 16",
    "Group Stage":            "Team to Advance from Group",
}


# ============================================================================
# TAB 1 — HIGH-YIELD PICKS
# ============================================================================
def _prob_bar(used_pct: float, market_pct: float, gap_pp: float) -> str:
    """Compact stacked model-vs-market bar as HTML."""
    return (
        "<div style='margin:2px 0 4px'>"
        # model (used) bar
        "<div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#9CA3AF;margin-bottom:1px'>"
        f"<span>Model (used)</span><span style='color:#10B981;font-weight:700'>{used_pct:.1f}%</span></div>"
        "<div style='background:#1f2937;border-radius:4px;height:7px;margin-bottom:6px'>"
        f"<div style='width:{min(used_pct,100):.1f}%;background:#10B981;height:7px;border-radius:4px'></div></div>"
        # market bar
        "<div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#9CA3AF;margin-bottom:1px'>"
        f"<span>Polymarket implied</span><span style='font-weight:700'>{market_pct:.1f}%</span></div>"
        "<div style='background:#1f2937;border-radius:4px;height:7px'>"
        f"<div style='width:{min(market_pct,100):.1f}%;background:#6B7280;height:7px;border-radius:4px'></div></div>"
        "</div>"
    )


def render_high_yield_tab(
    top_5_df: pd.DataFrame,
    all_qualifying_df: pd.DataFrame,
    bankroll_zar: float = 400.0,
    kelly_tilt: float = 0.5,
    adj_info: dict | None = None,
) -> None:
    """Display the risk-adjusted picks and the bundle simulator."""
    st.markdown("## Tournament Futures — High-Yield Picks")
    st.caption(
        "🏆 Team to reach a stage across the **whole tournament** (resolve in 1–4 weeks). "
        "Single-game W/D/L is the **⚽ Single-Match Bets** tab. "
        + (f"Filters: <40¢ · hit-rate ≥ {adj_info.get('min_win_prob', 0.20):.0%} · "
           f"horizon {adj_info.get('max_days', 40)}d · ¼-Kelly ×{adj_info['kelly_multiplier']:.0%} (sidebar)."
           if adj_info else "")
    )
    st.info(
        "**Survive to learn** — once-off event: expect to lose most longshots, size small, and log "
        "bets in System Overview to validate the model.",
        icon="🎯",
    )

    if top_5_df.empty:
        md = (adj_info or {}).get("max_days")
        soonest = days_until("Group Stage")  # earliest future to resolve (group qualification)
        if md is not None and soonest is not None and md < soonest:
            st.warning(
                f"No **futures** resolve within your **{md}-day horizon** — the earliest "
                f"tournament future (group qualification) is ~{soonest} days out, and the rest run "
                f"to ~29 days (champion). This isn't an error: raise the **Betting Horizon** slider "
                f"to ≥ {soonest} to see futures, or open the **⚽ Match Bets** tab for bets that "
                f"resolve in the next few days.",
                icon="🗓️",
            )
        else:
            st.warning(
                "No qualifying picks after adjustments. Loosen the sidebar knobs (hit-rate floor, "
                "betting horizon) or widen the price band.",
                icon="🔍",
            )
        return

    # ── KPI strip ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Picks", len(top_5_df))
    k2.metric("Best hit rate", f"{top_5_df['model_shrunk'].max() * 100:.0f}%",
              help="Highest used (bet-on) win probability among the picks. All clear the floor; "
                   "picks are ranked by this, most-likely-to-resolve first.")
    k3.metric("Top bet size", f"{top_5_df['recommended_bet_size'].max() * 100:.1f}%",
              help=f"Largest recommended single-bet stake (% of bankroll), capped at {MAX_BET_FRAC:.0%}.")
    k4.metric("Best adj EV",  f"{top_5_df['ev_adj'].max() * 100:+.1f}%",
              help="EV after shrinkage, haircut, and executable-price buffer.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — THE EDGE: where the model and market disagree
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 1 · Where the model and market disagree")
    st.caption("Green bar = our probability, grey = the market; the gap is the edge. "
               "**hit %** = single-shot win chance · **bet size** = Kelly-capped stake.")

    for rank, (_, row) in enumerate(top_5_df.iterrows(), start=1):
        price_cents  = float(row["polymarket_cents"])
        ev_raw       = float(row["EV"]) * 100
        ev_adj       = float(row["ev_adj"]) * 100
        model_pct    = float(row["model_prob"]) * 100
        shrunk_pct   = float(row["model_shrunk"]) * 100
        market_pct   = float(row["implied_prob"]) * 100
        mc_se_pp     = float(row["mc_se"]) * 100
        gap_pp       = model_pct - market_pct
        bet_frac     = float(row["recommended_bet_size"])      # % of bankroll
        bet_zar      = bet_frac * bankroll_zar
        market_label = _MARKET_TYPE_LABELS.get(row["match"], row["match"])

        with st.container(border=True):
            head, badges = st.columns([5, 2])
            with head:
                st.markdown(f"**{rank}. {row['selection']}**")
                st.caption(market_label)
            with badges:
                st.markdown(
                    f"<div style='text-align:right;display:flex;gap:6px;justify-content:flex-end'>"
                    f"<span style='background:#10B981;color:#fff;padding:2px 9px;border-radius:11px;"
                    f"font-size:0.78rem;font-weight:700'>hit {shrunk_pct:.0f}%</span>"
                    f"<span style='background:#6B7280;color:#fff;padding:2px 9px;border-radius:11px;"
                    f"font-size:0.78rem;font-weight:700'>adj EV {ev_adj:+.0f}%</span></div>",
                    unsafe_allow_html=True,
                )

            bar_col, stat_col = st.columns([3, 2])
            with bar_col:
                st.markdown(_prob_bar(shrunk_pct, market_pct, gap_pp), unsafe_allow_html=True)
                st.caption(
                    f"🗓 Resolves {resolves_label(row['match'])} · model {model_pct:.0f}%"
                    f" (±{mc_se_pp:.1f}pp) → {shrunk_pct:.0f}% used vs {market_pct:.0f}% market"
                )
            with stat_col:
                s1, s2, s3 = st.columns(3)
                s1.metric("Price", f"{price_cents:.0f}¢")
                s2.metric("Payout", f"{float(row['payout_multiplier']):.1f}x")
                s3.metric("Bet size", f"{bet_frac*100:.1f}%",
                          help=f"≈ R{bet_zar:.2f} of R{bankroll_zar:.0f} — fractional-Kelly stake "
                               f"on this single bet, capped at {MAX_BET_FRAC:.0%} of bankroll. "
                               f"Raw EV {ev_raw:+.0f}%.")

    with st.expander("Compare all picks as a table"):
        tbl = top_5_df.copy()
        tbl["Model %"]   = (tbl["model_prob"] * 100).round(1)
        tbl["Used %"]    = (tbl["model_shrunk"] * 100).round(1)
        tbl["Market %"]  = (tbl["implied_prob"] * 100).round(1)
        tbl["Price (¢)"] = tbl["polymarket_cents"]
        tbl["Payout"]    = tbl["payout_multiplier"].map(lambda v: f"{v:.1f}x")
        tbl["Bet %"]     = tbl["recommended_bet_size"].map(lambda v: f"{v*100:.1f}%")
        tbl["Adj EV"]    = tbl["ev_adj"].map(lambda v: f"{v*100:+.1f}%")
        tbl["Raw EV"]    = tbl["EV"].map(lambda v: f"{v*100:+.1f}%")
        st.dataframe(
            tbl[["selection", "Model %", "Used %", "Market %", "Price (¢)", "Payout", "Bet %", "Adj EV", "Raw EV"]]
            .rename(columns={"selection": "Selection"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Model %":  st.column_config.ProgressColumn("Model %", min_value=0, max_value=100, format="%.1f%%"),
                "Market %": st.column_config.ProgressColumn("Market %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 & 3 — build/simulate a bundle, then place it
    # ════════════════════════════════════════════════════════════════════════
    _render_bundle_builder(all_qualifying_df, top_5_df, bankroll_zar, kelly_tilt)


def _render_bundle_builder(
    all_qualifying_df: pd.DataFrame,
    top_5_df: pd.DataFrame,
    bankroll_zar: float,
    kelly_tilt: float = 0.5,
) -> None:
    # ── STAGE 2 — simulate ────────────────────────────────────────────────────
    st.markdown("### 2 · Build a bundle and simulate the payoff")
    _engine = "exact joint matrix" if joint_matrix_available() else "intra-team approximation"
    _split = ("an **equal split**" if kelly_tilt == 0
              else f"a **bounded Kelly blend** (tilt {kelly_tilt:.0%})")
    st.caption(
        f"Pick the contracts you'd actually back (defaults to the picks above). We split your "
        f"R{bankroll_zar:.0f} bankroll using {_split} and simulate every outcome via the "
        f"**{_engine}** — so you see the *whole* range of results, not just the jackpot."
    )

    options = all_qualifying_df["selection"].tolist()
    default = [s for s in top_5_df["selection"].tolist() if s in options][:5]
    selected = st.multiselect(
        "Bets in the bundle", options=options, default=default, max_selections=8,
        help="All pass the adjusted filter. Add or remove to reshape the risk.",
    )
    if not selected:
        st.info("Select at least one bet to simulate and place.", icon="👆")
        return

    sel = all_qualifying_df[all_qualifying_df["selection"].isin(selected)].copy()
    n = len(sel)

    # Stake split: bounded Kelly blend (tilt 0 = equal, 1 = fully Kelly-proportional).
    weights = bounded_kelly_weights(sel["kelly_frac"].to_numpy(), kelly_tilt=kelly_tilt)
    sel["stake_zar"] = bankroll_zar * weights
    sel["stake_usd"] = (bankroll_zar / ZAR_USD) * weights
    stakes_usd = sel["stake_usd"].to_numpy()

    if sel["stake_usd"].min() < 1.0:
        st.warning(
            f"Smallest stake is ${sel['stake_usd'].min():.2f} — below Polymarket's ~$1 minimum. "
            "Drop a leg, lower the Kelly tilt, or raise the bankroll.",
            icon="⚠️",
        )

    redundant = find_redundant_legs(sel)
    if redundant:
        msg = "; ".join(f"**{t}** ({len(s)} legs)" for t, s in redundant.items())
        st.warning(
            f"Correlated legs on the same team: {msg}. These are nested, not independent — "
            "the bundle is less diversified than the leg count suggests.",
            icon="🔗",
        )

    # Outcome distribution — exact joint matrix when available.
    win_matrix = bundle_win_matrix(sel)
    if win_matrix is not None:
        sim = simulate_bundle(sel, stakes_usd, ZAR_USD, win_matrix=win_matrix)
        st.caption(f"🎯 Exact joint distribution from {sim['n_sims']:,} simulated tournaments "
                   "(true cross-team correlation).")
        all_win_delta = None
    else:
        sim = simulate_bundle(sel, stakes_usd, ZAR_USD, prob_col="model_shrunk")
        sim_raw = simulate_bundle(sel, stakes_usd, ZAR_USD, prob_col="model_prob")
        st.caption("Approximation: intra-team correlation exact, different teams independent "
                   "(joint matrix not found).")
        all_win_delta = f"raw model {sim_raw['p_all_win']*100:.1f}%"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median outcome", f"R{sim['median_profit_zar']:+.2f}",
              help="The typical result — half of simulations do worse than this.")
    m2.metric("Chance of profit", f"{sim['p_profit']*100:.1f}%",
              help="Bundle nets positive overall — winners cover the losers.")
    m3.metric("Chance ALL win", f"{sim['p_all_win']*100:.1f}%", delta=all_win_delta, delta_color="off",
              help="Every leg lands — the jackpot case.")
    m4.metric("Chance total loss", f"{sim['p_total_loss']*100:.1f}%",
              help="Every leg loses — the full stake is gone.")

    p10, p90 = sim["p10_profit_zar"], sim["p90_profit_zar"]
    st.caption(
        f"These come from simulating the bundle on our model's probabilities, and **don't sum to "
        f"100%** — most outcomes are in between (some win, some lose). 80% of results land between "
        f"**R{p10:+.2f}** and **R{p90:+.2f}**; mean R{sim['mean_profit_zar']:+.2f}."
    )

    samples = sim["profit_samples_zar"]
    counts, edges = np.histogram(samples, bins=30)
    centers = ((edges[:-1] + edges[1:]) / 2).round(2)
    st.bar_chart(
        pd.DataFrame({"Profit (R)": centers, "Frequency": counts}).set_index("Profit (R)"),
        height=200,
    )
    st.caption(
        "⚠️ On a longshot bundle the median is usually a **loss** even when the mean is positive — "
        "that's the nature of asymmetric bets, and why bankroll survival matters more than EV."
    )

    st.divider()

    # ── STAGE 3 — place ───────────────────────────────────────────────────────
    _tilt_label = "equal split" if kelly_tilt == 0 else f"Kelly-tilted {kelly_tilt:.0%}"
    st.markdown(f"### 3 · Place the bets &nbsp;·&nbsp; R{bankroll_zar:.0f} across {n} ({_tilt_label})")
    st.caption("Execute these on Polymarket exactly as listed. Do not pay above the max price.")

    place_rows, total_if_all_win_zar = [], 0.0
    leg_days = []
    for _, r in sel.iterrows():
        exec_price = float(r["exec_price"])
        stake_zar  = float(r["stake_zar"])
        shares = float(r["stake_usd"]) / exec_price
        win_zar = shares * ZAR_USD
        total_if_all_win_zar += win_zar
        dd = days_until(r["match"])
        if dd is not None:
            leg_days.append(dd)
        place_rows.append({
            "Bet":        r["selection"],
            "Market":     _MARKET_TYPE_LABELS.get(r["match"], r["match"]),
            "Search":     r["team"],
            "Max price":  f"{float(r['polymarket_cents']):.0f}¢",
            "Shares":     f"{shares:.0f}",
            "Stake":      f"R{stake_zar:.2f}",
            "If it wins": f"R{win_zar:.2f}",
            "Resolves":   resolves_label(r["match"]),
        })
    st.dataframe(pd.DataFrame(place_rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total staked", f"R{sel['stake_zar'].sum():.2f}")
    c2.metric("If every leg wins", f"R{total_if_all_win_zar:.2f}",
              delta=f"+R{total_if_all_win_zar - bankroll_zar:.2f}")
    c3.metric("…but its chance is", f"{sim['p_all_win']*100:.1f}%")

    # ── Time to payoff ────────────────────────────────────────────────────────
    if leg_days:
        soonest, latest = min(leg_days), max(leg_days)
        if soonest == latest:
            st.caption(f"🗓 **Time to payoff:** the whole bundle resolves in ~{latest} days.")
        else:
            st.caption(
                f"🗓 **Time to payoff:** first leg resolves in ~{soonest} days; the bundle is "
                f"**fully settled in ~{latest} days** (the deepest-stage leg). Your capital is "
                "tied up until then — longer-dated legs lock the bankroll longest."
            )

    if st.button("📒 Log this bundle to the ledger", type="primary"):
        if "ledger" not in st.session_state:
            st.session_state.ledger = load_ledger()
        added = 0
        for _, r in sel.iterrows():
            bet = r["selection"]
            if any(e["bet"] == bet and e["outcome"] == "Pending" for e in st.session_state.ledger):
                continue
            st.session_state.ledger.append({
                "bet":           bet,
                "market_type":   r["match"],
                "model_p":       round(float(r["model_prob"]), 4),
                "market_p":      round(float(r["implied_prob"]), 4),
                "entry_cents":   float(r["polymarket_cents"]),
                "stake_zar":     round(float(r["stake_zar"]), 2),
                "payout_x":      float(r["payout_multiplier"]),
                "outcome":       "Pending",
                "closing_cents": None,
                "clv_cents":     None,
                "pnl_zar":       None,
            })
            added += 1
        save_ledger(st.session_state.ledger)
        st.success(f"Logged {added} new bet(s) (saved to disk). Settle them in System Overview to validate the model.")


# ============================================================================
# TAB 2 — SYSTEM OVERVIEW (validation scoreboard)
# ============================================================================
def render_system_overview_tab(
    model_age_str: str | None = None,
    match_probs_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
) -> None:
    st.markdown("## System Overview")
    st.caption(
        "The validation scoreboard — the strategy has edge only if the model beats the market "
        "(Brier) and the bets beat the close (CLV). Settle bets below to find out."
    )

    if model_age_str:
        st.info(f"Model probabilities last refreshed: **{model_age_str}**. "
                "A stale model vs. a live market manufactures fake 'disagreement'.", icon="🕒")

    # Ledger persists to disk (ledger.json) so it survives restarts.
    if "ledger" not in st.session_state:
        st.session_state.ledger = load_ledger()
    ledger = st.session_state.ledger

    # ── Record outcomes (with closing price for CLV) ───────────────────────────
    st.markdown("### Record outcomes")
    pending = [e for e in ledger if e["outcome"] == "Pending"]
    if not pending:
        st.info("No pending bets. Log a bundle from the High-Yield Picks tab.", icon="📋")
    else:
        st.caption(
            "For each bet: enter the **closing price** (the market's price at kickoff/resolution) "
            "and set the **outcome**. Won/Lost computes P&L and Closing-Line Value, then saves to disk."
        )
        changed = False
        for i, entry in enumerate(ledger):
            if entry["outcome"] != "Pending":
                continue
            c_label, c_close, c_outcome = st.columns([4, 2, 2])
            with c_label:
                st.markdown(f"**{entry['bet']}**")
                st.caption(f"Entry {entry['entry_cents']:.0f}¢ · stake R{entry['stake_zar']:.2f} · {entry['payout_x']:.1f}x")
            with c_close:
                closing = st.number_input(
                    "Closing ¢", min_value=0.0, max_value=100.0,
                    value=float(entry["entry_cents"]), step=1.0,
                    key=f"close_{i}", label_visibility="collapsed",
                )
            with c_outcome:
                outcome = st.selectbox(
                    "Outcome", ["Pending", "Won", "Lost"],
                    key=f"outcome_{i}", label_visibility="collapsed",
                )
            if outcome != "Pending":
                entry["outcome"] = outcome
                entry["closing_cents"] = closing
                entry["clv_cents"] = clv_cents(entry["entry_cents"], closing)
                if outcome == "Won":
                    entry["pnl_zar"] = round(entry["stake_zar"] * entry["payout_x"] - entry["stake_zar"], 2)
                else:
                    entry["pnl_zar"] = round(-entry["stake_zar"], 2)
                changed = True
        if changed:
            save_ledger(ledger)

    st.divider()
    settled = [e for e in ledger if e["outcome"] != "Pending"]

    # ── Model validation (Brier / log-loss / CLV) ──────────────────────────────
    st.markdown("### Model vs. market — is there actually edge?")
    report = calibration_report(settled)
    clv = clv_summary(settled)

    if not report:
        st.info("Settle at least one bet to score the model against the market.", icon="🔬")
    else:
        beats = report["brier_skill"] > 0
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Brier — model",  f"{report['brier_model']:.3f}",
                  help="Lower is better. Mean squared error of probability vs outcome.")
        v2.metric("Brier — market", f"{report['brier_market']:.3f}")
        v3.metric("Brier skill", f"{report['brier_skill']:+.3f}",
                  delta="model beats market" if beats else "market beats model",
                  delta_color="normal" if beats else "inverse",
                  help="Positive = the model is sharper than the market. This is the whole ballgame.")
        if clv:
            v4.metric("Avg CLV", f"{clv['mean_clv']:+.1f}¢",
                      delta=f"{clv['pct_beat']:.0f}% beat close",
                      help="Closing-Line Value. Positive over many bets is the strongest proof of edge.")
        else:
            v4.metric("Avg CLV", "—", help="Enter closing prices to compute CLV.")

        if not beats:
            st.warning(
                f"Over {report['n']} settled bet(s), the model's Brier score is **worse** than the "
                "market's. On this sample there is no demonstrated edge — treat picks as hypotheses.",
                icon="⚠️",
            )

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Model reliability** (predicted vs observed)")
            st.dataframe(report["reliability"], use_container_width=True, hide_index=True)
        with cc2:
            st.markdown("**Calibration by price band** (favourite–longshot check)")
            st.dataframe(report["by_price_band"], use_container_width=True, hide_index=True)
        st.caption(
            f"Scored on {report['n']} bet(s). Log loss — model {report['logloss_model']:.3f} "
            f"vs market {report['logloss_market']:.3f}. A meaningful verdict needs dozens of bets."
        )

    st.divider()

    # ── Trust calibration — what should model_trust actually be? ────────────
    st.markdown("### Trust calibration")
    st.caption(
        "Sweeps `model_trust` from 0→1 on **clean** resolved comparisons (a real pre-match market "
        "price AND the outcome) and finds the Brier-minimising value. Only the forward forecast log "
        "and settled ledger bets qualify — already-played matches are excluded (their live price is "
        "gone and the model has seen the result). Needs ≥15 for a stable estimate."
    )
    import altair as alt

    _obs = resolved_observations(load_forecast_log(), load_results())
    cal_result = calibrate_trust(observations=_obs, ledger_settled=settled)
    n_tot = cal_result.get("n_total", 0)
    n_l   = cal_result.get("n_ledger", 0)
    n_f   = cal_result.get("n_forecast", 0)
    st.caption(
        f"Clean comparisons: **{n_f}** from the forecast log + **{n_l}** settled ledger bet(s) "
        f"= **{n_tot}** total (need ≥15 — "
        f"{'✅ reliable' if cal_result.get('reliable') else f'⏳ {max(0,15-n_tot)} more needed'})."
    )

    if not cal_result.get("reliable"):
        st.info(cal_result["recommendation"], icon="📊")
    else:
        t_opt = cal_result["trust_optimal"]
        t_cur = 0.35
        improvement = cal_result["brier_at_current"] - cal_result["brier_at_optimal"]
        # A move is only "actionable" if the optimum is meaningfully away AND the Brier gain is real.
        actionable = abs(t_opt - t_cur) >= 0.05 and improvement >= 0.003
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Current MODEL_TRUST", f"{t_cur:.2f}", help="Set in strategy_config.py")
        tc2.metric("Data-optimal trust", f"{t_opt:.2f}",
                   delta=(("raise ↑" if t_opt > t_cur else "lower ↓") if actionable else "keep ✓"),
                   delta_color="normal" if actionable else "off")
        tc3.metric("Brier gain at optimal", f"{improvement:+.5f}",
                   help="How much lower Brier would be at the optimum vs 0.35. Below ~0.003 the "
                        "curve is effectively flat — a boundary optimum then is an artifact, not signal.")
        if actionable:
            st.warning(cal_result["recommendation"], icon="⚙️")
        else:
            st.success(cal_result["recommendation"], icon="✅")

    if cal_result.get("brier_curve") is not None:
        curve = cal_result["brier_curve"]
        t_opt = cal_result.get("trust_optimal", 0.35)
        ch = alt.Chart(curve).transform_fold(
            ["brier_blend", "brier_market", "brier_model"],
            as_=["Source", "Brier"]
        ).mark_line().encode(
            x=alt.X("trust:Q", title="model_trust →"),
            y=alt.Y("Brier:Q", title="Brier score (lower = better)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("Source:N", legend=alt.Legend(title=""),
                            scale=alt.Scale(
                                domain=["brier_blend", "brier_market", "brier_model"],
                                range=["#4c9aff", "#888", "#ff7f50"])),
        )
        vline = alt.Chart(pd.DataFrame({"trust": [t_opt]})).mark_rule(
            color="#4c9aff", strokeDash=[4, 3]).encode(x="trust:Q")
        cur_line = alt.Chart(pd.DataFrame({"trust": [0.35]})).mark_rule(
            color="#ff4b4b", strokeDash=[4, 3]).encode(x="trust:Q")
        title = "Trust calibration curve" + ("" if cal_result.get("reliable")
                                              else f"  (early-look — n={n_tot}, noisy)")
        st.altair_chart(
            (ch + vline + cur_line).properties(height=220, title=title),
            use_container_width=True,
        )
        st.caption(
            "**Blue** = shrunk-blend Brier as trust varies · **grey** = pure market · "
            "**orange** = pure model · blue dashed = data-optimal · red dashed = current 0.35. "
            "A flat curve means the data doesn't strongly prefer one value — keep the default."
        )

    # ── Model vs market positioning (current snapshot) ─────────────────────
    if match_probs_df is not None and not match_probs_df.empty and \
            market_df is not None and not market_df.empty:
        pos = model_vs_market_positioning(match_probs_df, market_df)
        if not pos.empty:
            st.markdown("#### Current model–market disagreements")
            st.caption(
                "Where the model most disagrees with live Polymarket prices right now. "
                "Positive edge = model is higher than market (potential buy). "
                "Draws systematically skewed positive = the model over-rates draws (known bias)."
            )
            draws = pos[pos["type"] == "draw"]
            wins  = pos[pos["type"] != "draw"]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Win/loss markets** (model edge = model − market)")
                disp = wins[["selection", "model_prob", "market_price", "model_edge"]].copy()
                disp.columns = ["Selection", "Model %", "Market %", "Edge"]
                disp["Model %"]  = (disp["Model %"]  * 100).round(1)
                disp["Market %"] = (disp["Market %"] * 100).round(1)
                disp["Edge"]     = (disp["Edge"]     * 100).round(1)
                st.dataframe(disp.head(12), use_container_width=True, hide_index=True)
            with d2:
                st.markdown("**Draw markets** (known model bias: DC over-rates draws)")
                disp2 = draws[["selection", "model_prob", "market_price", "model_edge"]].copy()
                disp2.columns = ["Selection", "Model %", "Market %", "Edge"]
                disp2["Model %"]  = (disp2["Model %"]  * 100).round(1)
                disp2["Market %"] = (disp2["Market %"] * 100).round(1)
                disp2["Edge"]     = (disp2["Edge"]     * 100).round(1)
                st.dataframe(disp2, use_container_width=True, hide_index=True)

    st.divider()

    # ── Full ledger ─────────────────────────────────────────────────────────
    st.markdown("### Full ledger")
    if not ledger:
        st.info("No bets logged yet.", icon="📒")
    else:
        rows = []
        for e in ledger:
            pnl = e["pnl_zar"]
            rows.append({
                "Bet":       e["bet"],
                "Model %":   f"{e['model_p']*100:.1f}%",
                "Market %":  f"{e['market_p']*100:.1f}%",
                "Entry ¢":   f"{e['entry_cents']:.0f}",
                "Close ¢":   f"{e['closing_cents']:.0f}" if e["closing_cents"] is not None else "—",
                "CLV":       f"{e['clv_cents']:+.1f}¢" if e["clv_cents"] is not None else "—",
                "Stake (R)": f"R{e['stake_zar']:.2f}",
                "Outcome":   e["outcome"],
                "P&L (R)":   (f"+R{pnl:.2f}" if pnl and pnl > 0 else (f"−R{abs(pnl):.2f}" if pnl is not None else "—")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Equity curve ────────────────────────────────────────────────────────
    st.markdown("### Equity curve")
    if not settled:
        st.info("Settle at least one bet to see the equity curve.", icon="📈")
    else:
        starting = 400.0
        cumulative = starting
        curve = []
        for idx, e in enumerate(settled, start=1):
            cumulative += e["pnl_zar"] or 0
            curve.append({"Bet #": idx, "Bankroll (R)": round(cumulative, 2)})
        st.line_chart(pd.DataFrame(curve).set_index("Bet #")["Bankroll (R)"])

        net = cumulative - starting
        wins = sum(1 for e in settled if e["outcome"] == "Won")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Starting", f"R{starting:.0f}")
        m2.metric("Current", f"R{cumulative:.2f}", delta=f"{'+' if net >= 0 else ''}R{net:.2f}")
        m3.metric("Settled", len(settled))
        m4.metric("Win rate", f"{wins/len(settled)*100:.0f}%")

    st.divider()
    st.caption(f"Ledger saved to `{ledger_path().name}` — it survives restarts and browser reloads.")
    if ledger and st.button("🗑️ Clear ledger", type="secondary"):
        st.session_state.ledger = []
        save_ledger([])
        st.rerun()


# ============================================================================
# TAB 3 — FULL EDGE SCAN (top 20, no stake cap)
# ============================================================================
def render_edge_scan_tab(all_qualifying_df: pd.DataFrame) -> None:
    st.markdown("## Full Edge Scan")
    st.caption(
        "Every contract that passes the adjusted filter, ranked by ¼-Kelly. "
        "This is for interest — the High-Yield Picks tab already surfaces the best 5 for action. "
        "Use this to spot patterns, sector concentrations, or bets worth watching."
    )

    if all_qualifying_df.empty:
        st.warning("No qualifying contracts found with the current sidebar settings.", icon="🔍")
        return

    top20 = all_qualifying_df.head(20).copy()

    # ── Bar chart — ¼-Kelly across all picks ─────────────────────────────────
    chart_df = (
        top20[["selection", "kelly_frac", "ev_adj", "EV"]]
        .copy()
        .assign(
            Label=top20["selection"].map(lambda s: s[:35]),
            **{"¼-Kelly %": (top20["kelly_frac"] * 100).round(2)},
            **{"Adj EV %":  (top20["ev_adj"] * 100).round(2)},
            **{"Raw EV %":  (top20["EV"] * 100).round(2)},
        )
        .set_index("Label")
    )
    st.markdown("### ¼-Kelly by contract")
    st.bar_chart(chart_df["¼-Kelly %"], height=260)

    st.divider()

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown("### Top 20 by ¼-Kelly")

    tbl = top20.copy()
    tbl["Rank"]      = range(1, len(tbl) + 1)
    tbl["Model %"]   = (tbl["model_prob"] * 100).round(1)
    tbl["Used %"]    = (tbl["model_shrunk"] * 100).round(1)
    tbl["Market %"]  = (tbl["implied_prob"] * 100).round(1)
    tbl["Gap (pp)"]  = (tbl["model_prob"] * 100 - tbl["implied_prob"] * 100).round(1)
    tbl["Price (¢)"] = tbl["polymarket_cents"]
    tbl["Payout"]    = tbl["payout_multiplier"].map(lambda v: f"{v:.1f}x")
    tbl["¼-Kelly"]   = tbl["kelly_frac"].map(lambda v: f"{v*100:.2f}%")
    tbl["Adj EV"]    = tbl["ev_adj"].map(lambda v: f"{v*100:+.1f}%")
    tbl["Raw EV"]    = tbl["EV"].map(lambda v: f"{v*100:+.1f}%")
    tbl["Resolves"]  = tbl["match"].map(lambda m: resolves_label(m))

    st.dataframe(
        tbl[["Rank", "selection", "match", "Model %", "Used %", "Market %",
             "Gap (pp)", "Price (¢)", "Payout", "¼-Kelly", "Adj EV", "Raw EV", "Resolves"]]
        .rename(columns={"selection": "Contract", "match": "Market type"}),
        use_container_width=True, hide_index=True,
        column_config={
            "Model %":  st.column_config.ProgressColumn("Model %",  min_value=0, max_value=100, format="%.1f%%"),
            "Market %": st.column_config.ProgressColumn("Market %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    st.divider()

    # ── Sector breakdown ─────────────────────────────────────────────────────
    st.markdown("### Concentration by market type")
    st.caption("How many of the top 20 edges fall in each tournament stage.")
    sector = (
        all_qualifying_df.head(20)["match"]
        .value_counts()
        .reset_index()
        .rename(columns={"match": "Market type", "count": "Count"})
    )
    st.dataframe(sector, use_container_width=True, hide_index=True)
    if sector["Count"].max() > 8:
        st.caption(
            "⚠️ High concentration in one market type. The same model view is driving multiple "
            "picks — less diversification than the count suggests."
        )


# ============================================================================
# TAB 4 — DICTIONARY (ELI5 concepts)
# ============================================================================
def render_dictionary_tab() -> None:
    st.markdown("## Dictionary")
    st.caption("Plain-English explanations of every concept used in this dashboard. No maths degree required.")

    concepts = [
        (
            "Once-off event (no Law of Large Numbers)",
            "🎲",
            "The World Cup happens **once**. In normal betting you place thousands of bets and the "
            "Law of Large Numbers means a +EV edge plays out as profit *on average*. Here you get "
            "**one tournament** — maybe a handful of bets — so the average never has time to show up. "
            "A model probability is therefore a **single-shot hit chance**: 10% means a one-in-ten "
            "that this one bet *ever* lands, NOT 'wins 10% of the time'. A +EV 10% longshot is still "
            "a 90%-likely loss with no second chance to recover. That's why the strategy is built "
            "around **survival**: a hit-rate floor (drop unlikely bets), small Kelly-capped stakes, "
            "and the honest expectation that most longshots lose. The real goal this tournament is to "
            "**log bets and validate the model** (CLV/Brier) before ever betting bigger.",
        ),
        (
            "Betting horizon (time to resolution)",
            "🗓️",
            "How long until a bet settles. 'Brazil to advance from the group' resolves in ~a week; "
            "'Brazil to win the World Cup' takes the full ~month. On a small finite bankroll, capital "
            "locked in a far-off future can't be reused, and a lot can change before it settles. The "
            "**Betting Horizon slider** lets you show only bets that resolve within N days — lower it "
            "for near-term, fast-feedback bets; raise it to include the long-dated futures.",
        ),
        (
            "Polymarket & prediction markets",
            "🏪",
            "A prediction market is like a stock exchange, but instead of company shares you're buying "
            "a contract that pays **\\$1 if something happens**, and \\$0 if it doesn't. "
            "Polymarket is the biggest one. If you think Brazil will win the World Cup, you buy the "
            "'Brazil to win' contract. If they win, each contract pays \\$1. If they don't, you lose "
            "what you paid. The price of the contract right now (say 18¢) is what the crowd collectively "
            "thinks the probability is — 18¢ means 18% chance.",
        ),
        (
            "A bet as a digital option",
            "🎰",
            "The cleanest way to think about these contracts: **a Polymarket bet is a digital "
            "(binary) option**. A digital option pays a fixed amount — here **\\$1** — if an event "
            "occurs, and **\\$0** if it doesn't. So buying 'Morocco to reach the semis' at 18¢ is "
            "exactly buying a binary option struck on that outcome.\n\n"
            "Three things follow, and they're the whole strategy:\n\n"
            "• **The price IS the implied probability.** Just like an option's price encodes the "
            "market's view of the underlying, 18¢ = an 18% implied chance. No greeks, no conversion.\n\n"
            "• **Edge = buying optionality below your own probability.** If your model says 30% and "
            "the option costs 18¢, you're paying 18¢ for something you value at 30¢ — the same trade "
            "an options desk makes when implied is below their fair value. The bet stops being "
            "'gambling' and becomes 'is this contract mispriced?'\n\n"
            "• **The payoff is asymmetric.** A 20¢ contract risks 20¢ to make 80¢ — a small, "
            "lottery-like ticket on a big move, like cheap out-of-the-money optionality. That "
            "asymmetry is why sizing matters more than picking: you want lots of small, positive-edge "
            "options, not one big swing (see *Once-off event* and *Kelly fraction*).\n\n"
            "Unlike a financial option, the contract has a **known, fixed expiry and a true terminal "
            "value** (the match result) that no amount of trading can change — which actually makes it "
            "a *cleaner* test of mispricing than most assets.",
        ),
        (
            "Implied probability",
            "🔢",
            "The probability the **market believes** is baked into the current price. On Polymarket "
            "it's simply the price in cents divided by 100. A contract at **25¢** implies a **25% "
            "chance**. That's it — no conversion needed. The market is saying 'we think this happens "
            "1-in-4 times.'",
        ),
        (
            "Model probability",
            "🤖",
            "The probability **our Monte Carlo simulation** assigns to the same outcome. It runs the "
            "tournament 100,000+ times (up to 1,000,000) using a statistical model (Dixon-Coles + "
            "XGBoost + Elo ratings) and counts how often each team reaches each stage. If Brazil wins "
            "about 1 in 5 of those simulated tournaments, the model probability is ~20%. This number "
            "is our opinion; the implied probability is the market's opinion.",
        ),
        (
            "Edge / the mispricing",
            "📐",
            "The **gap between what we think will happen and what the market is pricing in**. "
            "If our model says Morocco has a 30% chance of reaching the semis, but Polymarket is "
            "only charging 18¢ (18% implied), there's a 12-percentage-point gap. We think the "
            "contract is *undervalued* — the market is paying us as if the chance is 18% when we "
            "believe it's really 30%. That gap is the edge. No gap → no bet.",
        ),
        (
            "Expected Value (EV)",
            "📊",
            "A way to measure how mispriced a contract is, as a percentage. The formula is:\n\n"
            "**EV = (our probability ÷ implied probability) − 1**\n\n"
            "Example: model says 30%, market prices at 18¢ (18% implied). "
            "EV = (0.30 ÷ 0.18) − 1 = **+67%**. That means our model thinks the contract is "
            "worth 67% more than the market is charging. "
            "EV > 0 = underpriced (buy). EV ≤ 0 = fairly priced or overpriced (skip). "
            "**Caution:** EV is only as good as the model. If the model is wrong, the EV is meaningless.",
        ),
        (
            "Adjusted EV",
            "🔧",
            "Raw EV is optimistic — it assumes our model is perfectly correct and we get the mid-price. "
            "Neither is true. Adjusted EV shrinks the number down by:\n\n"
            "- **Shrinkage** — we blend the model and market together, because we don't fully trust the model\n"
            "- **Selection haircut** — the best-looking edges are often where the model is most wrong\n"
            "- **Spread buffer** — in practice you pay a bit more than the displayed price\n\n"
            "Adjusted EV is the honest estimate. Raw EV is the ceiling.",
        ),
        (
            "Kelly Criterion",
            "🎯",
            "A formula for **how much to bet** to grow your bankroll fastest over time, without risking "
            "ruin. Bigger edge or a smaller payout → bet more. Tiny edge or a huge longshot → bet less. "
            "It naturally says 'don't throw equal money at every bet.' The formula is:\n\n"
            "**f = (odds × probability − (1 − probability)) ÷ odds**\n\n"
            "where *odds* = how much you profit per \\$1 staked. The output `f` is the fraction of "
            "your bankroll to stake. A result of 0.08 means 'bet 8% of your bankroll on this.'",
        ),
        (
            "¼-Kelly (Quarter-Kelly)",
            "🛡️",
            "Kelly assumes the probability is *exactly* right. Our model is unproven, so using full "
            "Kelly could blow up the bankroll if the edge estimates are wrong. **Quarter-Kelly bets "
            "25% of what full Kelly recommends** — you grow more slowly but survive model errors. "
            "We use it here as a *ranking* signal (higher = better bet) rather than for strict sizing, "
            "because Polymarket's minimum stake (~\\$1) already forces roughly equal splitting at R400.",
        ),
        (
            "Bounded Kelly blend",
            "⚖️",
            "A way to split the bankroll across several bets that is smarter than just dividing equally. "
            "It mixes two approaches:\n\n"
            "- **Equal split (1/N)** — every bet gets the same amount, regardless of edge\n"
            "- **Kelly-proportional** — bets with bigger edges get more of the bankroll\n\n"
            "The 'Kelly tilt' slider controls the blend (0 = pure equal, 1 = pure Kelly-proportional). "
            "A cap of 35% stops any single bet from taking too big a slice. Result: the *typical* "
            "(median) outcome improves vs. equal splitting, without putting too many eggs in one basket.",
        ),
        (
            "Closing-Line Value (CLV)",
            "📈",
            "The best long-run proof that a betting strategy actually works. When you place a bet, "
            "the price is X. By the time the match starts, the market has had more time to adjust "
            "and the price might be different — that's the *closing price*. "
            "**If you consistently buy at a lower price than the closing price** (i.e. you got in "
            "before the market moved toward your view), you are beating the closing line. "
            "Over many bets, positive CLV is the strongest evidence of real edge. "
            "P&L alone is too noisy — you could win by luck. CLV controls for luck.",
        ),
        (
            "Brier Score",
            "🎯",
            "A way to score how accurate probability estimates are. For every bet, you square the "
            "difference between the predicted probability and the actual outcome (1=happened, 0=didn't). "
            "**Lower is better.** We compare the model's Brier score against the market's Brier score. "
            "If the model's score is lower, the model is more accurate than the market. "
            "If the model's score is higher, the market is actually a better predictor — and the "
            "strategy has no demonstrated edge. We need many settled bets for this to be meaningful.",
        ),
        (
            "Favourite–longshot bias",
            "⚠️",
            "A well-known pattern in betting markets: **longshots (cheap contracts) tend to be "
            "systematically overpriced**, and favourites (expensive contracts) tend to be "
            "slightly underpriced. Punters overpay for the thrill of big payouts. "
            "This strategy specifically targets contracts under 40¢ — exactly where this bias "
            "is strongest. That means we're fishing in the zone where the market is *least* likely "
            "to be wrong, which is why the model must be validated before trusting the numbers.",
        ),
        (
            "Winner's curse / selection bias",
            "🎰",
            "When we scan 60+ contracts and pick the 5 with the biggest apparent edge, we're "
            "disproportionately picking the ones where the **model made the biggest errors** — "
            "not where the market made them. The most extreme-looking edges are often the most "
            "wrong. The 'selection haircut' in the adjusted EV tries to correct for this by "
            "shrinking the apparent edge on the best-looking picks.",
        ),
        (
            "Monte Carlo simulation",
            "🎲",
            "A way to estimate probabilities by running many random experiments. "
            "We simulate the entire WC2026 tournament 100,000+ times (up to 1,000,000). In each run, "
            "match outcomes are determined by the statistical model (which itself has a random "
            "component). We then count: in what fraction did Morocco reach the semis? If it's 15%, "
            "the model probability is 15%. "
            "More runs = a smoother distribution, but past ~100k the probabilities barely move — the "
            "main source of uncertainty is the model "
            "itself, not the number of runs.",
        ),
        (
            "Shrinkage (model trust)",
            "🤝",
            "Instead of betting on our model's probability outright, we blend it with the market's: "
            "**used probability = (model trust × model) + ((1 − model trust) × market)**. "
            "At 35% model trust (the default), the probability we actually bet is 65% market and 35% model. "
            "The idea: the market aggregates information from many sources; a single model is unlikely "
            "to be dramatically better across the board. Shrinkage keeps us humble. "
            "As the model proves itself (via Brier score / CLV), trust can be increased.",
        ),
    ]

    for name, icon, text in concepts:
        with st.expander(f"{icon}  {name}"):
            st.markdown(text)


# ============================================================================
# TAB — MATCH BETS (single-match 3-way moneyline)
# ============================================================================
def _resolves_in(days: int) -> str:
    d = int(days)
    return "today" if d <= 0 else ("tomorrow" if d == 1 else f"in {d} days")


def _match_outcome_label(selection: str) -> str:
    """Plain W/D/L label of the outcome a match contract buys."""
    s = selection.strip()
    if "draw" in s.lower():
        return "the Draw"
    if " to beat " in s:
        return f"{s.split(' to beat ')[0].strip()} to WIN"
    if " to win" in s.lower():
        return f"{s.split(' to win')[0].strip()} to WIN"
    return s


def _match_skip_reason(row, s: dict) -> str:
    """Why a scanned match outcome is / isn't a recommended bet (binding reason first)."""
    ev = float(row["ev_adj"]); days = int(row["days_to_resolve"])
    cents = float(row["polymarket_cents"]); used = float(row["model_shrunk"])
    modelp = float(row["model_prob"]); sel = str(row["selection"]).lower()
    if ev <= 0:                                     return "✗ no edge"
    if cents >= s["max_cents"]:                     return "✗ too pricey"
    if used < s["min_win_prob"]:                    return "✗ below used floor"
    if days > s["max_days"]:                        return "✗ beyond horizon"
    if s["exclude_draws"] and "draw" in sel:        return "✗ draw (model bias)"
    if modelp < s["contender_floor"]:               return "✗ longshot (below contender floor)"
    if s["exclude_inplay"] and days < 1:            return "✗ in-play today"
    return "✓ recommended"


def render_match_bets_tab(
    match_top_df: pd.DataFrame,
    match_scan_df: pd.DataFrame,
    bankroll_zar: float = 400.0,
    strategy: dict | None = None,
) -> None:
    s = strategy or {"contender_floor": 0.45, "exclude_draws": True, "exclude_inplay": True,
                     "min_win_prob": 0.20, "max_days": 40, "max_cents": 95.0}

    st.markdown("## Single-Match Bets")
    st.caption(
        "⚽ **One game's result**, not the tournament. Each contract buys a single outcome — a team "
        "**to Win** or **the Draw** (no 'Loss' bet; to back a loss, buy the *other* team's Win). "
        "Resolves in **days** — the model's calibration-validated layer."
    )
    st.info(
        f"**Disciplined filter ON** — only outcomes the model rates **≥ {s['contender_floor']:.0%}** "
        f"(genuine contenders)"
        f"{', no draws' if s['exclude_draws'] else ''}"
        f"{', no in-play games' if s['exclude_inplay'] else ''}, because the model's longshot/draw "
        "'edges' are usually its own error, not the market's. Tune in the sidebar.",
        icon="🎯",
    )

    if match_scan_df is None or match_scan_df.empty:
        st.info("No match data loaded yet.", icon="⚽")
        return

    # ── Recommended picks ─────────────────────────────────────────────────────
    n_rec = 0 if match_top_df is None else len(match_top_df)
    st.markdown(f"### Recommended bets ({n_rec})")
    if n_rec == 0:
        st.info(
            "No match outcome clears the disciplined filters right now — usually the right answer "
            "(the market is sharp on single games, and the model's 'value' is mostly draws/longshots "
            "it over-rates). Loosen the rules in the sidebar, or see the full scan below for why each "
            "was skipped.",
            icon="🔍",
        )
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Bets", n_rec)
        k2.metric("Best hit rate", f"{match_top_df['model_shrunk'].max()*100:.0f}%")
        k3.metric("Soonest", _resolves_in(match_top_df["days_to_resolve"].min()))
        if "ledger" not in st.session_state:
            st.session_state.ledger = load_ledger()
        for i, (_, row) in enumerate(match_top_df.iterrows()):
            model_raw = float(row["model_prob"]) * 100
            market = float(row["implied_prob"]) * 100
            ev_adj = float(row["ev_adj"]) * 100
            price  = float(row["polymarket_cents"])
            fair   = model_raw                       # model fair value in cents (convergence target)
            days   = int(row["days_to_resolve"])
            outcome = _match_outcome_label(row["selection"])
            slug = str(row.get("event_slug", "")).strip()
            with st.container(border=True):
                h, b = st.columns([5, 2])
                with h:
                    st.markdown(f"🛒 **Buy: {outcome}**")
                    st.caption(f"{row['selection']}  ·  🗓 resolves {_resolves_in(days)}")
                with b:
                    st.markdown(
                        f"<div style='text-align:right;display:flex;gap:6px;justify-content:flex-end'>"
                        f"<span style='background:#10B981;color:#fff;padding:2px 9px;border-radius:11px;"
                        f"font-size:0.78rem;font-weight:700'>edge +{model_raw - market:.0f}pp</span>"
                        f"<span style='background:#6B7280;color:#fff;padding:2px 9px;border-radius:11px;"
                        f"font-size:0.78rem;font-weight:700'>adj EV {ev_adj:+.0f}%</span></div>",
                        unsafe_allow_html=True,
                    )
                c1, c2, c3 = st.columns(3)
                c1.metric("Our model", f"{model_raw:.0f}%", help="The model's probability for this outcome.")
                c2.metric("Market", f"{market:.0f}%", help="Polymarket's implied probability (= the price).")
                c3.metric("Price", f"{price:.0f}¢", help="Buy YES at or below this.")
                st.caption(
                    f"🎯 **Edge-and-exit**: buy ≤ **{price:.0f}¢**, model fair value ≈ **{fair:.0f}¢**. "
                    f"If the price rises toward {fair:.0f}¢ before kickoff, **sell and bank the move** "
                    "— no need to wait for the result."
                )
                cols = st.columns([3, 2])
                with cols[0]:
                    if slug:
                        st.link_button(
                            f"⚽ Open {row['group_id'].split(':',1)[-1].replace('|',' v ')} on Polymarket ↗",
                            f"https://polymarket.com/event/{slug}", use_container_width=True)
                with cols[1]:
                    if st.button("📒 Log for CLV", key=f"logmatch_{i}", use_container_width=True,
                                 help="Record at a flat 2% unit so System Overview can track "
                                      "Closing-Line Value and Brier — the real edge test."):
                        bet_id = row["selection"]
                        if not any(e["bet"] == bet_id and e["outcome"] == "Pending"
                                   for e in st.session_state.ledger):
                            st.session_state.ledger.append({
                                "bet": bet_id, "market_type": "match",
                                "model_p": round(float(row["model_prob"]), 4),
                                "market_p": round(float(row["implied_prob"]), 4),
                                "entry_cents": round(price, 1),
                                "stake_zar": round(bankroll_zar * 0.02, 2),
                                "payout_x": round(float(row["payout_multiplier"]), 2),
                                "outcome": "Pending", "closing_cents": None,
                                "clv_cents": None, "pnl_zar": None,
                            })
                            save_ledger(st.session_state.ledger)
                            st.success("Logged. Settle it in System Overview to score CLV.")
                        else:
                            st.info("Already logged (pending).")

    # ── Full scan (every outcome + WHY each was skipped) ──────────────────────
    with st.expander("All match outcomes (full scan — and why each is / isn't a bet)"):
        scan = match_scan_df.copy()
        scan["Status"]   = scan.apply(lambda r: _match_skip_reason(r, s), axis=1)
        scan["Model %"]  = (scan["model_prob"] * 100).round(0)
        scan["Market %"] = (scan["implied_prob"] * 100).round(0)
        scan["EV"]       = scan["ev_adj"].map(lambda v: f"{v*100:+.0f}%")
        scan["Price ¢"]  = scan["polymarket_cents"].round(0)
        scan["Resolves"] = scan["days_to_resolve"].map(_resolves_in)
        has_link = "event_slug" in scan.columns
        if has_link:
            scan["Polymarket"] = scan["event_slug"].map(
                lambda x: f"https://polymarket.com/event/{x}" if str(x).strip() else None
            )
        cols = ["selection", "Model %", "Market %", "Price ¢", "EV", "Resolves", "Status"]
        if has_link:
            cols.append("Polymarket")
        st.dataframe(
            scan[cols].rename(columns={"selection": "Outcome"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Model %":  st.column_config.ProgressColumn("Model %", min_value=0, max_value=100, format="%.0f%%"),
                "Market %": st.column_config.ProgressColumn("Market %", min_value=0, max_value=100, format="%.0f%%"),
                "Polymarket": st.column_config.LinkColumn("Polymarket", display_text="open ↗"),
            },
        )
        st.caption("The **Status** column shows the binding reason each outcome is or isn't a "
                   "recommended bet — draws and longshots are filtered as likely model error.")


# ============================================================================
# COMBINED PICKS TAB — futures + single matches in one place, with blend payoff
# ============================================================================
_WEIGHTING_HELP = {
    "Probability-weighted": "More stake on the likelier winners — best chance of profit and a "
                            "positive median, at the cost of some 'if all win' upside.",
    "Equal":                "Bankroll split evenly. Most upside if longshots land, but the median "
                            "is usually a loss (you over-stake the unlikely legs).",
    "Kelly-weighted":       "Sized for long-run growth (expected log-wealth). On a one-off event "
                            "this is swingy — it can lower your chance of profit.",
}


def _blend_stakes(sel: pd.DataFrame, bankroll: float, scheme: str) -> "np.ndarray":
    """Per-leg ZAR stakes for the chosen weighting scheme (always sums to the bankroll)."""
    n = len(sel)
    if n == 0:
        return np.array([])
    if scheme == "Probability-weighted":
        w = sel["model_shrunk"].to_numpy(dtype=float)
    elif scheme == "Kelly-weighted":
        w = sel["kelly_frac"].clip(lower=0.0).to_numpy(dtype=float)
    else:  # Equal
        w = np.ones(n)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n)
    return bankroll * w / w.sum()


def render_picks_tab(
    combined_df: pd.DataFrame,
    combined_scan_df: pd.DataFrame,
    bankroll_zar: float = 400.0,
) -> None:
    st.markdown("## Picks — Futures & Single Matches")
    st.caption(
        "All +EV bets in one list. **🏆 Future** = a team's tournament-long outcome (resolves in "
        "weeks); **⚽ Match** = one game's Win/Draw (resolves in days). Ranked by hit rate — "
        "most-likely-to-resolve first. **Model %** = our (shrunk) probability; **Market %** = "
        "Polymarket's price. A bet is listed only when Model % > Market %."
    )
    st.info(
        "**Survive to learn** — once-off event: expect to lose most longshots, size small, and log "
        "your blend below so System Overview can validate the model (CLV / Brier).",
        icon="🎯",
    )

    if combined_df is None or combined_df.empty:
        st.warning(
            "No +EV bets clear the filters right now. Lower **Minimum Win Probability** or raise the "
            "**Betting Horizon** in the sidebar — or there's simply no edge today.", icon="🔍",
        )
    else:
        n_fut = int((combined_df["bet_type"] == "Future").sum())
        n_mat = int((combined_df["bet_type"] == "Match").sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Picks", len(combined_df))
        k2.metric("Futures / Matches", f"{n_fut} / {n_mat}")
        k3.metric("Best hit rate", f"{combined_df['model_shrunk'].max()*100:.0f}%")
        k4.metric("Soonest", _resolves_in(combined_df["days_to_resolve"].min()))

        tbl = combined_df.copy()
        tbl["Type"]     = tbl["bet_type"].map({"Future": "🏆 Future", "Match": "⚽ Match"})
        tbl["Model %"]  = (tbl["model_shrunk"] * 100).round(0)
        tbl["Market %"] = (tbl["implied_prob"] * 100).round(0)
        tbl["Price ¢"]  = tbl["polymarket_cents"].round(0)
        tbl["Payout"]   = tbl["payout_multiplier"].map(lambda v: f"{v:.1f}x")
        tbl["EV"]       = tbl["ev_adj"].map(lambda v: f"{v*100:+.0f}%")
        tbl["Resolves"] = tbl["days_to_resolve"].map(_resolves_in)
        cols = ["Type", "selection", "Model %", "Market %", "Price ¢", "Payout", "EV", "Resolves"]
        if "event_url" in tbl.columns:
            cols.append("event_url")
        st.dataframe(
            tbl[cols].rename(columns={"selection": "Buy", "event_url": "Polymarket"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Model %":  st.column_config.ProgressColumn("Model %", min_value=0, max_value=100, format="%.0f%%"),
                "Market %": st.column_config.ProgressColumn("Market %", min_value=0, max_value=100, format="%.0f%%"),
                "Polymarket": st.column_config.LinkColumn("Polymarket", display_text="open ↗"),
            },
        )

        # ── Quick log — track individual picks for CLV ───────────────────────
        with st.expander("📒 Log picks for CLV tracking"):
            if "ledger" not in st.session_state:
                st.session_state.ledger = load_ledger()
            pending_bets = {e["bet"] for e in st.session_state.ledger if e["outcome"] == "Pending"}
            already = [r["selection"] for _, r in combined_df.iterrows() if r["selection"] in pending_bets]
            to_log = st.multiselect(
                "Select picks to log",
                options=combined_df["selection"].tolist(),
                default=[],
                help="Select any picks you've bought — they'll appear in System Overview for CLV/Brier tracking.",
            )
            log_stake = st.number_input("Stake per bet (R)", min_value=1.0, max_value=bankroll_zar,
                                        value=round(bankroll_zar / max(len(combined_df), 1), 2), step=1.0)
            if already:
                st.caption(f"Already pending: {', '.join(already)}")
            if st.button("📒 Log selected picks", disabled=not to_log):
                added = 0
                for sel_name in to_log:
                    row = combined_df[combined_df["selection"] == sel_name]
                    if row.empty or sel_name in pending_bets:
                        continue
                    r = row.iloc[0]
                    st.session_state.ledger.append({
                        "bet": sel_name,
                        "market_type": r["bet_type"].lower(),
                        "model_p": round(float(r["model_prob"]), 4),
                        "market_p": round(float(r["implied_prob"]), 4),
                        "entry_cents": round(float(r["polymarket_cents"]), 1),
                        "stake_zar": round(float(log_stake), 2),
                        "payout_x": round(float(r["payout_multiplier"]), 2),
                        "outcome": "Pending", "closing_cents": None, "clv_cents": None, "pnl_zar": None,
                    })
                    added += 1
                save_ledger(st.session_state.ledger)
                if added:
                    st.success(f"Logged {added} bet(s). Settle them in System Overview.")
                else:
                    st.info("All selected bets already logged.")

        # ── Blend — expected payoff ──────────────────────────────────────────
        st.divider()
        st.markdown("### Blend — expected payoff")
        st.caption("Pick any mix of futures and matches, then choose how your bankroll is weighted "
                   "across them.")
        opts = combined_df["selection"].tolist()
        chosen = st.multiselect("Bets in the blend", options=opts, default=opts[:min(5, len(opts))])
        if chosen:
            sel = combined_df[combined_df["selection"].isin(chosen)].copy()
            n = len(sel)

            scheme = st.radio(
                "Stake weighting", list(_WEIGHTING_HELP.keys()), horizontal=True,
                help="How the bankroll is divided across the blend. " +
                     " · ".join(f"**{k}** — {v}" for k, v in _WEIGHTING_HELP.items()),
            )
            sel["stake_zar"] = _blend_stakes(sel, bankroll_zar, scheme)
            st.caption(f"ℹ️ {scheme}: {_WEIGHTING_HELP[scheme]}")

            exp_profit = float((sel["stake_zar"] * sel["ev_adj"]).sum())
            exp_wins   = float(sel["model_shrunk"].sum())
            if_all_win = float((sel["stake_zar"] * sel["payout_exec"]).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total staked", f"R{sel['stake_zar'].sum():.0f}",
                      help=f"Spread across {n} bets, weighted by '{scheme}'. See the Stake column below.")
            m2.metric("Expected profit", f"R{exp_profit:+.2f}", delta=f"{exp_profit/bankroll_zar*100:+.1f}%",
                      help="Σ stake × adjusted EV. Expectation is additive across futures and matches, "
                           "even though they resolve at different times.")
            m3.metric("Expected wins", f"{exp_wins:.1f} / {n}", help="Σ of the used win probabilities.")
            m4.metric("If ALL win", f"R{if_all_win:.0f}", delta=f"+R{if_all_win - bankroll_zar:.0f}")

            rows = []
            for _, r in sel.iterrows():
                rows.append({
                    "Type":        "⚽" if r["bet_type"] == "Match" else "🏆",
                    "Bet":         r["selection"],
                    "Stake":       f"R{r['stake_zar']:.2f}",
                    "Used %":      f"{r['model_shrunk']*100:.0f}%",
                    "Payout":      f"{r['payout_multiplier']:.1f}x",
                    "If it wins":  f"R{r['stake_zar'] * r['payout_exec']:.2f}",
                    "Exp. profit": f"R{r['stake_zar'] * r['ev_adj']:+.2f}",
                    "Resolves":    _resolves_in(int(r["days_to_resolve"])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Full probability distribution of the blend ───────────────────
            st.markdown("#### Full probability distribution")
            fut_legs = sel[sel["bet_type"] == "Future"]
            mat_legs = sel[sel["bet_type"] == "Match"]
            fut_wm = bundle_win_matrix(fut_legs) if not fut_legs.empty else None
            dist = simulate_blend(
                fut_legs, mat_legs,
                fut_legs["stake_zar"].to_numpy(), mat_legs["stake_zar"].to_numpy(),
                fut_win_matrix=fut_wm,
            )
            if dist:
                exp_ret_pct = dist["mean"] / dist["total_stake"] * 100 if dist["total_stake"] else 0.0
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Lose everything", f"{dist['p_lose_all']*100:.1f}%",
                          help=f"All {n} bets lose — you forfeit the full R{bankroll_zar:.0f}.")
                d2.metric("Break even or better", f"{dist['p_break_even']*100:.1f}%",
                          help="End with at least your stake back (net profit ≥ R0).")
                d3.metric("Expected return", f"{exp_ret_pct:+.1f}%",
                          delta=f"R{dist['mean']:+.0f} avg",
                          help="Average profit ÷ total staked, across all simulations. This is the "
                               "long-run mean — on a once-off event the median (below) is the more "
                               "realistic single outcome.")
                d4.metric("Hit the max payout", f"{dist['p_max']*100:.2f}%",
                          delta=f"R{dist['max_profit']:+.0f} if every bet wins")
                samples = dist["samples"]
                counts, edges = np.histogram(samples, bins=30)
                centers = ((edges[:-1] + edges[1:]) / 2).round(0)
                st.bar_chart(
                    pd.DataFrame({"Profit (R)": centers, "Frequency": counts}).set_index("Profit (R)"),
                    height=200,
                )
                st.caption(
                    f"Stakes weighted by **{scheme}**. Median outcome **R{dist['median']:+.0f}** · "
                    f"80% of the time between **R{dist['p10']:+.0f}** and **R{dist['p90']:+.0f}**. "
                    f"From {dist['n_sims']:,} simulations on the **shrunk (used) probabilities** — "
                    "futures keep their joint-matrix correlation, matches are independent. "
                    "Try switching the weighting above: *Probability-weighted* usually lifts the "
                    "median and chance of profit; *Equal* adds upside but a heavier left tail."
                )

            if st.button("📒 Log this blend for CLV", type="primary"):
                if "ledger" not in st.session_state:
                    st.session_state.ledger = load_ledger()
                added = 0
                for _, r in sel.iterrows():
                    if any(e["bet"] == r["selection"] and e["outcome"] == "Pending"
                           for e in st.session_state.ledger):
                        continue
                    st.session_state.ledger.append({
                        "bet": r["selection"], "market_type": r["bet_type"].lower(),
                        "model_p": round(float(r["model_prob"]), 4),
                        "market_p": round(float(r["implied_prob"]), 4),
                        "entry_cents": round(float(r["polymarket_cents"]), 1),
                        "stake_zar": round(float(r["stake_zar"]), 2),
                        "payout_x": round(float(r["payout_multiplier"]), 2),
                        "outcome": "Pending", "closing_cents": None, "clv_cents": None, "pnl_zar": None,
                    })
                    added += 1
                save_ledger(st.session_state.ledger)
                st.success(f"Logged {added} bet(s). Settle them in System Overview to score CLV.")

        # ── ⚙️ Optimal blend — search stakes to best meet an objective ─────────
        st.divider()
        st.markdown("### ⚙️ Optimise the blend")
        st.caption(
            "Equal/Probability/Kelly weightings sit *inside* the achievable frontier. This searches "
            "stakes across **all** your picks — choosing which to back *and* how much — over the same "
            "shrunk, correlated distribution, to best meet your objective."
        )
        _OBJ = {
            "Best chance of profit": "max_pprofit",
            "Most profit (with a floor on chance of profit)": "max_return_floor",
            "Best typical (median) outcome": "max_median",
        }
        oc1, oc2, oc3 = st.columns([2, 1, 1])
        obj_label = oc1.selectbox("Objective", list(_OBJ.keys()))
        floor = oc2.slider("Min chance of profit", 0.40, 0.85, 0.60, 0.01,
                           help="Used by the 'Most profit' objective: maximise expected profit while "
                                "keeping the chance of profit at least this high.")
        cap = oc3.slider("Max per bet", 0.20, 1.0, 0.60, 0.05,
                         help="Cap on any one bet's share of the bankroll — controls concentration / ruin.")

        fut_all = combined_df[combined_df["bet_type"] == "Future"]
        fut_wm_all = bundle_win_matrix(fut_all) if not fut_all.empty else None
        opt = optimize_blend(combined_df, fut_wm_all, bankroll_zar,
                             objective=_OBJ[obj_label], min_pprofit=floor, max_leg_frac=cap)
        if opt:
            ch = opt["chosen"]
            if _OBJ[obj_label] == "max_return_floor" and not opt["feasible_any"]:
                st.warning(f"No allocation reaches a {floor*100:.0f}% chance of profit with these picks — "
                           "showing the highest achievable instead.", icon="⚠️")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Chance of profit", f"{ch['pprofit']*100:.1f}%")
            o2.metric("Expected return", f"{ch['expret']*100:+.1f}%",
                      delta=f"R{ch['expret']*bankroll_zar:+.0f} avg")
            o3.metric("Median outcome", f"R{ch['median']:+.0f}")
            o4.metric("Lose everything", f"{ch['ploseall']*100:.1f}%")

            picks = opt["picks"]
            order = np.argsort(-ch["stakes"])
            orows = []
            for i in order:
                s = float(ch["stakes"][i])
                if s < bankroll_zar * 0.01:        # hide ~zero-stake legs the optimiser dropped
                    continue
                r = picks.iloc[i]
                orows.append({
                    "Type":       "⚽" if r["bet_type"] == "Match" else "🏆",
                    "Bet":        r["selection"],
                    "Stake":      f"R{s:.2f}",
                    "Used %":     f"{r['model_shrunk']*100:.0f}%",
                    "EV":         f"{(r['model_shrunk']*r['payout_exec']-1)*100:+.0f}%",
                    "If it wins": f"R{s*r['payout_exec']:.2f}",
                    "Resolves":   _resolves_in(int(r["days_to_resolve"])),
                })
            st.dataframe(pd.DataFrame(orows), use_container_width=True, hide_index=True)

            import altair as alt
            cloud = pd.DataFrame({"Expected return %": opt["cloud"]["return_pct"],
                                  "Chance of profit %": opt["cloud"]["pprofit_pct"]})
            front = pd.DataFrame({"Expected return %": opt["frontier"]["return_pct"],
                                  "Chance of profit %": opt["frontier"]["pprofit_pct"]}
                                 ).sort_values("Expected return %")
            chosen_pt = pd.DataFrame({"Expected return %": [ch["expret"] * 100],
                                      "Chance of profit %": [ch["pprofit"] * 100]})
            pts = alt.Chart(cloud).mark_circle(size=14, opacity=0.15, color="#8895a7").encode(
                x=alt.X("Expected return %", scale=alt.Scale(zero=False)),
                y=alt.Y("Chance of profit %", scale=alt.Scale(zero=False)))
            line = alt.Chart(front).mark_line(point=True, color="#4c9aff").encode(
                x="Expected return %", y="Chance of profit %")
            star = alt.Chart(chosen_pt).mark_point(
                size=240, shape="diamond", filled=True, color="#ff4b4b").encode(
                x="Expected return %", y="Chance of profit %")
            st.altair_chart((pts + line + star).properties(height=260), use_container_width=True)
            st.caption("Grey = candidate allocations · blue = efficient frontier (nothing beats it on "
                       "both axes) · red ◆ = your objective's pick. Optimised over "
                       f"{opt['n_sims']:,} simulations on the shrunk, correlated distribution.")

            if st.button("📒 Log this optimal blend for CLV", type="primary", key="log_opt"):
                if "ledger" not in st.session_state:
                    st.session_state.ledger = load_ledger()
                added = 0
                for i in order:
                    s = float(ch["stakes"][i])
                    if s < bankroll_zar * 0.01:
                        continue
                    r = picks.iloc[i]
                    if any(e["bet"] == r["selection"] and e["outcome"] == "Pending"
                           for e in st.session_state.ledger):
                        continue
                    st.session_state.ledger.append({
                        "bet": r["selection"], "market_type": r["bet_type"].lower(),
                        "model_p": round(float(r["model_prob"]), 4),
                        "market_p": round(float(r["implied_prob"]), 4),
                        "entry_cents": round(float(r["polymarket_cents"]), 1),
                        "stake_zar": round(s, 2),
                        "payout_x": round(float(r["payout_multiplier"]), 2),
                        "outcome": "Pending", "closing_cents": None, "clv_cents": None, "pnl_zar": None,
                    })
                    added += 1
                save_ledger(st.session_state.ledger)
                st.success(f"Logged {added} bet(s). Settle them in System Overview to score CLV.")

    # ── Full scan (everything computed, incl. no-edge / filtered) ─────────────
    if combined_scan_df is not None and not combined_scan_df.empty:
        with st.expander("All scanned outcomes (futures + matches, including no-edge)"):
            rec = set(combined_df["selection"]) if (combined_df is not None and not combined_df.empty) else set()
            scan = combined_scan_df.copy()
            scan["Bet?"]     = scan["selection"].map(lambda x: "✓" if x in rec else "·")
            scan["Type"]     = scan["bet_type"].map({"Future": "🏆", "Match": "⚽"})
            scan["Model %"]  = (scan["model_shrunk"] * 100).round(0)
            scan["Market %"] = (scan["implied_prob"] * 100).round(0)
            scan["Price ¢"]  = scan["polymarket_cents"].round(0)
            scan["EV"]       = scan["ev_adj"].map(lambda v: f"{v*100:+.0f}%")
            scan["Resolves"] = scan["days_to_resolve"].map(_resolves_in)
            scan = scan.sort_values("ev_adj", ascending=False)
            st.dataframe(
                scan[["Bet?", "Type", "selection", "Model %", "Market %", "Price ¢", "EV", "Resolves"]]
                .rename(columns={"selection": "Outcome"}),
                use_container_width=True, hide_index=True,
                column_config={
                    "Model %":  st.column_config.ProgressColumn("Model %", min_value=0, max_value=100, format="%.0f%%"),
                    "Market %": st.column_config.ProgressColumn("Market %", min_value=0, max_value=100, format="%.0f%%"),
                },
            )
            st.caption("✓ = recommended · · = filtered (no edge, longshot/draw, too pricey, in-play, "
                       "or beyond your horizon).")


# ============================================================================
# VERDICT TAB — the scientific scoreboard: did we find edge?
# ============================================================================
_VERDICT_COLOR = {"EDGE FOUND": "#1a7f37", "PROMISING": "#9a6700",
                  "NO EDGE": "#cf222e", "INCONCLUSIVE": "#57606a"}


@st.cache_data(ttl=900, show_spinner=False)
def _q1_cached(results_sig: str):
    """Q1 is stable until results change; `results_sig` (e.g. row count) keys the cache."""
    return market_efficiency_report(load_price_history(), load_results())


def render_verdict_tab() -> None:
    import altair as alt
    st.markdown("## 📋 Verdict — did we find edge?")
    st.caption(
        "A pre-specified scoreboard, not a post-hoc story. **Q1** asks whether the *market* is "
        "beatable (answerable now). **Q2** asks whether *our model* beats it (accumulates forward "
        "from the forecast log — no look-ahead). The final call is colour-coded with a confidence "
        "level driven by sample size and significance."
    )

    results = load_results()
    fl = load_forecast_log()
    n_results = len(results)
    q1 = _q1_cached(f"{n_results}") if n_results else {}
    q2 = model_edge_report(fl, results) if not fl.empty else {"n_resolved": 0, "n_logged": int(fl.shape[0])}
    verdict = final_verdict(q1, q2)

    # ── Headline verdict banner ──────────────────────────────────────────────
    color = _VERDICT_COLOR.get(verdict["verdict"], "#57606a")
    st.markdown(
        f"<div style='background:{color};border-radius:10px;padding:18px 22px;margin:6px 0 14px'>"
        f"<span style='color:#fff;font-size:1.6rem;font-weight:800'>{verdict['verdict']}</span>"
        f"<span style='color:#fff;opacity:0.85;font-size:1rem;font-weight:600'> &nbsp;·&nbsp; "
        f"confidence: {verdict['confidence']}</span><br>"
        f"<span style='color:#fff;opacity:0.92;font-size:0.92rem'>{verdict['detail']}</span></div>",
        unsafe_allow_html=True,
    )

    # progress toward a verdict (Q2 data accumulation)
    n, nmin, nstrong = verdict["n_resolved"], verdict["n_min"], verdict["n_strong"]
    pmin = min(1.0, n / nmin) if nmin else 0
    st.progress(pmin, text=f"Model-edge data: {n} resolved forecast(s) · "
                           f"{nmin} for a first read · {nstrong} for confidence")

    tab_q2, tab_q1 = st.tabs(["🎯 Q2 — Does our model beat the market?",
                              "🌍 Q1 — Is the market beatable at all?"])

    # ── Q2: model edge ───────────────────────────────────────────────────────
    with tab_q2:
        n_logged = q2.get("n_logged", 0)
        st.caption(f"Forecast log: **{n_logged}** markets tracked · **{q2.get('n_resolved',0)}** "
                   "resolved and scored. Snapshots are captured automatically each time the "
                   "dashboard loads (and by the daily task, if installed).")
        if q2.get("n_resolved", 0) == 0:
            st.info(
                "No resolved forecasts yet. The log captures the model's probability + the market "
                "price for **every** upcoming market daily; as matches finish they become scored "
                "data points here. This is forward-only — already-played matches are excluded "
                "because the current model has seen their results (look-ahead bias).", icon="⏳")
        else:
            skill = q2["brier_skill"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Resolved", q2["n_resolved"])
            c2.metric("Brier — model", f"{q2['brier_model']:.3f}")
            c3.metric("Brier — market", f"{q2['brier_market']:.3f}")
            c4.metric("Brier skill", f"{skill:+.4f}",
                      delta="model sharper" if skill > 0 else "market sharper",
                      delta_color="normal" if skill > 0 else "inverse")
            clv = q2.get("clv_test")
            paper = q2.get("paper")
            if clv:
                d1, d2, d3 = st.columns(3)
                d1.metric("Backable markets", q2["n_backable"],
                          help="Markets where the model exceeded the market by ≥5pp — the ones we'd bet.")
                d2.metric("Mean CLV", f"{clv['mean']:+.2f}¢",
                          delta=f"p={clv['p']:.3f}",
                          delta_color="normal" if clv['mean'] > 0 and clv['p'] < 0.1 else "off",
                          help="Closing-line value: did the market move toward our entry? "
                               "Positive + significant is the strongest proof of edge.")
                if paper:
                    d3.metric("Paper-trade ROI", f"{paper['roi']*100:+.1f}%",
                              delta=f"{paper['wins']}/{paper['n']} won",
                              help="Flat R1 on every backable market, settled at outcome. "
                                   "Realised return, not expectation.")
            rel = q2.get("reliability")
            if rel is not None and not rel.empty:
                st.markdown("**Model reliability** — predicted vs observed (is a 45% pick a real 45%?)")
                st.dataframe(rel, use_container_width=True, hide_index=True)

    # ── Q1: market efficiency ─────────────────────────────────────────────────
    with tab_q1:
        if not q1:
            st.info("No results / price history available yet.", icon="🌍")
        else:
            st.caption(f"Scored on **{q1['n_markets']}** market outcomes from **{q1['n_matches']}** "
                       "resolved matches — using real Polymarket open/close prices.")
            b1, b2, b3 = st.columns(3)
            b1.metric("Market Brier (close)", f"{q1['brier_close']:.3f}")
            b2.metric("Market Brier (open)", f"{q1['brier_open']:.3f}")
            b3.metric("Naive 1/3 baseline", f"{q1['brier_naive']:.3f}",
                      delta="market is informative" if q1['brier_close'] < q1['brier_naive'] else "no better than guessing",
                      delta_color="off")

            st.markdown("**Favourite–longshot bias** — implied price vs what actually happened")
            band = q1["band_df"].copy()
            chart = alt.Chart(band.assign(idx=range(len(band)))).transform_fold(
                ["Implied %", "Observed %"], as_=["Series", "pct"]).mark_bar().encode(
                x=alt.X("Price band:N", sort=list(band["Price band"]), title=None),
                xOffset="Series:N",
                y=alt.Y("pct:Q", title="%"),
                color=alt.Color("Series:N", scale=alt.Scale(
                    domain=["Implied %", "Observed %"], range=["#888", "#4c9aff"])),
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(band, use_container_width=True, hide_index=True)
            st.caption("If observed > implied in a band, that band is **under-priced** (value); "
                       "observed < implied means **over-priced**.")

            dt = q1.get("draw_test")
            if dt:
                sig = "✅ significant" if dt["p"] < 0.05 else ("≈ borderline" if dt["p"] < 0.15 else "✗ not significant")
                st.markdown("**Draw mispricing test**")
                st.write(f"Across {dt['n']} matches, draws actually occurred **{dt['actual_rate']*100:.1f}%** "
                         f"of the time, but the market opened them at **{dt['market_open']*100:.1f}%** "
                         f"— a {(dt['actual_rate']-dt['market_open'])*100:+.1f}pp gap (p={dt['p']:.3f}, **{sig}**).")
                if dt["p"] >= 0.15:
                    st.caption("⚠️ The gap is real in-sample but the sample is too small to rule out "
                               "luck. Treat the draw edge as a hypothesis, not a proven bias, until "
                               "more matches resolve.")
            dy, dn = q1.get("drift_yes"), q1.get("drift_no")
            if dy and dn:
                st.markdown("**Open→close drift** (Moskowitz momentum test)")
                st.write(f"Markets that ended YES drifted **{dy['mean']*100:+.1f}pp** open→close "
                         f"(p={dy['p']:.2f}); those that ended NO drifted **{dn['mean']*100:+.1f}pp** "
                         f"(p={dn['p']:.2f}). Large opposite drifts would mean the market 'learns' "
                         "pre-match; near-zero means it's already efficient at open.")

    st.divider()
    st.caption(f"Forecast log: `{FORECAST_LOG_PATH.name}`. Verdict thresholds: ≥{verdict['n_min']} "
               f"resolved for a first read, ≥{verdict['n_strong']} for confidence. Q1 cached 15 min.")
