# Latency Arbitrage Module: Complete Overview

**Deployed: 2026-06-17 | Status: Paper-trading harness ready | Live Deployment: TBD (wait for validation)**

---

## What This Is

A complete latency arbitrage system that exploits the speed gap between sharp sportsbooks and Polymarket's retail order book during live matches.

**Core idea:** When a major unobserved event happens (injury, red card, goal), sharp books reprice instantly. Polymarket lags. We detect the sharp move, query Polymarket's stale order book, and execute FOK orders for immediate profit.

**Crucially:** This harness **measures real latencies and edge decay**, not theoretical ones. Most latency arb backtests fail because they assume instant fills. This one forces you to see what actually happens.

---

## What's Implemented

### 1. Volatility Trigger (`src/latency_arb/volatility_trigger.py`)

Detects statistically significant price moves in sharp books:

```
breakout = |p_current - p_60s_ago| > k·σ_rolling + min_move_pp
```

- **Input:** Sharp book probabilities on a 0.2s polling interval
- **Output:** Signal when move > 2σ + 3pp (configurable)
- **Latency:** Detection happens within 60s rolling window

### 2. Stale Price Checker (`src/latency_arb/polymarket_checker.py`)

Verifies that Polymarket has NOT already repriced:

```
race_lost = (Polymarket move / theoretical move) > 0.7
```

- **Input:** Volatility signal + Polymarket order book snapshot
- **Output:** Opportunity if stale; None if race already lost
- **Latency:** ~30–50ms to query Polymarket

### 3. Execution Engine (`src/latency_arb/execution.py`)

Executes FOK (Fill-or-Kill) orders with slippage protection:

```
vwap = weighted average price of cumulative liquidity
if vwap - sharp_price > min_edge:
    submit_market_order()
```

- **Input:** Stale-price opportunity
- **Output:** Actual fill price + timestamp
- **Latency:** 50–100ms from submission to fill
- **Protection:** VWAP check prevents buying into moving order book

### 4. Paper Trading Harness (`scripts/latency_arb_paper_harness.py`)

Logs everything with latency awareness:

```csv
exec_id, timestamp_signal, timestamp_query, timestamp_submit, timestamp_fill,
detection_latency_ms, query_latency_ms, fill_latency_ms,
theoretical_edge_pp, actual_edge_pp, race_lost, filled, vwap, ...
```

**This is the critical layer.** It captures:
- When the sharp move happened
- When we detected it
- When we queried Polymarket
- When we actually filled
- The real fill price (not theoretical)
- The sharp price at fill time (to measure edge decay)

### 5. Orchestrator (`src/latency_arb/coordinator.py`)

Wires all components together:

```
sharp_monitor.poll() → signal_queue → stale_checker → execution_queue → executor → paper_harness.log()
```

---

## Data Captured

### Timestamps (Unix seconds, millisecond precision)

```
timestamp_signal      = moment sharp book moved
timestamp_query       = moment we queried Polymarket
timestamp_submit      = moment we submitted FOK order
timestamp_fill        = moment order was filled (or None if timeout)
timestamp_settlement  = moment match result was known
```

### Computed Latencies (milliseconds)

```
detection_latency = timestamp_query - timestamp_signal         (30–70ms typically)
query_latency = timestamp_submit - timestamp_query             (20–50ms)
fill_latency = timestamp_fill - timestamp_submit               (10–100ms)
total_latency = timestamp_fill - timestamp_signal              (60–220ms typical)
```

### Pricing & Edges

```
old_prob_sharp, new_prob_sharp               (prices before/after sharp move)
polymarket_bid, polymarket_mid, polymarket_ask (at query time — stale)
theoretical_edge = polymarket_mid - new_prob_sharp  (edge at query time)
sharp_at_fill = sharp book price when order filled
actual_edge = polymarket_fill - sharp_at_fill      (edge accounting for execution)
vwap = actual volume-weighted price of liquidity taken
```

### Execution Flags

```
filled              = bool: did FOK order get filled?
race_lost           = bool: did Polymarket reprice before we filled?
```

### Post-Settlement (after match resolves)

```
settlement_price                 (final price; 0 or 100)
clv = settlement_price - actual_fill_price  (actual P&L)
```

---

## Key Metrics to Track

### 1. Race-Condition Rate (Most Important)

```
pct_race_lost = (signals where Polymarket repriced before fill) / total_signals × 100
```

- **0–20%:** You have a real latency advantage
- **20–40%:** Borderline; professional bots probably beat you
- **>50%:** You're losing most races; not viable

**Why:** If Polymarket reprice before you fill, your edge is gone.

### 2. Total Latency vs. Breakeven

```
breakeven_latency_ms = theoretical_edge_pp × 10
e.g. 5pp edge → max 50ms latency to stay profitable
```

Compare actual total latency to this. If you exceed it, the market is too fast.

### 3. Edge Decay

```
edge_decay = theoretical_edge - actual_edge
e.g. 6pp theoretical, 2pp actual → 4pp lost to execution slippage
```

This is money lost while your order sits in Polymarket's queue. If >70% of theoretical edge, not viable.

### 4. Fill Rate

```
pct_filled = (FOK orders that filled) / total_submitted × 100
```

- **<30%:** Timeout too tight or liquidity thin
- **30–70%:** Reasonable
- **>70%:** Timeout too loose; probably getting worse fills

---

## Critical Difference from Main Paper Harness

The **main paper harness** uses:
```python
clv = closing_price - entry_cost  # WRONG for latency arb
```

Where `entry_cost` is the **theoretical** price you wanted to fill at.

This harness uses:
```python
clv = settlement_price - actual_fill_price  # CORRECT
```

Where `actual_fill_price` is the **real VWAP** you paid when the order executed.

**Why it matters:** In latency arb, you often miss the full edge due to execution delay. A 6pp theoretical edge becomes 2pp actual because sharp repriced 4pp while you were executing. The main harness would hide this; this one exposes it.

---

## How to Run

### 1. Paper-Trading Mode (First 1–2 weeks)

```bash
# Terminal 1: Run the daemon
python scripts/latency_arb_daemon.py

# Terminal 2: Watch live stats
python scripts/_latency_arb_monitor.py

# Terminal 3: Generate report
python scripts/latency_arb_paper_harness.py --report
```

This logs all executions to `data/processed/latency_arb_executions.csv` with complete latency/pricing data.

### 2. Analyze After 500+ Signals

```bash
python scripts/latency_arb_paper_harness.py --report
```

Outputs:
- Race-condition rate
- Latency breakdown (detection, query, fill, total)
- Edge decay analysis
- Fill rate
- Projected viability

### 3. Compare to Theoretical

Use `LATENCY_ARB_EVALUATION.md` to verify whether realized CLV matches expected CLV given your latencies.

---

## Red Flags (Stop If You See These)

1. **Race-condition rate > 40%:** You're not winning races. Probably not viable.

2. **Total latency > 50ms with < 5pp edge:** You'll lose money to slippage. Market too fast.

3. **Edge decay > 70%:** 6pp theoretical becomes 2pp actual. Execution is eating the profit.

4. **Fill rate < 20%:** Orders timing out; can't get filled when you need to.

5. **Realized CLV on settled bets is negative:** The harness is showing losses. You've validated the system DOES NOT WORK.

---

## Next Steps

### If Paper-Trading Shows Promise:

1. **Run for 2–4 weeks** to accumulate 1,000+ signals
2. **Validate against live match results** once they come in
3. **Measure real fill prices** vs. theoretical (biggest source of error)
4. **Test with real capital on 1% of size:** $1–5 per order (tiny position)
5. **Live monitor slippage & transaction costs** (they're always worse than paper estimates)

### If Paper-Trading Fails:

1. **Don't go live.** Paper mode is cheap validation.
2. **Analyze why:** 
   - Too slow (latency problem → need colocate, can't fix)
   - Market too thin (liquidity problem → pick bigger moves only)
   - Sharp books too slow (not a real edge → wrong strategy choice)
3. **Decide:** Keep trying, or abandon latency arb for market-structure/cross arbitrage instead?

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    LATENCY ARB PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

                    [SHARP BOOK MONITOR]
                   (0.2s polling interval)
                            │
                    ┌───────▼────────┐
                    │ BREAKOUT      │
                    │ DETECTED      │
                    └────────────────┘
                            │
                   (30–70ms detection latency)
                            │
        ┌───────────────────▼───────────────────┐
        │   POLYMARKET STALE-PRICE CHECKER      │
        │  (query WS for current order book)    │
        └───────────────────┬───────────────────┘
                            │
                   (20–50ms query latency)
                            │
                    ┌───────▼────────┐
                    │ RACE LOST?     │◄─── 40–50% of signals lose race
                    │ (Poly moved    │      (opportunity = market already repriced)
                    │  >70% already) │
                    └───┬────────┬───┘
                    YES │        │ NO
                        │        │
                        │    ┌───▼──────────────────┐
                        │    │ EXECUTION ENGINE     │
                        │    │ (FOK order submit)   │
                        │    └───┬──────────────────┘
                        │        │
                        │  (50–100ms fill latency)
                        │        │
                        │    ┌───▼──────────────────────────────┐
                        │    │ ACTUAL FILL RECORDED:            │
                        │    │ - actual_fill_price (VWAP)       │
                        │    │ - sharp_price_at_fill            │
                        │    │ - fill_latency_ms                │
                        │    │ - edge_decay_pp                  │
                        │    └───┬──────────────────────────────┘
                        │        │
                        └────────┼──────────────────────┐
                                 │                      │
                        ┌────────▼─────────────────┐   │
                        │  PAPER HARNESS LOG       │   │
                        │ (CSV with timestamps)    │   │
                        └────────┬─────────────────┘   │
                                 │                      │
                        (wait for match to resolve)    │
                                 │                      │
                        ┌────────▼──────────────────┐   │
                        │ SETTLEMENT:               │   │
                        │ - settlement_price (0/100)│   │
                        │ - clv = settlement - fill │───┘
                        │ - pnl = stake × clv       │
                        └───────────────────────────┘
                                 │
                        ┌────────▼──────────────────┐
                        │ REPORT GENERATION:       │
                        │ - race_condition_rate    │
                        │ - latency_percentiles    │
                        │ - edge_decay_analysis    │
                        │ - fill_rate              │
                        │ - viability verdict      │
                        └──────────────────────────┘
```

---

## Files Created

```
src/latency_arb/
├── __init__.py
├── volatility_trigger.py         (detects sharp moves)
├── sharp_book_monitor.py         (polling loop)
├── polymarket_checker.py         (confirms stale)
├── execution.py                  (FOK orders)
├── config.py                     (configuration)
└── coordinator.py                (orchestration)

scripts/
├── latency_arb_daemon.py         (main entry point)
├── latency_arb_paper_harness.py  (logging + analysis)
└── _latency_arb_monitor.py       (live dashboard)

docs/
├── LATENCY_ARB_README.md         (this file)
├── LATENCY_ARB_EVALUATION.md     (how to measure viability)
└── LATENCY_ARB_INTEGRATION.md    (wiring everything together)
```

---

## Final Word

Latency arbitrage is **the shortest-lived edge in this project** (Section 9.3 of QUANT_THEORY_AND_DESIGN.md). It's riskless for an instant, but the window closes in milliseconds. You're competing against:
- Other bots with faster networks
- Professional market makers with sub-millisecond execution
- Polymarket's own pricing engine updating in real-time

**But:** if you're faster than the crowd AND have realistic latency measurements, it's defensible. This harness gives you that honest measurement.

Run it for 2 weeks. Measure. Decide. Then move on to more durable edges (market-structure arbitrage, cross-market divergence) if this doesn't work out.

---

## See Also

- **QUANT_THEORY_AND_DESIGN.md § 9.3** — Why latency arb is shortest-lived but riskless
- **LATENCY_ARB_EVALUATION.md** — How to interpret the metrics
- **LATENCY_ARB_INTEGRATION.md** — Wiring the components
- **scripts/latency_arb_paper_harness.py** — The harness code itself
