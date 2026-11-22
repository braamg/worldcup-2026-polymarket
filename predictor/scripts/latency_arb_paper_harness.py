"""
latency_arb_paper_harness.py — specialized paper-trading harness for latency arbitrage.

Unlike the main paper_harness (which assumes instant fills), this harness:
  1. Captures volatility-trigger detection latency (how long from sharp move to detection)
  2. Logs actual order-fill prices & times, not theoretical entry costs
  3. Measures stale-price lifespan (how long was Polymarket stale before we filled)
  4. Tracks race-condition rate (% of opportunities lost to Polymarket repricing)
  5. Computes real CLV using the actual fill time, not the entry signal time

The core idea: in latency arb, edge is TIME-DEPENDENT. A 5pp theoretical edge is worth
ZERO if the order takes 150ms to fill and Polymarket reprices in 50ms. This harness
quantifies that decay.

Usage:
  python scripts/latency_arb_paper_harness.py --live   # run the arb daemon in paper mode
  python scripts/latency_arb_paper_harness.py --report # analyze execution log
"""

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
LATENCY_ARB_LOG = DATA / 'latency_arb_executions.csv'
LATENCY_ARB_STATS = DATA / 'latency_arb_stats.json'


# ============================================================================
# Data classes for latency-aware execution tracking
# ============================================================================

@dataclass
class VolatilitySignal:
    """Captured when volatility trigger fires."""
    signal_id: str
    timestamp_signal: float  # Unix timestamp (seconds with millisecond precision)
    market_id: str
    old_prob_sharp: float
    new_prob_sharp: float
    move_pp: float
    direction: str  # 'up' or 'down'


@dataclass
class PolymarketQueryResult:
    """Captured when we query Polymarket order book."""
    timestamp_query: float  # Unix timestamp
    polymarket_mid: float
    polymarket_bid: float
    polymarket_ask: float
    polymarket_stale_vs_sharp: float  # how far behind is Polymarket's price


@dataclass
class OrderExecution:
    """Captured at order submission and fill."""
    timestamp_submit: float  # When we sent the order
    timestamp_fill: Optional[float] = None  # When we got filled (None if FOK timeout)
    timestamp_settlement: Optional[float] = None  # When match result was known

    actual_fill_price: Optional[float] = None
    actual_fill_size: Optional[float] = None
    vwap: Optional[float] = None  # Volume-weighted average price of the liquidity we took

    # For post-hoc analysis
    sharp_prob_at_fill: Optional[float] = None  # Sharp price when we filled
    polymarket_price_at_fill: Optional[float] = None  # Polymarket price at our fill

    filled: bool = False
    race_lost: bool = False  # True if Polymarket repriced before we filled


@dataclass
class LatencyArbExecution:
    """Complete record of one latency arb opportunity from detection through settlement."""
    exec_id: str
    signal: VolatilitySignal
    query_result: PolymarketQueryResult
    order: OrderExecution

    # Computed latencies (milliseconds)
    detection_latency_ms: float = field(init=False)
    query_latency_ms: float = field(init=False)
    fill_latency_ms: Optional[float] = field(init=False, default=None)

    # Computed edges
    theoretical_edge_pp: float = field(init=False)  # edge at time of Polymarket query
    actual_edge_pp: Optional[float] = field(init=False, default=None)  # edge accounting for fill time

    # CLV (Closing Line Value) — using ACTUAL fill time, not entry time
    clv_pp: Optional[float] = field(init=False, default=None)  # Settlement price vs actual fill price
    clv_dollar: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        # Latencies (in milliseconds)
        self.detection_latency_ms = (self.query_result.timestamp_query - self.signal.timestamp_signal) * 1000
        self.query_latency_ms = (
            (self.order.timestamp_submit - self.query_result.timestamp_query) * 1000
        )
        if self.order.timestamp_fill:
            self.fill_latency_ms = (self.order.timestamp_fill - self.order.timestamp_submit) * 1000

        # Theoretical edge at time of Polymarket query
        # (this is what we thought the edge was when we checked the stale price)
        self.theoretical_edge_pp = self.query_result.polymarket_mid - self.query_result.polymarket_stale_vs_sharp

        # Actual edge accounting for fill time
        # (if sharp repriced while we were executing, our edge shrinks)
        if self.order.filled and self.order.sharp_prob_at_fill is not None:
            self.actual_edge_pp = abs(self.order.polymarket_price_at_fill - self.order.sharp_prob_at_fill)

        # CLV: settlement price vs actual fill price
        # (this is the realized edge after the match resolved)
        # TODO: when you settle, populate order.timestamp_settlement and compute clv


class LatencyArbPaperTrader:
    """
    Orchestrates latency arb execution in paper-trading mode.

    Simulates the full latency-aware pipeline:
      1. Sharp book monitor detects volatility
      2. Query Polymarket order book with realistic latency
      3. Decide whether to execute FOK
      4. Log actual fill (simulated) with realistic latencies
    """

    def __init__(self, execution_log_path: Path = LATENCY_ARB_LOG):
        self.log_path = execution_log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV with headers if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
                writer.writeheader()

        self.executions = []

    @staticmethod
    def _get_fieldnames() -> list[str]:
        """Column names for the execution log CSV."""
        return [
            # IDs
            'exec_id', 'market_id', 'direction',

            # Timestamps (Unix seconds)
            'timestamp_signal', 'timestamp_query', 'timestamp_submit', 'timestamp_fill', 'timestamp_settlement',

            # Latencies (milliseconds)
            'detection_latency_ms', 'query_latency_ms', 'fill_latency_ms',

            # Pricing at signal time
            'old_prob_sharp', 'new_prob_sharp', 'move_pp',

            # Polymarket state at query time
            'polymarket_mid', 'polymarket_bid', 'polymarket_ask', 'polymarket_stale_vs_sharp',

            # Edges
            'theoretical_edge_pp', 'actual_edge_pp',

            # Execution
            'filled', 'race_lost', 'actual_fill_price', 'actual_fill_size', 'vwap',
            'sharp_prob_at_fill', 'polymarket_price_at_fill',

            # Outcome (post-settlement)
            'clv_pp', 'clv_dollar',
        ]

    async def process_breakout(
        self,
        signal: VolatilitySignal,
        polymarket_state: dict,
        sharp_current: dict,
    ) -> Optional[LatencyArbExecution]:
        """
        Simulate the full latency-arb pipeline for one breakout signal.

        Args:
            signal: VolatilitySignal from the sharp book monitor
            polymarket_state: {'mid': float, 'bid': float, 'ask': float}
            sharp_current: {'mid': float} current sharp fair value

        Returns:
            LatencyArbExecution object if processing completed, else None on race condition
        """
        exec_id = f"arb_{int(signal.timestamp_signal * 1000) % 1000000}"

        # Step 1: Simulate latency to query Polymarket (typically 20-50ms)
        await asyncio.sleep(0.03)  # 30ms simulation
        query_ts = datetime.now(timezone.utc).timestamp()

        query_result = PolymarketQueryResult(
            timestamp_query=query_ts,
            polymarket_mid=polymarket_state['mid'],
            polymarket_bid=polymarket_state['bid'],
            polymarket_ask=polymarket_state['ask'],
            polymarket_stale_vs_sharp=polymarket_state['mid'] - sharp_current['mid'],
        )

        # Step 2: Check if Polymarket is still stale (race condition detection)
        # If Polymarket has already moved >70% of the way to the new sharp price, abort
        move_magnitude = abs(signal.new_prob_sharp - signal.old_prob_sharp)
        polymarket_move = abs(query_result.polymarket_mid - signal.old_prob_sharp)

        race_lost = polymarket_move > 0.7 * move_magnitude
        if race_lost:
            # Log the missed opportunity
            order = OrderExecution(
                timestamp_submit=query_ts,
                race_lost=True,
                filled=False,
            )
            execution = LatencyArbExecution(exec_id, signal, query_result, order)
            self.executions.append(execution)
            self._write_execution(execution)
            return execution

        # Step 3: Simulate order submission and fill
        # Typically 10-80ms depending on network and Polymarket's queue depth
        await asyncio.sleep(0.05)  # 50ms simulation
        submit_ts = datetime.now(timezone.utc).timestamp()

        # Simulate fill: we take liquidity at the ask (if going long) or bid (if going short)
        side = 'short' if signal.direction == 'down' else 'long'
        if side == 'long':
            fill_price = query_result.polymarket_ask
        else:
            fill_price = query_result.polymarket_bid

        # Simulate 20-40ms to fill
        await asyncio.sleep(0.03)
        fill_ts = datetime.now(timezone.utc).timestamp()

        order = OrderExecution(
            timestamp_submit=submit_ts,
            timestamp_fill=fill_ts,
            actual_fill_price=fill_price,
            actual_fill_size=100.0,  # Placeholder
            vwap=fill_price,  # Simplified; real VWAP would walk the book
            sharp_prob_at_fill=sharp_current['mid'],  # In paper mode, sharp hasn't moved much
            polymarket_price_at_fill=fill_price,
            filled=True,
            race_lost=False,
        )

        execution = LatencyArbExecution(exec_id, signal, query_result, order)
        self.executions.append(execution)
        self._write_execution(execution)

        return execution

    def _write_execution(self, execution: LatencyArbExecution) -> None:
        """Append one execution record to the CSV log."""
        row = asdict(execution.signal)
        row.update(asdict(execution.query_result))
        row.update(asdict(execution.order))
        row.update({
            'exec_id': execution.exec_id,
            'market_id': execution.signal.market_id,
            'detection_latency_ms': execution.detection_latency_ms,
            'query_latency_ms': execution.query_latency_ms,
            'fill_latency_ms': execution.fill_latency_ms,
            'theoretical_edge_pp': execution.theoretical_edge_pp,
            'actual_edge_pp': execution.actual_edge_pp,
            'clv_pp': execution.clv_pp,
            'clv_dollar': execution.clv_dollar,
        })

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
            # Filter to only fields in fieldnames
            filtered_row = {k: row.get(k) for k in self._get_fieldnames()}
            writer.writerow(filtered_row)

    def generate_report(self) -> dict:
        """Generate summary statistics from the execution log."""
        if not self.log_path.exists():
            return {'status': 'no_executions_yet'}

        df = pd.read_csv(self.log_path)
        if df.empty:
            return {'status': 'no_executions_yet'}

        # Filter to executions that actually attempted to fill
        attempted = df[df['timestamp_submit'].notna()].copy()
        filled = df[df['filled'] == True].copy()

        report = {
            'total_signals': len(df),
            'total_attempts': len(attempted),
            'total_filled': len(filled),
            'race_condition_rate': (
                (df['race_lost'].sum() / len(attempted) * 100) if len(attempted) > 0 else 0
            ),
            'fill_rate': (
                (len(filled) / len(attempted) * 100) if len(attempted) > 0 else 0
            ),

            # Latency statistics (milliseconds)
            'detection_latency_stats': {
                'mean_ms': float(df['detection_latency_ms'].mean()) if len(df) > 0 else 0,
                'p50_ms': float(df['detection_latency_ms'].median()) if len(df) > 0 else 0,
                'p95_ms': float(df['detection_latency_ms'].quantile(0.95)) if len(df) > 0 else 0,
                'max_ms': float(df['detection_latency_ms'].max()) if len(df) > 0 else 0,
            },
            'query_latency_stats': {
                'mean_ms': float(attempted['query_latency_ms'].mean()) if len(attempted) > 0 else 0,
                'p50_ms': float(attempted['query_latency_ms'].median()) if len(attempted) > 0 else 0,
                'p95_ms': float(attempted['query_latency_ms'].quantile(0.95)) if len(attempted) > 0 else 0,
                'max_ms': float(attempted['query_latency_ms'].max()) if len(attempted) > 0 else 0,
            },
            'fill_latency_stats': {
                'mean_ms': float(filled['fill_latency_ms'].mean()) if len(filled) > 0 else 0,
                'p50_ms': float(filled['fill_latency_ms'].median()) if len(filled) > 0 else 0,
                'p95_ms': float(filled['fill_latency_ms'].quantile(0.95)) if len(filled) > 0 else 0,
                'max_ms': float(filled['fill_latency_ms'].max()) if len(filled) > 0 else 0,
            },

            # Edge statistics (percentage points)
            'theoretical_edge_stats': {
                'mean_pp': float(attempted['theoretical_edge_pp'].mean()) if len(attempted) > 0 else 0,
                'median_pp': float(attempted['theoretical_edge_pp'].median()) if len(attempted) > 0 else 0,
                'min_pp': float(attempted['theoretical_edge_pp'].min()) if len(attempted) > 0 else 0,
                'max_pp': float(attempted['theoretical_edge_pp'].max()) if len(attempted) > 0 else 0,
            },
            'actual_edge_stats': {
                'mean_pp': float(filled['actual_edge_pp'].mean()) if len(filled) > 0 else 0,
                'median_pp': float(filled['actual_edge_pp'].median()) if len(filled) > 0 else 0,
                'min_pp': float(filled['actual_edge_pp'].min()) if len(filled) > 0 else 0,
                'max_pp': float(filled['actual_edge_pp'].max()) if len(filled) > 0 else 0,
            },

            # Edge decay from submission to fill
            'edge_decay': {
                'mean_pp': (
                    float((attempted['theoretical_edge_pp'] - filled['actual_edge_pp']).mean())
                    if len(filled) > 0 else 0
                ),
                'explanation': 'Positive = edge shrank while we were executing (losing money to slippage/repricing)',
            },
        }

        return report


def print_report(trader: LatencyArbPaperTrader) -> None:
    """Pretty-print the latency arb report."""
    stats = trader.generate_report()

    if stats['status'] == 'no_executions_yet':
        print("No executions logged yet.")
        return

    print(f"""
{'='*80}
LATENCY ARBITRAGE PAPER-TRADING REPORT
{'='*80}

EXECUTION SUMMARY
  Total signals detected:     {stats['total_signals']:>6}
  Attempted (checked stale):  {stats['total_attempts']:>6}
  Filled:                     {stats['total_filled']:>6}
  Fill rate:                  {stats['fill_rate']:>6.1f}%
  Race-condition rate:        {stats['race_condition_rate']:>6.1f}%  ← Key metric: how often did Polymarket reprice before us?

LATENCY BREAKDOWN (milliseconds)
  Detection (sharp→detect):
    Mean:  {stats['detection_latency_stats']['mean_ms']:>6.0f} ms
    P50:   {stats['detection_latency_stats']['p50_ms']:>6.0f} ms
    P95:   {stats['detection_latency_stats']['p95_ms']:>6.0f} ms
    Max:   {stats['detection_latency_stats']['max_ms']:>6.0f} ms

  Query (detect→Polymarket query):
    Mean:  {stats['query_latency_stats']['mean_ms']:>6.0f} ms
    P50:   {stats['query_latency_stats']['p50_ms']:>6.0f} ms
    P95:   {stats['query_latency_stats']['p95_ms']:>6.0f} ms
    Max:   {stats['query_latency_stats']['max_ms']:>6.0f} ms

  Fill (submit→filled):
    Mean:  {stats['fill_latency_stats']['mean_ms']:>6.0f} ms
    P50:   {stats['fill_latency_stats']['p50_ms']:>6.0f} ms
    P95:   {stats['fill_latency_stats']['p95_ms']:>6.0f} ms
    Max:   {stats['fill_latency_stats']['max_ms']:>6.0f} ms

  TOTAL (signal→fill):
    Mean:  {stats['detection_latency_stats']['mean_ms'] + stats['query_latency_stats']['mean_ms'] + stats['fill_latency_stats']['mean_ms']:>6.0f} ms
    P95:   {stats['detection_latency_stats']['p95_ms'] + stats['query_latency_stats']['p95_ms'] + stats['fill_latency_stats']['p95_ms']:>6.0f} ms

EDGE ANALYSIS (percentage points)
  Theoretical edge at query time:
    Mean:   {stats['theoretical_edge_stats']['mean_pp']:>6.3f} pp
    Median: {stats['theoretical_edge_stats']['median_pp']:>6.3f} pp
    Range:  {stats['theoretical_edge_stats']['min_pp']:>6.3f} – {stats['theoretical_edge_stats']['max_pp']:<6.3f} pp

  Actual edge at fill time:
    Mean:   {stats['actual_edge_stats']['mean_pp']:>6.3f} pp
    Median: {stats['actual_edge_stats']['median_pp']:>6.3f} pp
    Range:  {stats['actual_edge_stats']['min_pp']:>6.3f} – {stats['actual_edge_stats']['max_pp']:<6.3f} pp

  Edge decay (slippage from submission to fill):
    Mean:   {stats['edge_decay']['mean_pp']:>6.3f} pp
    Why: As you execute, Polymarket's order book moves against you, eating into the edge.

{'='*80}
KEY INSIGHTS
{'='*80}

1. Race-condition rate = {stats['race_condition_rate']:.1f}%
   → This is your core success metric. If > 50%, you're losing most races (probably not viable).
   → If < 20%, you have a real latency advantage.

2. Total latency (signal → fill) = {stats['detection_latency_stats']['mean_ms'] + stats['query_latency_stats']['mean_ms'] + stats['fill_latency_stats']['mean_ms']:.0f} ms (mean)
   → Breakeven speed: if total latency > (theoretical edge × 1000) ms, you're already losing money.
   → Example: 5pp edge = max 50ms latency to profit. You're at {stats['detection_latency_stats']['mean_ms'] + stats['query_latency_stats']['mean_ms'] + stats['fill_latency_stats']['mean_ms']:.0f}ms.

3. Edge decay = {stats['edge_decay']['mean_pp']:.3f} pp during execution
   → This is slippage from order submission to fill. It's money lost to Polymarket's moving book.
   → If edge_decay > theoretical_edge / 2, the market is too fast for profitable execution.

4. Fill rate = {stats['fill_rate']:.1f}% of attempted
   → If this is low (<30%), FOK timeout is too tight or Polymarket liquidity too thin.
   → If this is high (>80%), you're winning races consistently.

{'='*80}
""")


async def main_live() -> None:
    """Run the latency arb daemon in paper-trading mode."""
    print("[latency_arb_paper_harness] Initializing paper-trading mode...")

    trader = LatencyArbPaperTrader()

    # TODO: Wire this into the actual sharp_book_monitor + polymarket_checker + executor
    # For now, this is a template for how data flows through the harness

    # Example: simulate one execution
    signal = VolatilitySignal(
        signal_id='sim_001',
        timestamp_signal=datetime.now(timezone.utc).timestamp(),
        market_id='Brazil_vs_Serbia',
        old_prob_sharp=0.60,
        new_prob_sharp=0.52,
        move_pp=-0.08,
        direction='down',
    )

    polymarket_state = {
        'mid': 0.60,  # Stale (sharp is 0.52)
        'bid': 0.59,
        'ask': 0.61,
    }

    sharp_current = {'mid': 0.52}

    execution = await trader.process_breakout(signal, polymarket_state, sharp_current)

    if execution:
        print(f"[Execution {execution.exec_id}]")
        if execution.order.race_lost:
            print(f"  Race lost. Polymarket repriced {execution.query_result.polymarket_stale_vs_sharp*100:.2f}pp before we filled.")
        else:
            print(f"  Filled at {execution.order.actual_fill_price:.4f}")
            print(f"  Theoretical edge: {execution.theoretical_edge_pp*100:.2f}pp")
            print(f"  Actual edge (after latency): {execution.actual_edge_pp*100:.2f}pp")
            print(f"  Total latency: {execution.detection_latency_ms + execution.query_latency_ms + (execution.fill_latency_ms or 0):.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true', help="Run the latency arb daemon in paper-trading mode")
    ap.add_argument('--report', action='store_true', help="Print latency arb analysis report")
    args = ap.parse_args()

    if args.live:
        asyncio.run(main_live())
    elif args.report:
        trader = LatencyArbPaperTrader()
        print_report(trader)
    else:
        print("Use --live (run daemon) or --report (print analysis)")


if __name__ == '__main__':
    main()
