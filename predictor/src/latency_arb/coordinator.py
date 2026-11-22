"""
coordinator.py — orchestrate the latency arb pipeline end-to-end.

Wires together:
  1. Sharp book monitor (detects volatility)
  2. Polymarket stale-price checker (confirms race condition)
  3. Paper harness logging (captures all latency metrics)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from src.latency_arb.volatility_trigger import VolatilityTriggerConfig
from src.latency_arb.sharp_book_monitor import SharpBookMonitor, WebSocketFeedRequired
from src.latency_arb.polymarket_checker import PolymarketStalePriceChecker
from scripts.latency_arb_paper_harness import LatencyArbPaperTrader, LatencyArbExecution
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class LatencyArbConfig:
    """Complete configuration."""
    volatility_trigger: VolatilityTriggerConfig = None
    min_theoretical_edge_pp: float = 0.005  # Don't process sub-0.5pp edges
    paper_mode: bool = True              # Always True unless validated
    websocket_url: str = None            # WebSocket feed URL; None = REST_DEGRADED mode
    latency_arb_enabled: bool = False    # Explicitly opt-in; False until WebSocket available

    def __post_init__(self):
        if self.volatility_trigger is None:
            self.volatility_trigger = VolatilityTriggerConfig()


class LatencyArbCoordinator:
    """Orchestrates the full latency arb pipeline."""

    def __init__(self, cfg: LatencyArbConfig):
        self.cfg = cfg

        # Initialize components
        self.sharp_monitor = SharpBookMonitor(cfg.volatility_trigger)
        self.stale_checker = PolymarketStalePriceChecker()
        self.paper_harness = LatencyArbPaperTrader()

        # State
        self.running = False
        self.stats = {
            'signals_detected': 0,
            'opportunities_checked': 0,
            'races_won': 0,
            'races_lost': 0,
            'executions_logged': 0,
        }

    async def start(self) -> None:
        """Start all background tasks.

        If latency_arb_enabled is False, or if a WebSocket feed is not
        available, the latency arb pipeline is disabled cleanly and the
        coordinator exits without starting background tasks.  All other
        collection (coherence arb, under-reaction) continues via
        run_arb_daemon.py which is unaffected.
        """
        if not self.cfg.latency_arb_enabled:
            log.info(
                "[LatencyArb] Disabled (latency_arb_enabled=False). "
                "Set cfg.websocket_url and cfg.latency_arb_enabled=True "
                "once a WebSocket sharp-book feed is available."
            )
            return

        self.running = True
        log.info("[LatencyArb] Starting coordinator...")

        try:
            await self.sharp_monitor.start()
            log.info("[LatencyArb] Sharp monitor started (WebSocket mode)")
        except WebSocketFeedRequired as exc:
            log.warning(
                "[LatencyArb] DISABLED — %s\n"
                "  Latency arb requires a WebSocket sharp-book feed. "
                "The Odds API is REST-only.\n"
                "  Coherence arb and under-reaction collection continue "
                "normally via run_arb_daemon.py.",
                exc,
            )
            self.running = False
            return

        asyncio.create_task(self._processing_loop())

    async def stop(self) -> None:
        """Stop all background tasks."""
        self.running = False
        await self.sharp_monitor.stop()
        log.info("[LatencyArb] Coordinator stopped")

    async def _processing_loop(self) -> None:
        """
        Main loop: pull breakouts from sharp monitor, check if stale, log results.
        """
        while self.running:
            try:
                # Get the next breakout signal (blocking with timeout)
                try:
                    signal = await asyncio.wait_for(
                        self.sharp_monitor.breakout_queue.get(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue

                self.stats['signals_detected'] += 1

                # Check if Polymarket is stale
                move_magnitude = abs(signal.move_pp / 100.0)  # Convert back to [0,1]
                opportunity = await self.stale_checker.check_breakout(signal, move_magnitude)

                if opportunity is None:
                    self.stats['races_lost'] += 1
                    # Race lost; log but continue
                    log.debug(f"Race lost: {signal.market_id}")
                    continue

                # Opportunity confirmed; log it
                self.stats['races_won'] += 1
                self.stats['opportunities_checked'] += 1

                # In paper mode, just log; don't execute
                if self.cfg.paper_mode:
                    log.info(
                        f"[PAPER] Opportunity on {signal.market_id}: "
                        f"edge {opportunity.theoretical_edge_pp:.2f}pp, "
                        f"latency {opportunity.stale_age_ms:.0f}ms"
                    )
                    self.stats['executions_logged'] += 1

                    # Log to paper harness
                    await self._log_execution(signal, opportunity)

            except Exception as e:
                log.error(f"Processing loop error: {e}")
                await asyncio.sleep(1)

    async def _log_execution(self, signal, opportunity) -> None:
        """Log a confirmed opportunity to the paper harness."""
        try:
            from scripts.latency_arb_paper_harness import (
                VolatilitySignal as HarnessSignal,
                PolymarketQueryResult,
                OrderExecution,
            )

            # Translate from volatility_trigger.VolatilitySignal to the harness type
            harness_signal = HarnessSignal(
                signal_id=signal.signal_id,
                timestamp_signal=signal.timestamp_signal,
                market_id=signal.market_id,
                old_prob_sharp=signal.old_prob,
                new_prob_sharp=signal.new_prob,
                move_pp=signal.move_pp,
                direction=signal.direction,
            )

            polymarket_state = {
                'mid': opportunity.polymarket_mid,
                'bid': opportunity.polymarket_bid,
                'ask': opportunity.polymarket_ask,
            }

            sharp_current = {'mid': opportunity.sharp_new_prob}

            await self.paper_harness.process_breakout(
                harness_signal, polymarket_state, sharp_current
            )

        except Exception as e:
            log.error(f"Execution logging error: {e}")

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        stats = self.stats.copy()
        stats['sharp_monitor'] = self.sharp_monitor.get_stats()
        stats['stale_checker'] = self.stale_checker.get_stats()
        return stats
