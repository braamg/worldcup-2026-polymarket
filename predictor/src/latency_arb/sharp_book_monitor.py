"""
sharp_book_monitor.py — sharp book price monitor for latency arbitrage.

ARCHITECTURAL CONSTRAINT — READ BEFORE MODIFYING
-------------------------------------------------
Latency arbitrage requires detecting a sharp-book price move within milliseconds
and confirming Polymarket has not yet repriced. This is only achievable with a
persistent WebSocket connection to the sharp data source.

The Odds API (our current sharp data source) is a REST-only API. It has no
WebSocket endpoint. Polling it at any interval — even 60 seconds — cannot
support true latency arbitrage:

  - At 60s REST polling: breakout detected ~30s after it happens on average.
    Polymarket reprices in 2–10s. The opportunity is always gone.
  - At <1s REST polling: free-tier quota (500 req/month) exhausted in 8 hours.

This module therefore implements two modes:

  REST_DEGRADED (default) — raises WebSocketFeedRequired at start(), keeping
      the coordinator alive but disabling all latency arb signal generation.
      The under-reaction and coherence arb collection in run_arb_daemon.py
      continues unaffected.

  WEBSOCKET (not yet wired) — if a caller provides a websocket_url, the
      polling loop is replaced by a persistent async WebSocket listener.
      Stub is ready; requires a data provider that offers a WS endpoint
      (e.g. Betfair Exchange Streaming API, or a self-hosted feed aggregator).

To move to live latency arb, wire a WebSocket data source here and set
coordinator.cfg.latency_arb_enabled = True.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from src.latency_arb.volatility_trigger import VolatilityTrigger, VolatilityTriggerConfig, VolatilitySignal
from src.utils import get_logger

log = get_logger(__name__)


class WebSocketFeedRequired(RuntimeError):
    """
    Raised when the SharpBookMonitor is started without a WebSocket feed.

    The Odds API is REST-only; REST polling cannot support sub-second latency
    arbitrage. The coordinator catches this and disables the latency arb
    signal pipeline while keeping everything else running.
    """


class SharpBookMonitor:
    """
    Monitor sharp book prices and detect volatility breakouts.

    In REST_DEGRADED mode (websocket_url=None), start() raises
    WebSocketFeedRequired immediately — no polling, no quota burn.

    In WEBSOCKET mode (websocket_url provided), connects to a persistent
    async WebSocket feed and emits VolatilitySignal objects via
    self.breakout_queue whenever a market moves >2σ + 3pp.
    """

    def __init__(
        self,
        cfg: VolatilityTriggerConfig,
        websocket_url: Optional[str] = None,
    ):
        self.cfg = cfg
        self.websocket_url = websocket_url

        # Per-market breakout detectors, keyed by market_id
        self.triggers: dict[str, VolatilityTrigger] = {}

        # Downstream consumers pull signals from this queue
        self.breakout_queue: asyncio.Queue[VolatilitySignal] = asyncio.Queue(maxsize=1000)

        self.running = False
        self.stats = {
            'total_updates': 0,
            'breakouts_detected': 0,
            'ws_reconnects': 0,
            'mode': 'REST_DEGRADED',
        }

    async def start(self) -> None:
        """
        Start monitoring.

        Raises WebSocketFeedRequired immediately if no websocket_url was
        supplied. The coordinator is expected to catch this, log it, and
        continue running without latency arb.
        """
        if self.websocket_url is None:
            raise WebSocketFeedRequired(
                "SharpBookMonitor requires a WebSocket data feed for latency "
                "arbitrage. The Odds API is REST-only and cannot support "
                "sub-second price monitoring. Provide websocket_url= or "
                "disable latency arb (coordinator.cfg.latency_arb_enabled=False)."
            )

        self.running = True
        self.stats['mode'] = 'WEBSOCKET'
        log.info("[SharpMonitor] WebSocket mode: %s", self.websocket_url)
        asyncio.create_task(self._websocket_loop())

    async def stop(self) -> None:
        self.running = False

    async def _websocket_loop(self) -> None:
        """
        Persistent WebSocket listener. Reconnects automatically on disconnect.

        Expected message format (JSON):
            {"market_id": "match:BRA|MOR:Brazil", "prob": 0.52, "ts": 1718600000.123}

        Each message is fed into the per-market VolatilityTrigger. Breakouts
        are pushed to self.breakout_queue for the coordinator to consume.
        """
        import json
        try:
            import websockets
        except ImportError:
            log.error(
                "[SharpMonitor] 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            self.running = False
            return

        while self.running:
            try:
                log.info("[SharpMonitor] Connecting to %s", self.websocket_url)
                async with websockets.connect(self.websocket_url, ping_interval=20) as ws:
                    log.info("[SharpMonitor] Connected.")
                    async for raw in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw)
                            market_id = str(msg['market_id'])
                            prob = float(msg['prob'])
                            ts = float(msg.get('ts', datetime.now(timezone.utc).timestamp()))

                            if market_id not in self.triggers:
                                self.triggers[market_id] = VolatilityTrigger(market_id, self.cfg)

                            signal = self.triggers[market_id].update(ts, prob)
                            self.stats['total_updates'] += 1

                            if signal:
                                self.stats['breakouts_detected'] += 1
                                await self.breakout_queue.put(signal)
                                log.info(
                                    "[BREAKOUT] %s: %s %.2fpp (%.1fσ)",
                                    signal.market_id, signal.direction,
                                    signal.move_pp, signal.sigma_mult,
                                )
                        except (KeyError, ValueError, json.JSONDecodeError) as exc:
                            log.debug("[SharpMonitor] Bad message: %s — %s", raw[:80], exc)

            except Exception as exc:
                if self.running:
                    self.stats['ws_reconnects'] += 1
                    log.warning(
                        "[SharpMonitor] WebSocket disconnected (%s). "
                        "Reconnecting in 5s (attempt %d).",
                        exc, self.stats['ws_reconnects'],
                    )
                    await asyncio.sleep(5)

    def get_stats(self) -> dict:
        return self.stats.copy()
