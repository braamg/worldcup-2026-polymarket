"""Latency arbitrage module: detect and log sharp book breakouts vs stale Polymarket prices."""

from src.latency_arb.volatility_trigger import VolatilityTrigger, VolatilityTriggerConfig, VolatilitySignal
from src.latency_arb.sharp_book_monitor import SharpBookMonitor
from src.latency_arb.polymarket_checker import PolymarketStalePriceChecker, StalePriceOpportunity
from src.latency_arb.coordinator import LatencyArbCoordinator, LatencyArbConfig

__all__ = [
    'VolatilityTrigger',
    'VolatilityTriggerConfig',
    'VolatilitySignal',
    'SharpBookMonitor',
    'PolymarketStalePriceChecker',
    'StalePriceOpportunity',
    'LatencyArbCoordinator',
    'LatencyArbConfig',
]
