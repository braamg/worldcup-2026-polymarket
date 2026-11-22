"""
polymarket_checker.py — check if Polymarket prices are stale vs sharp fair value.

Given a breakout signal from sharp books, query Polymarket's CLOB and verify
it hasn't already repriced (race condition detection).
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from src.betting.polymarket import fetch_wc_match_prices, fetch_wc_futures_prices
from src.latency_arb.volatility_trigger import VolatilitySignal
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class StalePriceOpportunity:
    """Confirmed latency arb opportunity: Polymarket is stale vs sharp."""
    market_id: str
    timestamp_query: float  # When we queried Polymarket
    polymarket_mid: float
    polymarket_bid: float
    polymarket_ask: float
    sharp_new_prob: float
    theoretical_edge_pp: float
    stale_age_ms: float  # How long has Polymarket been stale


class PolymarketStalePriceChecker:
    """Check if Polymarket has repriced in response to a sharp move."""

    def __init__(self):
        self.latest_prices = {}
        self.stats = {
            'checks_attempted': 0,
            'opportunities_found': 0,
            'races_lost': 0,
        }

    async def check_breakout(
        self,
        signal: VolatilitySignal,
        sharp_move_magnitude: float,
    ) -> Optional[StalePriceOpportunity]:
        """
        Verify Polymarket is still stale after a sharp breakout.

        Args:
            signal: VolatilitySignal from sharp book monitor
            sharp_move_magnitude: |new_prob - old_prob|

        Returns:
            StalePriceOpportunity if confirmed stale; None if race lost.
        """
        self.stats['checks_attempted'] += 1

        query_ts = datetime.now(timezone.utc).timestamp()

        try:
            # Fetch current Polymarket prices
            match_prices, _ = fetch_wc_match_prices()
            futures_prices, _ = fetch_wc_futures_prices()

            # market_id is a column, not the index — set it before combining
            def _idx(df):
                return df.set_index('market_id') if 'market_id' in df.columns else df

            all_prices = _idx(match_prices).combine_first(_idx(futures_prices))

            # Look up this market
            if signal.market_id not in all_prices.index:
                log.warning(f"Market not found in Polymarket: {signal.market_id}")
                return None

            mid = float(all_prices.loc[signal.market_id, 'market_mid'])

            # Check if Polymarket has repriced
            polymarket_move = abs(mid - signal.old_prob)

            # RACE CONDITION: if Polymarket has moved >70% of the sharp move, we lost
            if polymarket_move > 0.7 * sharp_move_magnitude:
                self.stats['races_lost'] += 1
                log.debug(
                    f"Race lost on {signal.market_id}: "
                    f"Polymarket already moved {polymarket_move*100:.2f}pp"
                )
                return None

            # Not a race condition; opportunity exists
            theoretical_edge = abs(mid - signal.new_prob) * 100  # in pp

            if theoretical_edge < 0.5:  # Don't bother with tiny edges
                return None

            self.stats['opportunities_found'] += 1

            return StalePriceOpportunity(
                market_id=signal.market_id,
                timestamp_query=query_ts,
                polymarket_mid=mid,
                polymarket_bid=mid - 0.005,  # TODO: fetch actual bid/ask from CLOB
                polymarket_ask=mid + 0.005,
                sharp_new_prob=signal.new_prob,
                theoretical_edge_pp=theoretical_edge,
                stale_age_ms=(query_ts - signal.timestamp_signal) * 1000,
            )

        except Exception as e:
            log.error(f"Polymarket check error: {e}")
            return None

    def get_stats(self) -> dict:
        """Return checking statistics."""
        return self.stats.copy()
