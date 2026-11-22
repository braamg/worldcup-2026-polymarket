"""
volatility_trigger.py — detect statistically significant sharp book moves.

When a major unobserved event happens (injury, red card, goal), sharp books reprice instantly.
This module detects those moves via a rolling-window volatility test.

Breakout = |move| > k * rolling_sigma + min_move_pp
  - rolling_sigma = std dev of price deltas over last N time windows
  - k = threshold multiplier (e.g. 2.0 for 2-sigma moves)
  - min_move_pp = minimum absolute move in percentage points (e.g. 3pp)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class VolatilityTriggerConfig:
    """Configuration for volatility detection."""
    window_size: int = 60  # seconds: rolling window of price history
    rolling_window_count: int = 20  # number of time intervals for sigma estimation
    threshold_sigma: float = 2.0  # k: 2σ breakout
    min_move_pp: float = 0.03  # 3 percentage points minimum
    polling_interval: float = 0.2  # seconds between sharp-book polls


@dataclass
class VolatilitySignal:
    """Emitted when breakout is detected."""
    signal_id: str
    timestamp_signal: float  # Unix seconds (millisecond precision)
    market_id: str
    old_prob: float  # price 60s ago (or older, depending on window)
    new_prob: float  # price now
    move_pp: float  # (new - old) in percentage points
    direction: str  # 'up' or 'down'
    rolling_sigma: float  # volatility at time of detection
    sigma_mult: float  # how many sigmas was this move (z-score)


class VolatilityTrigger:
    """Detect volatility breakouts in a single market's sharp-book price."""

    def __init__(self, market_id: str, cfg: VolatilityTriggerConfig):
        self.market_id = market_id
        self.cfg = cfg

        # Circular buffer of (timestamp, implied_prob) tuples
        # Sized to hold window_size / polling_interval snapshots, plus a bit extra
        max_snapshots = max(int(cfg.window_size / cfg.polling_interval) + 5, 200)
        self.price_history = deque(maxlen=max_snapshots)

        # Diagnostics
        self.last_sigma = 0.0
        self.last_move = 0.0
        self.last_signal_time = 0.0

    def update(self, timestamp: float, sharp_prob: float) -> Optional[VolatilitySignal]:
        """
        Ingest a sharp book snapshot and check for breakouts.

        Args:
            timestamp: Unix timestamp (seconds)
            sharp_prob: De-vigged implied probability (0–1)

        Returns:
            VolatilitySignal if breakout detected; None otherwise
        """
        self.price_history.append((timestamp, sharp_prob))

        if len(self.price_history) < 3:
            return None  # Need minimum history

        # Rolling window: oldest to newest within window_size
        oldest_ts, oldest_prob = self.price_history[0]
        newest_ts, newest_prob = self.price_history[-1]

        time_elapsed = newest_ts - oldest_ts
        if time_elapsed < self.cfg.window_size * 0.5:
            # Not enough time elapsed yet
            return None

        # Compute rolling sigma from deltas
        deltas = []
        for i in range(1, min(len(self.price_history), self.cfg.rolling_window_count + 1)):
            delta = abs(self.price_history[-i][1] - self.price_history[-i - 1][1])
            deltas.append(delta)

        rolling_sigma = float(np.std(deltas)) if deltas else 1e-6
        self.last_sigma = rolling_sigma

        # Actual move in the rolling window
        move = newest_prob - oldest_prob
        self.last_move = move

        # Breakout threshold
        trigger_threshold = self.cfg.threshold_sigma * rolling_sigma + self.cfg.min_move_pp

        if abs(move) > trigger_threshold:
            # BREAKOUT!
            signal_id = f"sig_{int(timestamp * 1000) % 1_000_000}"
            self.last_signal_time = timestamp

            sigma_mult = abs(move) / (rolling_sigma + 1e-10)

            return VolatilitySignal(
                signal_id=signal_id,
                timestamp_signal=timestamp,
                market_id=self.market_id,
                old_prob=oldest_prob,
                new_prob=newest_prob,
                move_pp=move * 100,  # Convert to percentage points
                direction='up' if move > 0 else 'down',
                rolling_sigma=rolling_sigma,
                sigma_mult=sigma_mult,
            )

        return None
