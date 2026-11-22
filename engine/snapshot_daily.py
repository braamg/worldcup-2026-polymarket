"""
snapshot_daily.py
=================
Headless daily forecast snapshot for the scientific verdict. Fetches live Polymarket prices +
the current model probabilities and appends one row per market to forecast_log.csv (deduped per
day). Designed to run from a scheduled task so the model's track record builds automatically,
with no dashboard interaction.

    python snapshot_daily.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from model_loader import load_model_probs, load_match_probs
from polymarket_client import fetch_futures_prices, fetch_match_prices
from utils.forecast_log import snapshot_forecasts, FORECAST_LOG_PATH

_FUT_RESOLVE = {
    "advance": "2026-06-27", "reach_r16": "2026-07-03", "reach_qf": "2026-07-07",
    "reach_sf": "2026-07-11", "reach_final": "2026-07-15", "champion": "2026-07-19",
}


def main() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] snapshot starting...")

    match_model = load_match_probs()
    match_market = fetch_match_prices()
    fut_model = load_model_probs()
    fut_market = fetch_futures_prices()

    print(f"  match: model={len(match_model)} market={len(match_market)} | "
          f"futures: model={len(fut_model)} market={len(fut_market)}")

    added = snapshot_forecasts(
        match_model, match_market, today,
        futures_model_df=fut_model, futures_market_df=fut_market,
        stage_resolve_dates=_FUT_RESOLVE,
    )
    print(f"  +{added} new rows -> {FORECAST_LOG_PATH}")
    print(f"  done. ({today})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
