"""Download and disk-cache raw international results from martj42/international_results."""
from pathlib import Path

import pandas as pd
import requests

from src.utils import get_logger

log = get_logger(__name__)

_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
RESULTS_URL   = f"{_BASE}/results.csv"
SHOOTOUTS_URL = f"{_BASE}/shootouts.csv"


def _fetch_csv(url: str, dest: Path) -> pd.DataFrame:
    if dest.exists():
        log.info("Cache hit: %s", dest.name)
        return pd.read_csv(dest)
    log.info("Downloading %s …", url)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach {url}.\n"
            "Fallback: download manually and place at:\n"
            f"  {dest}\n"
            f"Original error: {exc}"
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    log.info("Saved → %s  (%d bytes)", dest.name, len(r.content))
    return pd.read_csv(dest)


def fetch_results(raw_dir: str | Path) -> pd.DataFrame:
    return _fetch_csv(RESULTS_URL, Path(raw_dir) / "results.csv")


def fetch_shootouts(raw_dir: str | Path) -> pd.DataFrame:
    return _fetch_csv(SHOOTOUTS_URL, Path(raw_dir) / "shootouts.csv")
