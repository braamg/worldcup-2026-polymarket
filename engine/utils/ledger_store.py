"""
utils/ledger_store.py
====================
Disk persistence for the bet ledger.

st.session_state is wiped whenever the Streamlit session resets (restart, new browser, idle
timeout). For tracking real bets across a month-long tournament that is useless, so the ledger
is mirrored to a local JSON file. Load it once into session_state on startup; save after every
change (log / settle / clear).
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "ledger.json"


def load_ledger() -> list[dict]:
    if _PATH.exists():
        try:
            data = json.loads(_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def save_ledger(ledger: list[dict]) -> None:
    try:
        _PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except Exception:
        pass  # persistence is best-effort; never crash the UI over it


def ledger_path() -> Path:
    return _PATH
