"""
utils/schedule.py
=================
WC2026 round-completion dates, used to show when each bet is expected to resolve.

A "reach stage X" YES bet is decided by the time that round completes (the team is either
through or eliminated). These are the *approximate* round-end dates — verify against the
official FIFA schedule before relying on them. The tournament runs 11 Jun – 19 Jul 2026.
"""

from __future__ import annotations

from datetime import date

# match label (engine/views) -> date the bet's outcome is locked in
_RESOLVE_BY: dict[str, date] = {
    "Group Stage":          date(2026, 6, 27),  # group stage complete  → advance/eliminated
    "Reach Round of 16":    date(2026, 7, 3),   # Round of 32 complete
    "Reach Quarter-Finals": date(2026, 7, 7),   # Round of 16 complete
    "Reach Semi-Finals":    date(2026, 7, 11),  # quarter-finals complete
    "Reach Final":          date(2026, 7, 15),  # semi-finals complete
    "Champion market":      date(2026, 7, 19),  # final
}


def resolution_date(match_label: str) -> date | None:
    return _RESOLVE_BY.get(match_label)


def days_until(match_label: str, today: date | None = None) -> int | None:
    """Days from `today` (default: system date) until this market resolves. None if unknown."""
    d = _RESOLVE_BY.get(match_label)
    if d is None:
        return None
    today = today or date.today()
    return (d - today).days


def resolves_label(match_label: str, today: date | None = None) -> str:
    """Human string like 'by 19 Jul (~29 days)' / 'resolved' for a market type."""
    d = resolution_date(match_label)
    if d is None:
        return "date TBC"
    n = days_until(match_label, today)
    if n is None:
        return f"by {d:%d %b}"
    if n < 0:
        return f"resolved ({d:%d %b})"
    if n == 0:
        return f"today ({d:%d %b})"
    return f"by {d:%d %b} (~{n} day{'s' if n != 1 else ''})"
