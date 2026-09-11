from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    # Gregorian Meeus/Jones/Butcher algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@dataclass(frozen=True, slots=True)
class SessionClose:
    session: date
    close_utc: datetime
    early_close: bool


class USMarketCalendar:
    """Dependency-free U.S. equity-market calendar for current option expiries.

    Standard recurring full-day holidays are calculated directly. Optional JSON
    overrides support future ad-hoc closures and special close times without a
    code change. The override format is documented in assets/calendar_overrides.json.
    """

    def __init__(self, overrides_path: Path | None = None) -> None:
        self._closed_overrides: set[date] = set()
        self._close_overrides: dict[date, time] = {}
        if overrides_path and overrides_path.exists():
            self._load_overrides(overrides_path)

    def holidays(self, year: int) -> set[date]:
        days = {
            _observed(date(year, 1, 1)),
            _nth_weekday(year, 1, 0, 3),  # MLK Day
            _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
            _easter_sunday(year) - timedelta(days=2),  # Good Friday
            _last_weekday(year, 5, 0),  # Memorial Day
            _observed(date(year, 7, 4)),
            _nth_weekday(year, 9, 0, 1),  # Labor Day
            _nth_weekday(year, 11, 3, 4),  # Thanksgiving
            _observed(date(year, 12, 25)),
        }
        if year >= 2022:
            days.add(_observed(date(year, 6, 19)))
        # A following year's New Year can be observed on Dec 31 of this year.
        observed_next_new_year = _observed(date(year + 1, 1, 1))
        if observed_next_new_year.year == year:
            days.add(observed_next_new_year)
        return days | {d for d in self._closed_overrides if d.year == year}

    def is_session(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays(day.year)

    def previous_session(self, day: date) -> date:
        cursor = day
        while not self.is_session(cursor):
            cursor -= timedelta(days=1)
        return cursor

    def session_close(self, day: date) -> SessionClose:
        session = self.previous_session(day)
        close_time = self._close_overrides.get(session)
        early = close_time is not None
        if close_time is None:
            early = self._is_standard_early_close(session)
            close_time = time(13, 0) if early else time(16, 0)
        local = datetime.combine(session, close_time, tzinfo=NEW_YORK)
        return SessionClose(session, local.astimezone(UTC), early)

    @staticmethod
    def _is_standard_early_close(day: date) -> bool:
        thanksgiving = _nth_weekday(day.year, 11, 3, 4)
        if day == thanksgiving + timedelta(days=1):
            return True
        if day.month == 7 and day.day == 3 and day.weekday() < 5:
            return True
        if day.month == 12 and day.day == 24 and day.weekday() < 5:
            return True
        return False

    def _load_overrides(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload.get("closed", []):
            self._closed_overrides.add(date.fromisoformat(value))
        for key, value in payload.get("closes", {}).items():
            parsed = datetime.strptime(value, "%H:%M").time()
            self._close_overrides[date.fromisoformat(key)] = parsed
