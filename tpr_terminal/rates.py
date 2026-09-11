from __future__ import annotations

import logging
import math
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime

from .models import RateCurve

logger = logging.getLogger(__name__)

_TENORS: dict[str, float] = {
    "BC_1MONTH": 30.4375,
    "BC_1_5MONTH": 45.65625,
    "BC_2MONTH": 60.875,
    "BC_3MONTH": 91.3125,
    "BC_4MONTH": 121.75,
    "BC_6MONTH": 182.625,
    "BC_1YEAR": 365.25,
    "BC_2YEAR": 730.5,
    "BC_3YEAR": 1_095.75,
    "BC_5YEAR": 1_826.25,
    "BC_7YEAR": 2_556.75,
    "BC_10YEAR": 3_652.5,
    "BC_20YEAR": 7_305.0,
    "BC_30YEAR": 10_957.5,
}


class TreasuryCurveService:
    def __init__(
        self,
        *,
        enabled: bool,
        static_fallback_rate: float,
        ttl_seconds: int,
    ) -> None:
        self._enabled = enabled
        self._fallback_rate = static_fallback_rate
        self._ttl = ttl_seconds
        self._cached: RateCurve | None = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def get_curve(self) -> RateCurve:
        now = time.monotonic()
        if self._cached is not None and now < self._expires:
            return self._cached
        with self._lock:
            now = time.monotonic()
            if self._cached is not None and now < self._expires:
                return self._cached
            curve = self._fetch() if self._enabled else self._fallback("Treasury curve disabled")
            self._cached = curve
            self._expires = now + self._ttl
            return curve

    def _fetch(self) -> RateCurve:
        year = datetime.now(UTC).year
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml?data=daily_treasury_yield_curve&"
            f"field_tdr_date_value={year}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "TPR-Terminal/3.0 Discord analytics",
                "Accept": "application/atom+xml,application/xml,text/xml",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                return self.parse_xml(response.read())
        except Exception as exc:
            logger.warning("Treasury curve unavailable; using fallback: %s", exc)
            return self._fallback("Static fallback after Treasury fetch failure")

    def parse_xml(self, payload: bytes) -> RateCurve:
        root = ET.fromstring(payload)
        observations: list[tuple[date, dict[str, float]]] = []
        for entry in root.iter():
            if self._local(entry.tag) != "entry":
                continue
            values: dict[str, str] = {}
            for element in entry.iter():
                text = (element.text or "").strip()
                if text:
                    values[self._local(element.tag)] = text
            raw_date = values.get("NEW_DATE") or values.get("updated")
            if not raw_date:
                continue
            try:
                day = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
            except ValueError:
                continue
            points: dict[str, float] = {}
            for field in _TENORS:
                try:
                    value = float(values[field])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0:
                    points[field] = value
            if points:
                observations.append((day, points))
        if not observations:
            raise ValueError("Treasury XML contained no usable curve observations.")
        as_of, latest = max(observations, key=lambda item: item[0])
        curve = sorted(
            (_TENORS[field], math.log1p(percent / 100.0))
            for field, percent in latest.items()
        )
        if len(curve) < 2:
            raise ValueError("Treasury XML contained fewer than two tenor points.")
        return RateCurve(
            tenor_days=tuple(x for x, _ in curve),
            rates_cc=tuple(y for _, y in curve),
            source="U.S. Treasury par yield curve (continuous-rate approximation)",
            as_of=as_of,
            used_fallback=False,
        )

    def _fallback(self, source: str) -> RateCurve:
        return RateCurve(
            tenor_days=(0.0, 36_500.0),
            rates_cc=(self._fallback_rate, self._fallback_rate),
            source=source,
            as_of=None,
            used_fallback=True,
        )

    @staticmethod
    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]
