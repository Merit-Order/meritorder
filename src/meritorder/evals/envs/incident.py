"""Incident investigation over synthetic observability data: logs, metrics, deploys, alerts.

A scenario is a YAML file in fixtures/incidents/ describing one day: services and their dependencies, deploys (with
their config/code changes), alerts, per-minute metric series (a base value with deterministic noise and segments that
replace it) and log streams (templated lines at a rate per minute). Everything is generated deterministically when the
environment is built. The model has to find the deploy that caused the incident among red herrings and give its id,
the service, the config key and the impact window in a final JSON object; grading is mechanical.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path

import yaml

from . import Env, tool

SCENARIOS = Path(__file__).with_name("fixtures") / "incidents"
MAX_POINTS = 240
DEFAULT_LOG_LIMIT = 50
MAX_LOG_LIMIT = 100
LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}
_TIME = re.compile(r"^(?:(\d{4}-\d{2}-\d{2})[T ])?(\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?\s*(?:Z|UTC|\+00:?00)?$", re.IGNORECASE)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    t = tool(name, description, properties)
    t["function"]["parameters"]["required"] = list(required)
    return t


def _err(message: str) -> str:
    return json.dumps({"error": message})


def _h(*parts: object) -> int:
    return zlib.crc32(":".join(map(str, parts)).encode())


def hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


class IncidentEnv(Env):
    grading = "The final answer's JSON names the deploy, service and config key that caused it, and the impact window"  # shown on the console's Tasks tab
    tools = [
        _tool("list_services", "Services, their owners, what each depends on, and the metrics available for each.", {}, []),
        _tool("get_alerts", "Alerts that fired during the day, with when they fired and resolved (UTC).", {}, []),
        _tool("get_deploys", "Deploys and other production changes, oldest first. Filter by service and/or time range.",
              {"service": {"type": "string", "description": "Only this service (default: all)"},
               "start": {"type": "string", "description": "UTC time, e.g. 13:00 or 2026-11-03T13:00Z (default: start of data)"},
               "end": {"type": "string", "description": "UTC time (default: end of data)"}}, []),
        _tool("get_deploy", "Full details of one deploy or change: config diff, dependency changes, migrations, commits.",
              {"deploy_id": {"type": "string"}}, ["deploy_id"]),
        _tool("get_metrics", f"One metric of one service as a time series between start and end (UTC). Each point is the "
                             f"average over step_minutes; at most {MAX_POINTS} points per call.",
              {"service": {"type": "string"}, "metric": {"type": "string", "description": "e.g. error_rate_pct (see list_services)"},
               "start": {"type": "string", "description": "UTC time, e.g. 13:00"}, "end": {"type": "string", "description": "UTC time"},
               "step_minutes": {"type": "integer", "description": "Minutes per point (default 1)"}},
              ["service", "metric", "start", "end"]),
        _tool("search_logs", f"Search log lines, oldest first. Matches text case-insensitively; returns at most "
                             f"{MAX_LOG_LIMIT} lines and the total number of matches.",
              {"service": {"type": "string", "description": "A service name, or 'all'"},
               "query": {"type": "string", "description": "Text the line must contain (default: any line)"},
               "start": {"type": "string", "description": "UTC time (default: start of data)"},
               "end": {"type": "string", "description": "UTC time (default: end of data)"},
               "level": {"type": "string", "description": "Only this level: INFO, WARN or ERROR"},
               "limit": {"type": "integer", "description": f"Lines to return (default {DEFAULT_LOG_LIMIT}, max {MAX_LOG_LIMIT})"}},
              ["service"]),
    ]

    def __init__(self, sandbox=None, scenario: str = "pricing_pool", **args):
        super().__init__(sandbox, scenario=scenario, **args)
        if not re.fullmatch(r"[a-z0-9_-]+", str(scenario)) or not (SCENARIOS / f"{scenario}.yaml").exists():
            raise ValueError(f"unknown incident scenario {scenario!r}")
        s = yaml.safe_load((SCENARIOS / f"{scenario}.yaml").read_text(encoding="utf-8"))
        self.date = str(s["date"])
        self.lo, self.hi = self._minute(s["data_from"]), self._minute(s["data_to"])
        self.expected = s["expected"]
        self.services = s["services"]
        self.deploys = sorted(s["deploys"], key=lambda d: (self._minute(d["at"]), d["id"]))
        self.alerts = s["alerts"]
        self.metrics = s["metrics"]
        self.lines = self._render_logs(s["logs"])

    # ----------------------------------------------------------------- data
    @staticmethod
    def _minute(value: object) -> int:
        h, m = str(value).split(":")
        return int(h) * 60 + int(m)

    def _iso(self, minute: int, second: int = 0) -> str:
        return f"{self.date}T{hhmm(minute)}:{second:02d}Z"

    def _parse_time(self, value: object, default: int) -> tuple[int | None, str | None]:
        if value in (None, ""):
            return default, None
        m = _TIME.match(str(value).strip())
        if not m:
            return None, f"can't read the time {value!r}; use HH:MM or {self.date}THH:MMZ (UTC)"
        if m.group(1) and m.group(1) != self.date:
            return None, f"no data for {m.group(1)}: data covers {self.date} {hhmm(self.lo)}-{hhmm(self.hi)} UTC"
        hour, minute = int(m.group(2)), int(m.group(3))
        if hour > 24 or minute > 59:
            return None, f"can't read the time {value!r}"
        return hour * 60 + minute, None

    def _range(self, args: dict) -> tuple[int, int, str | None]:
        start, error = self._parse_time(args.get("start"), self.lo)
        if error:
            return 0, 0, error
        end, error = self._parse_time(args.get("end"), self.hi)
        if error:
            return 0, 0, error
        if end <= start:
            return 0, 0, "end must be after start"
        return max(start, self.lo), min(end, self.hi), None

    def _value(self, service: str, metric: str, minute: int) -> float:
        spec = self.metrics[service][metric]
        value, noise = spec["base"], spec.get("noise", 0)
        for seg in spec.get("segments", []):
            if self._minute(seg["from"]) <= minute < self._minute(seg["to"]):
                value, noise = seg["value"], seg.get("noise", 0)
        value += noise * ((_h(service, metric, minute) % 2001) / 1000 - 1)
        return max(value, 0.0)

    def _render_logs(self, streams: list[dict]) -> list[tuple[int, int, str, str, str]]:
        lines = []
        for i, st in enumerate(streams):
            start = self._minute(st.get("from", hhmm(self.lo)))
            end = self._minute(st.get("to", hhmm(self.hi)))
            rate = float(st.get("count", 1))
            for minute in range(start, end):
                n = math.floor((minute + 1 - start) * rate + 1e-9) - math.floor((minute - start) * rate + 1e-9)
                for k in range(n):
                    second = _h(i, minute, k, "s") % 60
                    lo, hi = st.get("ms", [1, 999])
                    text = (st["msg"].replace("{ms}", str(lo + _h(i, minute, k, "ms") % (hi - lo + 1)))
                            .replace("{id}", f"{_h(i, minute, k, 'id') % 16**6:06x}")
                            .replace("{n}", str(1 + _h(i, minute, k, "n") % int(st.get("n", 50)))))
                    lines.append((minute, second, st["service"], st["level"], text))
        lines.sort(key=lambda x: (x[0], x[1], x[2]))
        return lines

    def _service(self, name: object) -> tuple[str | None, str | None]:
        key = str(name or "").strip().lower()
        if key not in self.services:
            return None, f"unknown service {name!r}; services: {', '.join(self.services)}"
        return key, None

    # ----------------------------------------------------------------- tools
    def call(self, name: str, arguments: dict) -> str:
        return getattr(self, f"_t_{name}")(arguments)

    def _t_list_services(self, args: dict) -> str:
        return json.dumps({"services": [{"name": n, **info, "metrics": sorted(self.metrics.get(n, {}))} for n, info in self.services.items()],
                           "data": f"{self.date} {hhmm(self.lo)}-{hhmm(self.hi)} UTC"})

    def _t_get_alerts(self, args: dict) -> str:
        return json.dumps({"alerts": [{**a, "fired": self._iso(self._minute(a["fired"])), "resolved": self._iso(self._minute(a["resolved"]))}
                                      for a in self.alerts]})

    def _summary(self, d: dict) -> dict:
        return {"id": d["id"], "service": d["service"], "at": self._iso(self._minute(d["at"])), "kind": d["kind"],
                "author": d["author"], "title": d["title"]}

    def _t_get_deploys(self, args: dict) -> str:
        service = None
        if args.get("service") not in (None, "", "all", "*"):
            service, error = self._service(args["service"])
            if error:
                return _err(error)
        start, end, error = self._range(args)
        if error:
            return _err(error)
        hits = [self._summary(d) for d in self.deploys
                if (service is None or d["service"] == service) and start <= self._minute(d["at"]) < end]
        return json.dumps({"deploys": hits})

    def _t_get_deploy(self, args: dict) -> str:
        want = str(args.get("deploy_id") or "").strip().lower()
        for d in self.deploys:
            if d["id"] == want:
                return json.dumps({**self._summary(d), "changes": d.get("changes", {})})
        return _err(f"no deploy {args.get('deploy_id')!r}; use get_deploys to list them")

    def _t_get_metrics(self, args: dict) -> str:
        service, error = self._service(args.get("service"))
        if error:
            return _err(error)
        metric = str(args.get("metric") or "").strip()
        if metric not in self.metrics.get(service, {}):
            return _err(f"{service} has no metric {metric!r}; it has: {', '.join(sorted(self.metrics.get(service, {})))}")
        for key in ("start", "end"):
            if args.get(key) in (None, ""):
                return _err(f"{key} is required")
        start, end, error = self._range(args)
        if error:
            return _err(error)
        step = args.get("step_minutes", 1)
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            return _err("step_minutes must be a positive integer")
        if math.ceil((end - start) / step) > MAX_POINTS:
            return _err(f"that's more than {MAX_POINTS} points; use a larger step_minutes or a shorter range")
        spec = self.metrics[service][metric]
        digits = 2 if "%" in spec["unit"] else 0
        points = []
        for t in range(start, end, step):
            values = [self._value(service, metric, m) for m in range(t, min(t + step, end))]
            avg = round(sum(values) / len(values), digits)
            points.append([hhmm(t), avg if digits else int(avg)])
        return json.dumps({"service": service, "metric": metric, "unit": spec["unit"], "step_minutes": step,
                           "points": points})

    def _t_search_logs(self, args: dict) -> str:
        service = None
        if str(args.get("service") or "").strip().lower() not in ("", "all", "*"):
            service, error = self._service(args.get("service"))
            if error:
                return _err(error)
        start, end, error = self._range(args)
        if error:
            return _err(error)
        level = str(args.get("level") or "").strip().upper().replace("WARNING", "WARN")
        if level and level not in LEVELS:
            return _err("level must be INFO, WARN or ERROR")
        limit = args.get("limit", DEFAULT_LOG_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return _err("limit must be a positive integer")
        limit = min(limit, MAX_LOG_LIMIT)
        query = str(args.get("query") or "").lower()
        hits = [ln for ln in self.lines if start <= ln[0] < end and (service is None or ln[2] == service)
                and (not level or ln[3] == level) and query in ln[4].lower()]
        out = {"total_matches": len(hits),
               "lines": [f"{self._iso(m, s)} {svc} {lvl} {text}" for m, s, svc, lvl, text in hits[:limit]]}
        if len(hits) > limit:
            out["note"] = f"showing the first {limit} of {len(hits)} matching lines; narrow the time range or the query"
        return json.dumps(out)

    # ----------------------------------------------------------------- grading
    @staticmethod
    def _last_object(text: str) -> dict | None:
        decoder, found = json.JSONDecoder(), None
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    found = obj
        return found

    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        want = self.expected
        got = self._last_object(final_answer or "")
        if got is None:
            return False, ["no JSON object in the final answer"]
        reasons = []
        for key in ("deploy_id", "service"):
            if str(got.get(key, "")).strip().lower() != str(want[key]).lower():
                reasons.append(f"{key}: expected {want[key]!r}, got {got.get(key)!r}")
        if not re.search(rf"(?<![\w-]){re.escape(want['config_key'])}(?![\w-])", str(got.get("config_key", "")), re.IGNORECASE):
            reasons.append(f"config_key: expected {want['config_key']!r}, got {got.get('config_key')!r}")
        tol = int(want.get("tolerance_minutes", 2))
        for key in ("impact_start", "impact_end"):
            minute, error = self._parse_time(got.get(key) or "-", -1)
            if error or minute is None:
                reasons.append(f"{key}: can't read a UTC time from {got.get(key)!r}")
            elif abs(minute - self._minute(want[key])) > tol:
                reasons.append(f"{key}: expected {want[key]} (+-{tol} min), got {hhmm(minute)}")
        return not reasons, reasons
