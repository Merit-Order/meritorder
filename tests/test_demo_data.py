"""The console demo's sample data (ui/src/demo/data.json, from scripts/build_demo_data.py): its numbers add up as the
reports' do, and nothing in it is real."""

import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "ui" / "src" / "demo" / "data.json"
CAUSES = ("routing", "caching", "self_hosting", "batch", "compression", "compression_model", "response_cache")


@pytest.fixture(scope="module")
def answers() -> dict:
    return json.loads(DATA.read_text())["answers"]


def test_savings_by_cause_add_up(answers):
    summaries = [v for k, v in answers.items() if k.startswith("GET /v1/reports/summary?")]
    assert summaries
    for s in summaries:
        for row in [*s["rows"], s["totals"]]:
            parts = sum(row[f"savings_{c}"] for c in CAUSES)
            scale = sum(abs(row[f"savings_{c}"]) for c in CAUSES)  # the file keeps six significant digits of each
            assert row["savings_total"] == pytest.approx(parts, abs=1e-5 * scale + 1e-4), row["key"]
    month = answers["GET /v1/console/overview"]["month"]
    assert month["saved"] == pytest.approx(sum(month["savings_by_cause"].values()), rel=1e-4)
    assert month["saved"] > 0 and month["spend"] < month["baseline"]


def test_reports_agree_with_each_other(answers):
    summaries = [v for k, v in answers.items() if k.startswith("GET /v1/reports/summary?")]
    by_team = next(s for s in summaries if s["group_by"] == "team")
    by_model = next(s for s in summaries if s["group_by"] == "model" and (s["start"], s["end"]) == (by_team["start"], by_team["end"]))
    assert by_team["totals"]["actual_cost"] == pytest.approx(by_model["totals"]["actual_cost"], rel=1e-4)
    assert sum(r["requests"] for r in by_team["rows"]) == by_team["totals"]["requests"]


def test_every_person_and_address_is_invented(answers):
    text = DATA.read_text()
    emails = set(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text))
    assert emails and all(e.endswith("@example.com") for e in emails), sorted(e for e in emails if not e.endswith("@example.com"))
    assert not re.search(r"sk-ant-|sk-proj-|AKIA[0-9A-Z]{16}", text)
