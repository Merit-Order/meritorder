"""Hard caps can come from the environment (.env), so each install sets its own without editing the shared
config file. Unset or blank means no limit."""

import copy
from pathlib import Path

import pytest
from conftest import TEST_CONFIG

from meritorder.config import build_config, load_config

VARS = ("MERITORDER_ORG_BUDGET_USD", "MERITORDER_GPU_BUDGET_USD", "MERITORDER_MAX_GPU_USD_PER_HOUR", "MERITORDER_MAX_GPU_REPLICAS",
        "MERITORDER_MAX_REQUEST_USD")
SHIPPED = Path(__file__).resolve().parents[1] / "config" / "meritorder.yaml"


def with_guards(guards: dict):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["safeguards"] = guards
    return build_config(raw, quiet=True).safeguards


def test_caps_from_the_environment(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    refs = {"org_budget_usd_month": "${MERITORDER_ORG_BUDGET_USD:-}", "max_gpu_replicas": "${MERITORDER_MAX_GPU_REPLICAS:-}",
            "max_request_usd": "${MERITORDER_MAX_REQUEST_USD:-25}"}
    g = with_guards(refs)
    assert g.org_budget_usd_month is None and g.max_gpu_replicas is None and g.max_request_usd == 25
    monkeypatch.setenv("MERITORDER_ORG_BUDGET_USD", "50")
    monkeypatch.setenv("MERITORDER_MAX_GPU_REPLICAS", "1")
    monkeypatch.setenv("MERITORDER_MAX_REQUEST_USD", "3")
    g = with_guards(refs)
    assert (g.org_budget_usd_month, g.max_gpu_replicas, g.max_request_usd) == (50, 1, 3)
    assert with_guards({"gpu_budget_usd_month": "  "}).gpu_budget_usd_month is None
    with pytest.raises(ValueError):
        with_guards({"org_budget_usd_month": "lots"})
    with pytest.raises(ValueError):
        with_guards({"org_budget_usd_month": "-1"})


def test_shipped_config_reads_its_caps_from_the_environment(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    g = load_config(SHIPPED).safeguards
    assert g.org_budget_usd_month is None and g.gpu_budget_usd_month is None and g.max_request_usd == 25
    monkeypatch.setenv("MERITORDER_ORG_BUDGET_USD", "50")
    monkeypatch.setenv("MERITORDER_GPU_BUDGET_USD", "25")
    monkeypatch.setenv("MERITORDER_MAX_GPU_USD_PER_HOUR", "5")
    g = load_config(SHIPPED).safeguards
    assert (g.org_budget_usd_month, g.gpu_budget_usd_month, g.max_gpu_usd_per_hour) == (50, 25, 5)
