"""The per-cloud differences of SkyPilot replicas (sky_clouds.py): which clouds can run one, how each is reached, and the
warnings about what SkyPilot copies to the machines."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from meritorder import sky_clouds
from meritorder.config import build_config


def test_the_table_covers_every_cloud_and_its_extra():
    usable = {n for n, c in sky_clouds.CLOUDS.items() if not c.unusable}
    assert {"aws", "gcp", "azure", "lambda", "runpod", "nebius", "kubernetes", "vast", "fluidstack", "paperspace", "do",
            "mithril", "seeweb"} <= usable
    assert {"oci", "cudo", "hyperbolic", "primeintellect", "shadeform", "verda", "yotta", "coreweave"}.isdisjoint(usable)
    assert sky_clouds.cloud("DigitalOcean").name == "do" and sky_clouds.cloud("Lambda").name == "lambda"
    assert sky_clouds.cloud(None) is None and sky_clouds.cloud("nowhere") is None and sky_clouds.reach("nowhere") == "direct"
    assert sky_clouds.reach("RunPod") == "endpoint" and not sky_clouds.opens_ports("vast") and sky_clouds.opens_ports("aws")
    assert sky_clouds.unusable(["aws", "verda"]) == [f"verda: {sky_clouds._NO_AUTODOWN}"]


def test_dockerfile_default_extras_match_the_table():
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
    found = re.search(r'ARG SKYPILOT_EXTRAS="([^"]+)"', dockerfile)
    assert found and found.group(1).split(",") == list(sky_clouds.DEFAULT_EXTRAS)  # every cloud a replica can run on
    assert "oci" not in sky_clouds.DEFAULT_EXTRAS and "lambda" in sky_clouds.DEFAULT_EXTRAS


def write(home: Path, rel: str) -> None:
    path = home / rel.removeprefix("~/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("secret")


def test_warnings_name_the_keys_on_the_machines(tmp_path):
    assert sky_clouds.warnings(["lambda"], use_spot=False, home=str(tmp_path)) == []  # no key here: nothing to copy
    write(tmp_path, "~/.lambda_cloud/lambda_keys")
    write(tmp_path, "~/.aws/credentials")
    write(tmp_path, "~/.azure/msal_token_cache.json")
    found = sky_clouds.warnings(["lambda"], use_spot=False, home=str(tmp_path))
    assert len(found) == 2 and found[0].startswith("lambda: SkyPilot copies ~/.lambda_cloud/lambda_keys") and "revoke" in found[0]
    assert "~/.aws/credentials" in found[1] and "NO_UPLOAD for aws" in found[1]  # copied to Lambda's machines too
    anywhere = sky_clouds.warnings([], use_spot=True, home=str(tmp_path))
    assert [w.split(":")[0] for w in anywhere] == ["aws", "azure", "lambda", "azure", "nebius"]
    assert "stopped, not deleted" in anywhere[-1]


def test_config_load_warns_once_per_key(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    write(tmp_path, "~/.lambda_cloud/lambda_keys")
    serve = {"provisioner": "skypilot", "model": "ow", "accelerators": "A10:1", "clouds": ["lambda"]}
    raw = {
        "backends": {n: {"kind": "self_hosted", "format": "openai", "gpu_cost_per_hour": 1.0, "serve": serve} for n in ("a", "b")},
        "models": {"ow": {"hf_id": "o/m", "quality": {"coding": 3}, "routes": [{"backend": "a", "model": "m"}]}},
    }
    with caplog.at_level(logging.WARNING, logger="meritorder.config"):
        build_config(raw, tmp_path, prices_data={})
    said = [r.getMessage() for r in caplog.records if "SkyPilot" in r.getMessage()]
    assert len(said) == 1 and said[0].startswith("SkyPilot (a): lambda: SkyPilot copies ~/.lambda_cloud/lambda_keys")
    caplog.clear()
    build_config(raw, tmp_path, prices_data={}, quiet=True)
    assert not [r for r in caplog.records if "SkyPilot" in r.getMessage()]
