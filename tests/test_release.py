"""What a release publishes agrees with itself: one version everywhere it's written, the chart's default image tag is one
the release workflow pushes, and the image variants never take the main image's tags. And anyone using a gateway can
find its source (GET /about, console.source_url)."""

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from meritorder import __version__
from meritorder.config import SOURCE_URL, ConsoleConfig, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_one_version_everywhere():
    chart = yaml.safe_load((ROOT / "charts/meritorder/Chart.yaml").read_text())
    package = json.loads((ROOT / "ui/package.json").read_text())
    lock = json.loads((ROOT / "ui/package-lock.json").read_text())
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    uv_lock = re.search(r'name = "meritorder"\nversion = "([^"]+)"', (ROOT / "uv.lock").read_text())
    assert uv_lock is not None
    versions = {pyproject["project"]["version"], __version__, package["version"], lock["version"], lock["packages"][""]["version"],
                chart["version"], chart["appVersion"], uv_lock.group(1)}
    assert versions == {__version__}


def test_the_charts_default_image_tag_is_one_the_release_pushes():
    """The chart's image tag defaults to its appVersion; tag v<version> pushes <version>, <version>-gpu and
    <version>-compressor (docker/metadata-action's type=semver, pattern {{version}}, which drops the v)."""
    values = yaml.safe_load((ROOT / "charts/meritorder/values.yaml").read_text())
    helpers = (ROOT / "charts/meritorder/templates/_helpers.tpl").read_text()
    assert values["image"]["tag"] == "" and ".Values.image.tag | default .Chart.AppVersion" in helpers
    assert values["compressor"]["imageSuffix"] == "-compressor"
    steps = {s.get("id"): s for s in yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]["image"]["steps"]}
    for meta, suffix in (("meta", ""), ("meta-gpu", "-gpu"), ("meta-compressor", "-compressor")):
        w = steps[meta]["with"]
        assert w["tags"] == "type=semver,pattern={{version}}"
        if suffix:
            # Without onlatest=true the suffix skips `latest`, and a variant would push a plain `latest` over the main image's.
            assert w["flavor"] == f"suffix={suffix},onlatest=true"
        else:
            assert "flavor" not in w


async def test_about_says_where_the_source_is_without_a_key(client, gateway):
    r = await client.get("/about")
    assert r.status_code == 200
    assert r.json() == {"name": "Merit Order", "version": __version__, "license": "AGPL-3.0-only",
                        "source_url": SOURCE_URL, "third_party_notices": "/ui/third-party-notices.txt"}
    gateway.cfg.console = ConsoleConfig(source_url="https://git.example.com/our/meritorder")  # a changed version's own source
    assert (await client.get("/about")).json()["source_url"] == "https://git.example.com/our/meritorder"


def test_the_source_url_is_an_address(monkeypatch):
    assert ConsoleConfig(source_url="  ").source_url == SOURCE_URL
    with pytest.raises(ValidationError, match="must be an http"):
        ConsoleConfig(source_url="git@github.com:me/fork")
    monkeypatch.delenv("MERITORDER_SOURCE_URL", raising=False)
    assert load_config(ROOT / "config/meritorder.yaml").console.source_url == SOURCE_URL
    monkeypatch.setenv("MERITORDER_SOURCE_URL", "https://git.example.com/fork")
    assert load_config(ROOT / "config/meritorder.yaml").console.source_url == "https://git.example.com/fork"
