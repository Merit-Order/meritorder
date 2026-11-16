"""Provider keys from mounted files: picked up on change (no restart), with env fallback."""

import os

from meritorder.backends.auth import ApiKeyAuth
from meritorder.config import BackendAuth, BackendConfig, Config, read_secret_file


def write(path, text, bump=0):
    path.write_text(text)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump))  # make sure the change is visible


async def test_rotated_key_file_is_used_on_the_next_request(tmp_path):
    f = tmp_path / "OPENROUTER_API_KEY"
    write(f, "sk-old\n")
    auth = ApiKeyAuth(BackendAuth(type="api_key", value_file=str(f)))
    assert auth.available()
    assert (await auth.apply("POST", "u", {}, b""))["authorization"] == "Bearer sk-old"
    write(f, "sk-new-longer", bump=1_000_000)
    assert (await auth.apply("POST", "u", {}, b""))["authorization"] == "Bearer sk-new-longer"


def test_kubernetes_style_symlink_swap(tmp_path):
    """A mounted Secret's file is a symlink into a ..data directory that Kubernetes swaps atomically."""
    v1, v2 = tmp_path / "..2026_11_01", tmp_path / "..2026_11_09"
    v1.mkdir(), v2.mkdir()
    (v1 / "KEY").write_text("first")
    (v2 / "KEY").write_text("second")
    os.utime(v2 / "KEY", ns=(1, 2_000_000_000_000_000_000))
    data = tmp_path / "..data"
    data.symlink_to(v1.name)
    (tmp_path / "KEY").symlink_to("..data/KEY")
    assert read_secret_file(str(tmp_path / "KEY")) == "first"
    tmp = tmp_path / "..data_tmp"
    tmp.symlink_to(v2.name)
    os.replace(tmp, data)
    assert read_secret_file(str(tmp_path / "KEY")) == "second"


def test_file_falls_back_to_env_and_explains_what_is_missing(tmp_path, monkeypatch):
    missing = str(tmp_path / "nope")
    spec = BackendAuth(type="api_key", value_file=missing, value_env="MO_TEST_KEY")
    monkeypatch.delenv("MO_TEST_KEY", raising=False)
    assert spec.key() is None and spec.describe() == f"{missing} or MO_TEST_KEY"
    monkeypatch.setenv("MO_TEST_KEY", "from-env")
    assert spec.key() == "from-env"
    empty = tmp_path / "empty"
    empty.write_text("  \n")
    assert BackendAuth(type="api_key", value_file=str(empty), value_env="MO_TEST_KEY").key() == "from-env"

    monkeypatch.delenv("MO_TEST_KEY")
    from meritorder.config import _disable_unusable_backends

    cfg = Config(backends=[BackendConfig(name="p", format="openai", base_url="https://x.test/v1", auth=spec)], models=[])
    _disable_unusable_backends(cfg, quiet=True)
    assert not cfg.backends[0].enabled and cfg.backends[0].disabled_reason == f"{missing} or MO_TEST_KEY is not set"


def test_stored_config_may_reference_key_files():
    from meritorder.configstore import secret_problems

    doc = {"backends": {"p": {"format": "openai", "auth": {"type": "api_key", "value_file": "/run/secrets/meritorder/MERITORDER_BACKEND_K"}}}}
    assert secret_problems(doc) == []
    doc["backends"]["p"]["auth"]["value"] = "sk-literal"
    assert secret_problems(doc)


def test_stored_config_references_googles_client_secret():
    """The shipped config's ${MERITORDER_OIDC_CLIENT_SECRET} is where a secret belongs; a literal one isn't stored."""
    from meritorder.configstore import secret_problems

    doc = {"auth": {"oidc": {"issuer": "https://accounts.google.com", "public_client_secret": "${MERITORDER_OIDC_CLIENT_SECRET}"}}}
    assert secret_problems(doc) == []
    doc["auth"]["oidc"]["public_client_secret"] = "GOCSPX-literal"
    assert secret_problems(doc) == ["auth.oidc.public_client_secret holds a literal secret; use ${MERITORDER_OIDC_CLIENT_SECRET}"]
    doc["auth"]["oidc"] = {"issuer": "${MERITORDER_OIDC_CLIENT_SECRET}"}
    assert "outside a secret field" in secret_problems(doc)[0]


async def test_stored_config_cannot_pull_secrets_into_readable_fields(client, gateway):
    import copy

    import yaml
    from conftest import ADMIN_KEY, TEST_CONFIG, auth

    doc = copy.deepcopy(TEST_CONFIG)
    doc["auth"]["static_keys"] = [{"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
    doc["backends"]["mock-openai"]["base_url"] = "https://x.test/${MERITORDER_DATABASE_URL}"
    r = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY), json={"yaml": yaml.safe_dump(doc)})).json()
    assert r["ok"] is False and "MERITORDER_DATABASE_URL" in r["error"] and "outside a secret field" in r["error"]
    doc["backends"]["mock-openai"]["base_url"] = "https://${AZURE_OPENAI_ENDPOINT:-x.test}/v1"  # not a secret: fine
    r = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY), json={"yaml": yaml.safe_dump(doc)})).json()
    assert "outside a secret field" not in (r.get("error") or "")
