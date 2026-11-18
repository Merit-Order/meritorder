"""The console demo's AWS site (deploy/aws/demo-site.yaml): the CSP it sends is the one the demo's smoke test runs under,
and the deploy role trusts only main of the two repositories."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "deploy" / "aws" / "demo-site.yaml"


class _Loader(yaml.SafeLoader):
    """Reads CloudFormation's short tags (!Ref, !Sub, ...) as plain values."""


_Loader.add_multi_constructor("!", lambda loader, suffix, node: (
    loader.construct_scalar(node) if isinstance(node, yaml.ScalarNode)
    else loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_mapping(node)))


def template() -> dict:
    return yaml.load(TEMPLATE.read_text(), Loader=_Loader)


def test_csp_is_the_one_the_demo_smoke_test_runs_under():
    config = (ROOT / "ui" / "vite.config.ts").read_text()
    body = re.search(r"const DEMO_CSP = ((?:\s*\+?\s*\"[^\"]*\")+);", config)
    assert body
    vite = "".join(re.findall(r"\"([^\"]*)\"", body.group(1)))
    headers = template()["Resources"]["SecurityHeaders"]["Properties"]["ResponseHeadersPolicyConfig"]["SecurityHeadersConfig"]
    aws = " ".join(headers["ContentSecurityPolicy"]["ContentSecurityPolicy"].split())
    assert aws == vite
    assert "script-src 'self';" in aws  # 404.html's redirect is a file (404.js), so no inline script is allowed


def test_deploy_role_is_main_only_and_narrow():
    res = template()["Resources"]
    trust = res["DeployRole"]["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert len(trust) == 1
    cond = trust[0]["Condition"]["StringEquals"]
    assert cond["token.actions.githubusercontent.com:aud"] == "sts.amazonaws.com"
    subjects = template()["Parameters"]["GitHubSubjects"]["Default"].split(",")
    assert subjects == ["repo:therick@319326/meritorder@1459849005:ref:refs/heads/main",  # the immutable subject this repo sends
                        "repo:therick/meritorder:ref:refs/heads/main"]
    actions = sorted(a for s in res["DeployRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
                     for a in (s["Action"] if isinstance(s["Action"], list) else [s["Action"]]))
    assert actions == ["cloudfront:CreateInvalidation", "s3:DeleteObject", "s3:ListBucket", "s3:PutObject"]
