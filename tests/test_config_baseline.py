"""A baseline model that doesn't exist is a config error (it would silently zero every savings figure)."""

import copy

import pytest
from conftest import TEST_CONFIG

from meritorder.configstore import ConfigError, parse_and_validate


@pytest.mark.parametrize("where", ["org", "team"])
def test_unknown_baseline_model_is_rejected(tmp_path, where):
    doc = copy.deepcopy(TEST_CONFIG)
    doc["auth"] = {}
    if where == "org":
        doc["baseline_model"] = "no-such-model"
    else:
        doc["teams"]["default"]["baseline_model"] = "embed-small"  # exists, but isn't a chat model
    with pytest.raises(ConfigError, match="isn't a configured chat model"):
        parse_and_validate(None, str(tmp_path), None, document=doc)
    doc["baseline_model"], doc["teams"]["default"]["baseline_model"] = "big", "small"
    parse_and_validate(None, str(tmp_path), None, document=doc)
