"""D13: the index stage wrote finite corpus and workload features for all fifteen configurations, and key masking
lowers the extractor-free key-mention rate where it is applied."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "study1") != "study1", reason="Study 1 pilot check (run routewise-20261110T035239Z); v2 has its own pilot checks (D20-D21, A17-A20)")

import json
import math


def test_features(runs):
    f = json.loads((runs / "pilot" / "features.json").read_text())
    assert len(f) == 15, sorted(f)
    for c, v in f.items():
        assert v["n_chunks"] > 0, c
        for k in ("confusion", "near_dup_rate", "separability", "key_uniqueness", "mention_rate", "ambiguity"):
            assert v[k] is not None and not math.isnan(v[k]), (c, k)
        assert -1.0 <= v["confusion"] <= 1.0
    for s in ("fin", "fda", "cuad"):
        assert f[f"{s}.mask"]["mention_rate"] < f[f"{s}.base"]["mention_rate"], s
