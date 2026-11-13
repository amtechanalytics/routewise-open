"""A18: the cross-encoder (BAAI/bge-reranker-v2-m3) loads on the instance, its revision is recorded, and its
throughput is measured; the throughput guard picks top-50 or the pre-registered top-20."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2b", reason="launch-2 pilot check (run routewise-20261111T034648Z: PASS); study 3 launch 3a has its own pilot checks (D22, A21)")
import json

from conftest import fake


def test_rerank(runs):
    d = json.loads((runs / "pilot" / "diag_rerank.json").read_text())
    assert d["queries"] >= 1 and d["k"] in (50, 20), d
    if not fake():
        assert d["model"] == "BAAI/bge-reranker-v2-m3" and d["pairs_per_s"] > 0, d
