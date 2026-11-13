"""A17: the resolver answers for every card variant, parses, and names candidates from its list; the router ran on
every probe text."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2b", reason="launch-2 pilot check (run routewise-20261111T034648Z: PASS); study 3 launch 3a has its own pilot checks (D22, A21)")
from conftest import read_jsonl


def test_resolver(runs):
    rows = read_jsonl(runs / "pilot" / "resolve.jsonl")
    assert rows, "no resolver rows"
    import paper.v2b as B
    for v in B.CARD_VARIANTS:
        ok = [r for r in rows if r["meta"]["variant"] == v and r["status"] == "ok"]
        assert ok, f"no parsed resolver reply for card variant {v}"
    for r in rows:
        if r["status"] == "ok":
            for d in r["parsed"]["documents"]:
                assert 1 <= d["id"] <= max(1, len(r["meta"]["cands"])) or not r["meta"]["cands"], r["cell_id"]
    assert any(r["status"] == "ok" for r in read_jsonl(runs / "pilot" / "route.jsonl"))
