"""A16: L2 rewrites parse for every rewrite source, leave no placeholder, and the automatic checks ran on each
(L0, L1 and L2 rows per probe query; fin also yields fin-multiyear rows). The share passing is reported, not
asserted, in the pilot (2 probes per source). Launch 1 must finish unattended (anon, 10 Nov): one accepted L2
rewrite overall is enough here; shortfalls are warnings in the brief."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2a", reason="launch-1 pilot check (run routewise-20261111T002456Z: PASS); launch 2 has its own pilot checks (D20-D21, A17-A20)")

from conftest import read_jsonl


def test_rewrites(runs):
    rows = read_jsonl(runs / "pilot" / "rewrites.jsonl")
    assert rows, "rewritecheck wrote nothing"
    import paper.v2a as V
    for s in V.REWRITE_SOURCES:
        lv = {r["level"] for r in rows if r["source"] == s}
        assert lv == {"L0", "L1", "L2"}, (s, lv)
    for r in rows:
        if r["level"] == "L2" and r["status"] == "ok":
            assert "⟨" not in r["text"] and r["text"] != r["original"], r
    gen = read_jsonl(runs / "pilot" / "rewrite.jsonl")
    assert gen and any(g["status"] == "ok" for g in gen), [g.get("status") for g in gen]
