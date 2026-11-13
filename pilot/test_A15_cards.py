"""A15: profile cards come back and the grounding check runs on them. Launch 1 must finish unattended (anon,
10 Nov), so the pilot only requires that cards parse for most probe documents and most card sources, and that some
cards keep a grounded fact; per-source shortfalls are warnings in the brief, not failures."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2a", reason="launch-1 pilot check (run routewise-20261111T002456Z: PASS); launch 2 has its own pilot checks (D20-D21, A17-A20)")

import json

from conftest import fake, read_jsonl


def test_cards(runs):
    rows = read_jsonl(runs / "pilot" / "cards_checked.jsonl")
    assert rows, "cardcheck wrote no cards"
    srcs = {r["source"] for r in rows}
    import paper.v2a as V
    assert len(srcs) >= len(V.CARD_SOURCES) - 2, f"card sources {sorted(srcs)} (of {sorted(V.CARD_SOURCES)})"
    stats = json.loads((runs / "pilot" / "diag_cardcheck.json").read_text())
    docs, parsed = sum(st["docs"] for st in stats.values()), sum(st["parsed"] for st in stats.values())
    assert parsed >= 0.5 * docs, f"cards parsed for {parsed} of {docs} probe documents"
    if not fake():
        assert any(r["card"]["facts"] for r in rows), "no card kept a grounded fact"
