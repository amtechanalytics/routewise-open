"""A19: every arm of every launch-2 configuration retrieves for the probe queries, ROUTEWISE-R records a decision,
and the calibration file exists (LOSO, per held-out source)."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2b", reason="launch-2 pilot check (run routewise-20261111T034648Z: PASS); study 3 launch 3a has its own pilot checks (D22, A21)")
import json

from conftest import read_jsonl


def test_retrieve(runs):
    import paper.v2b as B
    rows = read_jsonl(runs / "pilot" / "retrieve2.jsonl")
    assert rows, "no retrieve rows"
    have = {(r["config"], r["arm"]) for r in rows}
    cfgs = {r["config"] for r in rows}
    missing = [(c, a) for c in cfgs for a in B.arms_for(c) if (c, a) not in have]
    assert not missing, missing[:20]
    assert all(r.get("decision") in ("hard", "soft", "fallback") for r in rows if r["arm"] in B.R_ARMS)
    cal = json.loads((runs / "pilot" / "calibration.json").read_text())
    assert "_all" in cal
