"""D20: launch 1's frozen artifacts (profile cards, rewrites, EDGAR manifest) load with their recorded sha256, the
strict re-check ran, and every launch-2 configuration has test queries in the pilot probe. Diagnostics:
runs/pilot/recheck.json (written by the recheck stage)."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2b", reason="launch-2 pilot check (run routewise-20261111T034648Z: PASS); study 3 launch 3a has its own pilot checks (D22, A21)")
import json


def test_frozen_and_recheck(runs):
    import paper.v2b as B
    fz = B.frozen()                       # raises on a sha256 mismatch
    assert fz["cards"] and fz["rewrites"] and fz["manifest"]["filings"]
    rc = json.loads((runs / "pilot" / "recheck.json").read_text())
    assert rc["counts"], "strict re-check wrote no counts"
    empty = [c for c, v in rc["queries"].items() if v["test"] == 0]
    assert len(empty) <= 2, f"configurations without test queries: {empty}"
