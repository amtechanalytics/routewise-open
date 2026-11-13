"""D17: EDGAR is reachable from the instance with the fair-access User-Agent, and prior-year 10-K decoys are found
and saved for the probe companies; the manifest records accession, report date, sha256 and size of every filing.
Diagnostics are in runs/pilot/diag_edgar.json (written by the edgar stage before its gate)."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2a", reason="launch-1 pilot check (run routewise-20261111T002456Z: PASS); launch 2 has its own pilot checks (D20-D21, A17-A20)")

import json

from conftest import fake


def test_edgar(runs):
    man = json.loads((runs / "pilot" / "edgar_manifest.json").read_text())
    assert man["user_agent"].startswith("anon Analytics ")
    fil = man["filings"]
    assert fil, f"no decoy filing fetched; skipped: {man['skipped']}"
    for f in fil:
        assert f["sha256"] and f["size"] > 0 and f["fy"] and f["file"], f
    if not fake():
        assert all(f["accession"] != f.get("matched_accession") for f in fil), "a decoy is the FinDER filing itself"
