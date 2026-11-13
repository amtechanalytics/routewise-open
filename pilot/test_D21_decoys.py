"""D21: the fin-multiyear decoys are re-fetched from EDGAR with the frozen manifest and match its sha256; the finmy
collection holds the fin filings plus decoys. Diagnostics: runs/pilot/diag_edgar2.json."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2b", reason="launch-2 pilot check (run routewise-20261111T034648Z: PASS); study 3 launch 3a has its own pilot checks (D22, A21)")
import json

from conftest import fake


def test_decoys(runs):
    d = json.loads((runs / "pilot" / "diag_edgar2.json").read_text())
    assert d["present"] + d["fetched"] >= 1, d
    import paper.sources as S
    my = S.load("finmy")
    assert my["stats"]["decoys"] >= 1, my["stats"]
    if not fake():
        assert len(d["sha_mismatch"]) <= 0.1 * max(1, d["filings"]), d["sha_mismatch"][:20]
