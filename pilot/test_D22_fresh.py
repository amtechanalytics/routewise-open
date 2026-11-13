"""D22: study 3 questions are fresh: none was used in Study 1 or Study 2 (test sample, dev pool or the frozen
launch-1 rewrites), every screened collection has a fresh pool, and the candidate stage found usable L1/L2
candidates. Diagnostics: runs/pilot/diag_cand3.json (written by the cand3 stage)."""
import json

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2c", reason="study 3 launch 3a pilot check")

from conftest import fake  # noqa: E402


def test_fresh_pool(runs):
    import paper.v2c as C
    d = json.loads((runs / "pilot" / "diag_cand3.json").read_text())
    assert d["used_overlap"] == 0, d
    for s in C.SOURCES3:
        assert d["originals"].get(s, 0) >= 1, f"{s}: no fresh original in the pilot probe"
    if not fake():
        used = C.used_qids()
        for s in C.SOURCES3:
            pool = C.fresh_pool(s)
            assert len(pool) >= 100, f"{s}: fresh pool of {len(pool)} questions"
            assert not {q["qid"] for q in pool} & used, f"{s}: fresh pool overlaps Study 1/2"
    ok_cfgs = [c for c, st in d["status"].items() if st.get("ok")]
    assert len(ok_cfgs) >= 5, f"too few configurations with usable candidates: {d['status']}"
