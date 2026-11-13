"""Tests for ROUTEWISE study 3 launch 3a (paper/v2c.py): the fresh pool is disjoint from Study 1/2, qasper L1 forms
are unique, the L2 leak and uniqueness checks, fact ranking by distinctiveness, the screen rule (amendment 1) with
its split and caps, and the blind rating sample. Run with pytest, or directly: python tests/test_paper_v2c.py"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))


def _load_paper():
    """Load this repository's paper/ under its own name (tests/demo has a `paper` package that would shadow it)."""
    name = "routewise_paper"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "paper" / "__init__.py",
                                                      submodule_search_locations=[str(REPO / "paper")])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_load_paper()
C = importlib.import_module("routewise_paper.v2c")
B = importlib.import_module("routewise_paper.v2b")
S = importlib.import_module("routewise_paper.sources")
A = importlib.import_module("routewise_paper.v2a")


def _q(qid, src="cuad"):
    return {"qid": qid, "source": src, "text": "q " + qid, "gold_docs": ["d1"], "evidence": ["e"], "answer": "a"}


def test_fresh_pool_disjoint():
    orig_load, orig_frozen, orig_fake = S.load, B.frozen, C.fake
    data = {s: {"docs": [], "queries": [_q(f"{s}:{i}", s) for i in range(3)],
                "dev": [_q(f"{s}:{i}", s) for i in range(3, 5)],
                "fresh": [_q(f"{s}:{i}", s) for i in range(5, 9)]} for s in C.SOURCES3}
    S.load = lambda s: data[s]
    B.frozen = lambda: {"rewrites": [{"qid": "cuad:1"}]}
    C.fake = lambda: False
    C.used_qids.cache_clear()
    try:
        used = C.used_qids()
        for s in C.SOURCES3:
            pool = C.fresh_pool(s)
            assert [q["qid"] for q in pool] == [f"{s}:{i}" for i in range(5, 9)]
            assert not {q["qid"] for q in pool} & used
        assert "cuad:1" in used and "fda:4" in used
    finally:
        S.load, B.frozen, C.fake = orig_load, orig_frozen, orig_fake
        C.used_qids.cache_clear()


def test_split_is_stable_and_about_one_in_five():
    sp = [C.split_of(f"q{i}") for i in range(2000)]
    assert sp == [C.split_of(f"q{i}") for i in range(2000)]
    assert 0.15 < sp.count("dev") / len(sp) < 0.25


def test_qasper_l1_unique():
    cat = [{"doc_id": "a", "title": "Neural Sarcasm Detection on Reddit"},
           {"doc_id": "b", "title": "Neural Machine Translation for Low Resource Languages"},
           {"doc_id": "c", "title": "Sarcasm in Tweets"}]
    form = C.qasper_l1(cat[0], cat)
    assert form == "reddit"                      # "neural" and "sarcasm" occur in other titles
    import re
    for d in cat[1:]:
        assert not re.search(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])", d["title"].lower())
    assert C.qasper_l1({"doc_id": "x", "title": "Short"}, cat) is None


def test_l2_fails():
    orig_rank, orig_ids = A.doc_rank, C.id_tokens
    A.doc_rank = lambda src, q, k: ["d1"] if "lab maker" in q else ["d9"]
    C.id_tokens = lambda src, d: {"biorad"}
    try:
        by_id = {"d1": {"doc_id": "d1", "title": "Bio-Rad 10-K", "aliases": ["Bio-Rad"],
                        "meta": {"company": "Bio-Rad Laboratories"}}}
        r = {"gold_docs": ["d1"], "orig": "What did Bio-Rad report?", "answer": "12 months"}
        assert C.l2_fails("fin", r, "What did the lab maker report?", by_id) == []
        assert "name_leak" in C.l2_fails("fin", r, "What did Bio-Rad, the lab maker, report?", by_id)
        assert "not_unique" in C.l2_fails("fin", r, "What did the company report?", by_id)
        assert "answer_leak" in C.l2_fails("fin", r, "Did the lab maker report 12 months?", by_id)
        assert "placeholder_left" in C.l2_fails("fin", r, "What did ⟨D1⟩ the lab maker report?", by_id)
    finally:
        A.doc_rank, C.id_tokens = orig_rank, orig_ids


def test_distinctiveness_and_attempts():
    nb = frozenset({"revenue", "increased", "company"})
    assert C.distinctiveness("Revenue increased for the company", nb, set()) == 0.0
    assert C.distinctiveness("The Zorvex implant received clearance", nb, {"zorvex"}) == 1.0
    orig = C._scored
    C._scored = lambda src, did: tuple((i, f"fact {i}", d, False) for i, d in
                                       enumerate([0.1, 0.9, 0.5, 0.7, 0.2, 0.8, 0.3]))
    try:
        r = C.ranked_facts("cuad", {"doc_id": "d1"}, seed="s")
        assert [x["distinct"] for x in r[:6]] == [0.9, 0.8, 0.7, 0.5, 0.3, 0.2]
        first, second = {x["pos"] for x in r[:3]}, {x["pos"] for x in r[3:6]}
        assert not first & second
    finally:
        C._scored = orig


class _Ctx:
    mode = "full"


def _rows_fixture():
    its = []
    for i in range(6):
        base = {"source": "cuad", "qid": f"cuad:{i}", "split": "dev" if i == 0 else "test", "orig": "o", "answer": "a",
                "evidence": ["e"], "gold_docs": ["d1"]}
        its.append({**base, "level": "L0", "text": "o", "status": "ok"})
        its.append({**base, "level": "L1", "text": f"t{i}", "status": "ok" if i != 5 else "answer_leak"})
    return its


def test_screen_rule_and_assign():
    o_items, o_oc, o_rows = C.items, C.oracle_correct, B._rows_by
    its = _rows_fixture()
    C.items = lambda ctx: its
    oc = {}
    for i in range(6):
        oc[f"o3|cuad|L0|cuad:{i}"] = i != 1          # 1: oracle wrong on the original
        oc[f"o3|cuad|L1|cuad:{i}"] = i != 2          # 2: oracle wrong on the rewrite
    C.oracle_correct = lambda ctx: oc
    votes = {}
    for i in range(6):
        for m in (C.SONNET, C.OSS):
            v = {"same_answer": True, "identifiable": False, "no_leak": True, "plausible": not (i == 3 and m == C.OSS)}
            votes[f"v3|cuad|L1|cuad:{i}|{m}"] = {"status": "ok", "parsed": v}
    B._rows_by = lambda ctx, stage: votes if stage == "panel3" else {}
    try:
        rows = {r["qid"]: r for r in C.screen_rows(_Ctx())}
        assert rows["cuad:0"]["accepted"] and rows["cuad:4"]["accepted"]       # identifiable=False does not gate
        assert rows["cuad:1"]["reason"] == "oracle_wrong_on_original"
        assert rows["cuad:2"]["reason"] == "oracle_wrong_on_rewrite"
        assert rows["cuad:3"]["reason"] == "panel_rejects"
        assert rows["cuad:5"]["reason"] == "answer_leak" and not rows["cuad:5"]["leak_ok"]
        cells = C.assign(list(rows.values()))
        assert [r["qid"] for r in cells["cuad.L1"]["dev"]] == ["cuad:0"]
        assert [r["qid"] for r in cells["cuad.L1"]["test"]] == ["cuad:4"]
    finally:
        C.items, C.oracle_correct, B._rows_by = o_items, o_oc, o_rows


def test_assign_caps():
    rows = [{"source": "cuad", "level": "L2", "qid": f"q{i}", "split": "test", "accepted": True} for i in range(450)]
    rows += [{"source": "cuad", "level": "L2", "qid": f"d{i}", "split": "dev", "accepted": True} for i in range(130)]
    c = C.assign(rows)["cuad.L2"]
    assert len(c["test"]) == C.TEST_PER_CELL and len(c["dev"]) == C.DEV_PER_CELL


def test_rating_sample_blind_mix():
    acc = {f"{s}.L1": {"test": [{"source": s, "level": "L1", "qid": f"{s}:{i}", "accepted": True, "leak_ok": True,
                                 "reason": "ok"} for i in range(60)], "dev": []} for s in ("cuad", "fda", "qasper")}
    rows = [r for c in acc.values() for r in c["test"]]
    rows += [{"source": "cuad", "level": "L2", "qid": f"rej{i}", "accepted": False, "leak_ok": True,
              "reason": "panel_rejects"} for i in range(80)]
    rows += [{"source": "cuad", "level": "L2", "qid": f"leak{i}", "accepted": False, "leak_ok": False,
              "reason": "name_leak"} for i in range(80)]
    s = C.rating_sample(rows, acc)
    assert sum(r["screened"] == "accepted" for r in s) == C.RATE_ACCEPTED
    assert sum(r["screened"] == "rejected" for r in s) == C.RATE_REJECTED
    assert not any(r["qid"].startswith("leak") for r in s)
    assert s != sorted(s, key=lambda r: r["screened"])           # mixed, not grouped


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {n}: {type(e).__name__}: {str(e)[:1500]}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
