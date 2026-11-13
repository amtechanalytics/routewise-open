"""Tests for ROUTEWISE v2 launch 1 (paper/v2a.py): the Study 1 test sample is unchanged by the dev pool, card
footprints and descriptor facts are disjoint, L1 forms are unique partial references, L2 leak checks fire, the
EDGAR client sends the fair-access User-Agent at <= 5 requests/s and picks the right prior filings, parsers accept
and reject the right shapes, and the launch-1 ladder runs on the fixture pins with reproducible frozen artifacts.
Run with pytest, or directly: python tests/test_paper_v2a.py"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))


def _load_paper():
    """Load this repository's paper/ under its own name: the harness tests run against tests/demo, whose
    `paper` package would otherwise shadow it under pytest."""
    import importlib.util
    name = "routewise_paper"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "paper" / "__init__.py",
                                                      submodule_search_locations=[str(REPO / "paper")])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_load_paper()
import importlib   # noqa: E402

P = importlib.import_module("routewise_paper.prompts")
S = importlib.import_module("routewise_paper.sources")
T = importlib.import_module("routewise_paper.text")
V = importlib.import_module("routewise_paper.v2a")


def _q(i, src="fin", gold="d1", text="What was Acme's revenue?", **kw):
    return {"qid": f"{src}:{i}", "source": src, "text": text, "gold_docs": [gold], "evidence": ["x"],
            "meta": {"category": kw.get("cat", "")}, **{k: v for k, v in kw.items() if k != "cat"}}


def test_dev_pool_keeps_study1_sample():
    qs = [_q(i) for i in range(900)]
    assert S._sample(qs, 400)[:300] == S._sample(qs)
    cu = [_q(i, "cuad", gold=f"c{i % 50}", cat=f"k{i % 7}") for i in range(900)]
    assert S._sample(cu, 400)[:300] == S._sample(cu)
    assert len({(q["gold_docs"][0], q["meta"]["category"]) for q in S._sample(cu, 340)}) == 340


def test_footprint_and_facts_are_disjoint_from_card_and_evidence():
    import random
    rng = random.Random(7)
    vocab = ["".join(chr(97 + (k // 26 ** j) % 26) for j in range(4)) + "word" for k in range(3000)]
    words = " ".join(f"Item {i} says Northfield Corp " + " ".join(rng.sample(vocab, 9)) + f" in {2000 + i % 30}."
                     for i in range(500))
    doc = {"doc_id": "d1", "text": words}
    fp = V.footprint(words)
    assert len(words) > 30000 and fp[0] == (0, 6000) and fp[1][1] - fp[1][0] == 2000
    short = "x" * 5000
    assert V.footprint(short)[0] == (0, 1500) and V.footprint(short)[1][1] - V.footprint(short)[1][0] == 500
    target = next(x for x in words.split(". ") if x.startswith("Item 420 "))
    ev = [target]
    facts = V.sample_facts(doc, ev, seed="q1")
    assert len(facts) == V.N_FACTS
    for f in facts:
        assert not any(a <= f["pos"] < b for a, b in fp)
        assert not f["text"].startswith("Item 420 ")
    assert facts == V.sample_facts(doc, ev, seed="q1")
    assert facts != V.sample_facts(doc, ev, seed="q2")


def test_l1_forms_are_unique_partial_references():
    cat = [{"doc_id": "fin:AAPL", "title": "Apple Inc. (AAPL) annual report on Form 10-K", "aliases": ["AAPL", "Apple Inc."],
            "meta": {"company": "Apple Inc."}},
           {"doc_id": "fin:APLE", "title": "Apple Hospitality REIT, Inc. (APLE) annual report", "aliases": ["APLE",
            "Apple Hospitality REIT, Inc."], "meta": {"company": "Apple Hospitality REIT, Inc."}}]
    f = V.l1_form(cat[0], "fin", cat)
    assert f == "AAPL"
    q = {"qid": "fin:1", "text": "What drove Apple Inc.'s services revenue?", "gold_docs": ["fin:AAPL"]}
    txt, st = V.make_l1("fin", q, {d["doc_id"]: d for d in cat}, cat)
    assert st == "ok" and "AAPL" in txt and "Apple Inc" not in txt
    fda = [{"doc_id": "fda:1", "title": "Atorvastatin Calcium prescribing information (FDA label)",
            "aliases": ["Atorvastatin Calcium"], "meta": {}},
           {"doc_id": "fda:2", "title": "Rosuvastatin Calcium prescribing information (FDA label)",
            "aliases": ["Rosuvastatin Calcium"], "meta": {}}]
    assert V.l1_form(fda[0], "fda", fda) == "atorvastatin"
    cu = [{"doc_id": "c1", "title": "x", "aliases": ["LIMEENERGYCO"], "meta": {"kind": "distributor agreement",
                                                                             "date": "1999-09-09", "party": "LIMEENERGYCO"}},
          {"doc_id": "c2", "title": "y", "aliases": ["ACME"], "meta": {"kind": "distributor agreement",
                                                                     "date": "1999-09-09", "party": "ACME"}},
          {"doc_id": "c3", "title": "z", "aliases": ["NORTH"], "meta": {"kind": "supply agreement",
                                                                      "date": "2012-11-02", "party": "NORTH"}}]
    assert V.cuad_l1(cu[0], cu) is None                       # same kind and date as c2: not unique
    assert V.cuad_l1(cu[2], cu) == "the supply agreement dated 2012-11-02"


def test_l2_checks_catch_name_answer_and_placeholder_leaks():
    assert V.answer_leak("What is the starting dose?", "What is the 10 mg once daily starting dose?", "10 mg once daily")
    assert not V.answer_leak("Is 10 mg once daily the dose?", "Is 10 mg once daily the dose of the statin?",
                             "10 mg once daily")
    assert not V.answer_leak("Is it approved?", "Is it approved?", "Yes")
    long_ans = " ".join(f"w{i}" for i in range(60))
    assert V.answer_leak("q", "q " + " ".join(f"w{i}" for i in range(10, 20)), long_ans)
    with_name = "What did Lisinopril's label say?"
    d = {"doc_id": "fda:1", "title": "Lisinopril prescribing information (FDA label)", "aliases": ["Lisinopril"],
         "meta": {}}
    assert any(T.mentions(with_name, a) for a in d["aliases"])


def test_identity_terms_name_like_words_only(monkeypatch=None):
    fda = {"doc_id": "fda:1", "title": "TPOXX prescribing information (FDA label)", "aliases": ["TPOXX"], "meta": {},
           "text": ("TPOXX (tecovirimat) is indicated for smallpox disease caused by variola virus. " * 3
                    + "Smallpox antivirals are stored dry. " * 4)}
    other = {"doc_id": "fda:2", "title": "Other", "aliases": ["Other"], "meta": {},
             "text": "Smallpox antivirals and other drugs. " * 3}
    orig = V._df
    V._df = lambda source: {"tecovirimat": 1, "tpoxx": 1, "smallpox": 1, "variola": 1, "antivirals": 2}
    try:
        terms = V.identity_terms("fda", fda)
        assert "tecovirimat" in terms and "tpoxx" in terms
        assert "antivirals" not in terms            # descriptive, not within 40 characters of the name
        cu = {"doc_id": "c1", "title": "license agreement — ACMEINC (2001-01-01)", "aliases": ["ACMEINC"],
              "meta": {"party": "ACMEINC"},
              "text": "Diplomat shall pay KI. " * 6 + "the royalty shall be paid monthly. " * 5}
        V._df = lambda source: {"diplomat": 1, "ki": 1, "royalty": 1, "monthly": 2}
        t2 = V.identity_terms("cuad", cu)
        assert {"acmeinc", "diplomat", "ki"} <= set(t2) and "royalty" not in t2
        assert V.scrub("Diplomat's fee to KI", t2) == "[name] fee to [name]"
    finally:
        V._df = orig
    del other


def test_duplicate_and_empty_ids_get_no_card():
    docs = [{"doc_id": "fda:"}, {"doc_id": "fda:"}, {"doc_id": "fda:a"}, {"doc_id": "fda:b"}, {"doc_id": "fda:b"},
            {"doc_id": "fda:c"}]
    assert [d["doc_id"] for d in V.unique_docs(docs)] == ["fda:a", "fda:c"]
    man = {"filings": [{"ticker": "AAA", "fy": "2023", "file": "x", "accession": "1"},
                       {"ticker": "AAA", "fy": "2023", "file": "y", "accession": "2"}]}
    assert V.decoy_docs(man, []) == []      # no base document: nothing, and never two with one id


def test_masked_question_per_source():
    docs = {"fin:AAPL": {"doc_id": "fin:AAPL", "aliases": ["AAPL", "Apple Inc."], "meta": {}, "title": "t"},
            "mh:1": {"doc_id": "mh:1", "aliases": ["TechCrunch"], "meta": {"publisher": "TechCrunch"}, "title": "a"},
            "mh:2": {"doc_id": "mh:2", "aliases": ["Reuters"], "meta": {"publisher": "Reuters"}, "title": "b"}}
    m, own = V.masked_question("fin", {"text": "How did AAPL grow?", "gold_docs": ["fin:AAPL"]}, docs)
    assert m == "How did ⟨D1⟩ grow?" and own == ["fin:AAPL"]
    m, own = V.masked_question("mhrag", {"text": "Did TechCrunch report it before Reuters?",
                                         "gold_docs": ["mh:1", "mh:2"]}, docs)
    assert "⟨D1⟩" in m and "⟨D2⟩" in m and own == ["mh:1", "mh:2"]
    m, own = V.masked_question("cuad", {"text": "For the supply agreement of ACME dated 2001-01-01: Is there a cap?",
                                        "text_mask": "Is there a cap?", "gold_docs": ["c1"]}, docs)
    assert m == "For ⟨D1⟩: Is there a cap?"
    assert V.masked_question("fin", {"text": "What is revenue?", "gold_docs": ["fin:AAPL"]}, docs) == ("", [])


def test_choose_prior_filings():
    recent = {"form": ["10-K", "10-Q", "10-K", "10-K/A", "10-K", "10-K", "8-K"],
              "accessionNumber": ["a2025", "q", "a2024", "amend", "a2023", "a2022", "e"],
              "reportDate": ["2025-12-31", "2025-09-30", "2024-12-31", "2024-12-31", "2023-12-31", "2022-12-31",
                             "2025-01-01"],
              "filingDate": ["2026-02-01", "x", "2025-02-01", "2025-03-01", "2024-02-01", "2023-02-01", "x"],
              "primaryDocument": ["d25.htm", "q", "d24.htm", "am.htm", "d23.htm", "d22.htm", "e"]}
    match, prior = V.choose_prior(recent, "2024")
    assert match["accession"] == "a2024"
    assert [p["accession"] for p in prior] == ["a2023", "a2022"]
    assert V.choose_prior(recent, "2019") == (None, [])


def test_edgar_client_headers_rate_and_gzip():
    import urllib.request
    seen = []

    class R:
        def __init__(self, body, enc):
            self.body, self.headers = body, {"Content-Encoding": enc} if enc else {}

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=60):
        seen.append((time.time(), dict(req.header_items())))
        return R(gzip.compress(b'{"ok": 1}'), "gzip")
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_open
    try:
        c = V.EdgarClient()
        for _ in range(4):
            assert json.loads(c.get("https://data.sec.gov/x")) == {"ok": 1}
    finally:
        urllib.request.urlopen = orig
    ua = {k.lower(): v for k, v in seen[0][1].items()}["user-agent"]
    assert ua == "anon Analytics anon@example.org"
    gaps = [b[0] - a[0] for a, b in zip(seen, seen[1:])]
    assert min(gaps) >= 1.0 / V.EDGAR_RATE - 0.01


def test_fiscal_period_from_dei_and_cover_text():
    raw = ('<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c">2024</ix:nonNumeric>'
           '<ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="c">December 31, 2024</ix:nonNumeric>')
    assert S.fiscal_period(raw, "") == {"fy": "2024", "period_end": "December 31, 2024", "how": "dei"}
    txt = "ANNUAL REPORT PURSUANT TO SECTION 13 For the fiscal year ended September 28, 2024 Commission file"
    assert S.fiscal_period("", txt)["fy"] == "2024" and S.fiscal_period("", txt)["how"] == "text"
    assert S.fiscal_period("", "no period here")["how"] == "none"


def test_card_and_rewrite_parsers_and_grounding():
    good = {"name": "Acme", "aliases": ["ACME"], "doc_type": "10-K", "date": "2024", "entities": ["Acme"],
            "facts": [{"fact": "f", "quote": "Acme shipped 400 units to Ohio"}], "summary": "An annual report."}
    c = P.parse_card(json.dumps(good))
    assert c["name"] == "Acme" and c["facts"][0]["quote"].startswith("Acme")
    # lenient: optional fields may be missing or malformed; only a non-object reply is rejected
    lax = P.parse_card(json.dumps({"name": "Xcel", "aliases": "XEL", "facts": [{"fact": "a", "quote": "Xcel is a holding"},
                                                                         {"fact": "no quote"}, 7]}))
    assert lax["aliases"] == ["XEL"] and len(lax["facts"]) == 1 and lax["summary"].startswith("Xcel")
    assert "summary" in lax["lenient"] and "doc_type" in lax["lenient"]
    assert P.parse_card(json.dumps({"summary": "x"}))["facts"] == []
    for bad in ("[1, 2]", "not json", ""):
        try:
            P.parse_card(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass
    g = V.ground_card(c, "In 2024 ACME SHIPPED  400 units to ohio and more.")
    assert len(g["facts"]) == 1 and g["facts_dropped"] == 0
    g = V.ground_card({**c, "facts": [{"fact": "f", "quote": "Acme shipped 900 units"}]}, "Acme shipped 400 units")
    assert not g["facts"] and g["facts_dropped"] == 1
    assert P.parse_rewrite('{"question": "What drove revenue at the CAD reseller?", "facts_used": [1, 2]}')
    for bad in ('{"question": "What drove ⟨D1⟩ revenue?", "facts_used": []}', '{"facts_used": [1]}', '{"question": "", "facts_used": []}'):
        try:
            P.parse_rewrite(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert P.parse_rewrite('{"question": "What drove revenue here?", "facts_used": ["1"]}')["facts_used"] == []
    item = P.rewrite_item("How did ⟨D1⟩ grow?", [("⟨D1⟩", "a report", ["f one", "f two"])],
                          ["Apple", "AAPL"], "")
    assert item.startswith("Question: How did") and "Forbidden names: Apple; AAPL" in item
    assert P.CARD_PREFIX and P.REWRITE_PREFIX and "⟨" not in P.REWRITE_PREFIX.split("such as")[0]


_LADDER = r'''
import json, os, sys
from pathlib import Path
from rh import runner, datapins, coverage
from rh.context import Context
from rh.heartbeat import Heartbeat
from rh.util import ensure_dirs
ensure_dirs()
ctx = Context(pins_file="data_pins.fixture.json")
assert not ctx.cfg.errors, ctx.cfg.errors
ctx.hb = Heartbeat()
datapins.verify_all(ctx.cfg.pins)
rep = coverage.report(ctx)
assert not coverage.problems(rep), coverage.problems(rep)
for mode in ("pilot", "canary", "full"):
    runner._stages(ctx, mode)
ctx.mode = "full"
runner._write_plan(ctx)
runner.rung_analysis(ctx)
print(json.dumps({"results": sorted(p.name for p in Path("results").iterdir())}))
'''


def test_launch1_ladder_on_fixtures_and_frozen_artifacts_reproduce():
    paper = sys.modules.get("routewise_paper")
    if getattr(paper, "STUDY", "v2a") != "v2a":
        try:
            import pytest
            pytest.skip("launch-1 ladder (run routewise-20261111T002456Z); launch 2 is covered by the L14 dry run")
        except ImportError:
            return
    tmp = Path(tempfile.mkdtemp(prefix="rw-v2a-")) / "repo"
    shutil.copytree(REPO, tmp, ignore=shutil.ignore_patterns(".git", ".venv", "runs", "results", "logs", ".data",
                                                             "data", "__pycache__", ".fake_aws"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("RH_")}
    env.update(RH_ROOT=str(tmp), RH_FAKE_VENDOR="1", RH_FAKE_AWS="1", RH_RUN_ID="v2a-test",
               RH_FAKE_AWS_DIR=str(tmp / ".fake_aws"))
    r = subprocess.run([sys.executable, "-c", _LADDER], cwd=tmp, env=env, capture_output=True, text=True,
                       timeout=1800)
    assert r.returncode == 0, (r.stdout[-2000:] + r.stderr[-4000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for f in ("v2a_cards.jsonl", "v2a_rewrites.jsonl", "v2a_edgar_manifest.json", "v2a_frozen_artifacts.json",
              "v2a_rewrite_audit_sheet.csv", "v2a_rewrite_summary.csv", "v2a_card_summary.csv"):
        assert f in out["results"], f
    res = tmp / "results"
    frozen = json.loads((res / "v2a_frozen_artifacts.json").read_text())
    for name, meta in frozen.items():
        assert hashlib.sha256((res / name).read_bytes()).hexdigest() == meta["sha256"], name
    rows = [json.loads(x) for x in (res / "v2a_rewrites.jsonl").read_text().splitlines()]
    assert {r["source"] for r in rows} == {"fin", "fda", "cuad", "mhrag", "finmy"}
    for x in rows:
        if x["level"] == "L2" and x["status"] == "ok":
            assert "⟨" not in x["text"]
    a = {p.name: p.read_bytes() for p in res.iterdir()}
    r2 = subprocess.run([sys.executable, "-c", "from rh.context import Context\nctx=Context(mode='full')\n"
                         "ctx.paper.analyze(ctx)"], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stderr[-3000:]
    b = {p.name: p.read_bytes() for p in res.iterdir()}
    assert a == b, sorted(k for k in a if a[k] != b.get(k))
    shutil.rmtree(tmp.parent, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {str(e)[:2000]}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
