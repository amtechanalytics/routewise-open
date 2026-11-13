"""Study tests for the routewise paper (pure functions + the study ladder on the fixture pins).

They load paper/ from this repository (the harness tests run against tests/demo), so the package is imported
under its own name. Run with pytest, or directly: python tests/test_paper_routewise.py"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _pkg():
    name = "routewise_paper"
    if name in sys.modules:
        return sys.modules[name]
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    spec = importlib.util.spec_from_file_location(name, REPO / "paper" / "__init__.py",
                                                  submodule_search_locations=[str(REPO / "paper")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mod(sub):
    _pkg()
    return importlib.import_module(f"routewise_paper.{sub}")


# --------------------------------------------------------------------------- text
def test_chunk_covers_text_with_overlap():
    T = _mod("text")
    words = " ".join(f"w{i}" for i in range(2000))
    ch = T.chunk(words, 300, 60)
    assert all(len(c) <= 300 for c in ch)
    assert ch[0].startswith("w0 ") and ch[-1].endswith("w1999")
    joined = set(" ".join(ch).split())
    assert joined == set(words.split())
    assert any(set(a.split()) & set(b.split()) for a, b in zip(ch, ch[1:]))


def test_mentions_and_mask():
    T = _mod("text")
    assert T.mentions("Capex for KO rose", "KO")
    assert not T.mentions("They know the cost", "KO")
    assert T.mentions("What did Apple Inc. report?", "Apple Inc.")
    m = T.mask_aliases("What did Apple Inc. (AAPL) report? AAPL's capex", ["AAPL", "Apple Inc."], "the company")
    assert "AAPL" not in m and "Apple" not in m and "the company" in m


def test_fuzzy_and_evidence_recall():
    T = _mod("text")
    assert T.fuzzy_score("Microsoft", "Microsoft Corporation") >= 0.9
    assert T.fuzzy_score("Coca-Cola", "The Coca-Cola Company") >= 0.9
    assert T.fuzzy_score("Pfizer", "Microsoft Corporation") < 0.5
    ev = ["Total net revenue was 394.3 billion dollars"]
    assert T.evidence_recall(ev, ["the total net revenue was 394.3 billion dollars in 2023"]) == 1.0
    assert T.evidence_recall(ev, ["unrelated passage about weather"]) == 0.0


# --------------------------------------------------------------------------- prompts / schemas
def test_schemas_accept_valid_and_reject_invalid():
    P = _mod("prompts")
    assert P.parse_router('{"entity": "AAPL", "qualifier": null, "confidence": 0.9}')["entity"] == "AAPL"
    assert P.parse_router('```json\n{"entity": null, "qualifier": null, "confidence": 0.1}\n```')["entity"] is None
    for bad in ('{"entity": 3, "qualifier": null, "confidence": 0.5}', '{"entity": "x", "confidence": 2}', "nope"):
        try:
            P.parse_router(bad)
            raise AssertionError(f"accepted {bad}")
        except (ValueError, TypeError):
            pass
    assert P.parse_gen('{"answer": "NOT FOUND", "found": false}')["found"] is False
    assert P.parse_judge('{"score": 4, "correct": true, "category": "correct"}')["score"] == 4
    for bad in ('{"score": 6, "correct": true, "category": "correct"}',
                '{"score": 3, "correct": "yes", "category": "partial"}',
                '{"score": 3, "correct": false, "category": "meh"}'):
        try:
            P.parse_judge(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_prefixes_are_stable_and_items_carry_the_variable_part():
    P = _mod("prompts")
    a = P.gen_item("Q1", [("T", "x")])
    b = P.gen_item("Q2", [("T", "y")])
    assert a != b and "Q1" in a and "(no passages were retrieved)" in P.gen_item("Q", [])
    for pre in (P.ROUTER_PREFIX, P.GEN_PREFIX, P.JUDGE_PREFIX):
        assert "{" in pre and "\n" in pre and len(pre) > 400


# --------------------------------------------------------------------------- stats
def test_stats_reference_values():
    St = _mod("stats")
    assert [round(x, 9) for x in St.holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]
    lo, hi = St.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert St.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert St.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert St.mcnemar_p(0, 0) == 1.0 and St.mcnemar_p(0, 10) < 0.01
    m = St.ridge_fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0], lam=1e-6)
    assert abs(float(St.ridge_predict(m, [[4.0]])[0]) - 4.0) < 1e-3


def test_registrant_name_from_10k_cover_pages():
    S = _mod("sources")
    hidden = "a-20241031 0001090872 false 2024 FY " + " ".join(f"us-gaap:Member{i} 2023-11-01 2024-10-31" for i in range(3000))
    head = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-K (Mark One) "
            "\u2612 ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the "
            "fiscal year ended October 31, 2024 OR \u2610 TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE "
            "SECURITIES EXCHANGE ACT OF 1934 For the transition period from to ")
    cases = {
        "AGILENT TECHNOLOGIES, INC.": "Commission File Number: 001-15405 AGILENT TECHNOLOGIES, INC. (Exact name of "
                                      "registrant as specified in its charter) Delaware 77-0518772",
        "Apple Inc.": "Commission File Number: 001-36743 Apple Inc.\xa0(Exact name of Registrant as specified in its "
                      "charter) California",
        "THE COCA-COLA COMPANY": "Commission File No. 001-02217 THE COCA-COLA COMPANY (Exact name of Registrant as "
                                 "Specified in its Charter)",
        "Johnson & Johnson": "Commission file number 1-3215 Johnson & Johnson (Exact name of registrant as "
                             "specified in its charter)",
        "3M COMPANY": "Commission file number: 1-3285 3M COMPANY (Exact name of registrant as specified in its "
                      "charter)",
        "MICROCHIP TECHNOLOGY INCORPORATED": "MICROCHIP TECHNOLOGY INCORPORATED (Exact Name of Registrant as "
                                             "Specified in Its Charter)",
    }
    for want, cover in cases.items():
        assert S.registrant_from_text(hidden + " " + head + cover) == want, (want, S.registrant_from_text(head + cover))
    assert S.registrant_from_text("no cover page here") == ""
    raw = ('<ix:nonNumeric contextRef="c-1" name="dei:EntityRegistrantName" id="f-3"><span style="x">AGILENT '
           'TECHNOLOGIES,&#160;INC.</span></ix:nonNumeric>')
    assert S.dei_registrant(raw) == "AGILENT TECHNOLOGIES, INC."


# --------------------------------------------------------------------------- API embedder (HTTP stubbed)

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
