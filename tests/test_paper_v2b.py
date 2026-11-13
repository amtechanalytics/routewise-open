"""Tests for ROUTEWISE v2 launch 2 (paper/v2b.py, paper/analysis_v2b.py): isotonic calibration and threshold rules,
the ROUTEWISE-R scoping decision, the strict leak re-check (Unicode hyphens, residual names), resolver/validity
parsers, and the stratified bootstrap. Run with pytest, or directly: python tests/test_paper_v2b.py"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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
P = importlib.import_module("routewise_paper.prompts")
B = importlib.import_module("routewise_paper.v2b")
AN = importlib.import_module("routewise_paper.analysis_v2b")


def test_pav_is_monotone_and_fits_means():
    fx, fy = B.pav([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1])
    assert all(a <= b + 1e-12 for a, b in zip(fy, fy[1:]))
    assert abs(B.iso_predict((fx, fy), 0.15) - 0.5) < 1e-9
    assert B.iso_predict((fx, fy), 0.4) == 1.0


def test_lowest_threshold_rule():
    s = [0.1] * 10 + [0.5] * 10 + [0.9] * 10
    lab = [0] * 10 + [1] * 6 + [0] * 4 + [1] * 10
    assert B.lowest_threshold(s, lab, 0.9) == 0.55       # first grid point where precision >= 0.9
    assert B.lowest_threshold(s, lab, 0.6) == 0.15
    assert B.lowest_threshold([0.2] * 3, [1] * 3, 0.9) == 1.01    # fewer than 5 selections: never


def test_r_decision_router_then_resolver():
    cal = {"fin": {"route_tau1": 0.8, "full": {"iso": [[0.0, 1.0], [0.0, 1.0]], "tau_high": 0.85, "tau_low": 0.5}}}
    d = B.r_decision("R", "fin", cal, {"docs": ["a"], "confidence": 0.9}, [])
    assert d == {"mode": "hard", "docs": ["a"], "p": [0.9], "path": "router"}
    d = B.r_decision("R", "fin", cal, {"docs": ["a", "b", "c", "d"], "confidence": 0.99}, [("x", 0.95), ("y", 0.4)])
    assert d["path"] == "resolver" and d["mode"] == "hard" and d["docs"] == ["x"]
    d = B.r_decision("R", "fin", cal, None, [("x", 0.6)])
    assert d["mode"] == "soft"
    d = B.r_decision("R", "fin", cal, None, [("x", 0.3)])
    assert d["mode"] == "fallback" and d["docs"] == []
    # R-uncal uses raw confidence against the same thresholds
    cal2 = {"fin": {"route_tau1": 0.8, "full": {"iso": [[0.0, 1.0], [0.0, 0.2]], "tau_high": 0.85, "tau_low": 0.5}}}
    assert B.r_decision("R", "fin", cal2, None, [("x", 0.9)])["mode"] == "fallback"
    assert B.r_decision("R-uncal", "fin", cal2, None, [("x", 0.9)])["mode"] == "hard"


def test_resolver_choice_caps_and_maps():
    parsed = {"documents": [{"id": 2, "confidence": 0.9}, {"id": 2, "confidence": 0.5}, {"id": 9, "confidence": 1},
                            {"id": 1, "confidence": 0.4}, {"id": 3, "confidence": 0.3}, {"id": 4, "confidence": 0.2}]}
    assert B.resolver_choice(parsed, ["a", "b", "c", "d"]) == [("b", 0.9), ("a", 0.4), ("c", 0.3)]
    assert B.resolver_choice(None, ["a"]) == []


def test_strict_recheck_catches_unicode_and_residual_names():
    assert B.ntok("Bio‑Rad's GM") == ["bio", "rad", "s", "gm"]
    doc = {"doc_id": "fin:BIO", "title": "Bio-Rad Laboratories, Inc. (BIO) annual report on Form 10-K",
           "aliases": ["BIO", "Bio-Rad Laboratories, Inc.", "Bio-Rad Laboratories"], "meta": {"company":
                                                                                          "Bio-Rad Laboratories, Inc."},
           "text": "x"}
    orig = B.card_for
    B.card_for = lambda corpus, doc_id: {"name": "Bio-Rad Laboratories", "aliases": ["Bio-Rad"]}
    try:
        row = {"status": "ok", "source": "fin", "level": "L2", "gold_docs": ["fin:BIO"],
               "masked": "⟨D1⟩ GM trends", "text": "Bio‑Rad GM trends in the 2023 report"}
        assert B.strict_status(row, {"fin:BIO": doc}) == "name_leak_strict"
        row2 = {**row, "masked": "Bio‑Rad GM trends - ⟨D1⟩", "text": "GM trends of the lab maker"}
        assert B.strict_status(row2, {"fin:BIO": doc}) == "mask_incomplete"
        row3 = {**row, "text": "GM trends of the life-science instrument maker in its 2023 report"}
        assert B.strict_status(row3, {"fin:BIO": doc}) == "ok"
        assert B.strict_status({**row, "source": "mhrag", "level": "L1"}, {"fin:BIO": doc}) == "mhrag_l1_dropped"
    finally:
        B.card_for = orig


def test_parsers():
    assert P.parse_resolver('{"documents": []}') == {"documents": []}
    assert P.parse_resolver('{"documents": [{"id": "2", "confidence": 1.7}]}')["documents"] == [{"id": 2,
                                                                                                "confidence": 1.0}]
    v = P.parse_validity('{"same_answer": true, "identifiable": "false", "no_leak": true, "plausible": true}')
    assert v["identifiable"] is False
    for bad in ('{"same_answer": 1}', "no"):
        try:
            P.parse_validity(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    item = P.resolver_item("filings", "Which report?", ["card one", "card two"])
    assert "[2] card two" in item and P.RESOLVER_PREFIX and P.VALIDITY_PREFIX


def test_stratified_bootstrap_and_ece():
    est, lo, hi = AN.strat_boot({"a": [1, 1, 1, 1], "b": [0, 0, 0, 0]}, n_boot=500)
    assert est == 0.5 and lo == 0.5 and hi == 0.5
    est, lo, hi = AN.strat_boot({"a": [1, 0] * 50}, n_boot=2000)
    assert lo < 0.5 < hi
    assert AN._ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) < 0.11
    assert abs(AN._token_f1("the cap is 12 months", "cap 12 months") - 0.75) < 1e-9


def test_shuffle_map_is_a_derangement():
    orig = B.S.load
    B.S.load = lambda c: {"docs": [{"doc_id": f"d{i}"} for i in range(7)]}
    try:
        m = B.shuffle_map("x")
        assert sorted(m) == sorted(m.values()) and all(k != v for k, v in m.items())
    finally:
        B.S.load = orig


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {str(e)[:1500]}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
