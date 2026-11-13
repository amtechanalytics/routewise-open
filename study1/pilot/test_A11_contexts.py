"""A11: retrieval produced contexts for every probe query and arm; oracle contexts come only from gold documents;
hard routing with nothing resolved sends an empty context (and generation still parsed)."""
import json

from conftest import read_jsonl


def test_contexts(runs):
    rows = read_jsonl(runs / "pilot" / "retrieve.jsonl")
    assert rows
    for r in rows:
        if r["arm"] in ("flat", "meta"):
            assert r["chunks"], f"no chunks for {r['config']} {r['qid']} {r['arm']}"
        if r["arm"] == "oracle":
            assert r["doc_hit"] == 1 or not r["chunks"], r
        if r["arm"] in ("route", "rule") and not r["n_resolved"]:
            assert r["chunks"] == []
    gens = {x["cell_id"]: x for x in read_jsonl(runs / "pilot" / "generate.jsonl")}
    assert gens and all(g["status"] == "ok" for g in gens.values())
    assert json.loads((runs / "pilot" / "corpora.json").read_text())["configs"]
