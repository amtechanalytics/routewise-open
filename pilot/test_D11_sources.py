"""D11: every source parses into documents and a full query sample with gold documents and evidence."""
from conftest import fake

MIN_DOCS = {"fin": 400, "fda": 690, "cuad": 500, "md2d": 480, "qasper": 400, "mhrag": 600}


def test_sources():
    import paper.sources as S
    for s in S.SOURCES:
        d = S.load(s)
        qs = d["queries"]
        assert qs, f"{s}: no queries"
        assert all(q["gold_docs"] and q["evidence"] for q in qs), f"{s}: query without gold or evidence"
        ids = {x["doc_id"] for x in d["docs"]}
        assert all(set(q["gold_docs"]) <= ids for q in qs), f"{s}: gold document not in corpus"
        assert len({q["qid"] for q in qs}) == len(qs), f"{s}: duplicate query ids"
        if not fake():
            assert len(d["docs"]) >= MIN_DOCS[s], f"{s}: {len(d['docs'])} documents < {MIN_DOCS[s]}"
            assert len(qs) >= 250, f"{s}: sample of {len(qs)} queries < 250 ({d['stats']})"
    assert len(S.config_names()) == 15
