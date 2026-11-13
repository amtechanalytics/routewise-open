"""D10: the FinDER 10-K archive (contents not described by the host) yields enough filings with ticker keys, the
FinDER parquet has its documented columns, enough HDRR split queries resolve to a gold filing, and the registrant
name is found for most filings. Diagnostics (stats, name sources, samples of unnamed filings) are written to
runs/pilot/fin_corpus_stats.json before any assertion, so a failure can be diagnosed from the results alone."""
import json
import re

from conftest import fake


def test_fin_corpus(runs):
    import paper.sources as S
    d = S.load("fin")
    st = d["stats"]
    docs = d["docs"]
    unnamed = [x for x in docs if not x["meta"]["company"]]
    samples = []
    for x in unnamed[:12]:
        t = x["text"]
        m = re.search(r"exact\s+name\s+of", t, re.I)
        samples.append({"ticker": x["meta"]["ticker"], "path": x["meta"]["path"][-80:],
                        "around_phrase": t[max(0, m.start() - 300):m.start() + 80] if m else None,
                        "head": t[:300]})
    diag = {"stats": st, "n_docs": len(docs), "unnamed": len(unnamed), "unnamed_samples": samples}
    out = runs / "pilot" / "fin_corpus_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diag, indent=1, default=str))
    assert all(x["meta"]["ticker"] for x in docs)
    if fake():
        assert len(docs) >= 4 and len(d["queries"]) >= 6
        return
    assert len(docs) >= 400, f"only {len(docs)} filings with ticker keys in 10-k.zip"
    assert st["finder_rows"] == 5703, st
    assert st["split_rows"] == 1500 and st["split_ids"] == 1354, st      # five overlapping 300-row splits
    assert len(d["queries"]) >= 250, f"only {len(d['queries'])} FinDER split queries resolved to a gold filing: {st}"
    named = len(docs) - len(unnamed)
    assert named >= 0.5 * len(docs), (f"registrant name found for only {named} of {len(docs)} filings "
                                      f"(sources {st.get('name_source')}); samples in runs/pilot/fin_corpus_stats.json")
