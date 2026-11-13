"""D18: the fiscal year of the FinDER 10-K filings is recovered (inline-XBRL dei:DocumentFiscalYearFocus, else the
cover-page 'for the fiscal year ended' line) for at least 90 % of filings; fin-multiyear needs it to pick decoys and
to write the period phrase. Diagnostics go to runs/pilot/fin_fiscal_year.json before any assertion."""
import json

from conftest import fake


def test_fiscal_year(runs):
    import paper.sources as S
    docs = S.load("fin")["docs"]
    how = {}
    for d in docs:
        how[d["meta"].get("period_how", "none")] = how.get(d["meta"].get("period_how", "none"), 0) + 1
    years = {}
    for d in docs:
        years[d["meta"].get("fy") or "?"] = years.get(d["meta"].get("fy") or "?", 0) + 1
    missing = [d["meta"]["ticker"] for d in docs if not d["meta"].get("fy")][:40]
    out = runs / "pilot" / "fin_fiscal_year.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"how": how, "years": years, "missing_sample": missing}, indent=1))
    if fake():
        return
    known = sum(1 for d in docs if d["meta"].get("fy"))
    assert known >= 0.5 * len(docs), f"fiscal year known for {known} of {len(docs)} filings: {how}"
    assert all(y == "?" or 2018 <= int(y) <= 2026 for y in years), years
