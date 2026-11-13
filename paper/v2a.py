"""ROUTEWISE v2, launch 1: decoy filings, document profile cards, rewritten questions, automatic checks and the
pre-run rewrite audit sheet (approach v2.1, sections 2 and 4).

  corpora (local) -> edgar (local) -> cards (llm, Haiku 5.5 batch) -> cardcheck (local)
  -> rewrite (llm, gpt-oss-120b) -> rewritecheck (local); analyze() writes the frozen artifacts that launch 2
  pins by sha256 (cards, rewrites, EDGAR manifest) and the stratified audit sheet.

Anti-circularity (approach 4): a card is written only from its document's *card footprint* (the first
FOOT_HEAD characters plus FOOT_MID characters from the middle). L2 descriptors are written by a different model
family from facts sampled outside the footprint and away from every gold evidence span of the document."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

from rh.util import RESULTS, ROOT, RUNS, journal, write_json

from . import prompts as P
from . import sources as S
from . import text as T

HAIKU, OSS = "claude-haiku-5-5", "openai/gpt-oss-120b"
REWRITE_SOURCES = ("fin", "fda", "cuad", "mhrag")
CARD_SOURCES = ("fin", "fda", "cuad", "md2d", "qasper", "mhrag", "finmy")
FOOT_HEAD, FOOT_MID = 6000, 2000
N_FACTS = 6
MIN_CARD_FACTS = 4
EDGAR_UA = "anon Analytics anon@example.org"
EDGAR_RATE = 5.0                         # requests per second (SEC fair-access limit is 10)
PRIOR_YEARS = 2
AUDIT_PER_CELL = 36
MAX_TOKENS = {"cards": {HAIKU: 4000}, "rewrite": {OSS: 4000}}
PH = "⟨D{}⟩"                   # placeholder ⟨D1⟩, ⟨D2⟩ ...
DOC_KIND = {"fin": "an annual report on Form 10-K of a US public company",
            "fda": "an FDA prescribing-information label of a drug",
            "cuad": "a commercial contract",
            "mhrag": "a news article from an online news outlet"}


def fake() -> bool:
    return os.environ.get("RH_FAKE_VENDOR", "") == "1"


def _edgar_dir() -> Path:
    d = Path(os.environ.get("RH_DATA_DIR", str(ROOT / "data"))) / "edgar"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =========================================================================== footprint and facts
def footprint(text: str) -> list[tuple[int, int]]:
    """Character spans the card is written from: the opening min(FOOT_HEAD, 30 %) of the document and
    min(FOOT_MID, 10 %) from its middle, so short documents (news articles) keep most text outside the card."""
    n = len(text or "")
    head = (0, min(FOOT_HEAD, int(0.3 * n)))
    mid0 = max(head[1], n // 2)
    mid = (mid0, min(n, mid0 + min(FOOT_MID, int(0.1 * n))))
    return [head] + ([mid] if mid[1] > mid[0] else [])


def card_source(text: str) -> str:
    return " … ".join(text[a:b] for a, b in footprint(text))


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def _sentences(text: str):
    pos = 0
    for s in _SENT.split(text or ""):
        i = text.find(s, pos)
        if i < 0:
            i = pos
        yield i, s
        pos = i + len(s)


def _junk(s: str) -> bool:
    """Signature blocks, tables of contents, rules and other non-prose lines make poor descriptor facts."""
    letters = sum(c.isalpha() for c in s)
    return ("/s/" in s or letters < 0.6 * len(s) or s.count("--") >= 2 or s.count("..") >= 3
            or sum(c.isupper() for c in s) > 0.5 * max(1, letters))


_CAND: dict = {}


def sample_facts(doc: dict, evidences: list[str], seed: str, n: int = N_FACTS) -> list[dict]:
    """n informative sentences from outside the card footprint, not overlapping any gold evidence of the
    document (content-token overlap < 0.5 of either side; the answer-leak check runs again on the rewrite),
    spread over n position bins, chosen with a seeded RNG. Candidates are cached per document."""
    text = doc["text"]
    key = (doc["doc_id"], len(text), len(evidences), T.stable_hash("|".join(evidences)))
    if key not in _CAND:
        _CAND[key] = _candidates(text, evidences)
    cand = _CAND[key]
    if not cand:
        return []
    rng = random.Random(T.stable_hash("facts|" + seed))
    n_len = max(1, len(text))
    bins = [[c for c in cand if k * n_len // n <= c[0] < (k + 1) * n_len // n] for k in range(n)]
    out = [rng.choice(b) for b in bins if b]
    rest = [c for c in cand if c not in out]
    rng.shuffle(rest)
    out += rest[:max(0, n - len(out))]
    out.sort()
    return [{"pos": i, "text": s} for i, s in out[:n]]


def _candidates(text: str, evidences: list[str]) -> list[tuple[int, str]]:
    fp = footprint(text)
    ev_tok = [set(T.content_tokens(e)) for e in evidences if e]
    cand = []
    for i, s in _sentences(text):
        if any(a <= i < b for a, b in fp):
            continue
        if not 60 <= len(s) <= 400 or _junk(s):
            continue
        ct = T.content_tokens(s)
        if len(ct) < 6:
            continue
        st = set(ct)
        if any(e and (len(st & e) / len(e) >= 0.5 or len(st & e) / len(st) >= 0.5) for e in ev_tok):
            continue
        if not re.search(r"\d|[A-Z][a-z]+\s+[A-Z][a-z]+", s):
            continue
        cand.append((i, s.strip()))
    return cand


# =========================================================================== identity masks
TITLE_GENERIC = frozenset("""prescribing information fda label annual report form agreement contract dated the of and
for on in a an amendment exhibit ex part news inc corp co company ltd llc plc""".split())


@lru_cache(maxsize=None)
def _df(source: str) -> dict:
    df: dict = {}
    for d in S.load(source)["docs"]:
        for t in set(T.tokens(d["text"])):
            df[t] = df.get(t, 0) + 1
    return df


_IDT: dict = {}


def identity_terms(source: str, doc: dict) -> list[str]:
    key = (source, doc["doc_id"], len(doc["text"]))
    if key not in _IDT:
        _IDT[key] = _identity_terms(source, doc)
    return _IDT[key]


def _identity_terms(source: str, doc: dict) -> list[str]:
    """Words that would name the document's subject (forbidden in L2 descriptors and scrubbed from the facts):
    the tokens of its title and aliases, plus words that occur in at most two documents of the collection and at
    least three times in this one, and look like names:
      fin, cuad: capitalised in >= 80 % of their occurrences (brands, party names), or short all-caps tokens
                 (party abbreviations such as "KI");
      fda:       written within 40 characters of the drug's name (the generic or brand name paired with it).
    Descriptive words (an indication, a product category) stay allowed. mhrag articles are named by their outlet."""
    out = set()
    for a in [canonical_name(doc, source)] + list(doc["aliases"]) + ([doc["title"]] if source != "mhrag" else []):
        for t in T.tokens(a):
            if t.isalpha() and len(t) >= 3 and t not in TITLE_GENERIC and t not in T.CORP_SUFFIX \
                    and t not in NON_IDENT and t not in T._STOP:
                out.add(t)
    if source not in ("fin", "fda", "cuad"):
        return sorted(out)
    df = _df(source)
    text = doc["text"]
    occ: dict = {}
    for m in re.finditer(r"[A-Za-z][A-Za-z\-]+", text):
        occ.setdefault(m.group(0).lower(), []).append((m.start(), m.group(0)))
    near = set()
    if source == "fda":
        for a in doc["aliases"] + [canonical_name(doc, source)]:
            for t in T.tokens(a):
                for pos, _ in occ.get(t, [])[:200]:
                    near |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text[max(0, pos - 40):pos + 40])}
    for t, lst in occ.items():
        if len(lst) < 3 or df.get(t, 0) > 2 or t in T._STOP or t in NON_IDENT or t in TITLE_GENERIC:
            continue
        caps = sum(1 for _, w in lst if w[0].isupper()) / len(lst)
        allcaps_short = all(w.isupper() for _, w in lst) and 2 <= len(t) <= 5 and len(lst) >= 5
        if source in ("fin", "cuad") and ((caps >= 0.8 and len(t) >= 4) or allcaps_short):
            out.add(t)
        elif source == "fda" and t in near and len(t) >= 4:
            out.add(t)
    return sorted(out)


def scrub(text: str, terms: list[str]) -> str:
    for t in sorted(terms, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?:'s)?(?![A-Za-z0-9])", "[name]", text, flags=re.I)
    return text
def canonical_name(doc: dict, source: str) -> str:
    m = doc.get("meta") or {}
    if source == "fin":
        return m.get("company") or doc["title"].split(" (")[0]
    if source == "fda":
        return doc["title"].replace(" prescribing information (FDA label)", "")
    if source == "cuad":
        return m.get("party") or ""
    if source == "mhrag":
        return m.get("publisher") or ""
    return doc["title"]


def masked_question(source: str, q: dict, docs_by_id: dict) -> tuple[str, list[str]]:
    """The question with each gold document's identity replaced by a placeholder; (text, gold doc per
    placeholder). Placeholders are numbered in order of appearance; ("", []) when the question names none."""
    if source == "cuad":
        core = q.get("text_mask") or ""
        return (f"For {PH.format(1)}: {core}", [q["gold_docs"][0]]) if core and q["text"] != core else ("", [])
    text, owners = q["text"], []
    groups: dict[str, list[str]] = {}
    for g in q["gold_docs"]:
        d = docs_by_id.get(g)
        if not d:
            continue
        key = canonical_name(d, source) if source == "mhrag" else g
        groups.setdefault(key, []).append(g)
    for key, gs in groups.items():
        aliases = sorted({a for g in gs for a in docs_by_id[g]["aliases"] if a}, key=len, reverse=True)
        ph = PH.format(len(owners) + 1)
        new = T.mask_aliases(text, aliases, ph)
        if new != text:
            text = new
            owners.append(gs[0])
    if not owners:
        return "", []
    return text, owners


NON_IDENT = frozenset("""hydrochloride hcl calcium sodium potassium sulfate sulphate acetate phosphate citrate maleate
tartrate mesylate besylate succinate fumarate bromide chloride magnesium tablets tablet capsules capsule injection
injectable oral solution suspension extended release delayed usp mg ml for and with of the cream ointment gel
patch film coated chewable powder kit""".split())


def l1_form(doc: dict, source: str, catalog: list[dict]) -> str | None:
    """A non-canonical surface form that still identifies the document uniquely in its collection:
    a registered alias or a strict token subset of the canonical name with token Jaccard <= 0.5 to the
    canonical name, matching no other document's name or aliases. None when no such form exists."""
    name = canonical_name(doc, source)
    nt = set(T.tokens(name))
    cands = []
    for a in doc["aliases"]:
        if a and T.norm(a) != T.norm(name):
            cands.append(a)
    toks = [t for t in T.tokens(name) if t not in T.CORP_SUFFIX]
    if source == "fda":
        cands = [a for a in cands if not all(t in NON_IDENT for t in T.tokens(a))]
    for k in range(1, len(toks)):
        for i in range(0, len(toks) - k + 1):
            cands.append(" ".join(toks[i:i + k]))
    best = None
    for c in sorted(dict.fromkeys(cands), key=lambda x: (len(x), x)):
        ct = set(T.tokens(c))
        if not ct or (len(ct & nt) / len(ct | nt) if nt else 1.0) > 0.5:
            continue
        if len(T.norm(c)) < 3 or all(t in T.CORP_SUFFIX or t in NON_IDENT for t in ct):
            continue
        if _matches_other(c, doc["doc_id"], source, catalog):
            continue
        best = c
        break
    return best


def _matches_other(form: str, doc_id: str, source: str, catalog: list[dict]) -> bool:
    f = T.norm(form)
    is_tk = bool(re.fullmatch(r"[A-Z][A-Z.\-]{0,5}", form))
    for d in catalog:
        if d["doc_id"] == doc_id:
            continue
        if source == "mhrag" and canonical_name(d, source) == canonical_name(_DOC_OF.get(doc_id, d), source):
            continue                                  # other articles of the same outlet share its name
        for a in [canonical_name(d, source)] + list(d["aliases"]):
            if not a:
                continue
            if is_tk:
                if a == form:
                    return True
            elif re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", T.norm(a)):
                return True
    return False


_DOC_OF: dict = {}


def make_l1(source: str, q: dict, docs_by_id: dict, catalog: list[dict]) -> tuple[str | None, str]:
    masked, owners = masked_question(source, q, docs_by_id)
    if not owners:
        return None, "no_identity"
    text = masked
    for k, g in enumerate(owners, 1):
        d = docs_by_id[g]
        _DOC_OF[g] = d
        if source == "cuad":
            form = cuad_l1(d, catalog)
        else:
            form = l1_form(d, source, catalog)
        if not form:
            return None, "no_unique_partial_form"
        text = text.replace(PH.format(k), form)
    return text, "ok"


def cuad_l1(doc: dict, catalog: list[dict]) -> str | None:
    """CUAD parties are single concatenated tokens taken from the file name, so no strict token subset exists.
    L1 for a contract keeps its kind and date and drops the party: "the distributor agreement dated 1999-09-09",
    accepted only when no other contract in the collection has the same kind and date."""
    m = doc.get("meta") or {}
    kind, date = (m.get("kind") or "").strip(), (m.get("date") or "").strip()
    if not kind or not date:
        return None
    same = [d for d in catalog if d["doc_id"] != doc["doc_id"]
            and T.norm((d.get("meta") or {}).get("kind", "")) == T.norm(kind)
            and ((d.get("meta") or {}).get("date") or "") == date]
    return None if same else f"the {kind} dated {date}"


# =========================================================================== EDGAR decoys (fin-multiyear)
class EdgarClient:
    def __init__(self, rate: float = EDGAR_RATE):
        self.gap = 1.0 / rate
        self.last = 0.0
        self.calls = 0

    def get(self, url: str) -> bytes:
        for attempt in range(6):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.calls += 1
            req = urllib.request.Request(url, headers={"User-Agent": EDGAR_UA, "Accept-Encoding": "gzip, deflate"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        data = gzip.decompress(data)
                    return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise
                if e.code in (403, 429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2 ** attempt)
        raise RuntimeError(f"EDGAR request kept failing: {url}")


def _date_year(s: str) -> str:
    m = re.search(r"(19|20)\d{2}", s or "")
    return m.group(0) if m else ""


def choose_prior(recent: dict, fy: str, n: int = PRIOR_YEARS) -> tuple[dict | None, list[dict]]:
    """From an EDGAR submissions 'recent' block: (the 10-K whose report year == fy, the n preceding 10-Ks)."""
    rows = []
    for i, form in enumerate(recent.get("form") or []):
        if form != "10-K":
            continue
        rows.append({"accession": recent["accessionNumber"][i], "report_date": recent["reportDate"][i],
                     "filing_date": recent["filingDate"][i], "primary": recent["primaryDocument"][i]})
    rows.sort(key=lambda r: r["report_date"], reverse=True)
    match = next((r for r in rows if _date_year(r["report_date"]) == fy), None)
    if match is None:
        return None, []
    older = [r for r in rows if r["report_date"] < match["report_date"]]
    return match, older[:n]


def fetch_decoys(fin_docs: list[dict], limit: int | None = None, hb=None) -> dict:
    """Two prior annual 10-Ks per FinDER company from EDGAR; texts saved under data/edgar, manifest returned."""
    out_dir = _edgar_dir()
    man = {"user_agent": EDGAR_UA, "accessed": time.strftime("%Y-%m-%d"), "filings": [], "skipped": {}}
    docs = sorted(fin_docs, key=lambda d: d["doc_id"])[:limit] if limit else sorted(fin_docs, key=lambda d: d["doc_id"])
    if fake():
        for d in docs:
            fy = d["meta"].get("fy") or "2024"
            for k in range(1, PRIOR_YEARS + 1):
                y = str(int(fy) - k)
                txt = re.sub(fy, y, d["text"])
                p = out_dir / f"{d['meta']['ticker']}_{y}.txt.gz"
                with gzip.open(p, "wt", encoding="utf-8") as f:
                    f.write(txt)
                raw = txt.encode()
                man["filings"].append({"ticker": d["meta"]["ticker"], "fy": y, "accession": f"fake-{y}",
                                       "report_date": f"{y}-12-31", "url": "fake://", "file": p.name,
                                       "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
        return man
    cl = EdgarClient()
    tick = json.loads(cl.get("https://www.sec.gov/files/company_tickers.json"))
    cik_of = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in tick.values()}
    for n, d in enumerate(docs):
        tk, fy = d["meta"]["ticker"], d["meta"].get("fy") or ""
        if hb is not None and n % 10 == 0:
            hb.touch(edgar=f"{n}/{len(docs)}")
        if not fy:
            man["skipped"][tk] = "fiscal year of the FinDER filing unknown"
            continue
        cik = cik_of.get(tk) or cik_of.get(tk.replace(".", "-"))
        if not cik:
            man["skipped"][tk] = "ticker not in SEC company_tickers.json"
            continue
        try:
            sub = json.loads(cl.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
        except Exception as e:  # noqa: BLE001
            man["skipped"][tk] = f"submissions: {type(e).__name__}"
            continue
        match, prior = choose_prior((sub.get("filings") or {}).get("recent") or {}, fy)
        if match is None:
            man["skipped"][tk] = f"no 10-K with report year {fy} in EDGAR submissions"
            continue
        for r in prior:
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{r['accession'].replace('-', '')}/"
                   f"{r['primary']}")
            try:
                raw = cl.get(url)
            except Exception as e:  # noqa: BLE001
                man["skipped"][f"{tk}:{r['accession']}"] = f"download: {type(e).__name__}"
                continue
            html_s = raw.decode("utf-8", errors="replace")
            txt = re.sub(r"\s+", " ", S._strip_html(html_s)).strip()
            if len(txt) < 20000:
                man["skipped"][f"{tk}:{r['accession']}"] = f"text too short ({len(txt)} chars)"
                continue
            y = S.dei_field(html_s, "DocumentFiscalYearFocus") or _date_year(r["report_date"])
            p = out_dir / f"{tk}_{y}.txt.gz"
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write(txt)
            man["filings"].append({"ticker": tk, "cik": cik, "fy": y, "accession": r["accession"],
                                   "report_date": r["report_date"], "matched_accession": match["accession"],
                                   "url": url, "file": p.name, "sha256": hashlib.sha256(raw).hexdigest(),
                                   "size": len(raw)})
    man["requests"] = cl.calls
    return man


def decoy_docs(manifest: dict, fin_docs: list[dict]) -> list[dict]:
    by_tk = {d["meta"]["ticker"]: d for d in fin_docs}
    out, seen = [], set()
    for f in manifest["filings"]:
        if (f["ticker"], f["fy"]) in seen:          # two filings labelled with the same fiscal year: keep the first
            continue
        seen.add((f["ticker"], f["fy"]))
        base = by_tk.get(f["ticker"])
        p = _edgar_dir() / f["file"]
        if base is None or not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            txt = fh.read()
        name = base["meta"].get("company") or f["ticker"]
        out.append({"doc_id": f"fin:{f['ticker']}:{f['fy']}",
                    "title": f"{name} ({f['ticker']}) annual report on Form 10-K for fiscal year {f['fy']}",
                    "text": txt, "aliases": list(base["aliases"]),
                    "meta": {**base["meta"], "fy": f["fy"], "accession": f["accession"], "decoy": True}})
    return out


# =========================================================================== stages
def _probe_n(ctx) -> int | None:
    if ctx.mode == "full":
        return None
    return int(ctx.cfg.plan["pilot_per_cell" if ctx.mode == "pilot" else "canary_per_cell"])


def _manifest_path(mode: str) -> Path:
    return RUNS / mode / "edgar_manifest.json"


def _card_docs(ctx) -> dict[str, list[dict]]:
    """Documents that get a card, per card source (probe subset outside the full run)."""
    k = _probe_n(ctx)
    out = {}
    for s in CARD_SOURCES:
        if s == "finmy":
            mp = _manifest_path(ctx.mode)
            man = json.loads(mp.read_text()) if mp.exists() else {"filings": []}
            docs = decoy_docs(man, S.load("fin")["docs"])
        else:
            docs = S.load(s)["docs"]
        docs = sorted(docs, key=lambda d: T.order_key(d["doc_id"]))
        out[s] = unique_docs(docs)
        out[s] = out[s][:k] if k else out[s]
    return out


def unique_docs(docs: list[dict]) -> list[dict]:
    """One card per document id. FDARxBench has 5 placeholder labels (titles tmp_<uuid>, empty set_id) that all
    parse to the id "fda:"; no question points at them. Documents whose id repeats, or whose id has an empty
    suffix, get no card (they stay in the collection, unchanged, for every retrieval arm)."""
    seen: dict = {}
    for d in docs:
        seen[d["doc_id"]] = seen.get(d["doc_id"], 0) + 1
    return [d for d in docs if seen[d["doc_id"]] == 1 and not d["doc_id"].endswith(":")]


def run_edgar(ctx) -> dict:
    ctx.register_diagnostics(_diag("edgar"))
    fin = S.load("fin")["docs"]
    k = _probe_n(ctx)
    man = fetch_decoys(fin, limit=(max(3, k) if k else None), hb=ctx.hb)
    write_json(_manifest_path(ctx.mode), man)
    have = {f["ticker"] for f in man["filings"]}
    n_two = sum(1 for tk in have if sum(1 for f in man["filings"] if f["ticker"] == tk) >= PRIOR_YEARS)
    m = {"companies": len(fin) if not k else max(3, k), "with_decoys": len(have), "with_two": n_two,
         "filings": len(man["filings"]), "skipped": len(man["skipped"]),
         "period_how": _count(d["meta"].get("period_how", "none") for d in fin)}
    write_json(RUNS / ctx.mode / "diag_edgar.json", {**m, "skipped_detail": dict(list(man["skipped"].items())[:200])})
    return m


def _check_edgar(ctx, m):
    """Never stops the run (anon, 10 Nov: launch 1 must finish unattended). Shortfalls are recorded as warnings
    for the brief; fin-multiyear is simply smaller (or absent) when EDGAR yields fewer decoys."""
    if m["filings"] == 0:
        warn(ctx, "edgar", "no decoy filing fetched from EDGAR; fin-multiyear will be empty (see diag_edgar.json)")
    elif ctx.mode == "full" and not _fixture(ctx) and m["with_two"] < 0.6 * m["companies"]:
        warn(ctx, "edgar", f"only {m['with_two']} of {m['companies']} companies have {PRIOR_YEARS} prior 10-Ks (< 60 %)")
    return []


def warn(ctx, stage: str, msg: str) -> None:
    """Quality shortfalls are reported, not gated: runs/<mode>/warnings.jsonl -> brief + results."""
    from rh.util import append_jsonl
    append_jsonl(RUNS / ctx.mode / "warnings.jsonl", {"stage": stage, "msg": msg})
    journal("WARN", f"{ctx.mode}/{stage}: {msg}")


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _diag(name):
    def writer(ctx, stage):
        p = RUNS / ctx.mode / f"diag_{name}.json"
        if not p.exists():
            write_json(p, {"stage": stage, "note": "stage failed before writing its diagnostics"})
    writer.__name__ = f"diag_{name}"
    return writer


def _cov_edgar(ctx, mode):
    return [{"source": "fin", "config": "finmy", "gen_model": "-"}]


def _dedupe_cells(ctx, stage: str, cells: list[dict]) -> list[dict]:
    out, seen, dup = [], set(), 0
    for c in cells:
        if c["cell_id"] in seen:
            dup += 1
            continue
        seen.add(c["cell_id"])
        out.append(c)
    if dup:
        warn(ctx, stage, f"{dup} duplicate cells dropped before sending")
    return out


def build_cards(ctx) -> list[dict]:
    cells = []
    for s, docs in _card_docs(ctx).items():
        src = "fin" if s == "finmy" else s
        for d in docs:
            cells.append({"cell_id": f"card|{s}|{d['doc_id']}", "model": HAIKU, "arm": "card",
                          "prefix": P.CARD_PREFIX,
                          "item": P.card_item(S.SOURCE_DESCRIPTION[src], d["title"], card_source(d["text"])),
                          "max_tokens": MAX_TOKENS["cards"][HAIKU],
                          "meta": {"source": src, "config": f"{s}.cards", "gen_model": "-", "doc": d["doc_id"]}})
    return _dedupe_cells(ctx, "cards", cells)


def ground_card(card: dict, src_text: str) -> dict:
    """Keep the facts whose quote occurs verbatim (whitespace- and case-normalised) in the card source."""
    hay = re.sub(r"\s+", " ", src_text).lower()
    ok, dropped = [], 0
    for f in card.get("facts") or []:
        qt = re.sub(r"\s+", " ", f.get("quote") or "").strip().lower()
        if len(qt) >= 15 and qt in hay:
            ok.append(f)
        else:
            dropped += 1
    return {**card, "facts": ok, "facts_dropped": dropped}


def run_cardcheck(ctx) -> dict:
    ctx.register_diagnostics(_diag("cardcheck"))
    docs = {(s, d["doc_id"]): d for s, ds in _card_docs(ctx).items() for d in ds}
    rows = {(r["meta"]["config"].split(".")[0], r["meta"]["doc"]): r for r in ctx.stage_rows("cards")
            if r.get("meta")}
    out_p = RUNS / ctx.mode / "cards_checked.jsonl"     # the llm stage keeps its own rows in cards.jsonl
    stats = {}
    with open(out_p, "w") as fo:
        for (s, did), d in sorted(docs.items()):
            r = rows.get((s, did))
            st = stats.setdefault(s, {"docs": 0, "parsed": 0, "grounded4": 0, "facts": 0, "gated": 0})
            st["docs"] += 1
            if not r or r.get("status") != "ok":
                continue
            st["parsed"] += 1
            card = ground_card(r["parsed"], card_source(d["text"]))
            st["facts"] += len(card["facts"])
            ex_len = len(card_source(d["text"]))
            if ex_len >= 1500:                       # very short documents are reported, not gated
                need = MIN_CARD_FACTS if ex_len >= 4000 else MIN_CARD_FACTS - 1
                st["gated"] += 1
                st["grounded4"] += int(len(card["facts"]) >= need)
            fo.write(json.dumps({"source": s, "doc_id": did, "card": card,
                                 "footprint": footprint(d["text"])}, sort_keys=True) + "\n")
    write_json(RUNS / ctx.mode / "diag_cardcheck.json", stats)
    return {"per_source": stats}


def _fixture(ctx) -> bool:
    return all(str(p.get("url", "")).startswith("file://") for p in ctx.cfg.pins)   # dry run on fixtures


def _check_cardcheck(ctx, m):
    """Stops only when no card at all parsed. Per-source parse and grounding shortfalls are warnings."""
    tot = sum(st["parsed"] for st in m["per_source"].values())
    if tot == 0:
        return ["no profile card parsed for any document"]
    for s, st in m["per_source"].items():
        if st["docs"] and st["parsed"] / st["docs"] < 0.9:
            warn(ctx, "cardcheck", f"{s}: cards parsed for {st['parsed']} of {st['docs']} documents (< 90 %)")
        if ctx.mode == "full" and not _fixture(ctx) and st["gated"] and st["grounded4"] / st["gated"] < 0.85:
            warn(ctx, "cardcheck", f"{s}: only {st['grounded4']} of {st['gated']} cards keep >= {MIN_CARD_FACTS} "
                                   f"grounded facts (>= {MIN_CARD_FACTS - 1} for excerpts under 4,000 characters) (< 85 %)")
    return []


def _cov_cardcheck(ctx, mode):
    return [{"source": "fin" if s == "finmy" else s, "config": f"{s}.cards", "gen_model": "-"} for s in CARD_SOURCES]


@lru_cache(maxsize=None)
def _evidence_by_doc(source: str) -> dict:
    data = S.load(source)
    out: dict = {}
    for q in data["queries"] + data.get("dev", []):
        for g in q["gold_docs"]:
            out.setdefault(g, []).extend(q.get("evidence") or [])
    return out


def pool(ctx, source: str) -> list[tuple[str, dict]]:
    """(split, query) for the test sample (= Study 1) and the dev pool. Outside the full run: the first k test
    queries and the first dev query that can be rewritten (they name a document and have facts outside the card
    footprint), so the pilot meets every rewrite source even when its first questions name no document."""
    data = S.load(source)
    k = _probe_n(ctx)
    test = [("test", q) for q in data["queries"]]
    dev = [("dev", q) for q in data.get("dev", [])]
    if not k:
        return test + dev
    return _rewritable(source, test, k) + _rewritable(source, dev, 1)


def _rewritable(source: str, items: list, k: int) -> list:
    by_id = {d["doc_id"]: d for d in S.load(source)["docs"]}
    ok = [x for x in items if rewrite_item_for(source, x[1], by_id)[0] is not None][:k]
    return ok if ok else items[:k]


def rewrite_item_for(source: str, q: dict, docs_by_id: dict) -> tuple[str | None, dict]:
    masked, owners = masked_question(source, q, docs_by_id)
    if not owners:
        return None, {"status": "no_identity"}
    ev = _evidence_by_doc(source)
    blocks, forbid = [], set()
    for k, g in enumerate(owners, 1):
        d = docs_by_id[g]
        facts = sample_facts(d, ev.get(g, []), seed=f"{q['qid']}|{g}")
        if len(facts) < 2:
            return None, {"status": "too_few_facts"}
        in_q = set(T.tokens(masked))
        terms = [t for t in identity_terms(source, d) if t not in in_q]
        forbid |= {a for a in d["aliases"] if a} | {canonical_name(d, source)} | set(terms)
        blocks.append((PH.format(k), DOC_KIND[source], [scrub(f["text"], terms) for f in facts]))
    ans = (q.get("answer") or "").strip()
    avoid = ans if 0 < len(ans) <= 120 and T.norm(ans) not in ("yes", "no") else ""
    return P.rewrite_item(masked, blocks, sorted(f for f in forbid if f), avoid), {"status": "ok", "masked": masked,
                                                                                   "owners": owners}


def build_rewrite(ctx) -> list[dict]:
    cells = []
    for s in REWRITE_SOURCES:
        by_id = {d["doc_id"]: d for d in S.load(s)["docs"]}
        for split, q in pool(ctx, s):
            item, info = rewrite_item_for(s, q, by_id)
            if item is None:
                continue
            cells.append({"cell_id": f"rewrite|{s}|{q['qid']}|L2", "model": OSS, "arm": "rewrite",
                          "prefix": P.REWRITE_PREFIX, "item": item, "max_tokens": MAX_TOKENS["rewrite"][OSS],
                          "meta": {"source": s, "config": f"{s}.L2", "gen_model": "-", "qid": q["qid"],
                                   "split": split}})
    return _dedupe_cells(ctx, "rewrite", cells)


# ---------------------------------------------------------------- checks
@lru_cache(maxsize=None)
def _doc_bm25(source: str):
    from .index import _bm25_matrix
    docs = S.load(source)["docs"]
    m, vocab = _bm25_matrix([d["text"] for d in docs])
    return m, vocab, [d["doc_id"] for d in docs]


def doc_rank(source: str, query: str, k: int) -> list[str]:
    m, vocab, ids = _doc_bm25(source)
    cols = sorted({vocab[w] for w in T.content_tokens(query) if w in vocab})
    if not cols:
        return []
    s = np.asarray(m[:, cols].sum(axis=1)).ravel()
    order = np.argsort(-s, kind="stable")[:k]
    return [ids[i] for i in order]


def _ngrams(s: str, n: int = 8) -> set:
    t = T.tokens(s)
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def answer_leak(original: str, rewritten: str, answer: str) -> bool:
    a = (answer or "").strip()
    if not a or T.norm(a) in ("yes", "no"):
        return False
    if len(a) <= 120:
        na = T.norm(a)
        return len(na) >= 3 and f" {na} " in f" {T.norm(rewritten)} " and f" {na} " not in f" {T.norm(original)} "
    return bool((_ngrams(a) & _ngrams(rewritten)) - _ngrams(original))


def check_l2(source: str, q: dict, rewritten: str, owners: list[str], docs_by_id: dict,
             masked: str = "") -> list[str]:
    fails = []
    in_q = set(T.tokens(masked or ""))
    rw_tok = set(T.tokens(rewritten))
    for g in owners:
        d = docs_by_id[g]
        for a in [canonical_name(d, source)] + list(d["aliases"]):
            if a and T.mentions(rewritten, a):
                fails.append("name_leak")
                break
        if [t for t in identity_terms(source, d) if t not in in_q and t in rw_tok]:
            fails.append("name_leak")
    if PH.format(1)[0] in rewritten:
        fails.append("placeholder_left")
    if answer_leak(q["text"], rewritten, q.get("answer") or ""):
        fails.append("answer_leak")
    k = 5 if source == "mhrag" else 3
    if not set(doc_rank(source, rewritten, k)) & set(q["gold_docs"]):
        fails.append("not_unique")
    return sorted(set(fails))


def period_phrase(fy: str) -> str:
    return f" (based on its annual report for fiscal year {fy})"


def run_rewritecheck(ctx) -> dict:
    ctx.register_diagnostics(_diag("rewritecheck"))
    rows = {(r["meta"]["source"], r["meta"]["qid"]): r for r in ctx.stage_rows("rewrite") if r.get("meta")}
    out_p = RUNS / ctx.mode / "rewrites.jsonl"
    cells: dict = {}

    def tally(src, level, split, status):
        c = cells.setdefault(f"{src}.{level}.{split}", {"items": 0, "ok": 0, "reasons": {}})
        c["items"] += 1
        if status == "ok":
            c["ok"] += 1
        else:
            c["reasons"][status] = c["reasons"].get(status, 0) + 1

    with open(out_p, "w") as fo:
        for s in REWRITE_SOURCES:
            docs = S.load(s)["docs"]
            by_id = {d["doc_id"]: d for d in docs}
            for n_q, (split, q) in enumerate(pool(ctx, s)):
                if n_q % 25 == 0:
                    ctx.hb.touch(rewritecheck=f"{s} {n_q}")
                base = {"source": s, "qid": q["qid"], "split": split, "gold_docs": q["gold_docs"],
                        "original": q["text"], "answer": q.get("answer") or ""}
                fo.write(json.dumps({**base, "level": "L0", "text": q["text"], "status": "ok", "fails": []},
                                    sort_keys=True) + "\n")
                tally(s, "L0", split, "ok")
                l1, why = make_l1(s, q, by_id, docs)
                st1 = "ok" if l1 else why
                if l1 and answer_leak(q["text"], l1, q.get("answer") or ""):
                    st1 = "answer_leak"
                fo.write(json.dumps({**base, "level": "L1", "text": l1, "status": st1, "fails": [] if l1 else [why]},
                                    sort_keys=True) + "\n")
                tally(s, "L1", split, st1)
                masked, owners = masked_question(s, q, by_id)
                r = rows.get((s, q["qid"]))
                if not owners:
                    st2, l2, fails = "no_identity", None, ["no_identity"]
                elif r is None:
                    st2, l2, fails = "no_rewrite_cell", None, ["too_few_facts_or_missing"]
                elif r.get("status") != "ok":
                    st2, l2, fails = "rewrite_unparsed", None, [r.get("status") or "error"]
                else:
                    l2 = r["parsed"]["question"]
                    fails = check_l2(s, q, l2, owners, by_id, masked)
                    st2 = "ok" if not fails else fails[0]
                fo.write(json.dumps({**base, "level": "L2", "text": l2, "status": st2, "fails": fails,
                                     "masked": masked}, sort_keys=True) + "\n")
                tally(s, "L2", split, st2)
                if s == "fin" and split == "test":
                    fy = (by_id[q["gold_docs"][0]].get("meta") or {}).get("fy") or ""
                    my_ok = st2 == "ok" and bool(fy)
                    fo.write(json.dumps({**base, "source": "finmy", "level": "L2",
                                         "text": (l2 + period_phrase(fy)) if my_ok else None,
                                         "status": "ok" if my_ok else (st2 if st2 != "ok" else "fy_unknown"),
                                         "fails": fails}, sort_keys=True) + "\n")
                    tally("finmy", "L2", split, "ok" if my_ok else (st2 if st2 != "ok" else "fy_unknown"))
                    fo.write(json.dumps({**base, "source": "finmy", "level": "L0", "text": q["text"], "status": "ok",
                                         "fails": []}, sort_keys=True) + "\n")
                    tally("finmy", "L0", split, "ok")
    write_json(RUNS / ctx.mode / "diag_rewritecheck.json", cells)
    return {"cells": cells}


def _check_rewritecheck(ctx, m):
    """Never stops the run; L2 pass-rate shortfalls are warnings (the human audit is the quality gate)."""
    if ctx.mode != "full" or _fixture(ctx):
        return []
    for c, v in m["cells"].items():
        src, level, split = c.split(".")
        if level == "L2" and src != "finmy" and split == "test" and v["items"] and v["ok"] / v["items"] < 0.5:
            warn(ctx, "rewritecheck", f"{c}: only {v['ok']} of {v['items']} rewrites pass the automatic checks "
                                      f"(< 50 %); reasons {v['reasons']}")
    return []


def _cov_rewritecheck(ctx, mode):
    return [{"source": s, "config": f"{s}.L2", "gen_model": "-"} for s in REWRITE_SOURCES]


# =========================================================================== analysis: frozen artifacts + audit
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def analyze(ctx) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    mode = "full"
    src = {"cards": RUNS / mode / "cards_checked.jsonl", "rewrites": RUNS / mode / "rewrites.jsonl",
           "edgar_manifest": _manifest_path(mode)}
    frozen = {}
    for name, p in src.items():
        if not p.exists():
            continue
        dst = RESULTS / (f"v2a_{name}.jsonl" if p.suffix == ".jsonl" else f"v2a_{name}.json")
        lines = p.read_text().splitlines()
        if p.suffix == ".jsonl":
            lines = sorted(lines)
            dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        else:
            d = json.loads(p.read_text())
            d.pop("requests", None)
            dst.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
        frozen[dst.name] = {"sha256": _sha(dst), "size": dst.stat().st_size}
    _summary_tables(RESULTS)
    _audit_sheet(RESULTS)
    (RESULTS / "v2a_warnings.json").write_text(json.dumps(_warnings(), indent=1, sort_keys=True) + "\n")
    (RESULTS / "v2a_frozen_artifacts.json").write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _summary_tables(res: Path) -> None:
    rw = _read_jsonl(res / "v2a_rewrites.jsonl")
    cells: dict = {}
    for r in rw:
        k = (r["source"], r["level"], r["split"])
        c = cells.setdefault(k, {"items": 0, "ok": 0, "reasons": {}})
        c["items"] += 1
        c["ok"] += int(r["status"] == "ok")
        if r["status"] != "ok":
            c["reasons"][r["status"]] = c["reasons"].get(r["status"], 0) + 1
    with open(res / "v2a_rewrite_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "level", "split", "items", "ok", "ok_rate", "reasons"])
        for k in sorted(cells):
            c = cells[k]
            w.writerow([*k, c["items"], c["ok"], f"{c['ok'] / c['items']:.4f}" if c["items"] else "",
                        json.dumps(c["reasons"], sort_keys=True)])
    cards = _read_jsonl(res / "v2a_cards.jsonl")
    st: dict = {}
    for r in cards:
        x = st.setdefault(r["source"], {"cards": 0, "facts": 0, "ge4": 0, "dropped": 0})
        x["cards"] += 1
        x["facts"] += len(r["card"]["facts"])
        x["ge4"] += int(len(r["card"]["facts"]) >= MIN_CARD_FACTS)
        x["dropped"] += int(r["card"].get("facts_dropped", 0))
    with open(res / "v2a_card_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "cards", "mean_grounded_facts", "share_ge4", "facts_dropped"])
        for s in sorted(st):
            x = st[s]
            w.writerow([s, x["cards"], f"{x['facts'] / max(1, x['cards']):.2f}",
                        f"{x['ge4'] / max(1, x['cards']):.4f}", x["dropped"]])


AUDIT_CELLS = [("fin", "L1"), ("fin", "L2"), ("fda", "L1"), ("fda", "L2"), ("cuad", "L1"), ("cuad", "L2"),
               ("mhrag", "L1"), ("mhrag", "L2"), ("finmy", "L2")]
CRITERIA = ("same_answer", "identifiable", "no_leak", "plausible")


def _audit_sheet(res: Path) -> None:
    rw = [r for r in _read_jsonl(res / "v2a_rewrites.jsonl") if r["split"] == "test" and r["status"] == "ok"]
    titles = {}
    for s in REWRITE_SOURCES:
        try:
            titles.update({d["doc_id"]: d["title"] for d in S.load(s)["docs"]})
        except Exception:  # noqa: BLE001
            pass
    pick = []
    for s, lv in AUDIT_CELLS:
        lst = [r for r in rw if r["source"] == s and r["level"] == lv]
        lst.sort(key=lambda r: T.stable_hash(f"audit2|{s}|{lv}|{r['qid']}"))
        pick += lst[:AUDIT_PER_CELL]
    pick.sort(key=lambda r: T.stable_hash(f"order2|{r['source']}|{r['level']}|{r['qid']}"))
    head = ["audit_id", "source", "level", "original_question", "rewritten_question", "reference_answer",
            "document_title"]
    for who in ("rater1", "rater2"):
        head += [f"{who}_{c}(y|n)" for c in CRITERIA]
    head.append("note")
    with open(res / "v2a_rewrite_audit_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for i, r in enumerate(pick):
            w.writerow([f"R{i + 1:03d}", r["source"], r["level"], r["original"], r["text"], r["answer"][:600],
                        " | ".join(titles.get(g, g) for g in r["gold_docs"])] + [""] * (2 * len(CRITERIA)) + [""])


def _warnings() -> list[dict]:
    p = RUNS / "full" / "warnings.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def brief_items(ctx) -> list[str]:
    w = _warnings()
    head = [f"Quality warnings (reported, not gated): " + "; ".join(f"{x['stage']}: {x['msg']}" for x in w)] if w \
        else ["Quality warnings: none."]
    return head + [
        "Rewrite audit (gate A, approach v2.1 section 4): anon fills the rater1_* columns and one external rater "
        "fills the rater2_* columns of results/v2a_rewrite_audit_sheet.csv (y or n per criterion), given only the "
        "sheet and the rewrite-audit instructions; attach the filled sheet in the chat.",
        "Check results/v2a_rewrite_summary.csv and results/v2a_card_summary.csv (automatic-check pass rates, "
        "grounded facts per card) before launch 2.",
        "Launch 2 pins results/v2a_cards.jsonl, results/v2a_rewrites.jsonl and results/v2a_edgar_manifest.json by "
        "the sha256 in results/v2a_frozen_artifacts.json; do not edit them.",
    ]
