"""The six source collections, parsed from the pinned files into one common shape, and the fifteen
configurations built from them.

Document: {"doc_id", "title", "text", "aliases": [...], "meta": {...}}
Query:    {"qid", "source", "text", "text_mask" (or None), "gold_docs": [...], "evidence": [...], "answer", "meta"}

Parsing is a pure function of the verified pins (cached on disk under the data cache), so every stage and
every cell builder can call it, in any rung, and get identical output."""
from __future__ import annotations

import csv
import gzip
import html
import json
import os
import pickle
import re
import zipfile
from functools import lru_cache
from pathlib import Path

from rh import datapins

from . import text as T

SOURCES = ("fin", "fda", "cuad", "md2d", "qasper", "mhrag")
N_QUERIES = 300                       # per source (fewer if the source has fewer usable questions)
N_DEV = 100                           # v2 dev pool per source: the next 100 in the same fixed order
N_FRESH = 2000                        # study 3: questions after those (never used in Study 1 or Study 2)
SUB_ALT = 100                         # second-family generator, alternate router, cross-judge subset
SUB_AA = 60                           # A/A replicate subset (fin.base, flat arm)
FAMILIES = {"base": SOURCES, "mask": ("fin", "fda", "cuad", "mhrag"), "dilute": ("fin", "fda", "cuad"),
            "merge": ("fin", "fda")}
MASK_PHRASE = {"fin": "the company", "fda": "this drug", "cuad": None, "mhrag": "a news outlet"}
SOURCE_DESCRIPTION = {
    "fin": "annual 10-K filings, one per company; a document is identified by the company name or stock ticker",
    "fda": "FDA prescription drug labels; a document is identified by the drug name",
    "cuad": "commercial contracts; a document is identified by the party name, the agreement type and the date",
    "md2d": "U.S. government help pages (SSA, VA, DMV, StudentAid); a document is identified by its page title",
    "qasper": "NLP research papers; a document is identified by the paper title",
    "mhrag": "news articles from many publishers; a document is identified by the publisher and publication date",
}
PARSE_VERSION = "6"
PROGRESS = [None]                     # set by the corpora stage: callable(str) -> heartbeat touch


def _progress(msg: str) -> None:
    if PROGRESS[0]:
        PROGRESS[0](msg)


def _cache_dir() -> Path:
    d = Path(os.environ.get("RH_DATA_DIR", str(datapins.DATA_DIR))) / "routewise" / "parsed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pin_sha(name: str) -> str:
    rep = json.loads((datapins.REPORT).read_text()) if datapins.REPORT.exists() else {"pins": []}
    for r in rep["pins"]:
        if r["name"] == name:
            return str(r.get("sha256") or "")[:16]
    return "nopin"


def _read_table(p: Path) -> list[dict]:
    """parquet (real pins) or jsonl/json (fixtures); the loader decides by suffix."""
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(p).to_pylist()
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    d = json.loads(p.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("data", [])


def _find(root: Path, name: str) -> Path:
    if root.is_file():
        return root
    hits = sorted(p for p in root.rglob(name) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


# --------------------------------------------------------------------------- fin
_PHRASE = re.compile(r"\(?\s*exact\s+name\s+of\s+(?:each\s+)?registrants?\b", re.I)
_DEI_NAME = re.compile(r"name=[\"']dei:EntityRegistrantName[\"'][^>]*>(.*?)</ix:nonNumeric>", re.I | re.S)
_CUT = re.compile(r"(?:\b(?:commission\s+file\s+(?:number|no\.?)|file\s+number|transition\s+period\s+from|"
                  r"act\s+of\s+1934|fiscal\s+year\s+ended|form\s+10-k|or)\b[\s:.#\u2610\u2612]*|\d[\d\-]{3,}\s*|[:;|)\]]\s*)",
                  re.I)


def dei_registrant(raw_html: str) -> str:
    """dei:EntityRegistrantName from an inline-XBRL filing (tags inside the fact are removed)."""
    m = _DEI_NAME.search(raw_html or "")
    if not m:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", m.group(1)))).strip().rstrip(" ,")


_DEI_FIELD = r"name=[\"']dei:{}[\"'][^>]*>(.*?)</ix:nonNumeric>"
_FYE = re.compile(r"for\s+the\s+fiscal\s+year\s+ended\s*:?\s*([A-Z][a-z]+\.?\s+\d{1,2}\s*,?\s*(?:19|20)\d{2})", re.I)


def dei_field(raw_html: str, field: str) -> str:
    """Any dei:<field> fact from an inline-XBRL filing, tags removed ("" if absent)."""
    m = re.search(_DEI_FIELD.format(field), raw_html or "", re.I | re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", m.group(1)))).strip()


def fiscal_period(raw_html: str, txt: str) -> dict:
    """{"fy": "2024", "period_end": "December 31, 2024", "how": "dei"|"text"|"none"} for a 10-K."""
    fy, pe = dei_field(raw_html, "DocumentFiscalYearFocus"), dei_field(raw_html, "DocumentPeriodEndDate")
    if re.fullmatch(r"(19|20)\d{2}", fy or ""):
        return {"fy": fy, "period_end": pe, "how": "dei"}
    m = _FYE.search((txt or "")[:60000])
    if m:
        pe = re.sub(r"\s+", " ", m.group(1))
        return {"fy": re.search(r"(19|20)\d{2}", pe).group(0), "period_end": pe, "how": "text"}
    return {"fy": "", "period_end": "", "how": "none"}


def registrant_from_text(txt: str) -> str:
    """Registrant name from the cover page text: the words just before '(Exact name of registrant ...)'.
    Searches the whole filing (inline-XBRL hidden headers can push the cover page far down)."""
    for m in _PHRASE.finditer(txt or ""):
        win = txt[max(0, m.start() - 220):m.start()].replace("\xa0", " ")
        win = win.rstrip(" (\t\n")
        cut = 0
        for c in _CUT.finditer(win):
            cut = c.end()
        cand = re.sub(r"\s+", " ", win[cut:]).lstrip(" ,.-").rstrip(" ,-")
        cand = re.sub(r"^(?:to|from|[^\w&]+)\s+", "", cand).lstrip(" ,.-").rstrip(" ,-")
        if 2 <= len(cand) <= 90 and re.search(r"[A-Za-z]{2}", cand):
            return cand
    return ""
_TICKER = re.compile(r"^[A-Z]{1,5}(?:[.\-][A-Z])?$")
_TEXT_SUFFIX = {".txt", ".md", ".htm", ".html", ".pdf", ".json", ".jsonl", ".xml"}


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return html.unescape(s)


def _pdf_text(p: Path) -> str:
    from pypdf import PdfReader
    r = PdfReader(str(p))
    return "\n".join((pg.extract_text() or "") for pg in r.pages)


def _ticker_from_path(p: Path) -> str | None:
    for part in [p.stem] + [x for x in reversed(p.parts[:-1])]:
        for tok in re.split(r"[_\-\s]+", part):
            t = tok.upper()
            if _TICKER.match(t) and not t.isdigit() and t not in {"K", "10K", "FY", "SEC", "FORM"}:
                if tok == tok.upper() or len(tok) <= 5:
                    return t
    return None


def _fin_doc_texts(root: Path) -> list[tuple[str, str, str, str, dict]]:
    """(ticker, path, text, inline-XBRL registrant name or "", fiscal period) for every filing in the unpacked zip."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _TEXT_SUFFIX
                   and not p.name.startswith("._") and "__MACOSX" not in p.parts)
    out = []
    for n, p in enumerate(files):
        if n % 10 == 0:
            _progress(f"fin parse {n}/{len(files)}")
        suf = p.suffix.lower()
        if suf in (".json", ".jsonl"):
            rows = _read_table(p) if suf == ".jsonl" else (lambda d: d if isinstance(d, list) else [d])(
                json.loads(p.read_text(encoding="utf-8", errors="replace")))
            for r in rows:
                if not isinstance(r, dict):
                    continue
                txt = r.get("text") or r.get("content") or r.get("body") or ""
                tk = (r.get("ticker") or r.get("symbol") or _ticker_from_path(p) or "")
                if txt and tk:
                    out.append((str(tk).upper(), str(p), txt, str(r.get("company") or r.get("name") or ""),
                                fiscal_period("", txt)))
            continue
        tk = _ticker_from_path(p)
        if not tk:
            continue
        dei, raw = "", ""
        if suf == ".pdf":
            txt = _pdf_text(p)
        else:
            raw = p.read_text(encoding="utf-8", errors="replace")
            if suf in (".htm", ".html", ".xml"):
                dei = dei_registrant(raw)
                txt = _strip_html(raw)
            else:
                txt = raw
        out.append((tk, str(p), txt, dei, fiscal_period(raw, txt)))
    return out


def _parse_fin() -> tuple[list[dict], list[dict], dict]:
    root = datapins.path("finder_10k")
    raw = _fin_doc_texts(root)
    docs, seen = [], {}
    name_by = {"dei": 0, "text": 0, "none": 0}
    for tk, path, txt, dei, period in raw:
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) < 2000:
            continue
        if tk in seen:                          # one filing per ticker: keep the longest
            if len(txt) <= len(docs[seen[tk]]["text"]):
                continue
            docs[seen[tk]] = None
        name = dei or registrant_from_text(txt)
        name_src = "dei" if dei else ("text" if name else "none")
        aliases = [tk] + ([name] if name else [])
        core = T.company_core(name)
        if core and len(core) >= 4 and core != T.norm(name):
            aliases.append(core)
        seen[tk] = len(docs)
        docs.append({"doc_id": f"fin:{tk}", "title": f"{name or tk} ({tk}) annual report on Form 10-K",
                     "text": txt, "aliases": aliases, "meta": {"ticker": tk, "company": name, "path": path,
                                                              "name_source": name_src, "fy": period["fy"],
                                                              "period_end": period["period_end"],
                                                              "period_how": period["how"]}})
    docs = [d for d in docs if d]
    for d in docs:
        name_by[d["meta"]["name_source"]] += 1
    # queries: FinDER rows restricted to HDRR's five fixed 300-query splits
    hdrr = datapins.path("hdrr_repo")
    split_ids = []
    for i in range(1, 6):
        p = _find(hdrr, f"Linq-AI-Research_FinDER-300-0{i}.csv")
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                split_ids.append((r["query_id"].strip(), f"300-0{i}"))
    split_of = {}
    for qid, grp in split_ids:                        # the five splits overlap; keep each id once, first split wins
        split_of.setdefault(qid, grp)
    rows = {r["_id"]: r for r in _read_table(datapins.path("finder_queries"))}
    toks = {d["doc_id"]: set(T.content_tokens(d["text"])) for d in docs}
    by_tk = {d["meta"]["ticker"]: d for d in docs}
    queries, stats = [], {"name_source": name_by, "split_rows": len(split_ids), "split_ids": len(split_of),
                          "finder_rows": len(rows),
                          "resolved_by_ticker": 0,
                          "resolved_by_scan": 0, "unresolved": 0}
    for qid, grp in split_of.items():
        r = rows.get(qid)
        if not r:
            continue
        refs = [x for x in (r.get("references") or []) if isinstance(x, str) and x.strip()]
        if not refs:
            stats["unresolved"] += 1
            continue
        ev_toks = [T.content_tokens(x) for x in refs]

        def cov(doc_id):
            have = toks[doc_id]
            vals = [sum(1 for t in e if t in have) / len(e) for e in ev_toks if e]
            return sum(vals) / len(vals) if vals else 0.0
        gold = None
        for tk, d in by_tk.items():
            if T.mentions(r["text"], tk) and cov(d["doc_id"]) >= 0.8:
                gold, how = d["doc_id"], "ticker"
                break
        if gold is None:
            scored = sorted(((cov(did), did) for did in toks), reverse=True)[:2]
            if scored and scored[0][0] >= 0.85 and (len(scored) < 2 or scored[0][0] - scored[1][0] >= 0.05):
                gold, how = scored[0][1], "scan"
        if gold is None:
            stats["unresolved"] += 1
            continue
        stats["resolved_by_" + how] += 1
        gd = next(d for d in docs if d["doc_id"] == gold)
        queries.append({"qid": f"fin:{qid}", "source": "fin", "text": r["text"],
                        "text_mask": T.mask_aliases(r["text"], gd["aliases"], MASK_PHRASE["fin"]),
                        "gold_docs": [gold], "evidence": refs, "answer": r.get("answer") or "",
                        "meta": {"finder_id": qid, "hdrr_split": grp, "category": r.get("category"),
                                 "gold_by": how}})
    return docs, queries, stats


# --------------------------------------------------------------------------- fda
_PASSAGE = re.compile(r"\|\|PASSAGE_\d+\|\|")


def _parse_fda():
    root = datapins.path("fdarxbench")
    labels = _read_table(_find(root, "labels.jsonl"))
    docs = []
    for r in labels:
        name = (r.get("drug_name") or "").strip()
        txt = _PASSAGE.sub(" ", r.get("label_raw") or "")
        if not txt.strip():
            txt = " ".join(r.get("chunks") or [])
        aliases = [name] + [p.strip() for p in re.split(r"\s+and\s+|/|,", name) if len(p.strip()) >= 4]
        docs.append({"doc_id": f"fda:{r['set_id']}", "title": f"{name} prescribing information (FDA label)",
                     "text": re.sub(r"\s+", " ", txt).strip(), "aliases": list(dict.fromkeys(aliases)),
                     "meta": {"set_id": r["set_id"], "drug_name": name}})
    ids = {d["doc_id"] for d in docs}
    by_id = {d["doc_id"]: d for d in docs}
    queries = []
    for r in _read_table(_find(root, "qa.jsonl")):
        if r.get("task") != "factual":
            continue
        did = f"fda:{r.get('set_id')}"
        if did not in ids:
            continue
        ev = [c.get("text", "") for c in (r.get("context") or []) if c.get("text")]
        if not ev:
            continue
        queries.append({"qid": f"fda:{r['qid']}", "source": "fda", "text": r["question"],
                        "text_mask": T.mask_aliases(r["question"], by_id[did]["aliases"], MASK_PHRASE["fda"]),
                        "gold_docs": [did], "evidence": ev, "answer": r.get("answer") or "",
                        "meta": {"drug_name": r.get("drug_name")}})
    return docs, queries, {"labels": len(docs), "factual": len(queries)}


# --------------------------------------------------------------------------- cuad
_SKIP_CUAD = {"document name", "parties"}


def _cuad_title(title: str) -> tuple[str, str, str]:
    """(party, date, agreement type) from a CUAD contract title. Three naming patterns occur in CUAD v1:
    PARTY_MM_DD_YYYY-EX-10.1-TYPE (301), Party_YYYYMMDD_FORM_EX-n_NNN_EX-n_Type (176), PARTY - TYPE (33)."""
    t = title.strip()
    m = re.match(r"^(?P<party>.+?)_(?P<d>\d{8})_[^_]+_EX-[^_]+_\d+_EX-[^_]+_(?P<kind>.+)$", t)
    if m:
        d = m.group("d")
        party, date, kind = m.group("party"), f"{d[:4]}-{d[4:6]}-{d[6:]}", m.group("kind")
    else:
        left, sep, right = t.partition("-EX-")
        m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{4})$", left)
        if sep and m:
            date = f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            party = left[:m.start()]
            kind = re.sub(r"^[\d.()A-Za-z]*-", "", right, count=1)
        elif " - " in t:
            party, _, kind = t.partition(" - ")
            date = ""
        else:
            party, date, kind = left, "", right
    party = re.sub(r"[_]+", " ", party).strip(" ,.-")
    kind = re.sub(r"[_]+", " ", kind).strip(" -_.").lower() or "agreement"
    kind = re.sub(r"\s+", " ", kind)
    return party, date, kind


def _parse_cuad():
    root = datapins.path("cuad")
    files = [p for p in (sorted(root.rglob("test.json")) + sorted(root.rglob("train_separate_questions.json")))
             if "__MACOSX" not in p.parts]
    docs, queries, seen = [], [], set()
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))["data"]
        for art in data:
            title = art["title"]
            ctx = art["paragraphs"][0]["context"]
            party, date, kind = _cuad_title(title)
            did = f"cuad:{T.stable_hash(title)[:12]}"
            if did not in seen:
                seen.add(did)
                docs.append({"doc_id": did, "title": f"{kind} — {party}" + (f" ({date})" if date else ""),
                             "text": re.sub(r"\s+", " ", ctx).strip(), "aliases": [party] if party else [],
                             "meta": {"cuad_title": title, "party": party, "date": date, "kind": kind}})
            for para in art["paragraphs"]:
                for qa in para["qas"]:
                    ans = [a["text"] for a in qa.get("answers") or [] if a.get("text", "").strip()]
                    if qa.get("is_impossible") or not ans:
                        continue
                    q = qa["question"]
                    cat = re.search(r'related to "([^"]+)"', q)
                    if cat and cat.group(1).lower() in _SKIP_CUAD:
                        continue
                    det = re.search(r"Details:\s*(.+)$", q, re.S)
                    core = (det.group(1) if det else q).strip()
                    where = f"the {kind} of {party}" + (f" dated {date}" if date else "")
                    queries.append({"qid": f"cuad:{qa['id']}", "source": "cuad", "text": f"For {where}: {core}",
                                    "text_mask": core, "gold_docs": [did], "evidence": ans[:5],
                                    "answer": "; ".join(dict.fromkeys(a.strip() for a in ans))[:800],
                                    "meta": {"category": cat.group(1) if cat else ""}})
    # one question per (contract, category) keeps the sample from piling onto a few long contracts
    return docs, queries, {"contracts": len(docs), "questions": len(queries)}


# --------------------------------------------------------------------------- md2d
def _parse_md2d():
    root = datapins.path("md2d")
    dd = json.loads(_find(root, "multidoc2dial_doc.json").read_text(encoding="utf-8"))["doc_data"]
    docs, spans = [], {}
    for domain, items in dd.items():
        for doc_id, d in items.items():
            title = d.get("title") or doc_id
            segs = [s.strip() for s in re.split(r"[|#]", title) if len(s.strip()) >= 8]
            did = f"md2d:{doc_id}"
            docs.append({"doc_id": did, "title": title.split("#")[0].strip(), "text": d.get("doc_text", ""),
                         "aliases": list(dict.fromkeys([title.split("#")[0].strip()] + segs[:2])),
                         "meta": {"domain": domain}})
            spans[did] = {k: v.get("text_sp", "") for k, v in (d.get("spans") or {}).items()}
    queries = []
    for split in ("test", "validation"):
        try:
            p = _find(root, f"multidoc2dial_dial_{split}.json")
        except FileNotFoundError:
            continue
        dial = json.loads(p.read_text(encoding="utf-8"))["dial_data"]
        for domain, lst in dial.items():
            for dg in lst:
                turns = dg.get("turns") or []
                for i, t in enumerate(turns):
                    if t.get("role") != "user" or not t.get("references"):
                        continue
                    ref = t["references"]
                    did = f"md2d:{ref[0]['doc_id']}"
                    ev = [spans.get(did, {}).get(r["id_sp"], "") for r in ref if r.get("doc_id") == ref[0]["doc_id"]]
                    ev = [e for e in ev if e.strip()]
                    nxt = next((u for u in turns[i + 1:] if u.get("role") == "agent"), None)
                    if ev and did in spans:
                        queries.append({"qid": f"md2d:{dg['dial_id']}", "source": "md2d", "text": t["utterance"],
                                        "text_mask": None, "gold_docs": [did], "evidence": ev,
                                        "answer": (nxt or {}).get("utterance", " ".join(ev))[:800],
                                        "meta": {"domain": domain, "split": split}})
                    break
    return docs, queries, {"docs": len(docs), "dialogues": len(queries)}


# --------------------------------------------------------------------------- qasper
def _as_records(struct, keys):
    """HF parquet stores nested sequences either as a dict of lists or as a list of dicts."""
    if isinstance(struct, dict):
        n = len(struct.get(keys[0]) or [])
        return [{k: (struct.get(k) or [None] * n)[i] for k in keys} for i in range(n)]
    return list(struct or [])


def _parse_qasper():
    rows = _read_table(datapins.path("qasper_test"))
    docs, queries = [], []
    for r in rows:
        secs = _as_records(r.get("full_text") or {}, ["section_name", "paragraphs"])
        body = [r.get("abstract") or ""]
        for s in secs:
            paras = s.get("paragraphs") or []
            body.append(f"{s.get('section_name') or ''}: " + " ".join(p for p in paras if p))
        did = f"qasper:{r['id']}"
        docs.append({"doc_id": did, "title": r.get("title") or r["id"], "text": re.sub(r"\s+", " ", " ".join(body)),
                     "aliases": [r.get("title") or ""], "meta": {"paper_id": r["id"]}})
        qas = _as_records(r.get("qas") or {}, ["question", "question_id", "answers"])
        for qa in qas:
            anns = _as_records(qa.get("answers") or {}, ["answer"])
            for a in anns:
                ans = a.get("answer") or {}
                if isinstance(ans, list):
                    ans = ans[0] if ans else {}
                if ans.get("unanswerable"):
                    continue
                ev = [e for e in (ans.get("evidence") or []) if e and not e.startswith("FLOAT SELECTED")]
                if not ev:
                    continue
                text = (ans.get("free_form_answer") or ", ".join(ans.get("extractive_spans") or [])
                        or ("yes" if ans.get("yes_no") else "no" if ans.get("yes_no") is False else ""))
                if not text:
                    continue
                queries.append({"qid": f"qasper:{qa['question_id']}", "source": "qasper", "text": qa["question"],
                                "text_mask": None, "gold_docs": [did], "evidence": ev, "answer": text[:800],
                                "meta": {}})
                break
    return docs, queries, {"papers": len(docs), "questions": len(queries)}


# --------------------------------------------------------------------------- mhrag
def _parse_mhrag():
    corpus = _read_table(datapins.path("mhrag_corpus"))
    docs, by_title = [], {}
    for i, r in enumerate(corpus):
        did = f"mhrag:{i:04d}"
        src = (r.get("source") or "").strip()
        docs.append({"doc_id": did, "title": f"{r.get('title', '')} — {src} ({str(r.get('published_at', ''))[:10]})",
                     "text": re.sub(r"\s+", " ", r.get("body") or "").strip(),
                     "aliases": [a for a in [src] if a],
                     "meta": {"publisher": src, "date": str(r.get("published_at") or "")[:10]}})
        by_title.setdefault(T.norm(r.get("title", "")), []).append(did)
    publishers = sorted({d["meta"]["publisher"] for d in docs if d["meta"]["publisher"]}, key=len, reverse=True)
    queries = []
    for i, r in enumerate(_read_table(datapins.path("mhrag_queries"))):
        if r.get("question_type") == "null_query":
            continue
        gold = []
        for e in r.get("evidence_list") or []:
            gold += by_title.get(T.norm(e.get("title", "")), [])
        gold = list(dict.fromkeys(gold))
        ev = [e.get("fact", "") for e in r.get("evidence_list") or [] if e.get("fact")]
        if not gold or not ev:
            continue
        queries.append({"qid": f"mhrag:{i:04d}", "source": "mhrag", "text": r["query"],
                        "text_mask": T.mask_aliases(r["query"], publishers, MASK_PHRASE["mhrag"]),
                        "gold_docs": gold, "evidence": ev, "answer": str(r.get("answer") or ""),
                        "meta": {"question_type": r.get("question_type")}})
    return docs, queries, {"articles": len(docs), "queries": len(queries)}


PARSERS = {"fin": _parse_fin, "fda": _parse_fda, "cuad": _parse_cuad, "md2d": _parse_md2d,
           "qasper": _parse_qasper, "mhrag": _parse_mhrag}
PINS_OF = {"fin": ("finder_10k", "finder_queries", "hdrr_repo"), "fda": ("fdarxbench",), "cuad": ("cuad",),
           "md2d": ("md2d",), "qasper": ("qasper_test",), "mhrag": ("mhrag_corpus", "mhrag_queries")}


def _sample(queries: list[dict], n: int | None = None) -> list[dict]:
    """The first n (default N_QUERIES) per source in the fixed order; for cuad at most one question per
    (contract, category). The first N_QUERIES are Study 1's test sample; the next N_DEV are the dev pool."""
    n = N_QUERIES if n is None else n
    seen, out = set(), []
    for q in sorted(queries, key=lambda q: T.order_key(q["qid"])):
        k = (q["gold_docs"][0], q["meta"].get("category")) if q["source"] == "cuad" else q["qid"]
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
        if len(out) >= n:
            break
    return out


EXTRA_LOADERS: dict = {}     # v2: derived collections (finmy = fin + EDGAR decoys), registered by paper.v2b


@lru_cache(maxsize=None)
def load(source: str) -> dict:
    """{"docs": [...], "queries": [...] (the sample, in fixed order), "stats": {...}}"""
    if source in EXTRA_LOADERS:
        return EXTRA_LOADERS[source]()
    key = "-".join(_pin_sha(p) for p in PINS_OF[source])
    cache = _cache_dir() / f"{source}-v{PARSE_VERSION}-{key}.pkl.gz"
    if cache.exists():
        with gzip.open(cache, "rb") as f:
            return pickle.load(f)
    docs, queries, stats = PARSERS[source]()
    pool = _sample(queries, N_QUERIES + N_DEV + N_FRESH)
    sample, dev, fresh = pool[:N_QUERIES], pool[N_QUERIES:N_QUERIES + N_DEV], pool[N_QUERIES + N_DEV:]
    stats.update(n_docs=len(docs), n_candidates=len(queries), n_sample=len(sample), n_dev=len(dev),
                 n_fresh=len(fresh))
    out = {"docs": docs, "queries": sample, "dev": dev, "fresh": fresh, "stats": stats}
    tmp = cache.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
    return out


# --------------------------------------------------------------------------- configurations
def config_names() -> list[str]:
    out = []
    for fam in ("base", "mask", "dilute", "merge"):
        out += [f"{s}.{fam}" for s in SOURCES if s in FAMILIES[fam]]
    return out


def text_variant(config: str) -> str:
    """Config whose query text this config uses (route cells exist only for base and mask)."""
    s, fam = config.split(".")
    return f"{s}.mask" if fam == "mask" else f"{s}.base"


@lru_cache(maxsize=None)
def config(name: str) -> dict:
    """Catalog of the configuration: doc_map original doc -> config doc; titles, aliases, meta per config doc."""
    src, fam = name.split(".")
    data = load(src)
    docs = data["docs"]
    gold = {g for q in data["queries"] for g in q["gold_docs"]}
    if fam == "dilute":
        keep = [d for d in docs if d["doc_id"] in gold or int(T.stable_hash("dilute|" + d["doc_id"])[:8], 16) % 2 == 0]
    else:
        keep = list(docs)
    doc_map, titles, aliases, meta = {}, {}, {}, {}
    if fam == "merge":
        order = sorted(keep, key=lambda d: T.stable_hash("merge|" + d["doc_id"]))
        for i in range(0, len(order), 2):
            pair = order[i:i + 2]
            cid = "m:" + "+".join(d["doc_id"] for d in pair)
            for d in pair:
                doc_map[d["doc_id"]] = cid
            titles[cid] = " / ".join(d["title"] for d in pair)
            aliases[cid] = [a for d in pair for a in d["aliases"]]
            meta[cid] = {"members": [d["doc_id"] for d in pair]}
    else:
        for d in keep:
            doc_map[d["doc_id"]] = d["doc_id"]
            titles[d["doc_id"]] = d["title"]
            aliases[d["doc_id"]] = list(d["aliases"])
            meta[d["doc_id"]] = dict(d["meta"])
    field = "text_mask" if fam == "mask" else "text"
    qs = []
    for q in data["queries"]:
        t = q.get(field)
        if t is None:
            continue
        qs.append({**q, "q": t, "gold_cfg": sorted({doc_map[g] for g in q["gold_docs"] if g in doc_map})})
    return {"name": name, "source": src, "family": fam, "doc_map": doc_map, "titles": titles, "aliases": aliases,
            "meta": meta, "queries": qs, "title_variant": "merge" if fam == "merge" else "base"}


def probe(queries: list[dict], mode: str, plan: dict) -> list[dict]:
    if mode == "full":
        return queries
    k = int(plan["pilot_per_cell" if mode == "pilot" else "canary_per_cell"])
    return queries[:k]
