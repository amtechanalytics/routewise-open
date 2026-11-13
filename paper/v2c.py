"""ROUTEWISE study 3, launch 3a: fresh questions at two levels of identity explicitness, screened for meaning
preservation before any arm is run (pre-registration eswa27-13, frozen 12 Nov 2026, amendment 1).

  corpora -> index3 (chunk indexes of the screened collections) -> cand3 (fresh originals; L1 partial forms; L2
  rewrite inputs with contrastive facts, two attempts) -> rewrite3a / rewrite3b (gpt-oss, the frozen rewrite
  prompt) -> leak3 (strict leak and uniqueness checks) -> oret3 (oracle passages: gold documents only) -> ogen3
  (Haiku) -> ojudge3 (Haiku) -> panel3 (Sonnet + gpt-oss validity panel) -> screen3 (accepted test/dev items,
  yields, the blind human rating sheet) -> analysis (results/s3_*)

Launch 3b (the arms) reads the accepted items frozen here. Profile cards are the frozen launch-1 cards
(paper/v2b.py frozen(), sha256-checked)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from rh.util import RESULTS, RUNS, write_json

from . import prompts as P
from . import sources as S
from . import text as T
from . import v2a as A
from . import v2b as B

HAIKU, OSS, SONNET = B.HAIKU, B.OSS, B.SONNET
SCREEN_SOURCES = ("cuad", "fda", "mhrag", "qasper")      # primary collections (P1, P2)
RESOLUTION_ONLY = ("fin",)                                # leak check only; resolution check in launch 3b (H5)
SOURCES3 = SCREEN_SOURCES + RESOLUTION_ONLY
LEVELS = {"cuad": ("L1", "L2"), "fda": ("L1", "L2"), "mhrag": ("L2",), "qasper": ("L1", "L2"),
          "fin": ("L1", "L2")}
N_FRESH = {"cuad": 1500, "fda": 1500, "mhrag": 1500, "qasper": 1500, "fin": 400}
TEST_PER_CELL, DEV_PER_CELL, MIN_CELL = 400, 100, 150
DEV_SHARE = 5                              # an original goes to dev when stable_hash % 5 == 0 (20 %)
N_NEIGHBOURS = 5
FACTS_PER_ATTEMPT = 3
RATE_ACCEPTED, RATE_REJECTED = 150, 50
CRITERIA = ("same_answer", "identifiable", "no_leak", "plausible")
GATING = ("same_answer", "no_leak", "plausible")          # amendment 1: identifiability is checked deterministically
MAX_TOKENS = {"rewrite": {OSS: 4000}, "generate": {HAIKU: 4000}, "judge": {HAIKU: 1500},
              "validity": {SONNET: 1500, OSS: 3000}}
A.DOC_KIND.setdefault("qasper", "a research paper in natural language processing")


def fake() -> bool:
    return B.fake()


def cfg_name(source: str, level: str) -> str:
    return f"{source}.{level}"


# =========================================================================== fresh pool
def fresh_pool(source: str) -> list[dict]:
    """Questions never used in Study 1 or Study 2: the ones after the first N_QUERIES + N_DEV of the fixed order.
    The fixture collections are smaller than that, so dry runs use their sample instead."""
    d = S.load(source)
    pool = d["queries"] if fake() else d.get("fresh", [])
    return pool[:N_FRESH[source]]


@lru_cache(maxsize=None)
def used_qids() -> frozenset:
    """Every question id used in Study 1 or Study 2 (test and dev samples, and the frozen launch-1 rewrites)."""
    out = set()
    for s in SOURCES3:
        d = S.load(s)
        out |= {q["qid"] for q in d["queries"] + d.get("dev", [])}
    out |= {r["qid"] for r in B.frozen()["rewrites"]}
    return frozenset(out)


def split_of(qid: str) -> str:
    return "dev" if int(T.stable_hash("s3split|" + qid)[:8], 16) % DEV_SHARE == 0 else "test"


# =========================================================================== L1 partial references
def qasper_l1(doc: dict, catalog: list[dict]) -> str | None:
    """A contiguous part of the paper's title (token Jaccard <= 0.5 to the title, at least one content word) that
    occurs in no other paper's title; the shortest such part."""
    toks = T.tokens(doc["title"])
    nt = set(toks)
    if len(toks) < 2:
        return None
    cands = []
    for k in range(1, len(toks)):
        for i in range(0, len(toks) - k + 1):
            cands.append(" ".join(toks[i:i + k]))
    others = [T.norm(d["title"]) for d in catalog if d["doc_id"] != doc["doc_id"]]
    for c in sorted(dict.fromkeys(cands), key=lambda x: (len(x), x)):
        ct = set(T.tokens(c))
        if not T.content_tokens(c) or len(T.norm(c)) < 4 or len(ct & nt) / len(ct | nt) > 0.5:
            continue
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(T.norm(c)) + r"(?![a-z0-9])")
        if any(pat.search(o) for o in others):
            continue
        return c
    return None


def make_l1(source: str, q: dict, by_id: dict, catalog: list[dict]) -> tuple[str | None, str]:
    if source == "qasper":
        d = by_id.get(q["gold_docs"][0])
        form = qasper_l1(d, catalog) if d else None
        return (f"For the paper on {form}: {q['text']}", "ok") if form else (None, "no_unique_partial_form")
    if source == "mhrag":
        return None, "mhrag_l1_dropped"
    return A.make_l1(source, q, by_id, catalog)


# =========================================================================== L2 inputs: contrastive facts
def masked_question(source: str, q: dict, by_id: dict) -> tuple[str, list[str]]:
    if source == "qasper":
        return f"For {A.PH.format(1)}: {q['text']}", [q["gold_docs"][0]]
    return A.masked_question(source, q, by_id)


def _candidates_all(text: str, evidences: list[str]) -> list[tuple[int, str]]:
    """Informative sentences from the whole document (launch 1 used only the part outside the card footprint),
    with the same filters: 60-400 characters, no table/signature junk, >= 6 content words, a number or a
    multi-word name, and no overlap with any gold evidence of the document."""
    ev_tok = [set(T.content_tokens(e)) for e in evidences if e]
    out = []
    for i, s in A._sentences(text):
        if not 60 <= len(s) <= 400 or A._junk(s):
            continue
        ct = T.content_tokens(s)
        if len(ct) < 6:
            continue
        st = set(ct)
        if any(e and (len(st & e) / len(e) >= 0.5 or len(st & e) / len(st) >= 0.5) for e in ev_tok):
            continue
        if not re.search(r"\d|[A-Z][a-z]+\s+[A-Z][a-z]+", s):
            continue
        out.append((i, s.strip()))
    return out


@lru_cache(maxsize=None)
def neighbours(source: str, doc_id: str) -> tuple:
    """The N_NEIGHBOURS documents most similar to this one (BM25 of its opening 3,000 characters)."""
    d = _doc(source, doc_id)
    ranked = A.doc_rank(source, d["text"][:3000], N_NEIGHBOURS + 1)
    return tuple(x for x in ranked if x != doc_id)[:N_NEIGHBOURS]


@lru_cache(maxsize=None)
def _neighbour_tokens(source: str, doc_id: str) -> frozenset:
    out = set()
    for n in neighbours(source, doc_id):
        out |= set(T.content_tokens(_doc(source, n)["text"]))
    return frozenset(out)


def distinctiveness(fact: str, nb_tokens: frozenset, ident: set) -> float:
    toks = [t for t in T.content_tokens(fact) if t not in ident]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t not in nb_tokens) / len(toks)


@lru_cache(maxsize=None)
def _card_tokens(source: str, doc_id: str) -> frozenset:
    card = B.card_for(source, doc_id) or {}
    parts = [card.get("summary") or ""] + [str(f.get("fact", "")) + " " + str(f.get("quote", ""))
                                          for f in card.get("facts") or [] if isinstance(f, dict)]
    return frozenset(T.content_tokens(" ".join(parts)))


def in_card(source: str, doc_id: str, fact: str, ident: set) -> bool:
    """A fact is in-card when >= 60 % of its content words (names excluded) occur in the document's profile card."""
    toks = [t for t in T.content_tokens(fact) if t not in ident]
    if not toks:
        return False
    ct = _card_tokens(source, doc_id)
    return sum(1 for t in toks if t in ct) / len(toks) >= 0.6


@lru_cache(maxsize=None)
def _scored(source: str, doc_id: str) -> tuple:
    """(position, sentence, distinctiveness, in_card) for every candidate fact of a document, computed once."""
    d = _doc(source, doc_id)
    ev = _evidence(source).get(doc_id, [])
    ident = id_tokens(source, d)
    nb = _neighbour_tokens(source, doc_id)
    return tuple((i, s, round(distinctiveness(s, nb, ident), 4), in_card(source, doc_id, s, ident))
                 for i, s in _candidates_all(d["text"], ev))


def ranked_facts(source: str, doc: dict, seed: str) -> list[dict]:
    """Candidate facts ranked by distinctiveness against the nearest documents (ties: seeded hash)."""
    sc = sorted(_scored(source, doc["doc_id"]), key=lambda x: (-x[2], T.stable_hash(f"fact|{seed}|{x[0]}")))
    return [{"pos": i, "text": s, "distinct": dd, "in_card": ic} for i, s, dd, ic in sc]


@lru_cache(maxsize=None)
def _docs_by_id(source: str) -> dict:
    return {d["doc_id"]: d for d in S.load(source)["docs"]}


def _doc(source: str, doc_id: str) -> dict:
    return _docs_by_id(source)[doc_id]


@lru_cache(maxsize=None)
def _evidence(source: str) -> dict:
    """Gold evidence of every question of the collection (sample, dev and fresh pool): facts never overlap it."""
    d = S.load(source)
    out: dict = {}
    for q in d["queries"] + d.get("dev", []) + d.get("fresh", []):
        for g in q["gold_docs"]:
            out.setdefault(g, []).extend(q.get("evidence") or [])
    return out


def id_tokens(source: str, doc: dict) -> set:
    """Name words of the document (the strict re-check's identity tokens, rarity-filtered, NFKC-folded)."""
    return set(B.identity_tokens(source, doc))


def forbidden_names(source: str, doc: dict, masked: str) -> list[str]:
    in_q = set(T.tokens(masked))
    names = {a for a in doc.get("aliases") or [] if a} | {A.canonical_name(doc, source)}
    if source == "qasper":
        names = {doc["title"]}
    return sorted({n for n in names if n} | {t for t in id_tokens(source, doc) if t not in in_q})


def l2_attempts(source: str, q: dict, by_id: dict) -> tuple[dict | None, str]:
    """Rewrite inputs for attempt 1 (facts ranked 1-3) and attempt 2 (facts ranked 4-6), per gold document."""
    masked, owners = masked_question(source, q, by_id)
    if not owners:
        return None, "no_identity"
    attempts = []
    for a in range(2):
        blocks, forbid, facts_meta = [], set(), []
        for k, g in enumerate(owners, 1):
            d = by_id[g]
            ranked = ranked_facts(source, d, seed=f"{q['qid']}|{g}")
            pick = ranked[a * FACTS_PER_ATTEMPT:(a + 1) * FACTS_PER_ATTEMPT]
            if len(pick) < 2:
                return (None, "too_few_facts") if a == 0 else ({"masked": masked, "owners": owners,
                                                               "attempts": attempts}, "ok")
            names = forbidden_names(source, d, masked)
            terms = [t for t in names if " " not in t]
            forbid |= set(names)
            blocks.append((A.PH.format(k), A.DOC_KIND[source], [A.scrub(f["text"], terms) for f in pick]))
            facts_meta.append([{"distinct": f["distinct"], "in_card": f["in_card"], "pos": f["pos"]} for f in pick])
        ans = (q.get("answer") or "").strip()
        avoid = ans if 0 < len(ans) <= 120 and T.norm(ans) not in ("yes", "no") else ""
        attempts.append({"item": P.rewrite_item(masked, blocks, sorted(forbid), avoid), "facts": facts_meta})
    return {"masked": masked, "owners": owners, "attempts": attempts}, "ok"


# =========================================================================== candidates (pure function of the data)
def _probe_k(ctx) -> int | None:
    if ctx.mode == "full":
        return None
    return int(ctx.cfg.plan["pilot_per_cell" if ctx.mode == "pilot" else "canary_per_cell"])


_CANDS: dict = {}


def candidates(ctx) -> dict:
    """{source: [original record with its L1 and L2 candidates]} for the fresh pool. Outside the full run the
    first originals are taken until every level of the source has k usable candidates (at most 200 scanned)."""
    k = _probe_k(ctx)
    if k in _CANDS:
        return _CANDS[k]
    out = {}
    for s in SOURCES3:
        docs = S.load(s)["docs"]
        by_id = {d["doc_id"]: d for d in docs}
        recs, have = [], {lv: 0 for lv in LEVELS[s]}
        pool = fresh_pool(s)
        for n, q in enumerate(pool):
            if k and (all(v >= k for v in have.values()) or n >= 200):
                break
            if not all(g in by_id for g in q["gold_docs"]):
                continue
            rec = {"source": s, "qid": q["qid"], "split": split_of(q["qid"]), "orig": q["text"],
                   "answer": q.get("answer") or "", "evidence": q.get("evidence") or [], "gold_docs": q["gold_docs"]}
            if "L1" in LEVELS[s]:
                l1, why = make_l1(s, q, by_id, docs)
                rec["L1"] = {"text": l1, "status": "ok" if l1 else why}
            if "L2" in LEVELS[s]:
                inp, why = l2_attempts(s, q, by_id)
                rec["L2"] = {"status": why, **(inp or {})}
            usable = [lv for lv in LEVELS[s] if (rec.get(lv) or {}).get("status") == "ok"]
            if k and not usable:
                continue
            for lv in usable:
                have[lv] += 1
            recs.append(rec)
        out[s] = recs
    _CANDS[k] = out
    return out


def _cov_sources(ctx, mode):
    return [{"source": s, "config": cfg_name(s, lv), "gen_model": "-"} for s in SOURCES3 for lv in LEVELS[s]]


def _cov_screen(ctx, mode):
    return [{"source": s, "config": cfg_name(s, lv), "gen_model": "-"} for s in SCREEN_SOURCES for lv in LEVELS[s]]


def _cov_index(ctx, mode):
    return [{"source": s, "config": f"{s}.base", "gen_model": "-"} for s in SCREEN_SOURCES]


# =========================================================================== stages
def run_index3(ctx) -> dict:
    from . import embed
    from . import index as X
    ctx.register_diagnostics(A._diag("index3"))
    embed.init(ctx)
    out = {}
    for corpus in SCREEN_SOURCES:
        X.build(corpus, hb=ctx.hb)
        X.get.cache_clear()
        out[corpus] = {"chunks": len(X.get(corpus).chunks), "docs": len(S.load(corpus)["docs"])}
        write_json(RUNS / ctx.mode / "diag_index3.json", out)
    return {"corpora": out}


def _check_index3(ctx, m):
    z = [c for c, v in m["corpora"].items() if v["chunks"] == 0]
    return [f"collections without chunks: {z}"] if z else []


def run_cand3(ctx) -> dict:
    ctx.register_diagnostics(A._diag("cand3"))
    cands = candidates(ctx)
    used = used_qids() if not fake() else frozenset()
    counts, overlap = {}, 0
    with open(RUNS / ctx.mode / "cand3.jsonl", "w") as fo:
        for s, recs in cands.items():
            for r in recs:
                overlap += int(r["qid"] in used)
                for lv in LEVELS[s]:
                    st = (r.get(lv) or {}).get("status", "missing")
                    c = counts.setdefault(cfg_name(s, lv), {})
                    c[st] = c.get(st, 0) + 1
                fo.write(json.dumps({k: v for k, v in r.items() if k != "L2"} |
                                    {"L2_status": (r.get("L2") or {}).get("status")}) + "\n")
            ctx.hb.touch(cand3=s)
    diag = {"originals": {s: len(v) for s, v in cands.items()}, "status": counts, "used_overlap": overlap}
    write_json(RUNS / ctx.mode / "diag_cand3.json", diag)
    return diag


def _check_cand3(ctx, m):
    errs = []
    if m["used_overlap"]:
        errs.append(f"{m['used_overlap']} fresh questions were already used in Study 1 or Study 2")
    for c, st in m["status"].items():
        if not st.get("ok"):
            A.warn(ctx, "cand3", f"{c}: no usable candidate ({st})")
    return errs


def _rw_cell(s: str, r: dict, attempt: int) -> dict:
    a = r["L2"]["attempts"][attempt]
    return {"cell_id": f"rw3|{s}|{r['qid']}|a{attempt + 1}", "model": OSS, "arm": f"rewrite_a{attempt + 1}",
            "prefix": P.REWRITE_PREFIX, "item": a["item"], "max_tokens": MAX_TOKENS["rewrite"][OSS],
            "meta": {"source": s, "config": cfg_name(s, "L2"), "gen_model": "-", "qid": r["qid"],
                     "attempt": attempt + 1}}


def build_rewrite3a(ctx) -> list[dict]:
    cells = []
    for s, recs in candidates(ctx).items():
        for r in recs:
            if (r.get("L2") or {}).get("status") == "ok" and r["L2"]["attempts"]:
                cells.append(_rw_cell(s, r, 0))
    return cells


def build_rewrite3b(ctx) -> list[dict]:
    """Second attempt (facts ranked 4-6) for every L2 question whose first rewrite fails the leak check; before the
    first attempt has run (coverage report) or outside the full run, for every L2 question that has one."""
    rows = B._rows_by(ctx, "rewrite3a")
    cells, spare = [], None
    for s, recs in candidates(ctx).items():
        by_id = {d["doc_id"]: d for d in S.load(s)["docs"]}
        for r in recs:
            l2 = r.get("L2") or {}
            if l2.get("status") != "ok" or len(l2.get("attempts") or []) < 2:
                continue
            if rows and ctx.mode == "full":
                row = rows.get(f"rw3|{s}|{r['qid']}|a1")
                if row and row.get("status") == "ok" and not l2_fails(s, r, row["parsed"]["question"], by_id):
                    spare = spare or _rw_cell(s, r, 1)
                    continue
            cells.append(_rw_cell(s, r, 1))
    # a stage may not be empty (gate); a spare second attempt is harmless (leak3 takes the first passing attempt)
    return cells or ([spare] if spare else [])


def l2_fails(source: str, r: dict, text: str, by_id: dict) -> list[str]:
    """Strict leak and uniqueness checks of an L2 rewrite: a name or name word of a gold document (NFKC- and
    hyphen-folded; rarity-filtered as in the Study 2 strict re-check), a placeholder left, the answer stated, or
    the gold document not among the top BM25 documents for the rewrite (3; 5 for mhrag)."""
    fails = []
    rw = set(B.ntok(text))
    for g in r["gold_docs"]:
        d = by_id.get(g)
        if not d:
            continue
        names = [d["title"]] if source == "qasper" else [A.canonical_name(d, source)] + list(d["aliases"])
        if any(a and T.mentions(text, a) for a in names):
            fails.append("name_leak")
        if id_tokens(source, d) & rw:
            fails.append("name_leak_strict")
    if "⟨" in text or "⟩" in text:
        fails.append("placeholder_left")
    if A.answer_leak(r["orig"], text, r["answer"]):
        fails.append("answer_leak")
    k = 5 if source == "mhrag" else 3
    if not set(A.doc_rank(source, text, k)) & set(r["gold_docs"]):
        fails.append("not_unique")
    return sorted(set(fails))


def run_leak3(ctx) -> dict:
    """Final L1 and L2 items per original: L1 needs no answer leak (its form is unique by construction); L2 takes
    the first attempt that passes l2_fails()."""
    ctx.register_diagnostics(A._diag("leak3"))
    ra, rb = B._rows_by(ctx, "rewrite3a"), B._rows_by(ctx, "rewrite3b")
    counts: dict = {}
    n = 0
    with open(RUNS / ctx.mode / "items3.jsonl", "w") as fo:
        for s, recs in candidates(ctx).items():
            by_id = {d["doc_id"]: d for d in S.load(s)["docs"]}
            for r in recs:
                base = {k: r[k] for k in ("source", "qid", "split", "orig", "answer", "evidence", "gold_docs")}
                fo.write(json.dumps({**base, "level": "L0", "text": r["orig"], "status": "ok"}) + "\n")
                if "L1" in LEVELS[s]:
                    l1 = r["L1"]
                    st = l1["status"]
                    if st == "ok" and A.answer_leak(r["orig"], l1["text"], r["answer"]):
                        st = "answer_leak"
                    fo.write(json.dumps({**base, "level": "L1", "text": l1["text"] if st == "ok" else None,
                                         "status": st}) + "\n")
                    c = counts.setdefault(cfg_name(s, "L1"), {})
                    c[st] = c.get(st, 0) + 1
                if "L2" in LEVELS[s]:
                    l2 = r["L2"]
                    st, text, used, attempt, fails_all = l2["status"], None, None, None, []
                    if st == "ok":
                        st = "rewrite_missing"
                        for a, rows in ((1, ra), (2, rb)):
                            row = rows.get(f"rw3|{s}|{r['qid']}|a{a}")
                            if not row or row.get("status") != "ok":
                                continue
                            f = l2_fails(s, r, row["parsed"]["question"], by_id)
                            fails_all.append(f)
                            if not f:
                                st, text, attempt = "ok", row["parsed"]["question"], a
                                used = row["parsed"].get("facts_used") or []
                                break
                            st = f[0]
                    facts = (l2.get("attempts") or [{}] * 2)[(attempt or 1) - 1].get("facts") if l2.get("attempts") \
                        else None
                    flat = [x for per_doc in (facts or []) for x in per_doc]
                    used_meta = [flat[i - 1] for i in (used or []) if 1 <= i <= len(flat)]
                    fo.write(json.dumps({**base, "level": "L2", "text": text, "status": st, "attempt": attempt,
                                         "masked": l2.get("masked"), "fails": fails_all, "facts_used": used,
                                         "in_card": any(x["in_card"] for x in used_meta) if used_meta else None,
                                         "facts_meta": facts}) + "\n")
                    c = counts.setdefault(cfg_name(s, "L2"), {})
                    c[st] = c.get(st, 0) + 1
                n += 1
                if n % 200 == 0:
                    ctx.hb.touch(leak3=f"{s} {n}")
    write_json(RUNS / ctx.mode / "diag_leak3.json", counts)
    return {"status": counts}


def _check_leak3(ctx, m):
    for c, st in m["status"].items():
        if not st.get("ok"):
            A.warn(ctx, "leak3", f"{c}: no item passes the leak check ({st})")
    return []


def items(ctx) -> list[dict]:
    """Rows of items3.jsonl; before leak3 has run (coverage report), every candidate as a placeholder item."""
    p = RUNS / ctx.mode / "items3.jsonl"
    if p.exists():
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    out = []
    for s, recs in candidates(ctx).items():
        for r in recs:
            base = {k: r[k] for k in ("source", "qid", "split", "orig", "answer", "evidence", "gold_docs")}
            out.append({**base, "level": "L0", "text": r["orig"], "status": "ok"})
            for lv in LEVELS[s]:
                if (r.get(lv) or {}).get("status") == "ok":
                    out.append({**base, "level": lv, "text": (r[lv].get("text") if lv == "L1" else r["orig"]),
                                "status": "ok"})
    return out


def oracle_items(ctx) -> list[dict]:
    """Items the oracle answers: screened collections only; L1/L2 items that passed the leak check and the L0
    original of every question that has at least one."""
    its = [x for x in items(ctx) if x["source"] in SCREEN_SOURCES]
    keep = {(x["source"], x["qid"]) for x in its if x["level"] != "L0" and x["status"] == "ok"}
    return [x for x in its if (x["source"], x["qid"]) in keep and x["status"] == "ok"]


def oid(x: dict) -> str:
    return f"o3|{x['source']}|{x['level']}|{x['qid']}"


def _octx_path(mode: str) -> Path:
    return RUNS / mode / "octx3.jsonl.gz"


def run_oret3(ctx) -> dict:
    """Oracle passages: hybrid retrieval restricted to the gold documents, for every oracle item."""
    from . import embed
    ctx.register_diagnostics(A._diag("oret3"))
    embed.init(ctx)
    its = oracle_items(ctx)
    n = 0
    with gzip.open(_octx_path(ctx.mode), "wt") as g:
        for s in SCREEN_SOURCES:
            sub = [x for x in its if x["source"] == s]
            if not sub:
                continue
            cf = S.config(B.base_cfg(s))
            from . import index as X
            ix = X.get(s)
            vecs = B._qvecs_cached(tuple(x["text"] for x in sub))
            for i, x in enumerate(sub):
                gold = {cf["doc_map"][d] for d in x["gold_docs"] if d in cf["doc_map"]}
                idx = B.hybrid(s, x["text"], vecs[i], "plain", gold)
                docs = [cf["doc_map"][ix.chunks[j]["doc"]] for j in idx]
                g.write(json.dumps({"id": oid(x), "passages": [[cf["titles"][d], ix.chunks[j]["text"]]
                                                               for d, j in zip(docs, idx)]}) + "\n")
                n += 1
            ctx.hb.touch(oret3=s)
    write_json(RUNS / ctx.mode / "diag_oret3.json", {"items": n})
    return {"items": n}


def _check_oret3(ctx, m):
    return [] if m["items"] else ["no oracle item retrieved"]


def _load_octx(mode: str) -> dict:
    p = _octx_path(mode)
    if not p.exists():
        return {}
    out = {}
    with gzip.open(p, "rt") as g:
        for line in g:
            r = json.loads(line)
            out[r["id"]] = [tuple(x) for x in r["passages"]]
    return out


def build_ogen3(ctx) -> list[dict]:
    ctxs = _load_octx(ctx.mode)
    cells = []
    for x in oracle_items(ctx):
        cells.append({"cell_id": oid(x), "model": HAIKU, "arm": "oracle", "prefix": P.GEN_PREFIX,
                      "item": P.gen_item(x["text"], ctxs.get(oid(x), [])),
                      "max_tokens": MAX_TOKENS["generate"][HAIKU],
                      "meta": {"source": x["source"], "config": cfg_name(x["source"], x["level"]), "gen_model": "-",
                               "qid": x["qid"]}})
    return cells


def build_ojudge3(ctx) -> list[dict]:
    gen = B._rows_by(ctx, "ogen3")
    cells = []
    for x in oracle_items(ctx):
        if gen:
            r = gen.get(oid(x))
            if not r or r["status"] != "ok":
                continue
            ans = r["parsed"]["answer"]
        else:
            ans = ""
        cells.append({"cell_id": "j|" + oid(x), "model": HAIKU, "arm": "oracle", "prefix": P.JUDGE_PREFIX,
                      "item": P.judge_item(x["orig"], x["answer"], x["evidence"], ans),
                      "max_tokens": MAX_TOKENS["judge"][HAIKU],
                      "meta": {"source": x["source"], "config": cfg_name(x["source"], x["level"]), "gen_model": HAIKU,
                               "qid": x["qid"]}})
    return cells


def oracle_correct(ctx) -> dict:
    """{oid: True/False} from the Haiku judge (missing when not judged)."""
    out = {}
    for cid, r in B._rows_by(ctx, "ojudge3").items():
        if r.get("status") == "ok":
            out[cid[2:]] = bool(r["parsed"]["correct"])
    return out


def panel_items(ctx) -> list[dict]:
    """L1/L2 items the validity panel rates: in the full run, those whose oracle answer is correct both for the
    rewrite and for the original (screen criterion c); before the oracle has run, or outside the full run, every
    oracle item (so the pilot meets every panel combination)."""
    its = [x for x in oracle_items(ctx) if x["level"] != "L0"]
    oc = oracle_correct(ctx)
    if not oc or ctx.mode != "full":
        return its
    out = []
    for x in its:
        l0 = {**x, "level": "L0"}
        if oc.get(oid(x)) and oc.get(oid(l0)):
            out.append(x)
    return out or its[:1]          # a stage may not be empty (gate); screen3 still applies criterion c


def _desc(source: str, gold: list[str]) -> list[str]:
    cf = S.config(B.base_cfg(source))
    out = []
    for g in gold:
        gc = cf["doc_map"].get(g, g)
        card = B.card_for(source, gc) or B.card_for(source, g) or {}
        out.append(f"{cf['titles'].get(gc, gc)}: {(card.get('summary') or '')[:500]}")
    return out


def build_panel3(ctx) -> list[dict]:
    cells = []
    for x in panel_items(ctx):
        item = P.validity_item(x["orig"], x["text"], x["answer"], _desc(x["source"], x["gold_docs"]))
        for m in (SONNET, OSS):
            cells.append({"cell_id": f"v3|{x['source']}|{x['level']}|{x['qid']}|{m}", "model": m, "arm": "validity",
                          "prefix": P.VALIDITY_PREFIX, "item": item, "max_tokens": MAX_TOKENS["validity"][m],
                          "meta": {"source": x["source"], "config": cfg_name(x["source"], x["level"]),
                                   "gen_model": "-", "qid": x["qid"]}})
    return cells


# =========================================================================== screen
def screen_rows(ctx) -> list[dict]:
    """Every L1/L2 item with its screen outcome: leak check (a), validity panel (b, amendment 1: same_answer,
    no_leak and plausible true for both models; identifiability is enforced by construction for L1 and by the
    BM25 uniqueness check for L2), meaning preservation (c: oracle correct on the original and on the rewrite)."""
    oc = oracle_correct(ctx)
    pv: dict = {}
    for cid, r in B._rows_by(ctx, "panel3").items():
        _, s, lv, qid, m = cid.split("|", 4)
        if r.get("status") == "ok":
            pv.setdefault((s, lv, qid), {})[m] = r["parsed"]
    out = []
    for x in items(ctx):
        if x["level"] == "L0":
            continue
        row = {**x, "leak_ok": x["status"] == "ok"}
        if x["source"] in RESOLUTION_ONLY:
            row.update(oracle_l0=None, oracle_level=None, panel=None, panel_ok=None,
                       accepted=row["leak_ok"], reason="ok" if row["leak_ok"] else x["status"])
            out.append(row)
            continue
        o_lv = oc.get(oid(x))
        o_l0 = oc.get(oid({**x, "level": "L0"}))
        votes = pv.get((x["source"], x["level"], x["qid"]), {})
        panel_ok = len(votes) == 2 and all(v[k] for v in votes.values() for k in GATING)
        row.update(oracle_l0=o_l0, oracle_level=o_lv, panel=votes or None, panel_ok=panel_ok if votes else None)
        if not row["leak_ok"]:
            reason = x["status"]
        elif not o_l0:
            reason = "oracle_wrong_on_original" if o_l0 is False else "oracle_missing"
        elif not o_lv:
            reason = "oracle_wrong_on_rewrite" if o_lv is False else "oracle_missing"
        elif not votes or len(votes) < 2:
            reason = "panel_missing"
        elif not panel_ok:
            reason = "panel_rejects"
        else:
            reason = "ok"
        row.update(accepted=reason == "ok", reason=reason)
        out.append(row)
    return out


def assign(rows: list[dict]) -> dict:
    """Accepted items per (source, level), in seeded order: up to DEV_PER_CELL dev and TEST_PER_CELL test items,
    by the original's split."""
    cells: dict = {}
    for r in sorted(rows, key=lambda r: T.stable_hash(f"s3order|{r['source']}|{r['level']}|{r['qid']}")):
        if not r["accepted"]:
            continue
        c = cells.setdefault(cfg_name(r["source"], r["level"]), {"test": [], "dev": []})
        cap = TEST_PER_CELL if r["split"] == "test" else DEV_PER_CELL
        if len(c[r["split"]]) < cap:
            c[r["split"]].append(r)
    return cells


def rating_sample(rows: list[dict], cells: dict) -> list[dict]:
    """RATE_ACCEPTED accepted test items (equal share per screened cell, seeded) plus RATE_REJECTED items the
    screen rejected for the rewrite itself (the panel rejected it, or the oracle failed on the rewrite while it
    answered the original), mixed in seeded order."""
    scr = [c for c in sorted(cells) if c.split(".")[0] in SCREEN_SOURCES]
    acc = []
    if scr:
        per = max(1, RATE_ACCEPTED // len(scr))
        for c in scr:
            acc += sorted(cells[c]["test"], key=lambda r: T.stable_hash("rate|" + r["qid"]))[:per]
        rest = [r for c in scr for r in cells[c]["test"] if r not in acc]
        acc += sorted(rest, key=lambda r: T.stable_hash("rate+|" + r["qid"]))[:max(0, RATE_ACCEPTED - len(acc))]
    rej_pool = [r for r in rows if r["source"] in SCREEN_SOURCES and r["leak_ok"] and not r["accepted"]
                and r["reason"] in ("panel_rejects", "oracle_wrong_on_rewrite")]
    rej = sorted(rej_pool, key=lambda r: T.stable_hash("rej|" + r["qid"] + r["level"]))[:RATE_REJECTED]
    mixed = [dict(r, screened="accepted") for r in acc[:RATE_ACCEPTED]] + [dict(r, screened="rejected") for r in rej]
    return sorted(mixed, key=lambda r: T.stable_hash(f"mix|{r['source']}|{r['level']}|{r['qid']}"))


def _csv(rows: list[list], header: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def run_screen3(ctx) -> dict:
    ctx.register_diagnostics(A._diag("screen3"))
    rows = screen_rows(ctx)
    cells = assign(rows)
    out_d = RUNS / ctx.mode / "s3"
    out_d.mkdir(parents=True, exist_ok=True)
    keep = []
    for c, v in sorted(cells.items()):
        for sp in ("test", "dev"):
            for r in v[sp]:
                keep.append({k: r.get(k) for k in ("source", "level", "qid", "split", "text", "orig", "answer",
                                                   "evidence", "gold_docs", "attempt", "facts_used", "in_card",
                                                   "oracle_l0", "oracle_level", "panel")} | {"config": c})
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep)
    (out_d / "s3_items.jsonl").write_text(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    yields = []
    for s in SOURCES3:
        for lv in LEVELS[s]:
            c = cfg_name(s, lv)
            rs = [r for r in rows if r["source"] == s and r["level"] == lv]
            reasons: dict = {}
            for r in rs:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            nt, nd = len(cells.get(c, {}).get("test", [])), len(cells.get(c, {}).get("dev", []))
            yields.append([c, len(rs), sum(r["leak_ok"] for r in rs),
                           sum(1 for r in rs if r.get("oracle_l0")), sum(1 for r in rs if r.get("oracle_level")),
                           sum(1 for r in rs if r.get("panel_ok")), sum(r["accepted"] for r in rs), nt, nd,
                           int(nt >= MIN_CELL) if s in SCREEN_SOURCES else "resolution-only",
                           json.dumps(reasons, sort_keys=True)])
    (out_d / "s3_yields.csv").write_text(_csv(yields, ["config", "items", "leak_ok", "oracle_correct_original",
                                                       "oracle_correct_rewrite", "panel_ok", "accepted",
                                                       "test_items", "dev_items", "eligible", "reasons"]))
    sample = rating_sample(rows, cells)
    sheet, key = [], []
    for n, r in enumerate(sample, 1):
        sheet.append([n, A.DOC_KIND.get(r["source"], r["source"]), r["orig"], r["text"], (r["answer"] or "")[:600],
                      " | ".join(_desc(r["source"], r["gold_docs"])), "", "", "", "", ""])
        key.append([n, r["source"], r["level"], r["qid"], r["screened"], r["reason"], json.dumps(r.get("panel"))])
    (out_d / "s3_rating_sheet.csv").write_text(_csv(sheet, ["item", "collection", "original_question",
                                                           "rewritten_question", "reference_answer",
                                                           "document_title_and_summary", "same_answer",
                                                           "identifiable", "no_leak", "plausible", "comment"]))
    (out_d / "s3_rating_key.csv").write_text(_csv(key, ["item", "source", "level", "qid", "screened", "reason",
                                                       "panel_votes"]))
    agree = []
    for s in SCREEN_SOURCES:
        for lv in LEVELS[s]:
            rs = [r for r in rows if r["source"] == s and r["level"] == lv and r.get("panel")
                  and len(r["panel"]) == 2]
            for k in CRITERIA:
                a = [r["panel"][SONNET][k] for r in rs]
                b = [r["panel"][OSS][k] for r in rs]
                if rs:
                    agree.append([cfg_name(s, lv), k, len(rs), round(sum(a) / len(a), 4), round(sum(b) / len(b), 4),
                                  round(sum(x == y for x, y in zip(a, b)) / len(rs), 4)])
    (out_d / "s3_panel_agreement.csv").write_text(_csv(agree, ["config", "criterion", "n", "sonnet_true",
                                                              "gpt_oss_true", "agreement"]))
    in_card = {}
    for r in keep:
        if r["level"] == "L2" and r["in_card"] is not None:
            k = r["config"]
            in_card.setdefault(k, [0, 0])
            in_card[k][0 if r["in_card"] else 1] += 1
    frozen = {"s3_items.jsonl": {"sha256": sha, "items": len(keep)}, "in_card_vs_out": in_card}
    write_json(out_d / "s3_frozen_items.json", frozen)
    eligible = [y[0] for y in yields if y[9] == 1]
    summary = {"accepted": len(keep), "eligible_cells": eligible, "sha256": sha, "rating_items": len(sample)}
    write_json(RUNS / ctx.mode / "diag_screen3.json", summary)
    return summary


def _check_screen3(ctx, m):
    if ctx.mode == "full" and not fake() and not m["eligible_cells"]:
        A.warn(ctx, "screen3", "no screened cell reaches 150 accepted test items; P1/P2 cannot be tested "
                               "(pre-registration section 4: stop-rule route)")
    return []


# =========================================================================== analysis and brief
def analyze(ctx) -> None:
    src = RUNS / "full" / "s3"
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for p in sorted(src.iterdir()):
        (RESULTS / p.name).write_bytes(p.read_bytes())
    w = A._warnings() if hasattr(A, "_warnings") else []
    write_json(RESULTS / "s3_warnings.json", w)


def brief_items(ctx) -> list[str]:
    out = []
    d = RESULTS / "s3_frozen_items.json"
    if d.exists():
        fz = json.loads(d.read_text())
        out.append(f"Study 3 launch 3a: {fz['s3_items.jsonl']['items']} accepted items frozen "
                   f"(sha256 {fz['s3_items.jsonl']['sha256'][:16]}); yields in results/s3_yields.csv.")
    out.append("Send results/s3_rating_sheet.csv (not the key) to the rater; the human check gates launch 3b "
               "(pre-registration section 5).")
    w = A._warnings() if hasattr(A, "_warnings") else []
    if w:
        out.append("Quality warnings (reported, not gated): " + "; ".join(f"{x['stage']}: {x['msg']}" for x in w[:8]))
    return out
