"""ROUTEWISE v2, launch 2: ROUTEWISE-R against equally informed baselines under graded identity explicitness
(approach v2.1 sections 2-6 and amendments 1-11).

  corpora -> edgar (decoys re-fetched from the frozen manifest, sha256-checked) -> recheck (strict leak re-check of
  the frozen rewrites) -> index (plain, sac, card indexes) -> route (Haiku key router) -> rerank (cross-encoder over
  cards) -> resolve (Haiku resolver over cards, 4 card variants) -> calibrate (isotonic, LOSO) -> retrieve (12 arms)
  -> generate (Haiku) -> judge (Haiku on all; Haiku + gpt-oss + Sonnet panel on a stratified subset) -> validity
  (Sonnet + gpt-oss rate the accepted rewrites) -> analysis (paper/analysis_v2b.py)

Launch 1 artifacts (profile cards, rewrites, EDGAR manifest) are read from frozen/ and checked against the sha256 in
frozen/v2a_frozen_artifacts.json; dry runs on the fixture pins read tests/fixtures/routewise/frozen/ instead."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from rh.util import ROOT, RUNS, journal, write_json

from . import prompts as P
from . import sources as S
from . import text as T
from . import v2a as A

HAIKU, OSS, SONNET = "claude-haiku-5-5", "openai/gpt-oss-120b", "claude-sonnet-5-5"
SRC4 = ("fin", "fda", "cuad", "mhrag")
CONFIGS = ("fin.L0", "fin.L1", "fin.L2", "fda.L0", "fda.L1", "fda.L2", "cuad.L0", "cuad.L1", "cuad.L2",
           "mhrag.L0", "mhrag.L2", "finmy.L0", "finmy.L2", "qasper.L0", "md2d.L0")
CORPORA = ("fin", "fda", "cuad", "md2d", "qasper", "mhrag", "finmy")
CARD_VARIANTS = ("full", "noans", "title", "shuf")
R_ARMS = {"R": "full", "R-uncal": "full", "R-noans": "noans", "R-title": "title", "R-shuf": "shuf"}
BASE_ARMS = ("flat", "meta", "hdrr", "sac", "pdoc", "pdoc-rr")
ARMS = BASE_ARMS + tuple(R_ARMS) + ("oracle",)
N_CAND = 10                       # cards shown to the resolver
N_RERANK, N_RERANK_FALLBACK = 50, 20
PDOC_RR_DOCS = 5                  # HiREC-style: passages from the top-5 documents
MAX_RESOLVE = 3
RERANK_MAX_HOURS = 3.0
RERANKER = "BAAI/bge-reranker-v2-m3"
SOFT_BONUS = 1.0 / 61             # = one top-rank RRF vote (k = 60)
PANEL_N = 3000                    # judge panel subset (stratified by config x arm)
MAX_TOKENS = {"route": {HAIKU: 1500}, "resolve": {HAIKU: 2000}, "generate": {HAIKU: 4000},
              "judge": {HAIKU: 1500, OSS: 3000, SONNET: 1500}, "validity": {SONNET: 1500, OSS: 3000}}
LOOSE_GATE = {"min_parse_rate": 0.5, "max_duplicate_cells": 10 ** 6, "max_vendor_error_rate": 0.2,
              "empty_reply_ceiling": 0.5}
GENERIC = frozenset("""annual report form fda prescribing information label agreement contract news article
document filing company corporation inc corp ltd llc plc group holdings the and for with from drug tablets
tablet capsules injection oral solution usp""".split())


def fake() -> bool:
    return os.environ.get("RH_FAKE_VENDOR", "") == "1"


def frozen_dir() -> Path:
    return ROOT / ("tests/fixtures/routewise/frozen" if fake() else "frozen")


# =========================================================================== frozen launch-1 artifacts
def _frozen_bytes(d: Path, name: str) -> bytes:
    """Artifacts are stored gzip-compressed (name + ".gz"); the recorded sha256 is of the uncompressed bytes."""
    gz = d / (name + ".gz")
    return gzip.decompress(gz.read_bytes()) if gz.exists() else (d / name).read_bytes()


@lru_cache(maxsize=None)
def frozen() -> dict:
    d = frozen_dir()
    meta = json.loads((d / "v2a_frozen_artifacts.json").read_text())
    data = {}
    for name, m in meta.items():
        raw = _frozen_bytes(d, name)
        got = hashlib.sha256(raw).hexdigest()
        if got != m["sha256"]:
            raise ValueError(f"frozen artifact {name}: sha256 {got} != {m['sha256']}")
        data[name] = raw.decode("utf-8")
    cards = {}
    for line in data["v2a_cards.jsonl"].splitlines():
        if line.strip():
            r = json.loads(line)
            cards[(r["source"], r["doc_id"])] = r["card"]
    rw = [json.loads(x) for x in data["v2a_rewrites.jsonl"].splitlines() if x.strip()]
    man = json.loads(data["v2a_edgar_manifest.json"])
    return {"cards": cards, "rewrites": rw, "manifest": man}


def card_for(corpus: str, doc_id: str) -> dict | None:
    c = frozen()["cards"]
    if corpus == "finmy":
        return c.get(("finmy", doc_id)) or c.get(("fin", doc_id))
    return c.get((corpus, doc_id))


# =========================================================================== finmy collection
def fetch_frozen_decoys(hb=None) -> dict:
    """Download every decoy of the frozen manifest that is not on disk and check its sha256 (raw bytes)."""
    man = frozen()["manifest"]
    out = A._edgar_dir()
    res = {"filings": len(man["filings"]), "present": 0, "fetched": 0, "sha_mismatch": [], "failed": []}
    if fake():
        A.fetch_decoys(S.load("fin")["docs"])        # deterministic stand-in texts, same names and hashes
        for f in man["filings"]:
            p = out / f["file"]
            if not p.exists():
                res["failed"].append(f["file"])
                continue
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                h = hashlib.sha256(fh.read().encode()).hexdigest()
            if h != f["sha256"]:
                res["sha_mismatch"].append(f["file"])
            res["present"] += 1
        return res
    cl = A.EdgarClient()
    for n, f in enumerate(man["filings"]):
        p = out / f["file"]
        if hb is not None and n % 20 == 0:
            hb.touch(edgar=f"{n}/{len(man['filings'])}")
        if p.exists() and (out / (f["file"] + ".ok")).exists():
            res["present"] += 1
            continue
        try:
            raw = cl.get(f["url"])
        except Exception as e:  # noqa: BLE001
            res["failed"].append(f"{f['file']}: {type(e).__name__}")
            continue
        if hashlib.sha256(raw).hexdigest() != f["sha256"]:
            res["sha_mismatch"].append(f["file"])
            continue
        txt = re.sub(r"\s+", " ", S._strip_html(raw.decode("utf-8", errors="replace"))).strip()
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write(txt)
        (out / (f["file"] + ".ok")).write_text(f["sha256"])
        res["fetched"] += 1
    return res


def _finmy_load() -> dict:
    fin = S.load("fin")
    man = frozen()["manifest"]
    good = [f for f in man["filings"] if (A._edgar_dir() / f["file"]).exists()]
    decoys = A.decoy_docs({"filings": good}, fin["docs"])
    docs = []
    for d in fin["docs"]:
        fy = (d.get("meta") or {}).get("fy") or ""
        docs.append({**d, "title": d["title"] + (f" for fiscal year {fy}" if fy else "")})
    have = {d["doc_id"] for d in docs}
    docs += [d for d in decoys if d["doc_id"] not in have]
    return {"docs": docs, "queries": fin["queries"], "dev": fin.get("dev", []),
            "stats": {"n_docs": len(docs), "decoys": len(docs) - len(fin["docs"])}}


S.EXTRA_LOADERS["finmy"] = _finmy_load
S.PINS_OF.setdefault("finmy", S.PINS_OF["fin"])
S.SOURCE_DESCRIPTION.setdefault("finmy", S.SOURCE_DESCRIPTION["fin"].replace(
    "one per company", "three consecutive fiscal years per company") + " and the fiscal year")


def corpus_of(cfg: str) -> str:
    return cfg.split(".")[0]


def level_of(cfg: str) -> str:
    return cfg.split(".")[1]


def source_of(cfg: str) -> str:
    """Source for LOSO grouping (finmy belongs to fin)."""
    c = corpus_of(cfg)
    return "fin" if c == "finmy" else c


def base_cfg(corpus: str) -> str:
    return f"{corpus}.base"


# =========================================================================== strict re-check (amendment 10 i)
_FOLD = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "’": "'", "‘": "'"}


def nfold(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(_FOLD.get(ch, ch) for ch in s)


def ntok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", nfold(s).lower())


def identity_tokens(corpus: str, doc: dict) -> set:
    src = "fin" if corpus == "finmy" else corpus
    names = [A.canonical_name(doc, src)] + list(doc.get("aliases") or [])
    if src != "mhrag":
        card = card_for(corpus, doc["doc_id"]) or {}
        names += [card.get("name") or ""] + list(card.get("aliases") or [])
    out = set()
    for n in names:
        for t in ntok(n):
            if len(t) >= 3 and not t.isdigit() and t not in GENERIC and t not in T.CORP_SUFFIX \
                    and t not in A.NON_IDENT and t not in T._STOP:
                out.add(t)
        # tickers and short all-caps forms (e.g. "XEL", "KI") as written
        for t in re.findall(r"\b[A-Z]{2,5}\b", nfold(n)):
            out.add(t.lower())
    try:
        out |= set(A.identity_terms(src, doc))
    except Exception:  # noqa: BLE001
        pass
    if src != "mhrag":
        # a name word is rare in its collection; common words of card names ("between", "software", "FDA",
        # "AND") are not names (10 Nov check on the real CUAD and FDA rewrites)
        try:
            df = A._df(src)
            cap = max(2, int(0.01 * len(S.load(src)["docs"])))
            out = {t for t in out if df.get(t, 0) <= cap}
        except Exception:  # noqa: BLE001
            pass
    return out


def strict_status(row: dict, docs_by_id: dict) -> str:
    """ok, or why a launch-1 L1/L2 rewrite is dropped by the strict re-check."""
    if row["status"] != "ok" or not row.get("text"):
        return row["status"] or "missing"
    src = row["source"]
    if src == "mhrag" and row["level"] == "L1":
        return "mhrag_l1_dropped"
    if row["level"] != "L2":
        return "ok"
    corpus = "fin" if src == "finmy" else src
    ids = set()
    for g in row["gold_docs"]:
        if g in docs_by_id:
            ids |= identity_tokens(corpus, docs_by_id[g])
    masked = set(ntok(row.get("masked") or ""))
    if ids & masked:
        return "mask_incomplete"
    if ids & set(ntok(row["text"])):
        return "name_leak_strict"
    return "ok"


@lru_cache(maxsize=None)
def strict_items() -> dict:
    """{(source, level, split): {qid: rewritten text}} after the strict re-check, plus status counts."""
    docs = {s: {d["doc_id"]: d for d in S.load(s)["docs"]} for s in SRC4}
    keep: dict = {}
    counts: dict = {}
    fin_l2_ok = set()
    rows = frozen()["rewrites"]
    for r in rows:
        if r["source"] == "finmy":
            continue
        st = strict_status(r, docs[r["source"]])
        key = (r["source"], r["level"], r["split"])
        counts.setdefault(key, {})
        counts[key][st] = counts[key].get(st, 0) + 1
        if st == "ok":
            keep.setdefault(key, {})[r["qid"]] = r["text"]
            if r["source"] == "fin" and r["level"] == "L2":
                fin_l2_ok.add((r["split"], r["qid"]))
    for r in rows:
        if r["source"] != "finmy":
            continue
        key = ("finmy", r["level"], r["split"])
        ok = r["status"] == "ok" and bool(r.get("text")) and (r["level"] == "L0" or (r["split"], r["qid"]) in fin_l2_ok)
        counts.setdefault(key, {})
        st = "ok" if ok else (r["status"] if r["status"] != "ok" else "fin_l2_dropped")
        counts[key][st] = counts[key].get(st, 0) + 1
        if ok:
            keep.setdefault(key, {})[r["qid"]] = r["text"]
    return {"keep": keep, "counts": {"|".join(k): v for k, v in counts.items()}}


# =========================================================================== workload queries
def _probe_k(ctx) -> int | None:
    if ctx.mode == "full":
        return None
    return int(ctx.cfg.plan["pilot_per_cell" if ctx.mode == "pilot" else "canary_per_cell"])


@lru_cache(maxsize=None)
def _all_queries(cfg: str, split: str) -> tuple:
    corpus, level = corpus_of(cfg), level_of(cfg)
    data = S.load(corpus)
    pool = data["queries"] if split == "test" else data.get("dev", [])
    c = S.config(base_cfg(corpus))
    src_key = "finmy" if corpus == "finmy" else corpus
    keep = strict_items()["keep"].get((src_key, level, split)) if level != "L0" else None
    if corpus == "finmy" and level == "L0":
        keep = None
    out = []
    for q in pool:
        if keep is not None:
            if q["qid"] not in keep:
                continue
            text = keep[q["qid"]]
        else:
            text = q["text"]
        out.append({"qid": q["qid"], "q": text, "orig": q["text"], "answer": q.get("answer") or "",
                    "evidence": q.get("evidence") or [], "gold_cfg": sorted({c["doc_map"][g] for g in q["gold_docs"]
                                                                           if g in c["doc_map"]}),
                    "cfg": cfg, "split": split})
    if corpus in ("qasper", "md2d") and split == "dev":
        out = out[:100]
    return tuple(out)


def queries(ctx, cfg: str, split: str = "test") -> list[dict]:
    qs = list(_all_queries(cfg, split))
    k = _probe_k(ctx)
    return qs[:k] if k else qs


def splits_for(cfg: str) -> tuple:
    return ("test", "dev")


# =========================================================================== card texts and indexes
def card_text(corpus: str, doc: dict, variant: str, noans_drop: dict | None = None,
              shuf_map: dict | None = None) -> str:
    card = card_for(corpus, doc["doc_id"]) or {}
    name = card.get("name") or doc["title"]
    head = [doc["title"], name, card.get("doc_type") or "", card.get("date") or ""]
    if variant == "title":
        return " | ".join(x for x in head if x)
    src_card = card
    if variant == "shuf" and shuf_map is not None:
        src_card = card_for(corpus, shuf_map.get(doc["doc_id"], doc["doc_id"])) or {}
    facts = [f.get("fact") or "" for f in (src_card.get("facts") or [])]
    summ = src_card.get("summary") or ""
    if variant == "noans" and noans_drop is not None:
        bad = noans_drop.get(doc["doc_id"], set())
        facts = [f for i, f in enumerate(facts) if i not in bad]
        summ = " ".join(s for s in re.split(r"(?<=[.!?])\s+", summ) if not _overlaps(s, _EVID.get((corpus, doc["doc_id"]), [])))
    parts = head + [", ".join(card.get("aliases") or []), ", ".join(card.get("entities") or []) if variant != "shuf"
                    else ", ".join(src_card.get("entities") or []), summ] + facts
    return " | ".join(x for x in parts if x)


_EVID: dict = {}


def _overlaps(s: str, evid: list[str]) -> bool:
    st = set(T.content_tokens(s))
    if not st:
        return False
    for e in evid:
        et = set(T.content_tokens(e))
        if et and (len(st & et) / len(et) >= 0.5 or len(st & et) / len(st) >= 0.5):
            return True
    return False


def _evidence_by_doc(corpus: str) -> dict:
    data = S.load(corpus)
    out: dict = {}
    for q in data["queries"] + data.get("dev", []):
        for g in q["gold_docs"]:
            out.setdefault(g, []).extend(q.get("evidence") or [])
    return out


def noans_drops(corpus: str) -> dict:
    ev = _evidence_by_doc(corpus)
    out = {}
    for d in S.load(corpus)["docs"]:
        _EVID[(corpus, d["doc_id"])] = ev.get(d["doc_id"], [])
        card = card_for(corpus, d["doc_id"]) or {}
        bad = {i for i, f in enumerate(card.get("facts") or [])
               if _overlaps((f.get("fact") or "") + " " + (f.get("quote") or ""), ev.get(d["doc_id"], []))}
        out[d["doc_id"]] = bad
    return out


def shuffle_map(corpus: str) -> dict:
    ids = sorted(d["doc_id"] for d in S.load(corpus)["docs"])
    perm = list(ids)
    random.Random(T.stable_hash("shuf|" + corpus)).shuffle(perm)
    # derangement-ish: rotate any fixed point
    out = dict(zip(ids, perm))
    for k in ids:
        if out[k] == k and len(ids) > 1:
            j = ids[(ids.index(k) + 1) % len(ids)]
            out[k], out[j] = out[j], out[k]
    return out


def _idx_dir(corpus: str) -> Path:
    from . import index as X
    return X._dir(corpus)


class CardIndex:
    def __init__(self, corpus: str, variant: str, ids: list[str], texts: list[str], emb, bm, vocab):
        self.corpus, self.variant, self.ids, self.texts, self.emb, self.bm, self.vocab = \
            corpus, variant, ids, texts, emb, bm, vocab

    def search(self, query: str, qvec, k: int, allowed: set | None = None) -> list[tuple[str, float]]:
        n = len(self.ids)
        dense = np.asarray(self.emb @ np.asarray(qvec, dtype=np.float32), dtype=np.float32)
        cols = sorted({self.vocab[w] for w in T.content_tokens(query) if w in self.vocab})
        lex = np.asarray(self.bm[:, cols].sum(axis=1)).ravel() if cols else np.zeros(n, dtype=np.float32)
        rows = np.arange(n) if allowed is None else np.array([i for i, d in enumerate(self.ids) if d in allowed])
        if len(rows) == 0:
            return []
        fused: dict = {}
        kk = min(100, len(rows))
        for scores, is_lex in ((dense[rows], False), (lex[rows], True)):
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top], kind="stable")]
            for r, j in enumerate(top):
                if is_lex and scores[j] <= 0:
                    continue
                i = int(rows[j])
                fused[i] = fused.get(i, 0.0) + 1.0 / (61 + r)
        best = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.ids[i], float(s)) for i, s in best]


def build_card_index(corpus: str, variant: str, hb=None) -> CardIndex:
    from scipy import sparse
    from . import embed
    from . import index as X
    d = _idx_dir(corpus) / f"cards_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    docs = sorted(S.load(corpus)["docs"], key=lambda x: x["doc_id"])
    drops = noans_drops(corpus) if variant == "noans" else None
    smap = shuffle_map(corpus) if variant == "shuf" else None
    ids = [x["doc_id"] for x in docs]
    texts = [card_text(corpus, x, variant, drops, smap) for x in docs]
    key = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
    done = d / f"done_{key}.json"
    if done.exists():
        emb = np.load(d / "emb.npy")
        bm = sparse.load_npz(d / "bm25.npz").tocsc()
        vocab = json.loads((d / "vocab.json").read_text())
        return CardIndex(corpus, variant, ids, texts, emb, bm, vocab)
    emb = embed.embedder().encode(texts, cache_dir=d, label=f"cards_{variant}_{key}")
    bm, vocab = X._bm25_matrix(texts)
    np.save(d / "emb.npy", emb.astype(np.float32))
    sparse.save_npz(d / "bm25.npz", bm.tocsr())
    (d / "vocab.json").write_text(json.dumps(vocab))
    done.write_text(json.dumps({"n": len(ids)}))
    return CardIndex(corpus, variant, ids, texts, emb, bm.tocsc(), vocab)


_CARDIX: dict = {}


def card_index(corpus: str, variant: str) -> CardIndex:
    if (corpus, variant) not in _CARDIX:
        _CARDIX[(corpus, variant)] = build_card_index(corpus, variant)
    return _CARDIX[(corpus, variant)]


# ---------------------------------------------------------------- SAC chunk index (summary-prefixed chunks)
class SacIndex:
    def __init__(self, emb, bm, vocab):
        self.emb, self.bm, self.vocab = emb, bm, vocab


def build_sac(corpus: str, hb=None) -> SacIndex:
    from scipy import sparse
    from . import embed
    from . import index as X
    ix = X.get(corpus)
    d = _idx_dir(corpus) / "sac"
    d.mkdir(parents=True, exist_ok=True)
    done = d / "done.json"
    if done.exists():
        return SacIndex(np.load(d / "emb.npy", mmap_mode="r"), sparse.load_npz(d / "bm25.npz").tocsc(),
                        json.loads((d / "vocab.json").read_text()))
    summ = {}
    for doc in S.load(corpus)["docs"]:
        card = card_for(corpus, doc["doc_id"]) or {}
        summ[doc["doc_id"]] = card.get("summary") or doc["title"]
    texts = [f"{summ.get(c['doc'], '')}\n\n{c['text']}" for c in ix.chunks]

    def prog(a, b):
        if hb:
            hb.touch(index={"source": corpus, "step": "embed_sac", "done": int(a), "total": int(b)})
    emb = embed.embedder().encode(texts, progress=prog, cache_dir=d, label="sac")
    np.save(d / "emb.npy", emb.astype(np.float32))
    bm, vocab = X._bm25_matrix(texts)
    sparse.save_npz(d / "bm25.npz", bm.tocsr())
    (d / "vocab.json").write_text(json.dumps(vocab))
    done.write_text(json.dumps({"chunks": len(texts)}))
    return SacIndex(np.load(d / "emb.npy", mmap_mode="r"), bm.tocsc(), vocab)


_SAC: dict = {}


def sac_index(corpus: str) -> SacIndex:
    if corpus not in _SAC:
        _SAC[corpus] = build_sac(corpus)
    return _SAC[corpus]


# ---------------------------------------------------------------- hybrid chunk retrieval with options
def hybrid(corpus: str, query: str, qvec, mode: str, allowed: set | None = None, bonus_docs: set | None = None
           ) -> list[int]:
    """Chunk retrieval over the base configuration of a corpus.
    mode: "plain" / "meta" (Study 1 arms, via paper.index.retrieve when no bonus is asked), "sac" (summary-prefixed
    chunk index), "meta+bonus" (meta scores plus SOFT_BONUS for chunks of bonus_docs: ROUTEWISE-R soft scoping)."""
    from . import index as X
    cfg = base_cfg(corpus)
    if mode in ("plain", "meta") and not bonus_docs:
        return X.retrieve(cfg, query, qvec, mode, allowed)
    c = S.config(cfg)
    ix = X.get(corpus)
    rows, docs = X.cfg_chunks(cfg)
    if allowed is not None:
        uniq, codes = X._cfg_codes(cfg)
        ok = np.isin(uniq, np.array(sorted(allowed), dtype=str))
        rows = rows[ok[codes]]
    if len(rows) == 0:
        return []
    qv = np.asarray(qvec, dtype=np.float32)
    if mode == "sac":
        si = sac_index(corpus)
        dense = np.asarray(si.emb @ qv, dtype=np.float32)[rows]
        cols = sorted({si.vocab[w] for w in T.content_tokens(query) if w in si.vocab})
        lex = (np.asarray(si.bm[:, cols].sum(axis=1)).ravel()[rows] if cols
               else np.zeros(len(rows), dtype=np.float32))
    else:
        dense = np.asarray(ix.emb @ qv, dtype=np.float32)[rows]
        tid, temb = ix.titles[c["title_variant"]]
        dense = dense + X.TITLE_WEIGHT * (temb @ qv)[tid[rows]]
        lex = X._bm25_scores(ix, c["title_variant"], query, rows)
    k = min(X.POOL, len(rows))
    fused: dict = {}
    for scores, is_lex in ((dense, False), (lex, True)):
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        for r, j in enumerate(top):
            if is_lex and scores[j] <= 0:
                continue
            fused[int(rows[j])] = fused.get(int(rows[j]), 0.0) + 1.0 / (X.RRF_K + r + 1)
    if bonus_docs:
        for i in list(fused):
            if c["doc_map"].get(ix.chunks[i]["doc"]) in bonus_docs:
                fused[i] += SOFT_BONUS
    best = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:X.TOP_K]
    return [i for i, _ in best]


# =========================================================================== stages
def _diag(name):
    return A._diag(name)


def warn(ctx, stage, msg):
    A.warn(ctx, stage, msg)


def _cov_cfgs(ctx, mode):
    return [{"source": source_of(c), "config": c, "gen_model": "-"} for c in CONFIGS]


def run_edgar(ctx) -> dict:
    ctx.register_diagnostics(_diag("edgar2"))
    res = fetch_frozen_decoys(ctx.hb)
    write_json(RUNS / ctx.mode / "diag_edgar2.json", res)
    S.load.cache_clear()
    return res


def _check_edgar(ctx, m):
    if m["sha_mismatch"] or m["failed"]:
        warn(ctx, "edgar", f"{len(m['sha_mismatch'])} decoys with a changed sha256 and {len(m['failed'])} failed "
                           f"downloads are left out of fin-multiyear")
    return []


def _cov_edgar(ctx, mode):
    return [{"source": "fin", "config": "finmy.L0", "gen_model": "-"}]


def run_recheck(ctx) -> dict:
    ctx.register_diagnostics(_diag("recheck"))
    si = strict_items()
    n = {c: {sp: len(_all_queries(c, sp)) for sp in ("test", "dev")} for c in CONFIGS}
    write_json(RUNS / ctx.mode / "recheck.json", {"counts": si["counts"], "queries": n})
    return {"queries": n}


def _check_recheck(ctx, m):
    empty = [c for c, v in m["queries"].items() if v["test"] == 0]
    if len(empty) == len(CONFIGS):
        return ["no test query in any configuration"]
    for c in empty:
        warn(ctx, "recheck", f"{c}: no test query left after the strict re-check")
    return []


def run_index(ctx) -> dict:
    from . import embed
    from . import index as X
    ctx.register_diagnostics(_diag("index2"))
    embed.init(ctx)
    out = {}
    for corpus in CORPORA:
        X.build(corpus, hb=ctx.hb)
        X.get.cache_clear()
        ix = X.get(corpus)
        if corpus != "finmy":
            build_sac(corpus, hb=ctx.hb)
        for v in CARD_VARIANTS:
            _CARDIX[(corpus, v)] = build_card_index(corpus, v, hb=ctx.hb)
            ctx.hb.touch(index={"source": corpus, "step": f"cards_{v}"})
        out[corpus] = {"chunks": len(ix.chunks), "docs": len(S.load(corpus)["docs"])}
        write_json(RUNS / ctx.mode / "diag_index2.json", out)
    return {"corpora": out}


def _check_index(ctx, m):
    z = [c for c, v in m["corpora"].items() if v["chunks"] == 0]
    return [f"corpora without chunks: {z}"] if z else []


def _cov_index(ctx, mode):
    return [{"source": "fin" if c == "finmy" else c, "config": f"{c}.index", "gen_model": "-"} for c in CORPORA]


# ---------------------------------------------------------------- route (Haiku key router on every query text)
def _qkeys(ctx):
    for c in CONFIGS:
        for sp in ("test", "dev"):
            for q in queries(ctx, c, sp):
                yield c, sp, q


def build_route(ctx) -> list[dict]:
    cells = []
    for c, sp, q in _qkeys(ctx):
        cells.append({"cell_id": f"route2|{c}|{sp}|{q['qid']}", "model": HAIKU, "arm": "route",
                      "prefix": P.ROUTER_PREFIX, "item": P.router_item(S.SOURCE_DESCRIPTION[corpus_of(c)], q["q"]),
                      "max_tokens": MAX_TOKENS["route"][HAIKU],
                      "meta": {"source": source_of(c), "config": c, "gen_model": "-", "qid": q["qid"], "split": sp}})
    return cells


def _rows_by(ctx, stage: str) -> dict:
    out = {}
    if not ctx.out(stage).exists():
        return out
    for r in ctx.stage_rows(stage):
        out[r["cell_id"]] = r
    return out


@lru_cache(maxsize=None)
def _qvecs_cached(texts: tuple) -> np.ndarray:
    from . import embed
    return embed.embedder().encode(list(texts), query=True)


def qvecs(ctx, cfg: str, split: str) -> tuple[list[dict], np.ndarray]:
    qs = queries(ctx, cfg, split)
    if not qs:
        return qs, np.zeros((0, 1), dtype=np.float32)
    from . import embed
    embed.init(ctx)
    return qs, _qvecs_cached(tuple(q["q"] for q in qs))


# ---------------------------------------------------------------- rerank (local cross-encoder over the full cards)
class _FakeCE:
    name = "fake-token-overlap"

    def predict(self, pairs, batch_size=32):
        out = []
        for a, b in pairs:
            ta, tb = set(T.content_tokens(a)), set(T.content_tokens(b))
            out.append(len(ta & tb) / (1 + len(ta)))
        return np.array(out, dtype=np.float32)


def _cross_encoder():
    if fake():
        return _FakeCE(), {"model": "fake", "revision": None}
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ce = CrossEncoder(RERANKER, max_length=512, device="cpu")
    rev = getattr(getattr(ce.model, "config", None), "_commit_hash", None)
    return ce, {"model": RERANKER, "revision": rev}


def run_rerank(ctx) -> dict:
    from . import embed
    ctx.register_diagnostics(_diag("rerank"))
    embed.init(ctx)
    ce, info = _cross_encoder()
    work = []
    for c in CONFIGS:
        for sp in ("test", "dev"):
            qs, qv = qvecs(ctx, c, sp)
            ci = card_index(corpus_of(c), "full")
            for i, q in enumerate(qs):
                cand = ci.search(q["q"], qv[i], N_RERANK)
                work.append((c, sp, q, cand))
    n_pairs = sum(len(w[3]) for w in work)
    k_use = N_RERANK
    t0 = time.time()
    probe = [(w[2]["q"], card_index(corpus_of(w[0]), "full").texts[card_index(corpus_of(w[0]), "full").ids.index(d)])
             for w in work[:8] for d, _ in w[3]]
    if probe:
        ce.predict(probe[:256], batch_size=32)
    rate = min(256, len(probe)) / max(1e-6, time.time() - t0) if probe else 1e9
    proj_h = n_pairs / max(rate, 1e-6) / 3600
    if proj_h > RERANK_MAX_HOURS:
        k_use = N_RERANK_FALLBACK
        warn(ctx, "rerank", f"cross-encoder at {rate:.1f} pairs/s projects {proj_h:.1f} h for top-{N_RERANK}; "
                            f"pre-registered fallback: rerank the top {N_RERANK_FALLBACK}")
    out_p = RUNS / ctx.mode / "rerank.jsonl"
    with open(out_p, "w") as fo:
        for n, (c, sp, q, cand) in enumerate(work):
            ci = card_index(corpus_of(c), "full")
            pos = {d: j for j, d in enumerate(ci.ids)}
            cand = cand[:k_use]
            pairs = [(q["q"], ci.texts[pos[d]]) for d, _ in cand]
            sc = ce.predict(pairs, batch_size=32) if pairs else np.zeros(0)
            order = sorted(range(len(cand)), key=lambda j: (-float(sc[j]), j))
            fo.write(json.dumps({"config": c, "split": sp, "qid": q["qid"],
                                 "ranked": [[cand[j][0], round(float(sc[j]), 5)] for j in order]}) + "\n")
            if n % 100 == 0:
                ctx.hb.touch(rerank=f"{n}/{len(work)}")
    write_json(RUNS / ctx.mode / "diag_rerank.json", {**info, "pairs_per_s": rate, "k": k_use, "queries": len(work)})
    return {"queries": len(work), "k": k_use, **info}


def _check_rerank(ctx, m):
    return [] if m["queries"] else ["no query reranked"]


def _load_rerank(mode: str) -> dict:
    p = RUNS / mode / "rerank.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[(r["config"], r["split"], r["qid"])] = r["ranked"]
    return out


# ---------------------------------------------------------------- resolve (Haiku over the top-10 cards)
def build_resolve(ctx) -> list[dict]:
    cells = []
    for c in CONFIGS:
        for sp in ("test", "dev"):
            qs, qv = qvecs(ctx, c, sp)
            for v in CARD_VARIANTS:
                ci = card_index(corpus_of(c), v) if qs else None
                pos = {d: j for j, d in enumerate(ci.ids)} if ci else {}
                for i, q in enumerate(qs):
                    cand = ci.search(q["q"], qv[i], N_CAND)
                    cands = [(d, ci.texts[pos[d]][:700]) for d, _ in cand]
                    cells.append({"cell_id": f"resolve|{v}|{c}|{sp}|{q['qid']}", "model": HAIKU, "arm": f"resolve_{v}",
                                  "prefix": P.RESOLVER_PREFIX,
                                  "item": P.resolver_item(S.SOURCE_DESCRIPTION[corpus_of(c)], q["q"],
                                                          [t for _, t in cands]),
                                  "max_tokens": MAX_TOKENS["resolve"][HAIKU],
                                  "meta": {"source": source_of(c), "config": c, "gen_model": "-", "qid": q["qid"],
                                           "split": sp, "variant": v, "cands": [d for d, _ in cands]}})
    return cells


# ---------------------------------------------------------------- calibrate (local)
def pav(xs, ys):
    """Isotonic (non-decreasing) regression by pool-adjacent-violators; returns (sorted x, fitted y)."""
    order = np.argsort(np.asarray(xs, dtype=float), kind="stable")
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    blocks = [[y[i], 1.0, x[i], x[i]] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            a = out.pop()
            out[-1][0] += a[0]
            out[-1][1] += a[1]
            out[-1][3] = a[3]
    fx, fy = [], []
    for s, n, lo, hi in out:
        fx += [lo, hi]
        fy += [s / n, s / n]
    return np.array(fx), np.array(fy)


def iso_predict(model, x: float) -> float:
    fx, fy = model
    if len(fx) == 0:
        return 0.0
    return float(np.interp(x, fx, fy))


GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))


def lowest_threshold(scores, labels, precision: float) -> float:
    s, l = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    for t in GRID:
        m = s >= t
        if m.sum() >= 5 and l[m].mean() >= precision:
            return t
    return 1.01


def resolver_choice(parsed: dict | None, cands: list[str]) -> list[tuple[str, float]]:
    """[(doc_id, self-confidence)] in the resolver's order, capped at MAX_RESOLVE, unknown ids dropped."""
    out = []
    for d in (parsed or {}).get("documents") or []:
        k = d["id"] - 1
        if 0 <= k < len(cands) and cands[k] not in [x for x, _ in out]:
            out.append((cands[k], float(d["confidence"])))
    return out[:MAX_RESOLVE]


def run_calibrate(ctx) -> dict:
    """LOSO isotonic calibration of resolver confidence and the router / pdoc thresholds, on dev queries only."""
    from . import index as X
    ctx.register_diagnostics(_diag("calibrate"))
    res_rows = _rows_by(ctx, "resolve")
    route_rows = _rows_by(ctx, "route")
    data: dict = {"resolve": {}, "route": [], "pdoc": []}
    for c in CONFIGS:
        qs, qv = qvecs(ctx, c, "dev")
        ci = card_index(corpus_of(c), "full") if qs else None
        for i, q in enumerate(qs):
            gold = set(q["gold_cfg"])
            for v in CARD_VARIANTS:
                r = res_rows.get(f"resolve|{v}|{c}|dev|{q['qid']}")
                if not r or r["status"] != "ok":
                    continue
                for d, conf in resolver_choice(r["parsed"], r["meta"]["cands"]):
                    data["resolve"].setdefault(v, []).append((source_of(c), conf, int(d in gold)))
            rr = route_rows.get(f"route2|{c}|dev|{q['qid']}")
            if rr and rr["status"] == "ok":
                p = rr["parsed"]
                resolved = X.resolve(base_cfg(corpus_of(c)), p.get("entity"), p.get("qualifier"))
                if resolved and len(resolved) <= MAX_RESOLVE:
                    data["route"].append((source_of(c), float(p["confidence"]), int(bool(set(resolved) & gold))))
            top = ci.search(q["q"], qv[i], 2)
            if top:
                margin = top[0][1] - (top[1][1] if len(top) > 1 else 0.0)
                data["pdoc"].append((source_of(c), margin * 61, int(top[0][0] in gold)))
    sources = sorted({source_of(c) for c in CONFIGS})
    cal: dict = {}
    for s in sources + ["_all"]:
        e: dict = {}
        for v in CARD_VARIANTS:
            tr = [(conf, y) for src, conf, y in data["resolve"].get(v, []) if src != s]
            model = pav([a for a, _ in tr], [b for _, b in tr]) if tr else (np.array([]), np.array([]))
            cal_scores = [iso_predict(model, a) for a, _ in tr]
            raw_scores = [a for a, _ in tr]
            labels = [b for _, b in tr]
            e[v] = {"iso": [model[0].tolist(), model[1].tolist()],
                    "tau_high": lowest_threshold(cal_scores, labels, 0.9),
                    "tau_low": lowest_threshold(cal_scores, labels, 0.6), "n_dev": len(tr),
                    "tau_high_raw_scale": lowest_threshold(raw_scores, labels, 0.9)}
            if fake() and not tr:          # dry run on fixtures (no dev pool): exercise every scoping branch
                e[v].update(iso=[[0.0, 1.0], [0.0, 1.0]], tau_high=0.7, tau_low=0.4)
        rt = [(conf, y) for src, conf, y in data["route"] if src != s]
        e["route_tau1"] = lowest_threshold([a for a, _ in rt], [b for _, b in rt], 0.9)
        if fake() and not rt:
            e["route_tau1"] = 0.5
        pdx = [(m, y) for src, m, y in data["pdoc"] if src != s]
        e["pdoc_margin"] = (lowest_threshold([min(0.95, a) for a, _ in pdx], [b for _, b in pdx], 0.9)
                            if pdx else 1.01)
        cal[s] = e
    write_json(RUNS / ctx.mode / "calibration.json", cal)
    return {"sources": sources, "n_dev_resolve": {v: len(x) for v, x in data["resolve"].items()},
            "n_dev_route": len(data["route"])}


def _check_calibrate(ctx, m):
    if not m["n_dev_resolve"]:
        warn(ctx, "calibrate", "no dev resolver output; R uses uncalibrated defaults")
    return []


def _cov_calibrate(ctx, mode):
    return [{"source": s, "config": f"{s}.calibrate", "gen_model": "-"} for s in sorted({source_of(c) for c in CONFIGS})]


def load_calibration(mode: str) -> dict:
    p = RUNS / mode / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- retrieve (all arms)
def arms_for(cfg: str) -> tuple:
    return tuple(a for a in ARMS if not (a == "sac" and corpus_of(cfg) == "finmy"))


def _ctx_dir(mode: str) -> Path:
    return RUNS / mode / "contexts2"


def r_decision(arm: str, src: str, cal: dict, router: dict | None, res: list[tuple[str, float]]) -> dict:
    """ROUTEWISE-R scoping decision for one query: {"mode": hard|soft|fallback, "docs": [...], "p": [...],
    "path": router|resolver|none}. R-uncal uses raw self-confidence against the same calibrated thresholds."""
    variant = R_ARMS[arm]
    e = cal.get(src) or cal.get("_all") or {}
    ev = e.get(variant) or {}
    tau_h, tau_l = ev.get("tau_high", 0.9), ev.get("tau_low", 0.6)
    tau1 = e.get("route_tau1", 0.9)
    if router and router["docs"] and len(router["docs"]) <= MAX_RESOLVE and router["confidence"] >= tau1:
        return {"mode": "hard", "docs": list(router["docs"]), "p": [router["confidence"]] * len(router["docs"]),
                "path": "router"}
    iso = ev.get("iso") or [[], []]
    model = (np.array(iso[0]), np.array(iso[1]))
    scored = []
    for d, conf in res:
        p = conf if arm == "R-uncal" or not len(model[0]) else iso_predict(model, conf)
        scored.append((d, p))
    keep = [(d, p) for d, p in scored if p >= tau_l][:MAX_RESOLVE]
    if not keep:
        return {"mode": "fallback", "docs": [], "p": [], "path": "resolver" if res else "none"}
    pmax = max(p for _, p in keep)
    return {"mode": "hard" if pmax >= tau_h else "soft", "docs": [d for d, _ in keep], "p": [p for _, p in keep],
            "path": "resolver"}


def run_retrieve(ctx) -> dict:
    from . import embed
    from . import index as X
    ctx.register_diagnostics(_diag("retrieve2"))
    embed.init(ctx)
    route_rows = _rows_by(ctx, "route")
    res_rows = _rows_by(ctx, "resolve")
    rer = _load_rerank(ctx.mode)
    cal = load_calibration(ctx.mode)
    out_p = RUNS / ctx.mode / "retrieve2.jsonl"
    cdir = _ctx_dir(ctx.mode)
    cdir.mkdir(parents=True, exist_ok=True)
    n = 0
    per_arm = {a: 0 for a in ARMS}
    with open(out_p, "w") as fo:
        for c in CONFIGS:
            corpus, src = corpus_of(c), source_of(c)
            cf = S.config(base_cfg(corpus))
            ix = X.get(corpus)
            qs, qv = qvecs(ctx, c, "test")
            ci = card_index(corpus, "full") if qs else None
            rec = []
            for i, q in enumerate(qs):
                gold = set(q["gold_cfg"])
                rr = route_rows.get(f"route2|{c}|test|{q['qid']}")
                router = None
                if rr and rr["status"] == "ok":
                    p = rr["parsed"]
                    router = {"docs": X.resolve(base_cfg(corpus), p.get("entity"), p.get("qualifier")),
                              "confidence": float(p["confidence"])}
                top_cards = ci.search(q["q"], qv[i], 2)
                e = cal.get(src) or cal.get("_all") or {}
                margin = (top_cards[0][1] - (top_cards[1][1] if len(top_cards) > 1 else 0.0)) * 61 if top_cards else 0
                ranked = rer.get((c, "test", q["qid"])) or []
                for arm in arms_for(c):
                    info: dict = {}
                    if arm == "flat":
                        idx = hybrid(corpus, q["q"], qv[i], "plain")
                    elif arm == "meta":
                        idx = hybrid(corpus, q["q"], qv[i], "meta")
                    elif arm == "oracle":
                        idx = hybrid(corpus, q["q"], qv[i], "plain", gold)
                    elif arm == "hdrr":
                        docs_r = set(router["docs"]) if router else set()
                        idx = hybrid(corpus, q["q"], qv[i], "plain", docs_r) if docs_r else \
                            hybrid(corpus, q["q"], qv[i], "plain")
                        info = {"top1": (router["docs"][0] if router and router["docs"] else None),
                                "scoped": bool(docs_r)}
                    elif arm == "sac":
                        idx = hybrid(corpus, q["q"], qv[i], "sac")
                    elif arm == "pdoc":
                        t1 = top_cards[0][0] if top_cards else None
                        use = t1 is not None and min(0.95, margin) >= e.get("pdoc_margin", 1.01)
                        idx = hybrid(corpus, q["q"], qv[i], "plain", {t1}) if use else hybrid(corpus, q["q"], qv[i], "meta")
                        info = {"top1": t1, "scoped": use}
                    elif arm == "pdoc-rr":
                        top5 = [d for d, _ in ranked[:PDOC_RR_DOCS]]
                        idx = hybrid(corpus, q["q"], qv[i], "plain", set(top5)) if top5 else \
                            hybrid(corpus, q["q"], qv[i], "meta")
                        info = {"top1": top5[0] if top5 else None, "scoped": bool(top5), "docs": top5}
                    else:
                        v = R_ARMS[arm]
                        r = res_rows.get(f"resolve|{v}|{c}|test|{q['qid']}")
                        res = resolver_choice(r["parsed"], r["meta"]["cands"]) if r and r["status"] == "ok" else []
                        dec = r_decision(arm, src, cal, router, res)
                        if dec["mode"] == "hard":
                            idx = hybrid(corpus, q["q"], qv[i], "plain", set(dec["docs"]))
                        elif dec["mode"] == "soft":
                            idx = hybrid(corpus, q["q"], qv[i], "meta", None, set(dec["docs"]))
                        else:
                            idx = hybrid(corpus, q["q"], qv[i], "meta")
                        info = {"top1": dec["docs"][0] if dec["docs"] else None, "decision": dec["mode"],
                                "path": dec["path"], "docs": dec["docs"], "p": dec["p"],
                                "raw_top": res[0][1] if res else None}
                    docs = [cf["doc_map"][ix.chunks[j]["doc"]] for j in idx]
                    texts = [ix.chunks[j]["text"] for j in idx]
                    gtexts = [t for t, d in zip(texts, docs) if d in gold]
                    row = {"config": c, "qid": q["qid"], "arm": arm, "chunks": idx, "docs": docs,
                           "doc_hit": int(bool(set(docs) & gold)),
                           "ev_any": round(T.evidence_recall(q["evidence"], texts), 4),
                           "ev_gold": round(T.evidence_recall(q["evidence"], gtexts), 4),
                           "top1_correct": (int(info.get("top1") in gold) if "top1" in info else None), **info}
                    fo.write(json.dumps(row) + "\n")
                    rec.append({"qid": q["qid"], "arm": arm, "passages": [[cf["titles"][d], t] for d, t in zip(docs, texts)]})
                    n += 1
                    per_arm[arm] += 1
                ctx.hb.touch(retrieve2={"config": c, "done": i + 1, "total": len(qs)})
            tmp = cdir / f"{c}.jsonl.gz.tmp"
            with gzip.open(tmp, "wt") as g:
                for r in rec:
                    g.write(json.dumps(r) + "\n")
            tmp.replace(cdir / f"{c}.jsonl.gz")
    return {"rows": n, "per_arm": per_arm}


def _check_retrieve(ctx, m):
    return [] if m["rows"] else ["retrieve wrote no row"]


def _cov_retrieve(ctx, mode):
    return [{"source": source_of(c), "config": c, "gen_model": "-", "arm": a} for c in CONFIGS for a in arms_for(c)]


def _load_contexts(mode: str) -> dict | None:
    d = _ctx_dir(mode)
    if not d.exists():
        return None
    out = {}
    for p in sorted(d.glob("*.jsonl.gz")):
        c = p.name[: -len(".jsonl.gz")]
        with gzip.open(p, "rt") as g:
            for line in g:
                r = json.loads(line)
                out[(c, r["qid"], r["arm"])] = [tuple(x) for x in r["passages"]]
    return out


# ---------------------------------------------------------------- generate / judge / validity
def gen_id(c, qid, arm) -> str:
    return f"gen2|{c}|{qid}|{arm}"


def build_generate(ctx) -> list[dict]:
    ctxs = _load_contexts(ctx.mode)
    cells = []
    for c in CONFIGS:
        for q in queries(ctx, c, "test"):
            for arm in arms_for(c):
                passages = [] if ctxs is None else ctxs.get((c, q["qid"], arm), [])
                cells.append({"cell_id": gen_id(c, q["qid"], arm), "model": HAIKU, "arm": arm, "prefix": P.GEN_PREFIX,
                              "item": P.gen_item(q["q"], passages), "max_tokens": MAX_TOKENS["generate"][HAIKU],
                              "meta": {"source": source_of(c), "config": c, "gen_model": "-", "qid": q["qid"]}})
    return cells


def panel_subset(ctx) -> set:
    """Stratified judge-panel subset: an equal share of answers per (config, arm), chosen by a stable hash."""
    keys = [(c, q["qid"], a) for c in CONFIGS for q in queries(ctx, c, "test") for a in arms_for(c)]
    if ctx.mode != "full":
        return {gen_id(c, qid, a) for c, qid, a in keys}
    strata: dict = {}
    for c, qid, a in keys:
        strata.setdefault((c, a), []).append(qid)
    per = max(1, PANEL_N // max(1, len(strata)))
    out = set()
    for (c, a), qids in strata.items():
        for qid in sorted(qids, key=lambda x: T.stable_hash(f"panel|{c}|{a}|{x}"))[:per]:
            out.add(gen_id(c, qid, a))
    return out


def build_judge(ctx) -> list[dict]:
    gen = _rows_by(ctx, "generate")
    have = bool(gen)
    panel = panel_subset(ctx)
    cells = []
    for c in CONFIGS:
        for q in queries(ctx, c, "test"):
            for arm in arms_for(c):
                gid = gen_id(c, q["qid"], arm)
                if have:
                    r = gen.get(gid)
                    if not r or r["status"] != "ok":
                        continue
                    ans = r["parsed"]["answer"]
                else:
                    ans = ""
                for jm in [HAIKU] + ([OSS, SONNET] if gid in panel else []):
                    cells.append({"cell_id": f"judge2|{gid}|{jm}", "model": jm, "arm": arm, "prefix": P.JUDGE_PREFIX,
                                  "item": P.judge_item(q["orig"], q["answer"], q["evidence"], ans),
                                  "max_tokens": MAX_TOKENS["judge"][jm],
                                  "meta": {"source": source_of(c), "config": c, "gen_model": HAIKU, "qid": q["qid"],
                                           "gen_cell": gid}})
    return cells


def build_validity(ctx) -> list[dict]:
    cells = []
    for c in CONFIGS:
        if level_of(c) == "L0" or corpus_of(c) == "finmy":
            continue
        corpus = corpus_of(c)
        titles = S.config(base_cfg(corpus))["titles"]
        for q in queries(ctx, c, "test"):
            desc = []
            for g in q["gold_cfg"]:
                card = card_for(corpus, g) or {}
                desc.append(f"{titles.get(g, g)}: {(card.get('summary') or '')[:500]}")
            for m in (SONNET, OSS):
                cells.append({"cell_id": f"valid|{c}|{q['qid']}|{m}", "model": m, "arm": "validity",
                              "prefix": P.VALIDITY_PREFIX,
                              "item": P.validity_item(q["orig"], q["q"], q["answer"], desc),
                              "max_tokens": MAX_TOKENS["validity"][m],
                              "meta": {"source": source_of(c), "config": c, "gen_model": "-", "qid": q["qid"]}})
    return cells


# =========================================================================== finmy index (fin chunks reused)
def build_finmy_index(hb=None):
    """fin-multiyear index: the fin chunks and their embeddings are reused unchanged (same chunker, same
    embedder); only the decoy filings are chunked and embedded. BM25 and the title variant are rebuilt over the
    whole collection, with fiscal-year titles."""
    import pickle
    from scipy import sparse
    from . import embed
    from . import index as X
    fin = X.get("fin")
    d = X._dir("finmy")
    done_p = d / "complete.json"
    if done_p.exists():
        with gzip.open(d / "chunks.pkl.gz", "rb") as f:
            chunks, vocab = pickle.load(f)
        emb = np.load(d / "emb_plain.npy", mmap_mode="r")
        bm = {v: sparse.load_npz(d / f"bm25_{v}.npz").tocsc() for v in ("plain", "base")}
        titles = {"base": (np.load(d / "title_ids_base.npy"), np.load(d / "title_emb_base.npy"))}
        return X.Index("finmy", chunks, vocab, bm, emb, titles)
    data = S.load("finmy")
    fin_ids = {doc["doc_id"] for doc in S.load("fin")["docs"]}
    new = []
    for doc in data["docs"]:
        if doc["doc_id"] in fin_ids:
            continue
        for i, piece in enumerate(T.chunk(doc["text"], X.CHUNK_SIZE, X.CHUNK_OVERLAP)):
            new.append({"doc": doc["doc_id"], "i": i, "text": piece})

    def prog(a, b, label):
        if hb:
            hb.touch(index={"source": "finmy", "step": label, "done": int(a), "total": int(b)})
    e = embed.embedder()
    emb_new = e.encode([c["text"] for c in new], progress=lambda a, b: prog(a, b, "embed_decoys"), cache_dir=d,
                       label="decoys") if new else np.zeros((0, fin.emb.shape[1]), dtype=np.float32)
    chunks = list(fin.chunks) + new
    emb = np.concatenate([np.asarray(fin.emb, dtype=np.float32), emb_new.astype(np.float32)], axis=0)
    np.save(d / "emb_plain.npy", emb)
    tmap = {doc["doc_id"]: doc["title"] for doc in data["docs"]}
    uniq = sorted(set(tmap.values()))
    pos = {t: i for i, t in enumerate(uniq)}
    tid = np.array([pos[tmap[c["doc"]]] for c in chunks], dtype=np.int64)
    temb = e.encode(uniq, cache_dir=d, label="title_base")
    np.save(d / "title_ids_base.npy", tid)
    np.save(d / "title_emb_base.npy", temb.astype(np.float32))
    bm = {}
    bm["plain"], vocab = X._bm25_matrix([c["text"] for c in chunks], progress=lambda a, b: prog(a, b, "bm25_plain"))
    texts = [f"{tmap.get(c['doc'], '')}\n{c['text']}" for c in chunks]
    bm["base"], vocab = X._bm25_matrix(texts, X._extend(vocab, texts), progress=lambda a, b: prog(a, b, "bm25_base"))
    for v, m in bm.items():
        sparse.save_npz(d / f"bm25_{v}.npz", m.tocsr())
    tmp = (d / "chunks.pkl.gz").with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump((chunks, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, d / "chunks.pkl.gz")
    done_p.write_text(json.dumps({"chunks": len(chunks), "decoy_chunks": len(new), "embedder": e.name}))
    return X.Index("finmy", chunks, vocab, {k: v.tocsc() for k, v in bm.items()},
                   np.load(d / "emb_plain.npy", mmap_mode="r"), {"base": (tid, temb)})


def _register_finmy_builder():
    from . import index as X
    X.BUILDERS["finmy"] = build_finmy_index


_register_finmy_builder()
