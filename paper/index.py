"""Per-source chunk index (dense, twice: plain and title-prefixed; BM25 likewise), cached on disk by the
content of the parsed source, plus the corpus-only features of every configuration and the hybrid
retriever every arm uses."""
from __future__ import annotations

import gzip
import json
import math
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

from . import embed
from . import sources as S
from . import text as T

CHUNK_SIZE, CHUNK_OVERLAP = 1200, 200
TOP_K = 8
POOL = 100
RRF_K = 60
K1, B = 1.2, 0.75
INDEX_VERSION = "2"


def _dir(source: str) -> Path:
    base = Path(os.environ.get("RH_DATA_DIR", str(S._cache_dir().parent.parent))) / "routewise" / "index"
    key = T.stable_hash("|".join(S._pin_sha(p) for p in S.PINS_OF[source]) + embed.embedder().name
                        + f"|{CHUNK_SIZE}|{CHUNK_OVERLAP}")[:12]
    d = base / f"{source}-v{INDEX_VERSION}-{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


TITLE_WEIGHT = 0.5      # meta arm, dense part: cos(q, chunk) + TITLE_WEIGHT * cos(q, document title)


class Index:
    def __init__(self, source: str, chunks: list[dict], vocab: dict, bm25: dict, emb, titles: dict):
        self.source, self.chunks, self.vocab = source, chunks, vocab
        self.bm25 = bm25                    # keyed by variant: "plain", "base" (title words added), "merge"
        self.emb = emb                      # plain chunk embeddings, one pass
        self.titles = titles                # variant -> (title ids array per chunk, title embedding matrix)
        self.doc_of = np.array([c["doc"] for c in chunks], dtype=object)


def _bm25_matrix(texts: list[str], vocab: dict | None = None, progress=None):
    rows, cols, vals = [], [], []
    build = vocab is None
    vocab = {} if build else vocab
    tfs, lens = [], []
    for k, t in enumerate(texts):
        tf = {}
        for w in T.content_tokens(t):
            if build and w not in vocab:
                vocab[w] = len(vocab)
            j = vocab.get(w)
            if j is not None:
                tf[j] = tf.get(j, 0) + 1
        tfs.append(tf)
        lens.append(sum(tf.values()))
        if progress and k % 5000 == 0:
            progress(k, len(texts))
    n = len(texts)
    avg = (sum(lens) / n) if n else 1.0
    df = np.zeros(len(vocab), dtype=np.float64)
    for tf in tfs:
        for j in tf:
            df[j] += 1
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    for i, tf in enumerate(tfs):
        norm = K1 * (1 - B + B * lens[i] / max(avg, 1e-9))
        for j, f in tf.items():
            rows.append(i)
            cols.append(j)
            vals.append(idf[j] * f * (K1 + 1) / (f + norm))
    m = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, len(vocab)))
    return m.tocsc(), vocab


def _variants(source: str) -> list[str]:
    return ["base"] + (["merge"] if f"{source}.merge" in S.config_names() else [])


def _variant_titles(source: str, variant: str) -> dict:
    data = S.load(source)
    if variant == "base":
        return {doc["doc_id"]: doc["title"] for doc in data["docs"]}
    c = S.config(f"{source}.merge")
    return {od: c["titles"][cd] for od, cd in c["doc_map"].items()}


BUILDERS: dict = {}     # v2: derived collections with their own builder (finmy reuses the fin chunk embeddings)


def build(source: str, hb=None) -> Index:
    if source in BUILDERS:
        return BUILDERS[source](hb)
    d = _dir(source)
    data = S.load(source)
    variants = _variants(source)
    done_p = d / "complete.json"

    def prog(done, total, label):
        if hb:
            hb.touch(index={"source": source, "step": label, "done": int(done), "total": int(total)})
    if done_p.exists():
        with gzip.open(d / "chunks.pkl.gz", "rb") as f:
            chunks, vocab = pickle.load(f)
        emb = np.load(d / "emb_plain.npy", mmap_mode="r")
        bm = {v: sparse.load_npz(d / f"bm25_{v}.npz").tocsc() for v in ["plain"] + variants}
        titles = {v: (np.load(d / f"title_ids_{v}.npy"), np.load(d / f"title_emb_{v}.npy")) for v in variants}
        return Index(source, chunks, vocab, bm, emb, titles)
    chunks = []
    for doc in data["docs"]:
        for i, piece in enumerate(T.chunk(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append({"doc": doc["doc_id"], "i": i, "text": piece})
    if not chunks:
        raise ValueError(f"{source}: no chunks")
    prog(0, len(chunks), "chunked")
    e = embed.embedder()
    emb = e.encode([c["text"] for c in chunks], progress=lambda a, b: prog(a, b, "embed_chunks"),
                   cache_dir=d, label="plain")
    np.save(d / "emb_plain.npy", emb.astype(np.float32))
    titles = {}
    for v in variants:
        tmap = _variant_titles(source, v)
        uniq = sorted(set(tmap.values()))
        pos = {t: i for i, t in enumerate(uniq)}
        tid = np.array([pos[tmap[c["doc"]]] for c in chunks], dtype=np.int64)
        temb = e.encode(uniq, progress=lambda a, b, v=v: prog(a, b, f"embed_titles_{v}"), cache_dir=d,
                        label=f"title_{v}")
        np.save(d / f"title_ids_{v}.npy", tid)
        np.save(d / f"title_emb_{v}.npy", temb.astype(np.float32))
        titles[v] = (tid, temb)
    bm, vocab = {}, None
    bm["plain"], vocab = _bm25_matrix([c["text"] for c in chunks], progress=lambda a, b: prog(a, b, "bm25_plain"))
    for v in variants:
        tmap = _variant_titles(source, v)
        texts = [f"{tmap.get(c['doc'], '')}\n{c['text']}" for c in chunks]
        bm[v], vocab = _bm25_matrix(texts, _extend(vocab, texts), progress=lambda a, b, v=v: prog(a, b, f"bm25_{v}"))
    for v, m in bm.items():
        sparse.save_npz(d / f"bm25_{v}.npz", m.tocsr())
    tmp = (d / "chunks.pkl.gz").with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump((chunks, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, d / "chunks.pkl.gz")
    done_p.write_text(json.dumps({"chunks": len(chunks), "dim": int(emb.shape[1]), "embedder": e.name}))
    return Index(source, chunks, vocab, bm, np.load(d / "emb_plain.npy", mmap_mode="r"), titles)


def _extend(vocab: dict, texts: list[str]) -> dict:
    """The title-prefixed matrices share the plain vocabulary, extended with title words."""
    v = dict(vocab)
    for t in texts:
        for w in T.content_tokens(t.split("\n", 1)[0]):
            if w not in v:
                v[w] = len(v)
    return v


@lru_cache(maxsize=None)
def get(source: str) -> Index:
    return build(source)


# --------------------------------------------------------------------------- retrieval
@lru_cache(maxsize=None)
def cfg_chunks(config: str):
    """(chunk indices allowed in the config, array of config doc id per allowed chunk)."""
    c = S.config(config)
    ix = get(c["source"])
    keep, docs = [], []
    for i, ch in enumerate(ix.chunks):
        cd = c["doc_map"].get(ch["doc"])
        if cd is not None:
            keep.append(i)
            docs.append(cd)
    return np.array(keep, dtype=np.int64), np.array(docs, dtype=object)


@lru_cache(maxsize=None)
def _cfg_codes(config: str):
    rows, docs = cfg_chunks(config)
    uniq, codes = np.unique(docs.astype(str), return_inverse=True)
    return uniq, codes


def _bm25_scores(ix: Index, variant: str, query: str, rows: np.ndarray) -> np.ndarray:
    m = ix.bm25[variant]
    vocab_n = m.shape[1]
    cols = sorted({ix.vocab[w] for w in T.content_tokens(query) if w in ix.vocab and ix.vocab[w] < vocab_n})
    if not cols:
        return np.zeros(len(rows), dtype=np.float32)
    s = np.asarray(m[:, cols].sum(axis=1)).ravel()
    return s[rows]


def retrieve(config: str, query: str, qvec: np.ndarray, variant: str, allowed_docs: set | None) -> list[int]:
    """Hybrid BM25 + dense with reciprocal-rank fusion; returns global chunk indices (best first)."""
    c = S.config(config)
    ix = get(c["source"])
    rows, docs = cfg_chunks(config)
    if allowed_docs is not None:
        uniq, codes = _cfg_codes(config)
        ok = np.isin(uniq, np.array(sorted(allowed_docs), dtype=str))
        rows = rows[ok[codes]]
    if len(rows) == 0:
        return []
    qv = np.asarray(qvec, dtype=np.float32)
    dense_all = np.asarray(ix.emb @ qv, dtype=np.float32)
    dense = dense_all[rows]
    if variant == "plain":
        lex = _bm25_scores(ix, "plain", query, rows)
    else:
        tv = c["title_variant"]
        tid, temb = ix.titles[tv]
        dense = dense + TITLE_WEIGHT * (temb @ qv)[tid[rows]]
        lex = _bm25_scores(ix, tv, query, rows)
    k = min(POOL, len(rows))
    fused = {}
    for scores, is_lex in ((dense, False), (lex, True)):
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        for r, j in enumerate(top):
            if is_lex and scores[j] <= 0:
                continue
            fused[int(rows[j])] = fused.get(int(rows[j]), 0.0) + 1.0 / (RRF_K + r + 1)
    best = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
    return [i for i, _ in best]


# --------------------------------------------------------------------------- corpus features
def features(config: str, sample: int = 1500, sil_sample: int = 800) -> dict:
    """Corpus-only features of a configuration (no queries involved)."""
    c = S.config(config)
    ix = get(c["source"])
    rows, docs = cfg_chunks(config)
    _, codes = _cfg_codes(config)
    E = np.asarray(ix.emb[rows], dtype=np.float32)
    order = sorted(range(len(rows)), key=lambda i: T.stable_hash(f"feat|{config}|{i}"))
    pick = np.array(order[:min(sample, len(rows))], dtype=np.int64)
    other_best, own_best, gaps = [], [], []
    for start in range(0, len(pick), 256):
        p = pick[start:start + 256]
        sims = E[p] @ E.T
        for a, i in enumerate(p):
            same = codes == codes[i]
            s = sims[a]
            o = s[~same]
            ob = float(o.max()) if o.size else float("nan")
            own = s[same].copy()
            own_idx = np.where(same)[0]
            own = own[own_idx != i]
            wb = float(own.max()) if own.size else float("nan")
            other_best.append(ob)
            own_best.append(wb)
            if not math.isnan(ob) and not math.isnan(wb):
                gaps.append(ob - wb)
    ob = np.array(other_best, dtype=np.float64)
    # silhouette (cosine distance) on a smaller sample, docs with >= 2 sampled chunks only
    sp = pick[:min(sil_sample, len(pick))]
    labs = docs[sp]
    uniq, inv = np.unique(labs.astype(str), return_inverse=True)
    D = 1.0 - (E[sp] @ E[sp].T)
    sil = []
    counts = np.bincount(inv, minlength=len(uniq))
    if len(uniq) >= 2:
        onehot = np.zeros((len(sp), len(uniq)), dtype=np.float64)
        onehot[np.arange(len(sp)), inv] = 1.0
        sums = D @ onehot
        for a in range(len(sp)):
            ca = inv[a]
            if counts[ca] < 2:
                continue
            ai = sums[a, ca] / (counts[ca] - 1)
            others = [sums[a, b] / counts[b] for b in range(len(uniq)) if b != ca and counts[b] > 0]
            bi = min(others)
            sil.append((bi - ai) / max(ai, bi, 1e-9))
    prim = [c["aliases"][d][0] if c["aliases"][d] else "" for d in c["titles"]]
    from collections import Counter
    cnt = Counter(T.norm(a) for a in prim if a)
    uniq_keys = sum(1 for a in prim if a and cnt[T.norm(a)] == 1) / max(1, len(prim))
    n_docs = len(c["titles"])
    return {"config": config, "n_docs": n_docs, "log_docs": round(math.log(max(n_docs, 1)), 4),
            "n_chunks": int(len(rows)), "chunks_per_doc": round(len(rows) / max(n_docs, 1), 2),
            "confusion": round(float(np.mean(gaps)) if gaps else 0.0, 4),
            "other_doc_sim": round(float(np.nanmean(ob)) if ob.size else 0.0, 4),
            "near_dup_rate": round(float(np.nanmean(ob >= 0.95)) if ob.size else 0.0, 4),
            "separability": round(float(np.mean(sil)) if sil else 0.0, 4),
            "key_uniqueness": round(uniq_keys, 4), "feature_sample": int(len(pick))}


def workload(config: str) -> dict:
    """Query-workload features, measured without any LLM (rule-based alias matching)."""
    c = S.config(config)
    hits = [rule_match(config, q["q"]) for q in c["queries"]]
    n = max(1, len(hits))
    mention = sum(1 for h in hits if h) / n
    amb = sum(1 for h in hits if len(h) > 1) / max(1, sum(1 for h in hits if h))
    correct = sum(1 for h, q in zip(hits, c["queries"]) if h and set(h) & set(q["gold_cfg"])) / n
    return {"config": config, "n_queries": len(c["queries"]), "mention_rate": round(mention, 4),
            "ambiguity": round(amb, 4), "rule_route_accuracy": round(correct, 4),
            "query_tokens": round(sum(len(T.tokens(q["q"])) for q in c["queries"]) / n, 2)}


@lru_cache(maxsize=None)
def _alias_table(config: str):
    c = S.config(config)
    tab = {}
    for d, al in c["aliases"].items():
        for a in al:
            if a:
                tab.setdefault(a, set()).add(d)
    return tab


def rule_match(config: str, query: str, cap: int = 20) -> list[str]:
    hits = set()
    for a, ds in _alias_table(config).items():
        if T.mentions(query, a):
            hits |= ds
    return sorted(hits)[:cap] if len(hits) <= cap else sorted(hits)[:cap]


def resolve(config: str, entity: str | None, qualifier: str | None, thr: float = 0.75, cap: int = 20) -> list[str]:
    """Docs of the configuration matched by the router's extracted entity (fuzzy alias match)."""
    if not entity or not str(entity).strip():
        return []
    c = S.config(config)
    best, out = 0.0, {}
    for a, ds in _alias_table(config).items():
        form, cs = T.alias_forms(a)
        if cs:
            sc = 1.0 if str(entity).strip().upper() == form else 0.0
        else:
            sc = T.fuzzy_score(str(entity), a)
        if sc >= thr:
            for d in ds:
                out[d] = max(out.get(d, 0.0), sc)
            best = max(best, sc)
    res = sorted(d for d, s in out.items() if s >= best - 0.02)
    if qualifier and c["source"] == "mhrag" and len(res) > 1:
        q = str(qualifier)
        dated = [d for d in res if c["meta"].get(d, {}).get("date") and c["meta"][d]["date"] in q]
        if dated:
            res = dated
    return res[:cap]


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
