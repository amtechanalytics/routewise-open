"""ROUTEWISE study: when does document routing improve retrieval-augmented generation?

Six source collections, fifteen configurations (base, mask, dilute, merge), five retrieval arms (flat, meta,
route, oracle, rule-route; hdrr is composed from route and flat), an LLM router, generators from two model
families and an LLM judge validated against humans. Stages:

  corpora (local) -> index (local) -> route (llm) -> retrieve (local) -> generate (llm) -> judge (llm)
  -> aggregate (local, derived) ; analyze() then fits the leave-one-source-out predictor and evaluates the
  corpus-aware decision policy against fixed policies and the oracle.

Every llm cell carries meta {source, config, gen_model, qid}; plan.yaml coverage_dims = [source, config,
gen_model], so the pilot meets every (model, arm, lane, source, config, generator) combination of the full run.
In pilot and canary every stage works on the same probe queries (the first k of each configuration's fixed
order), so later stages always find the rows they need."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from rh.context import Stage
from rh.util import RUNS, write_json

from . import prompts as P
from . import sources as S

HAIKU, SONNET, OSS = "claude-haiku-5-5", "claude-sonnet-5-5", "openai/gpt-oss-120b"
GEN_ARMS = ("flat", "meta", "route", "oracle")
RET_ARMS = ("flat", "meta", "route", "oracle", "rule")
AA_ARM = "flat_aa"
MAX_TOKENS = {"route": {HAIKU: 1500, OSS: 3000}, "generate": {HAIKU: 4000, SONNET: 4000, OSS: 4000},
              "judge": {HAIKU: 2000, OSS: 3000}}
fake_response = P.fake_response


def _queries(ctx, config: str) -> list[dict]:
    return S.probe(S.config(config)["queries"], ctx.mode, ctx.cfg.plan)


def _sub(config: str, n: int) -> set:
    return {q["qid"] for q in S.config(config)["queries"][:n]}


def _diag(name):
    def writer(ctx, stage):
        p = RUNS / ctx.mode / f"diag_{name}.json"
        if not p.exists():
            write_json(p, {"stage": stage, "note": "stage failed before writing its diagnostics"})
    writer.__name__ = f"diag_{name}"
    return writer


def _cov_configs(ctx, mode):
    return [{"source": c.split(".")[0], "config": c, "gen_model": "-"} for c in S.config_names()]


# --------------------------------------------------------------------------- S1 corpora
def run_corpora(ctx) -> dict:
    ctx.register_diagnostics(_diag("corpora"))
    S.PROGRESS[0] = lambda msg: ctx.hb.touch(corpora=msg)
    out = {"sources": {}, "configs": {}}
    for s in S.SOURCES:
        d = S.load(s)
        out["sources"][s] = d["stats"]
        ctx.hb.touch(corpora=s)
    for c in S.config_names():
        cf = S.config(c)
        qs = cf["queries"]
        out["configs"][c] = {"docs": len(cf["titles"]), "queries": len(qs),
                             "probe": len(S.probe(qs, ctx.mode, ctx.cfg.plan)),
                             "gold_missing": sum(1 for q in qs if not q["gold_cfg"])}
    write_json(RUNS / ctx.mode / "corpora.json", out)
    write_json(RUNS / ctx.mode / "diag_corpora.json", out)
    return {"sources": len(out["sources"]), "configs": len(out["configs"]),
            "queries": sum(v["queries"] for v in out["configs"].values()),
            "min_probe": min(v["probe"] for v in out["configs"].values()),
            "gold_missing": sum(v["gold_missing"] for v in out["configs"].values()), "detail": out["configs"]}


def _check_corpora(ctx, m):
    errs = []
    k = int(ctx.cfg.plan["pilot_per_cell" if ctx.mode == "pilot" else "canary_per_cell"])
    if ctx.mode != "full" and m["min_probe"] < k:
        errs.append(f"a configuration has only {m['min_probe']} probe queries < {k}")
    if m["gold_missing"]:
        errs.append(f"{m['gold_missing']} queries lost their gold document in a configuration")
    for c, v in m["detail"].items():
        if v["docs"] < 2 or v["queries"] < 1:
            errs.append(f"{c}: {v['docs']} documents, {v['queries']} queries")
    fixture = all(str(p.get("url", "")).startswith("file://") for p in ctx.cfg.pins)   # dry run on fixtures
    if ctx.mode == "full" and not fixture:
        small = [c for c, v in m["detail"].items() if v["queries"] < int(ctx.cfg.plan.get("min_queries_full", 1))]
        if small:
            errs.append(f"configurations below min_queries_full: {small}")
    return errs


# --------------------------------------------------------------------------- S2 index + corpus features
def run_index(ctx) -> dict:
    from . import embed
    from . import index as X
    ctx.register_diagnostics(_diag("index"))
    embed.init(ctx)
    feats = {}
    for s in S.SOURCES:
        X.build(s, hb=ctx.hb)
        X.get.cache_clear()
        X.get(s)
        ctx.hb.touch(index={"source": s, "built": True})
    for c in S.config_names():
        f = X.features(c)
        f.update(X.workload(c))
        feats[c] = f
        write_json(RUNS / ctx.mode / "diag_index.json", {"features_so_far": feats})
        ctx.hb.touch(features=c)
    write_json(RUNS / ctx.mode / "features.json", feats)
    return {"configs": len(feats), "chunks": sum(v["n_chunks"] for v in feats.values()),
            "zero_chunk_configs": [c for c, v in feats.items() if v["n_chunks"] == 0]}


def _check_index(ctx, m):
    return [f"configurations without chunks: {m['zero_chunk_configs']}"] if m["zero_chunk_configs"] else []


# --------------------------------------------------------------------------- S3 route (llm)
def _route_configs() -> list[str]:
    return [c for c in S.config_names() if c.split(".")[1] in ("base", "mask")]


def build_route(ctx) -> list[dict]:
    cells = []
    for c in _route_configs():
        src = c.split(".")[0]
        alt = _sub(c, S.SUB_ALT) if c.endswith(".base") else set()
        for q in _queries(ctx, c):
            for m in [HAIKU] + ([OSS] if q["qid"] in alt else []):
                cells.append({"cell_id": f"route|{c}|{q['qid']}|{m}", "model": m, "arm": "route",
                              "prefix": P.ROUTER_PREFIX, "item": P.router_item(S.SOURCE_DESCRIPTION[src], q["q"]),
                              "max_tokens": MAX_TOKENS["route"][m],
                              "meta": {"source": src, "config": c, "gen_model": "-", "qid": q["qid"]}})
    return cells


def _check_llm(min_ok: float):
    def check(ctx, m):
        attempted = m["ok"] + m["parse_fail"] + m.get("empty_reply", 0)
        if m["cells"] and ctx.mode == "full" and m["ok"] / m["cells"] < min_ok:
            return [f"ok rows {m['ok']} of {m['cells']} cells < {min_ok:.0%}"]
        if attempted == 0:
            return ["no replies at all"]
        return []
    return check


# --------------------------------------------------------------------------- S4 retrieve (local)
def _ctx_dir(mode: str) -> Path:
    return RUNS / mode / "contexts"


def _router_rows(ctx, model: str = HAIKU) -> dict:
    out = {}
    for r in ctx.stage_rows("route"):
        mt = r.get("meta") or {}
        if r["model"] == model:
            out[(mt.get("config"), mt.get("qid"))] = r
    return out


def run_retrieve(ctx) -> dict:
    from . import embed
    from . import index as X
    from . import text as T
    ctx.register_diagnostics(_diag("retrieve"))
    embed.init(ctx)
    routers = _router_rows(ctx)
    alt_routers = _router_rows(ctx, OSS)
    out_p = RUNS / ctx.mode / "retrieve.jsonl"
    out_p.unlink(missing_ok=True)
    cdir = _ctx_dir(ctx.mode)
    cdir.mkdir(parents=True, exist_ok=True)
    n_rows, missing_router, per_arm = 0, 0, {a: 0 for a in RET_ARMS}
    with open(out_p, "w") as fo:
        for c in S.config_names():
            cf = S.config(c)
            ix = X.get(cf["source"])
            qs = _queries(ctx, c)
            qv = embed.embedder().encode([q["q"] for q in qs], query=True)
            rec = []
            for qi, q in enumerate(qs):
                rr = routers.get((S.text_variant(c), q["qid"]))
                parsed = (rr or {}).get("parsed") if (rr or {}).get("status") == "ok" else None
                if rr is None:
                    missing_router += 1
                ent = (parsed or {}).get("entity")
                conf = float((parsed or {}).get("confidence") or 0.0)
                resolved = X.resolve(c, ent, (parsed or {}).get("qualifier"))
                rule = X.rule_match(c, q["q"])
                ar = alt_routers.get((S.text_variant(c), q["qid"]))
                alt = None
                if ar is not None and c.endswith(".base"):
                    ap = ar.get("parsed") if ar.get("status") == "ok" else None
                    ares = X.resolve(c, (ap or {}).get("entity"), (ap or {}).get("qualifier"))
                    alt = {"ok": ap is not None, "n_resolved": len(ares),
                           "resolved_correct": int(bool(set(ares) & set(q["gold_cfg"]))),
                           "confidence": float((ap or {}).get("confidence") or 0.0)}
                allowed = {"flat": None, "meta": None, "route": set(resolved), "oracle": set(q["gold_cfg"]),
                           "rule": set(rule)}
                for arm in RET_ARMS:
                    if arm in ("route", "rule") and not allowed[arm]:
                        idx = []
                    else:
                        idx = X.retrieve(c, q["q"], qv[qi], "plain" if arm != "meta" else "meta", allowed[arm])
                    docs = [cf["doc_map"][ix.chunks[i]["doc"]] for i in idx]
                    texts = [ix.chunks[i]["text"] for i in idx]
                    row = {"config": c, "source": cf["source"], "family": cf["family"], "qid": q["qid"], "arm": arm,
                           "chunks": idx, "docs": docs, "doc_hit": int(bool(set(docs) & set(q["gold_cfg"]))),
                           "ev_recall": round(T.evidence_recall(q["evidence"], texts), 4),
                           "ctx_chars": sum(len(t) for t in texts), "n_resolved": len(allowed[arm] or []) if
                           arm in ("route", "rule") else None,
                           "resolved_correct": (int(bool(set(allowed[arm]) & set(q["gold_cfg"])))
                                                if arm in ("route", "rule") else None),
                           "confidence": conf if arm == "route" else None, "router_ok": parsed is not None,
                           "alt_router": alt if arm == "route" else None}
                    fo.write(json.dumps(row) + "\n")
                    rec.append({"qid": q["qid"], "arm": arm,
                                "passages": [[cf["titles"][d], t] for d, t in zip(docs, texts)]})
                    n_rows += 1
                    per_arm[arm] += 1
                ctx.hb.touch(retrieve={"config": c, "done": qi + 1, "total": len(qs)})
            tmp = cdir / f"{c}.jsonl.gz.tmp"
            with gzip.open(tmp, "wt") as g:
                for r in rec:
                    g.write(json.dumps(r) + "\n")
            tmp.replace(cdir / f"{c}.jsonl.gz")
            write_json(RUNS / ctx.mode / "diag_retrieve.json", {"rows": n_rows, "last_config": c})
    expected = sum(len(_queries(ctx, c)) for c in S.config_names()) * len(RET_ARMS)
    return {"rows": n_rows, "expected": expected, "missing_router": missing_router, "per_arm": per_arm}


def _check_retrieve(ctx, m):
    errs = []
    if m["rows"] != m["expected"]:
        errs.append(f"retrieve wrote {m['rows']} rows, expected {m['expected']}")
    if m["missing_router"] > 0.05 * max(1, m["expected"] / len(RET_ARMS)):
        errs.append(f"{m['missing_router']} queries have no router row")
    return errs


def _cov_retrieve(ctx, mode):
    return [{"source": c.split(".")[0], "config": c, "gen_model": "-", "arm": a}
            for c in S.config_names() for a in RET_ARMS]


# --------------------------------------------------------------------------- S5 generate (llm)
def _gen_plan(ctx) -> list[tuple[str, dict, str, str]]:
    """(config, query, arm, model) for every generation of the mode."""
    plan = []
    for c in S.config_names():
        base = c.endswith(".base")
        alt = _sub(c, S.SUB_ALT) if base else set()
        aa = _sub(c, S.SUB_AA) if c == "fin.base" else set()
        for q in _queries(ctx, c):
            for arm in GEN_ARMS:
                plan.append((c, q, arm, HAIKU))
                if c == "fin.base":
                    plan.append((c, q, arm, SONNET))
                if q["qid"] in alt:
                    plan.append((c, q, arm, OSS))
            if q["qid"] in aa:
                plan.append((c, q, AA_ARM, HAIKU))
    return plan


def _load_contexts(mode: str) -> dict | None:
    cdir = _ctx_dir(mode)
    if not cdir.exists():
        return None
    out = {}
    for p in sorted(cdir.glob("*.jsonl.gz")):
        c = p.name[: -len(".jsonl.gz")]
        with gzip.open(p, "rt") as g:
            for line in g:
                r = json.loads(line)
                out[(c, r["qid"], r["arm"])] = [tuple(x) for x in r["passages"]]
    return out


def gen_cell_id(c, qid, arm, model) -> str:
    return f"gen|{c}|{qid}|{arm}|{model}"


def build_generate(ctx) -> list[dict]:
    ctxs = _load_contexts(ctx.mode)
    cells = []
    for c, q, arm, m in _gen_plan(ctx):
        src_arm = "flat" if arm == AA_ARM else arm
        if ctxs is None:
            passages = []                                  # coverage-only build (retrieve has not run yet)
        else:
            key = (c, q["qid"], src_arm)
            if key not in ctxs:
                raise ValueError(f"generate: no retrieved context for {key}")
            passages = ctxs[key]
        cells.append({"cell_id": gen_cell_id(c, q["qid"], arm, m), "model": m, "arm": arm, "prefix": P.GEN_PREFIX,
                      "item": P.gen_item(q["q"], passages), "max_tokens": MAX_TOKENS["generate"][m],
                      "meta": {"source": c.split(".")[0], "config": c, "gen_model": "-", "qid": q["qid"]}})
    return cells


# --------------------------------------------------------------------------- S6 judge (llm)
def build_judge(ctx) -> list[dict]:
    gen_out = ctx.out("generate")
    rows = {}
    if gen_out.exists():
        for r in ctx.stage_rows("generate"):
            rows[r["cell_id"]] = r
    qmap = {}
    cells = []
    for c, q, arm, gm in _gen_plan(ctx):
        gid = gen_cell_id(c, q["qid"], arm, gm)
        if gen_out.exists():
            r = rows.get(gid)
            if not r or r["status"] != "ok":
                continue                                     # not judged; scored as wrong in the analysis
            answer = r["parsed"]["answer"]
        else:
            answer = ""                                      # coverage-only build
        judges = [HAIKU]
        if gm == HAIKU and c.endswith(".base") and arm in GEN_ARMS and q["qid"] in qmap.setdefault(
                c, _sub(c, S.SUB_ALT)):
            judges.append(OSS)
        for jm in judges:
            cells.append({"cell_id": f"judge|{gid}|{jm}", "model": jm, "arm": arm, "prefix": P.JUDGE_PREFIX,
                          "item": P.judge_item(q["q"], q["answer"], q["evidence"], answer),
                          "max_tokens": MAX_TOKENS["judge"][jm],
                          "meta": {"source": c.split(".")[0], "config": c, "gen_model": gm, "qid": q["qid"],
                                   "gen_cell": gid}})
    return cells


# --------------------------------------------------------------------------- S7 aggregate (derived)
def run_aggregate(ctx) -> dict:
    from . import analysis
    ctx.register_diagnostics(_diag("aggregate"))
    n = analysis.build_outcomes(ctx)
    return {"outcomes": n}


def _check_aggregate(ctx, m):
    return [] if m["outcomes"] > 0 else ["aggregate built zero outcome rows"]


STUDY1_STAGES = [
    Stage("corpora", "local", run=run_corpora, gate={"check": _check_corpora}, coverage=_cov_configs),
    Stage("index", "local", run=run_index, gate={"check": _check_index}, coverage=_cov_configs),
    Stage("route", "llm", build=build_route, schema=P.parse_router,
          gate={"min_parse_rate": 0.9, "max_duplicate_cells": 0, "check": _check_llm(0.95)}),
    Stage("retrieve", "local", run=run_retrieve, gate={"check": _check_retrieve}, coverage=_cov_retrieve),
    Stage("generate", "llm", build=build_generate, schema=P.parse_gen,
          gate={"min_parse_rate": 0.9, "max_duplicate_cells": 0, "check": _check_llm(0.95)}),
    Stage("judge", "llm", build=build_judge, schema=P.parse_judge,
          gate={"min_parse_rate": 0.9, "max_duplicate_cells": 0, "check": _check_llm(0.95)}),
    Stage("aggregate", "local", run=run_aggregate, gate={"check": _check_aggregate}, coverage="derived"),
]


# --------------------------------------------------------------------------- v2 launch 1 (approach v2.1)
# Study 1 ran from STUDY1_STAGES (run routewise-20261110T035239Z); its results are kept as published there.
# This harness runs launch 1 of ROUTEWISE v2: decoy filings, profile cards, rewritten questions and the pre-run
# rewrite audit. Launch 2 is a later harness version that pins launch 1's frozen artifacts.
STUDY = "v2a"
from . import v2a as _v2a   # noqa: E402

V2A_STAGES = [
    Stage("corpora", "local", run=run_corpora, gate={"check": _check_corpora}, coverage=_cov_configs),
    Stage("edgar", "local", run=_v2a.run_edgar, gate={"check": _v2a._check_edgar}, coverage=_v2a._cov_edgar),
    Stage("cards", "llm", build=_v2a.build_cards, schema=P.parse_card,
          gate={"min_parse_rate": 0.5, "max_duplicate_cells": 0, "max_vendor_error_rate": 0.2,
                "empty_reply_ceiling": 0.5, "check": _check_llm(0.5)}),
    Stage("cardcheck", "local", run=_v2a.run_cardcheck, gate={"check": _v2a._check_cardcheck},
          coverage=_v2a._cov_cardcheck),
    Stage("rewrite", "llm", build=_v2a.build_rewrite, schema=P.parse_rewrite,
          gate={"min_parse_rate": 0.5, "max_duplicate_cells": 0, "max_vendor_error_rate": 0.2,
                "empty_reply_ceiling": 0.5, "check": _check_llm(0.5)}),
    Stage("rewritecheck", "local", run=_v2a.run_rewritecheck, gate={"check": _v2a._check_rewritecheck},
          coverage=_v2a._cov_rewritecheck),
]
# --------------------------------------------------------------------------- v2 launch 2 (approach v2.1 + amendments)
from . import v2b as _v2b   # noqa: E402

V2B_STAGES = [
    Stage("corpora", "local", run=run_corpora, gate={"check": _check_corpora}, coverage=_cov_configs),
    Stage("edgar", "local", run=_v2b.run_edgar, gate={"check": _v2b._check_edgar}, coverage=_v2b._cov_edgar),
    Stage("recheck", "local", run=_v2b.run_recheck, gate={"check": _v2b._check_recheck}, coverage=_v2b._cov_cfgs),
    Stage("index", "local", run=_v2b.run_index, gate={"check": _v2b._check_index}, coverage=_v2b._cov_index),
    Stage("route", "llm", build=_v2b.build_route, schema=P.parse_router,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("rerank", "local", run=_v2b.run_rerank, gate={"check": _v2b._check_rerank}, coverage=_v2b._cov_cfgs),
    Stage("resolve", "llm", build=_v2b.build_resolve, schema=P.parse_resolver,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("calibrate", "local", run=_v2b.run_calibrate, gate={"check": _v2b._check_calibrate},
          coverage=_v2b._cov_calibrate),
    Stage("retrieve", "local", run=_v2b.run_retrieve, gate={"check": _v2b._check_retrieve},
          coverage=_v2b._cov_retrieve),
    Stage("generate", "llm", build=_v2b.build_generate, schema=P.parse_gen,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("judge", "llm", build=_v2b.build_judge, schema=P.parse_judge,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("validity", "llm", build=_v2b.build_validity, schema=P.parse_validity,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
]
# --------------------------------------------------------------------------- study 3, launch 3a (pre-registration
# eswa27-13 with amendment 1): fresh questions, screened before any arm runs
from . import v2c as _v2c   # noqa: E402

V2C_STAGES = [
    Stage("corpora", "local", run=run_corpora, gate={"check": _check_corpora}, coverage=_cov_configs),
    Stage("index3", "local", run=_v2c.run_index3, gate={"check": _v2c._check_index3}, coverage=_v2c._cov_index),
    Stage("cand3", "local", run=_v2c.run_cand3, gate={"check": _v2c._check_cand3}, coverage=_v2c._cov_sources),
    Stage("rewrite3a", "llm", build=_v2c.build_rewrite3a, schema=P.parse_rewrite,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("rewrite3b", "llm", build=_v2c.build_rewrite3b, schema=P.parse_rewrite,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("leak3", "local", run=_v2c.run_leak3, gate={"check": _v2c._check_leak3}, coverage=_v2c._cov_sources),
    Stage("oret3", "local", run=_v2c.run_oret3, gate={"check": _v2c._check_oret3}, coverage=_v2c._cov_screen),
    Stage("ogen3", "llm", build=_v2c.build_ogen3, schema=P.parse_gen,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("ojudge3", "llm", build=_v2c.build_ojudge3, schema=P.parse_judge,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("panel3", "llm", build=_v2c.build_panel3, schema=P.parse_validity,
          gate={**_v2b.LOOSE_GATE, "check": _check_llm(0.5)}),
    Stage("screen3", "local", run=_v2c.run_screen3, gate={"check": _v2c._check_screen3}, coverage=_v2c._cov_screen),
]
STUDY = "v2c"
STAGES = {"v2a": V2A_STAGES, "v2b": V2B_STAGES, "v2c": V2C_STAGES}.get(STUDY, STUDY1_STAGES)


def analyze(ctx) -> None:
    if STUDY == "v2c":
        _v2c.analyze(ctx)
        return
    if STUDY == "v2b":
        from . import analysis_v2b
        analysis_v2b.analyze(ctx)
        return
    if STUDY == "v2a":
        _v2a.analyze(ctx)
        return
    from . import analysis
    analysis.analyze(ctx)


def brief(ctx) -> list[str]:
    if STUDY == "v2c":
        return _v2c.brief_items(ctx)
    if STUDY == "v2b":
        from . import analysis_v2b
        return analysis_v2b.brief_items(ctx)
    if STUDY == "v2a":
        return _v2a.brief_items(ctx)
    from . import analysis
    return analysis.brief_items(ctx)
