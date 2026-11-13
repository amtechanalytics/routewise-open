"""Outcomes table (aggregate stage, on the instance) and the analysis (results/), which reads only runs/ so
`rh.reproduce` can re-run it anywhere. Primary outcome: evidence recall@8; secondary: judge correctness."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

from rh.util import RESULTS, RUNS, read_jsonl, write_json

from . import stats as St

HAIKU, SONNET, OSS = "claude-haiku-5-5", "claude-sonnet-5-5", "openai/gpt-oss-120b"
GEN_ARMS = ("flat", "meta", "route", "oracle")
POLICY_FEATURES = ("confusion", "mention_rate", "ambiguity")          # pre-registered (approach doc, sec. 11)
H4_FEATURES = ("confusion", "router_resolution", "ambiguity")
TAU_GRID = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9)


# --------------------------------------------------------------------------- aggregate (instance)
def build_outcomes(ctx) -> int:
    from . import sources as S
    mode = ctx.mode
    ret = {(r["config"], r["qid"], r["arm"]): r for r in read_jsonl(RUNS / mode / "retrieve.jsonl")}
    gens = {r["cell_id"]: r for r in ctx.stage_rows("generate")}
    judges = defaultdict(dict)
    for r in ctx.stage_rows("judge"):
        mt = r.get("meta") or {}
        judges[mt.get("gen_cell")][r["model"]] = r
    from . import _gen_plan, gen_cell_id
    out = []
    qinfo = {}
    for c, q, arm, gm in _gen_plan(ctx):
        rr = ret.get((c, q["qid"], "flat" if arm == "flat_aa" else arm))
        gid = gen_cell_id(c, q["qid"], arm, gm)
        g = gens.get(gid) or {}
        js = judges.get(gid, {})
        jh, jo = js.get(HAIKU) or {}, js.get(OSS) or {}
        ph = jh.get("parsed") if jh.get("status") == "ok" else None
        po = jo.get("parsed") if jo.get("status") == "ok" else None
        gp = g.get("parsed") if g.get("status") == "ok" else None
        usage = g.get("usage") or {}
        out.append({"config": c, "source": c.split(".")[0], "family": c.split(".")[1], "qid": q["qid"], "arm": arm,
                    "gen_model": gm, "doc_hit": (rr or {}).get("doc_hit"), "ev_recall": (rr or {}).get("ev_recall"),
                    "ctx_chars": (rr or {}).get("ctx_chars"), "n_resolved": (rr or {}).get("n_resolved"),
                    "resolved_correct": (rr or {}).get("resolved_correct"), "confidence": (rr or {}).get("confidence"),
                    "gen_status": g.get("status", "missing"), "found": (gp or {}).get("found"),
                    "answer": (gp or {}).get("answer", ""),
                    "score": (ph or {}).get("score", 1), "correct": int(bool((ph or {}).get("correct", False))),
                    "category": (ph or {}).get("category", "missing" if not gp else "unjudged"),
                    "oss_score": (po or {}).get("score"), "oss_correct": (int(po["correct"]) if po else None),
                    "in_tokens": (usage.get("input", 0) or 0) + (usage.get("cache_read", 0) or 0)
                    + (usage.get("cache_write", 0) or 0), "out_tokens": usage.get("output", 0) or 0,
                    "cost": g.get("cost", 0.0)})
        qinfo[q["qid"]] = {"question": q["q"] if c.split(".")[1] != "mask" else q.get("text"),
                           "reference": q["answer"], "evidence": " | ".join(q["evidence"])[:1500]}
        qinfo[f"{c}|{q['qid']}"] = {"question": q["q"]}
    # retrieval-only rows for the rule arm
    for (c, qid, arm), r in ret.items():
        if arm == "rule":
            out.append({"config": c, "source": c.split(".")[0], "family": c.split(".")[1], "qid": qid, "arm": "rule",
                        "gen_model": "-", "doc_hit": r["doc_hit"], "ev_recall": r["ev_recall"],
                        "ctx_chars": r["ctx_chars"], "n_resolved": r["n_resolved"],
                        "resolved_correct": r["resolved_correct"]})
    with open(RUNS / mode / "outcomes.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    write_json(RUNS / mode / "questions.json", qinfo)
    # router accuracy table needs the retrieve rows of the route arm only
    rt = [{k: r.get(k) for k in ("config", "qid", "n_resolved", "resolved_correct", "confidence", "router_ok",
                                 "alt_router")} for (c, qid, arm), r in ret.items() if arm == "route"]
    write_json(RUNS / mode / "router_rows.json", rt)
    _ = S  # parsed sources are only needed above through _gen_plan
    return len(out)


# --------------------------------------------------------------------------- helpers
def _w(name: str, header: list[str], rows: list[list]):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(x) for x in r])


def _fmt(x):
    if isinstance(x, float):
        return "" if math.isnan(x) else f"{x:.4f}"
    return x


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _load(mode="full"):
    rows = list(read_jsonl(RUNS / mode / "outcomes.jsonl"))
    feats = json.loads((RUNS / mode / "features.json").read_text())
    router = json.loads((RUNS / mode / "router_rows.json").read_text())
    qinfo = json.loads((RUNS / mode / "questions.json").read_text())
    return rows, feats, router, qinfo


def _index(rows, gen_model=HAIKU):
    """(config, qid) -> {arm: row} for one generator, plus composed hdrr."""
    by = defaultdict(dict)
    for r in rows:
        if r["gen_model"] == gen_model or (r["arm"] == "rule"):
            by[(r["config"], r["qid"])][r["arm"]] = r
    for k, arms in by.items():
        if "route" in arms and "flat" in arms:
            arms["hdrr"] = arms["route"] if (arms["route"].get("n_resolved") or 0) > 0 else arms["flat"]
    return by


def _metric(r, metric):
    if r is None:
        return float("nan")
    v = r.get(metric)
    return float("nan") if v is None else float(v)


def _routefb(arms, tau, metric):
    r = arms.get("route")
    if r is not None and (r.get("n_resolved") or 0) > 0 and (r.get("confidence") or 0.0) >= tau:
        return _metric(r, metric)
    return _metric(arms.get("flat"), metric)


# --------------------------------------------------------------------------- analysis
def analyze(ctx) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, feats, router, qinfo = _load()
    configs = sorted(feats)
    by = _index(rows, HAIKU)
    per_cfg = defaultdict(list)
    for (c, qid), arms in by.items():
        per_cfg[c].append((qid, arms))
    for c in per_cfg:
        per_cfg[c].sort()
    # router resolution rate (LLM) per config, for H4 and T9
    rres = defaultdict(list)
    for r in router:
        rres[r["config"]].append(r)
    for c in configs:
        lst = rres.get(c, [])
        feats[c]["router_resolution"] = round(_mean([1.0 if (r["n_resolved"] or 0) > 0 else 0.0 for r in lst]), 4) \
            if lst else 0.0

    # ---- T1 configurations
    keys = ["n_docs", "n_chunks", "chunks_per_doc", "confusion", "other_doc_sim", "near_dup_rate", "separability",
            "key_uniqueness", "n_queries", "mention_rate", "ambiguity", "rule_route_accuracy", "router_resolution",
            "query_tokens"]
    _w("t1_configs.csv", ["config"] + keys, [[c] + [feats[c].get(k) for k in keys] for c in configs])

    # ---- T2 arm outcomes (Haiku generator)
    t2 = []
    arms_all = ("flat", "meta", "route", "hdrr", "oracle", "rule")
    for c in configs:
        for a in arms_all:
            rs = [arms[a] for _, arms in per_cfg.get(c, []) if a in arms]
            if not rs:
                continue
            k = sum(int(r.get("correct") or 0) for r in rs) if a != "rule" else None
            lo, hi = St.wilson(k, len(rs)) if k is not None else (None, None)
            t2.append([c, a, len(rs), _mean([_metric(r, "ev_recall") for r in rs]),
                       _mean([_metric(r, "doc_hit") for r in rs]),
                       (k / len(rs)) if k is not None else None, lo, hi,
                       _mean([_metric(r, "score") for r in rs]) if a != "rule" else None,
                       _mean([_metric(r, "ctx_chars") for r in rs])])
    _w("t2_arms.csv", ["config", "arm", "n", "ev_recall", "doc_hit", "correct", "correct_lo", "correct_hi",
                       "mean_score", "ctx_chars"], t2)

    # ---- T3 contrasts vs flat (primary: route, meta, hdrr; Holm across all primary tests)
    t3, ps = [], []
    for c in configs:
        for a in ("route", "meta", "hdrr", "oracle", "rule"):
            pairs = [(arms["flat"], arms[a]) for _, arms in per_cfg.get(c, []) if "flat" in arms and a in arms]
            if not pairs:
                continue
            d_ev = [_metric(y, "ev_recall") - _metric(x, "ev_recall") for x, y in pairs]
            row = [c, a, len(pairs), _mean(d_ev), St.wilcoxon_p(d_ev)]
            if a != "rule":
                b = sum(1 for x, y in pairs if x.get("correct") and not y.get("correct"))
                cc = sum(1 for x, y in pairs if y.get("correct") and not x.get("correct"))
                d_c = (sum(int(y.get("correct") or 0) for _, y in pairs) - sum(int(x.get("correct") or 0)
                                                                                 for x, _ in pairs)) / len(pairs)
                row += [d_c, b, cc, St.mcnemar_p(b, cc)]
            else:
                row += [None, None, None, None]
            t3.append(row)
            if a in ("route", "meta", "hdrr"):
                ps.append((len(t3) - 1, "ev", row[4]))
                if row[8] is not None:
                    ps.append((len(t3) - 1, "correct", row[8]))
    adj = St.holm([p for _, _, p in ps])
    holm_p = {(i, kind): q for (i, kind, _), q in zip(ps, adj)}
    for i, r in enumerate(t3):
        r += [holm_p.get((i, "ev")), holm_p.get((i, "correct"))]
    _w("t3_contrasts.csv", ["config", "arm", "n", "delta_ev_recall", "wilcoxon_p", "delta_correct", "flat_only",
                            "arm_only", "mcnemar_p", "holm_p_ev", "holm_p_correct"], t3)

    # ---- per-config deltas used by T4-T6
    delta = {}
    for c in configs:
        lst = per_cfg.get(c, [])

        def md(a, metric):
            v = [_metric(arms[a], metric) - _metric(arms["flat"], metric) for _, arms in lst
                 if a in arms and "flat" in arms]
            return _mean(v)
        delta[c] = {a: md(a, "ev_recall") for a in ("route", "meta", "hdrr", "rule")}
        delta[c].update({f"{a}_correct": md(a, "correct") for a in ("route", "meta", "hdrr")})

    # ---- T4 feature correlations (directional, n = configurations)
    t4 = []
    for target in ("route", "meta", "route_correct"):
        for f in ("confusion", "other_doc_sim", "near_dup_rate", "separability", "key_uniqueness", "log_docs",
                  "mention_rate", "ambiguity", "router_resolution"):
            r, p = St.spearman_perm([feats[c].get(f) for c in configs], [delta[c][target] for c in configs])
            t4.append([target, f, len(configs), r, p])
    _w("t4_feature_corr.csv", ["delta", "feature", "n_configs", "spearman", "perm_p"], t4)

    # ---- T5 leave-one-source-out predictor of delta_route
    sources = sorted({c.split(".")[0] for c in configs})
    t5 = []
    for target in ("route", "meta"):
        for name, fs in (("ridge_preregistered", POLICY_FEATURES), ("ridge_h4_router", H4_FEATURES),
                         ("mention_only", ("mention_rate",)), ("train_mean", ())):
            errs = []
            for s in sources:
                tr = [c for c in configs if not c.startswith(s + ".")]
                te = [c for c in configs if c.startswith(s + ".")]
                y = [delta[c][target] for c in tr]
                if not fs:
                    pred = [float(np.nanmean(y))] * len(te)
                else:
                    m = St.ridge_fit([[feats[c][f] for f in fs] for c in tr], y)
                    pred = list(St.ridge_predict(m, [[feats[c][f] for f in fs] for c in te]))
                for c, p in zip(te, pred):
                    errs.append(abs(p - delta[c][target]))
                    t5.append([target, name, s, c, round(float(p), 4), delta[c][target]])
            t5.append([target, name, "ALL", "MAE", round(float(np.nanmean(errs)), 4), None])
    _w("t5_loso.csv", ["delta", "model", "held_out_source", "config", "predicted", "observed"], t5)

    # ---- T6 policy evaluation on held-out sources
    def score_cfg(c, policy, metric, tau=0.0):
        vals = []
        for _, arms in per_cfg.get(c, []):
            if policy == "routefb":
                vals.append(_routefb(arms, tau, metric))
            else:
                vals.append(_metric(arms.get(policy), metric))
        return _mean(vals)

    t6, summary = [], defaultdict(lambda: defaultdict(dict))
    for metric in ("ev_recall", "correct"):
        for s in sources:
            tr = [c for c in configs if not c.startswith(s + ".")]
            te = [c for c in configs if c.startswith(s + ".")]
            tau = max(TAU_GRID, key=lambda t: (np.nanmean([score_cfg(c, "routefb", metric, t) for c in tr]), -t))
            fixed = ("flat", "meta", "route", "hdrr")
            best_fixed = max(fixed, key=lambda a: np.nanmean([score_cfg(c, a, metric) for c in tr]))
            dm = St.ridge_fit([[feats[c][f] for f in POLICY_FEATURES] for c in tr], [delta[c]["meta"] for c in tr])
            dr = St.ridge_fit([[feats[c][f] for f in POLICY_FEATURES] for c in tr],
                              [score_cfg(c, "routefb", "ev_recall", tau) - score_cfg(c, "flat", "ev_recall")
                               for c in tr])
            for c in te:
                x = [[feats[c][f] for f in POLICY_FEATURES]]
                pm, pr = float(St.ridge_predict(dm, x)[0]), float(St.ridge_predict(dr, x)[0])
                choice = "flat" if max(pm, pr) <= 0 else ("meta" if pm >= pr else "routefb")
                vals = {a: score_cfg(c, a, metric) for a in ("flat", "meta", "route", "hdrr", "oracle")}
                vals["rule"] = score_cfg(c, "rule", "ev_recall") if metric == "ev_recall" else float("nan")
                vals["best_fixed_train"] = vals[best_fixed]
                vals["routewise"] = score_cfg(c, choice, metric, tau)
                cand = {a: vals[a] for a in ("flat", "meta", "route", "hdrr")}
                cand["routefb"] = score_cfg(c, "routefb", metric, tau)
                oracle_arm = max(cand, key=lambda a: cand[a])
                vals["oracle_arm"] = cand[oracle_arm]
                qbest = []
                for _, arms in per_cfg.get(c, []):
                    qbest.append(max(_metric(arms.get(a), metric) for a in ("flat", "meta", "route")
                                     if arms.get(a) is not None))
                vals["oracle_query"] = _mean(qbest)
                router_calls = 1.0 if choice == "routefb" else 0.0
                for pol, v in vals.items():
                    t6.append([metric, s, c, pol, v, vals["oracle_arm"] - v if v == v else None,
                               choice if pol == "routewise" else "", tau if pol == "routewise" else "",
                               router_calls if pol == "routewise" else (1.0 if pol in ("route", "hdrr") else 0.0)])
                    summary[metric][pol].setdefault(s, []).append(v)
    _w("t6_policy.csv", ["metric", "held_out_source", "config", "policy", "score", "regret_to_oracle_arm",
                         "routewise_choice", "tau", "router_calls_per_query"], t6)
    t6s = []
    for metric, pols in summary.items():
        for pol, byc in sorted(pols.items()):
            clean = {k: [x for x in v if x == x] for k, v in byc.items()}
            clean = {k: v for k, v in clean.items() if v}
            m, lo, hi = St.cluster_bootstrap(clean) if clean else (float("nan"),) * 3
            t6s.append([metric, pol, len(clean), m, lo, hi])
    _w("t6_policy_summary.csv", ["metric", "policy", "sources", "mean", "ci_lo", "ci_hi"], t6s)

    # ---- T7 judges and generators
    pairs = [(r["score"], r["oss_score"]) for r in rows if r.get("oss_score") is not None and r["gen_model"] == HAIKU]
    agree = [(int(r["correct"]), int(r["oss_correct"])) for r in rows if r.get("oss_correct") is not None
             and r["gen_model"] == HAIKU]
    _w("t7_judges.csv", ["pairs", "weighted_kappa_score", "kappa_correct", "agreement_correct"],
       [[len(pairs), St.weighted_kappa([a for a, _ in pairs], [b for _, b in pairs]),
         St.cohen_kappa([a for a, _ in agree], [b for _, b in agree]),
         _mean([1.0 if a == b else 0.0 for a, b in agree])]])
    t7g = []
    for gm in (OSS, SONNET):
        byg = _index(rows, gm)
        for c in configs:
            ks = [k for k in byg if k[0] == c and "route" in byg[k] and "flat" in byg[k]]
            if not ks:
                continue
            dg = _mean([_metric(byg[k]["route"], "correct") - _metric(byg[k]["flat"], "correct") for k in ks])
            dh = _mean([_metric(by[k]["route"], "correct") - _metric(by[k]["flat"], "correct") for k in ks if k in by])
            t7g.append([gm, c, len(ks), dh, dg, int(np.sign(dh) == np.sign(dg)) if dh == dh and dg == dg else None])
    _w("t7_generators.csv", ["generator", "config", "n", "delta_route_correct_haiku", "delta_route_correct_other",
                             "same_sign"], t7g)
    aa = [(by[k]["flat"], by[k]["flat_aa"]) for k in by if "flat_aa" in by[k] and "flat" in by[k]]
    _w("t7_aa_replicate.csv", ["n", "correct_agreement", "score_mean_abs_diff"],
       [[len(aa), _mean([1.0 if a.get("correct") == b.get("correct") else 0.0 for a, b in aa]),
         _mean([abs(_metric(a, "score") - _metric(b, "score")) for a, b in aa])]])

    # ---- T8 HDRR replication (fin.base), descriptive
    t8 = []
    for gm in (HAIKU, SONNET):
        byg = _index(rows, gm)
        for a in ("flat", "meta", "route", "hdrr", "oracle"):
            rs = [byg[k][a] for k in byg if k[0] == "fin.base" and a in byg[k]]
            if rs:
                t8.append([gm, a, len(rs), _mean([_metric(r, "score") for r in rs]),
                           _mean([1.0 if (r.get("score") or 1) <= 1 else 0.0 for r in rs]),
                           _mean([1.0 if (r.get("score") or 1) >= 4 else 0.0 for r in rs]),
                           _mean([_metric(r, "ev_recall") for r in rs])])
    _w("t8_replication.csv", ["generator", "arm", "n", "mean_score_1to5", "fail_rate_score1", "good_rate_score4plus",
                              "ev_recall"], t8)

    # ---- T9 router accuracy
    t9 = []
    for c in configs:
        lst = rres.get(c, [])
        if not lst:
            continue
        alt = [r["alt_router"] for r in lst if r.get("alt_router")]
        t9.append([c, len(lst), _mean([1.0 if (r["n_resolved"] or 0) > 0 else 0.0 for r in lst]),
                   _mean([float(r["resolved_correct"] or 0) for r in lst]),
                   _mean([1.0 if (r["n_resolved"] or 0) > 1 else 0.0 for r in lst]),
                   len(alt), _mean([float(a["resolved_correct"]) for a in alt]) if alt else None,
                   feats[c].get("rule_route_accuracy")])
    _w("t9_router.csv", ["config", "n", "haiku_resolved", "haiku_correct", "haiku_multi_doc", "n_alt",
                         "oss_correct", "rule_correct"], t9)

    # ---- spend by stage x model (from the cost column of every llm row)
    sp = defaultdict(float)
    for st in ("route", "generate", "judge"):
        for r in read_jsonl(RUNS / "full" / f"{st}.jsonl"):
            sp[(st, r["model"])] += float(r.get("cost") or 0.0)
    _w("t10_spend.csv", ["stage", "model", "usd"], [[k[0], k[1], round(v, 4)] for k, v in sorted(sp.items())])

    write_json(RESULTS / "deltas.json", delta)
    _figures(configs, feats, delta, t6s)
    _audit(rows, qinfo)


def _figures(configs, feats, delta, t6s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "svg.hashsalt": "routewise", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    xs = np.arange(len(configs))
    ax.bar(xs - 0.2, [delta[c]["route"] for c in configs], 0.4, label="route − flat", color="#3b6ea5")
    ax.bar(xs + 0.2, [delta[c]["meta"] for c in configs], 0.4, label="meta − flat", color="#c9822e")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xs, configs, rotation=60, ha="right")
    ax.set_ylabel("Δ evidence recall@8")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_deltas.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    marks = {"base": "o", "mask": "x", "dilute": "s", "merge": "^"}
    for c in configs:
        fam = c.split(".")[1]
        ax.scatter(feats[c]["confusion"], delta[c]["route"], marker=marks[fam], color="#3b6ea5")
        ax.annotate(c.split(".")[0], (feats[c]["confusion"], delta[c]["route"]), fontsize=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("inter-document confusion")
    ax.set_ylabel("Δ evidence recall@8 (route − flat)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_confusion.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    pol = [r for r in t6s if r[0] == "ev_recall"]
    if pol:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        names = [r[1] for r in pol]
        m = np.array([r[3] for r in pol], float)
        lo = np.array([r[4] for r in pol], float)
        hi = np.array([r[5] for r in pol], float)
        ax.barh(names, m, xerr=[np.nan_to_num(m - lo), np.nan_to_num(hi - m)], color="#7a9e7e")
        ax.set_xlabel("held-out evidence recall@8")
        fig.tight_layout()
        fig.savefig(RESULTS / "fig3_policies.png", dpi=300, metadata={"Software": None})
        plt.close(fig)


def _audit(rows, qinfo, n_target: int = 300):
    """Stratified human-audit sheet (source x arm, half judged correct, half not); raters never see the judge."""
    import hashlib
    strata = defaultdict(list)
    for r in rows:
        if r["gen_model"] != HAIKU or r["arm"] not in GEN_ARMS or r.get("gen_status") != "ok":
            continue
        strata[(r["source"], r["arm"], int(r["correct"]))].append(r)
    per = max(1, n_target // max(1, len({(s, a) for s, a, _ in strata}) * 2))
    pick = []
    for k in sorted(strata):
        lst = sorted(strata[k], key=lambda r: hashlib.sha1(f"audit|{r['config']}|{r['qid']}|{r['arm']}".encode())
                     .hexdigest())
        pick += lst[:per]
    pick = pick[:n_target]
    sheet, key = [], []
    for i, r in enumerate(sorted(pick, key=lambda r: hashlib.sha1(f"order|{r['config']}|{r['qid']}|{r['arm']}"
                                                                  .encode()).hexdigest())):
        aid = f"A{i + 1:03d}"
        qi = qinfo.get(r["qid"], {})
        q = qinfo.get(f"{r['config']}|{r['qid']}", {}).get("question") or qi.get("question", "")
        sheet.append([aid, r["config"], q, qi.get("reference", ""), qi.get("evidence", ""), r.get("answer", ""),
                      "", "", "", "", ""])
        key.append([aid, r["config"], r["qid"], r["arm"], r["score"], r["correct"], r["category"]])
    _w("audit_sheet_raters.csv", ["audit_id", "config", "question", "reference_answer", "gold_evidence",
                                  "system_answer", "rater1_label(correct|partial|wrong)", "rater1_category",
                                  "rater2_label(correct|partial|wrong)", "rater2_category", "note"], sheet)
    _w("audit_key.csv", ["audit_id", "config", "qid", "arm", "judge_score", "judge_correct", "judge_category"], key)


def brief_items(ctx) -> list[str]:
    return [
        "Human audit: fill rater1_* (anon) and rater2_* (external rater, given only the sheet and the rating "
        "instructions) in results/audit_sheet_raters.csv; do not open results/audit_key.csv until both are done; "
        "attach the filled sheet in the chat.",
        "Read results/t3_contrasts.csv and results/t6_policy_summary.csv against hypotheses H1-H6 of the approach "
        "doc; report any pre-registered hypothesis whose sign is opposite.",
        "Check results/t9_router.csv (router accuracy) and results/t1_configs.csv (feature ranges) for implausible "
        "values before drafting.",
    ]
