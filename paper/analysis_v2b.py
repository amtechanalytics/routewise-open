"""Analysis of ROUTEWISE v2 launch 2 (approach v2.1 section 6, amendments 10-11). Writes results/v2_*.csv,
results/v2_hypotheses.json and results/v2_summary.json. Every statistic is computed from runs/full rows."""
from __future__ import annotations

import csv
import json
import re

import numpy as np

from rh.util import RESULTS, RUNS

from . import stats as ST
from . import text as T
from . import v2b as B

SEED = 20261110
N_BOOT = 10000
HAIKU, OSS, SONNET = B.HAIKU, B.OSS, B.SONNET
COMPARATORS = ("flat", "meta", "hdrr", "sac", "pdoc", "pdoc-rr")   # "prefix" = Study 1 meta (title + metadata)


def _jl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _w(name, header, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_f(x) for x in r])


def _f(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return "" if x is None else x


def strat_boot(by_src: dict, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of per-source mean differences; 95 % CI from a bootstrap that resamples queries within each source."""
    rng = np.random.default_rng(seed)
    arrs = {s: np.asarray(v, dtype=float) for s, v in by_src.items() if len(v)}
    if not arrs:
        return float("nan"), float("nan"), float("nan")
    est = float(np.mean([a.mean() for a in arrs.values()]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrs.values()])
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _ece(conf, lab, bins: int = 10) -> float:
    conf, lab = np.asarray(conf, dtype=float), np.asarray(lab, dtype=float)
    if len(conf) == 0:
        return float("nan")
    e = 0.0
    for k in range(bins):
        m = (conf >= k / bins) & (conf < (k + 1) / bins if k < bins - 1 else conf <= 1.0)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - lab[m].mean())
    return float(e)


def _token_f1(a: str, b: str) -> float:
    ta, tb = T.tokens(a or ""), T.tokens(b or "")
    if not ta or not tb:
        return 0.0
    common = {}
    for t in ta:
        common[t] = min(ta.count(t), tb.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(ta), same / len(tb)
    return 2 * p * r / (p + r)


def load(mode: str = "full") -> dict:
    run = RUNS / mode
    ret = {(r["config"], r["qid"], r["arm"]): r for r in _jl(run / "retrieve2.jsonl")}
    gen = {r["cell_id"]: r for r in _jl(run / "generate.jsonl")}
    jud: dict = {}
    for r in _jl(run / "judge.jsonl"):
        if r["status"] == "ok":
            jud.setdefault(r["meta"]["gen_cell"], {})[r["model"]] = r["parsed"]
    val: dict = {}
    for r in _jl(run / "validity.jsonl"):
        if r["status"] == "ok":
            val.setdefault((r["meta"]["config"], r["meta"]["qid"]), {})[r["model"]] = r["parsed"]
    res = {r["cell_id"]: r for r in _jl(run / "resolve.jsonl")}
    route = {r["cell_id"]: r for r in _jl(run / "route.jsonl")}
    return {"ret": ret, "gen": gen, "jud": jud, "val": val, "res": res, "route": route,
            "cal": B.load_calibration(mode)}


def outcomes(D: dict, ctx) -> dict:
    """(config, qid, arm) -> outcome record (judge correctness by Haiku; panel majority where judged)."""
    out = {}
    for c in B.CONFIGS:
        for q in B.queries(ctx, c, "test"):
            for a in B.arms_for(c):
                gid = B.gen_id(c, q["qid"], a)
                g = D["gen"].get(gid)
                j = D["jud"].get(gid, {})
                hk = j.get(HAIKU)
                rr = D["ret"].get((c, q["qid"], a), {})
                panel = [j[m]["correct"] for m in (HAIKU, OSS, SONNET) if m in j]
                out[(c, q["qid"], a)] = {
                    "correct": int(bool(hk and hk["correct"])), "judged": hk is not None,
                    "answered": bool(g and g["status"] == "ok"), "answer": (g or {}).get("parsed", {}) and
                    g["parsed"].get("answer", "") if g and g.get("parsed") else "",
                    "panel": panel, "majority": int(sum(panel) * 2 > len(panel)) if len(panel) == 3 else None,
                    "doc_hit": rr.get("doc_hit"), "ev_any": rr.get("ev_any"), "ev_gold": rr.get("ev_gold"),
                    "top1": rr.get("top1_correct"), "decision": rr.get("decision"), "docs": rr.get("docs"),
                    "cost": float((g or {}).get("cost") or 0.0), "q": q}
    return out


def _paired(O, cfg, a, b, key="correct"):
    xs, ys = [], []
    for (c, qid, arm), o in O.items():
        if c != cfg or arm != a:
            continue
        p = O.get((c, qid, b))
        if p is None or o[key] is None or p[key] is None:
            continue
        xs.append(o[key])
        ys.append(p[key])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _mcnemar(x, y):
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    return b, c, ST.mcnemar_p(b, c)


def _mean_arm(O, cfg, arm, key="correct"):
    v = [o[key] for (c, _, a), o in O.items() if c == cfg and a == arm and o[key] is not None]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def analyze(ctx) -> None:
    D = load("full")
    O = outcomes(D, ctx)
    H: dict = {}
    # ---------------------------------------------------------------- arms table
    rows = []
    for c in B.CONFIGS:
        for a in B.arms_for(c):
            m, n = _mean_arm(O, c, a)
            rows.append([c, a, n, m, _mean_arm(O, c, a, "doc_hit")[0], _mean_arm(O, c, a, "ev_any")[0],
                         _mean_arm(O, c, a, "ev_gold")[0], _mean_arm(O, c, a, "top1")[0],
                         _mean_arm(O, c, a, "cost")[0]])
    _w("v2_t2_arms.csv", ["config", "arm", "n", "correct", "doc_hit", "ev_any", "ev_gold", "top1_acc",
                          "gen_cost_per_query"], rows)
    # ---------------------------------------------------------------- R2 / R4 contrasts at L2
    l2 = [f"{s}.L2" for s in B.SRC4]
    l0 = [f"{s}.L0" for s in B.SRC4]
    strongest = {}
    for c in l2:
        best = max(COMPARATORS, key=lambda a: (_mean_arm(O, c, a)[0] if _mean_arm(O, c, a)[1] else -1, a))
        strongest[c] = best
    tests = []
    for c in l2:
        for a in COMPARATORS:
            x, y = _paired(O, c, "R", a)
            b, cc, p = _mcnemar(x, y) if len(x) else (0, 0, 1.0)
            tests.append([c, a, len(x), float(x.mean()) if len(x) else None, float(y.mean()) if len(y) else None,
                          b, cc, p])
    holm = ST.holm([t[-1] for t in tests]) if tests else []
    for t, h in zip(tests, holm):
        t.append(h)
    _w("v2_t4_contrasts_L2.csv", ["config", "comparator", "n", "R", "comparator_correct", "R_only", "comp_only",
                                  "mcnemar_p", "holm_p"], tests)

    def beats_strongest(arm):
        per_src, diffs = 0, {}
        for c in l2:
            x, y = _paired(O, c, arm, strongest[c])
            if not len(x):
                continue
            diffs[c] = x - y
            if arm == "R":
                hp = next((t[-1] for t in tests if t[0] == c and t[1] == strongest[c]), 1.0)
                if x.mean() > y.mean() and hp < 0.05:
                    per_src += 1
        return per_src, strat_boot(diffs)
    n_src, (est, lo, hi) = beats_strongest("R")
    H["R2"] = {"strongest": strongest, "sources_holm_sig": n_src, "pooled_diff": est, "ci": [lo, hi],
               "pass": n_src >= 3 and lo > 0}
    _, (e4, l4, h4) = beats_strongest("R-noans")
    H["R4"] = {"pooled_diff_vs_strongest": e4, "ci": [l4, h4], "pass": l4 > 0}
    # ---------------------------------------------------------------- R1 recall@1
    r1 = {}
    for comp in ("pdoc", "pdoc-rr"):
        sig, diffs = 0, {}
        for c in l2:
            x, y = _paired(O, c, "R", comp, "top1")
            if not len(x):
                continue
            b, cc, p = _mcnemar(x, y)
            r1[f"{c}|{comp}"] = {"R": float(x.mean()), comp: float(y.mean()), "p": p}
            diffs[c] = x - y
        ps = [v["p"] for k, v in r1.items() if k.endswith("|" + comp)]
        hp = ST.holm(ps) if ps else []
        keys = [k for k in r1 if k.endswith("|" + comp)]
        for k, h in zip(keys, hp):
            r1[k]["holm_p"] = h
            sig += int(r1[k]["R"] > r1[k][comp] and h < 0.05)
        e, lo1, hi1 = strat_boot(diffs)
        r1[comp] = {"sources_holm_sig": sig, "pooled": e, "ci": [lo1, hi1], "pass": sig >= 3 and lo1 > 0}
    H["R1"] = {**r1, "pass": all(r1[c]["pass"] for c in ("pdoc", "pdoc-rr"))}
    # ---------------------------------------------------------------- R3 non-inferiority at L0
    diffs = {}
    for c in l0:
        x, y = _paired(O, c, "R", "meta")
        if len(x):
            diffs[c] = x - y
    e3, lo3, hi3 = strat_boot(diffs)
    H["R3"] = {"pooled_R_minus_meta": e3, "ci": [lo3, hi3], "margin": -0.02, "pass": lo3 > -0.02}
    # ---------------------------------------------------------------- R5 (amended): wrong-scope rate and ECE
    ws, cal_c, raw_c, labs = {}, [], [], []
    for c in [f"{s}.{lv}" for s in B.SRC4 for lv in ("L0", "L1", "L2") if f"{s}.{lv}" in B.CONFIGS]:
        x, y = [], []
        for (cc, qid, a), o in O.items():
            if cc != c or a != "R":
                continue
            u = O.get((cc, qid, "R-uncal"))
            if u is None:
                continue
            gold = set(o["q"]["gold_cfg"])
            x.append(int(o["decision"] == "hard" and not (set(o["docs"] or []) & gold)))
            y.append(int(u["decision"] == "hard" and not (set(u["docs"] or []) & gold)))
            rr = D["ret"].get((cc, qid, "R"), {})
            if rr.get("path") == "resolver" and rr.get("p"):
                cal_c.append(rr["p"][0])
                raw_c.append(rr.get("raw_top") or 0.0)
                labs.append(int((rr.get("docs") or [None])[0] in gold))
        if x:
            ws[c] = np.asarray(x, float) - np.asarray(y, float)
    e5, lo5, hi5 = strat_boot({k: v for k, v in ws.items()})
    ece_c, ece_r = _ece(cal_c, labs), _ece(raw_c, labs)
    H["R5"] = {"wrong_scope_R_minus_uncal": e5, "ci": [lo5, hi5], "ece_calibrated": ece_c, "ece_raw": ece_r,
               "n_resolver_decisions": len(labs), "pass": hi5 < 0 and ece_c < ece_r}
    # ---------------------------------------------------------------- R6 cost / correctness frontier at L2
    sys_cost = {}
    for a in ("R", "sac", "pdoc", "pdoc-rr", "meta", "hdrr"):
        tot, n = 0.0, 0
        for c in l2:
            for (cc, qid, arm), o in O.items():
                if cc != c or arm != a:
                    continue
                extra = 0.0
                if a in ("R", "hdrr"):
                    extra += float((D["route"].get(f"route2|{c}|test|{qid}") or {}).get("cost") or 0.0)
                if a == "R":
                    extra += float((D["res"].get(f"resolve|full|{c}|test|{qid}") or {}).get("cost") or 0.0)
                tot += o["cost"] + extra
                n += 1
        sys_cost[a] = tot / max(1, n)
    dom = []
    for a in ("sac", "pdoc", "pdoc-rr"):
        diffs = {}
        for c in l2:
            x, y = _paired(O, c, a, "R")
            if len(x):
                diffs[c] = x - y
        e, lo, hi = strat_boot(diffs)
        if sys_cost[a] <= sys_cost["R"] and lo > 0:
            dom.append(a)
    H["R6"] = {"cost_per_query_usd": sys_cost, "dominated_by": dom, "pass": not dom,
               "note": "API cost per query (router + resolver + generation); the cross-encoder runs on CPU and is "
                       "reported in diag_rerank.json"}
    # ---------------------------------------------------------------- S1 manipulation check
    s1, ok_src = {}, 0
    for s in B.SRC4:
        lv = [lv for lv in ("L0", "L1", "L2") if f"{s}.{lv}" in B.CONFIGS]
        res_rate, gain = [], []
        for v in lv:
            c = f"{s}.{v}"
            hd = [o for (cc, _, a), o in O.items() if cc == c and a == "hdrr"]
            res_rate.append(float(np.mean([o["top1"] or 0 for o in hd])) if hd else float("nan"))
            gain.append(_mean_arm(O, c, "hdrr")[0] - _mean_arm(O, c, "flat")[0])
        mono = all(res_rate[i] >= res_rate[i + 1] for i in range(len(res_rate) - 1)) and \
            all(gain[i] >= gain[i + 1] for i in range(len(gain) - 1))
        s1[s] = {"levels": lv, "router_resolution": res_rate, "hdrr_gain": gain, "monotone": mono}
        ok_src += int(mono)
    H["S1"] = {**s1, "pass": ok_src >= 3}
    # ---------------------------------------------------------------- S2 natural workloads
    diffs = {}
    for c in ("qasper.L0", "md2d.L0"):
        x, y = _paired(O, c, "R", "meta")
        if len(x):
            diffs[c] = x - y
    e, lo, hi = strat_boot(diffs)
    gap = {}
    for c in ("qasper.L0", "md2d.L0"):
        r, m, o = (_mean_arm(O, c, a)[0] for a in ("R", "meta", "oracle"))
        gap[c] = (r - m) / (o - m) if o - m > 1e-9 else None
    H["S2"] = {"pooled_R_minus_meta": e, "ci": [lo, hi], "gap_recovered": gap, "pass": lo > 0}
    # ---------------------------------------------------------------- S3 decoys
    s3 = {}
    for lv in ("L0", "L2"):
        loss = {}
        for a in ("R", "flat", "meta", "hdrr"):
            x, _ = _paired(O, f"fin.{lv}", a, a)
            fin = {qid: o["correct"] for (c, qid, arm), o in O.items() if c == f"fin.{lv}" and arm == a}
            my = {qid: o["correct"] for (c, qid, arm), o in O.items() if c == f"finmy.{lv}" and arm == a}
            common = sorted(set(fin) & set(my))
            loss[a] = np.asarray([fin[q] - my[q] for q in common], dtype=float)
        s3[lv] = {}
        for a in ("flat", "meta", "hdrr"):
            if len(loss["R"]) and len(loss[a]) == len(loss["R"]):
                e, lo, hi = strat_boot({"fin": loss["R"] - loss[a]})
                s3[lv][a] = {"R_loss": float(loss["R"].mean()), f"{a}_loss": float(loss[a].mean()),
                             "diff": e, "ci": [lo, hi]}
    H["S3"] = {**s3, "pass": all(v["ci"][1] < 0 for lv in s3.values() for v in lv.values()) if s3 else False}
    # ---------------------------------------------------------------- S4 deep stratum
    deep, shallow = {}, {}
    docs_by = {s: {d["doc_id"]: d for d in B.S.load(s)["docs"]} for s in B.SRC4}
    for c in [x for x in B.CONFIGS if B.corpus_of(x) in B.SRC4]:
        s = B.corpus_of(c)
        for (cc, qid, a), o in O.items():
            if cc != c or a != "R":
                continue
            m = O.get((cc, qid, "meta"))
            if m is None:
                continue
            ids = set()
            for g in o["q"]["gold_cfg"]:
                if g in docs_by[s]:
                    ids |= B.identity_tokens(s, docs_by[s][g])
            ev = " ".join(o["q"]["evidence"])
            is_deep = not (ids & set(B.ntok(ev)))
            (deep if is_deep else shallow).setdefault(s, []).append(o["correct"] - m["correct"])
    ed, ld, hd = strat_boot(deep)
    es, ls, hs = strat_boot(shallow)
    rng = np.random.default_rng(SEED)
    inter = []
    for _ in range(2000):
        dv = np.mean([np.asarray(v)[rng.integers(0, len(v), len(v))].mean() for v in deep.values() if v]) if deep \
            else 0.0
        sv = np.mean([np.asarray(v)[rng.integers(0, len(v), len(v))].mean() for v in shallow.values() if v]) \
            if shallow else 0.0
        inter.append(dv - sv)
    H["S4"] = {"deep": [ed, ld, hd], "shallow": [es, ls, hs], "n_deep": sum(len(v) for v in deep.values()),
               "n_shallow": sum(len(v) for v in shallow.values()),
               "interaction_ci": [float(np.percentile(inter, 2.5)), float(np.percentile(inter, 97.5))],
               "pass": float(np.percentile(inter, 2.5)) > 0}
    # ---------------------------------------------------------------- S5 card facts
    s5 = {}
    for a in ("R-title", "R-shuf"):
        diffs = {}
        for c in l2:
            x, y = _paired(O, c, "R", a)
            if len(x):
                diffs[c] = x - y
        e, lo, hi = strat_boot(diffs)
        s5[a] = {"R_minus": e, "ci": [lo, hi], "pass": lo > 0}
    H["S5"] = {**s5, "pass": all(v["pass"] for v in s5.values())}
    # ---------------------------------------------------------------- descriptive: oracle gap recovered
    gap_rows = []
    for c in B.CONFIGS:
        r, m, o = (_mean_arm(O, c, a)[0] for a in ("R", "meta", "oracle"))
        gap_rows.append([c, r, m, o, (r - m) / (o - m) if o - m > 1e-9 else None])
    _w("v2_t5_gap_recovered.csv", ["config", "R", "meta", "oracle", "share_of_gap_recovered"], gap_rows)
    # ---------------------------------------------------------------- meaning preservation (oracle L0 vs L1/L2)
    mp = []
    flipped = set()
    for s in B.SRC4:
        base = {qid: o["correct"] for (c, qid, a), o in O.items() if c == f"{s}.L0" and a == "oracle"}
        for lv in ("L1", "L2"):
            c = f"{s}.{lv}"
            if c not in B.CONFIGS:
                continue
            cur = {qid: o["correct"] for (cc, qid, a), o in O.items() if cc == c and a == "oracle"}
            common = sorted(set(base) & set(cur))
            if not common:
                continue
            d = np.asarray([cur[q] - base[q] for q in common], dtype=float)
            e, lo, hi = strat_boot({c: d})
            fl = [q for q in common if base[q] == 1 and cur[q] == 0]
            flipped |= {(c, q) for q in fl}
            mp.append([c, len(common), float(np.mean([base[q] for q in common])),
                       float(np.mean([cur[q] for q in common])), e, lo, hi, len(fl)])
    _w("v2_t8_meaning_preservation.csv", ["config", "n_paired", "oracle_L0", "oracle_level", "diff", "ci_lo",
                                          "ci_hi", "flips_correct_to_wrong"], mp)
    # ---------------------------------------------------------------- validity panel
    vrows, both_valid = [], set()
    for c in B.CONFIGS:
        if B.level_of(c) == "L0" or B.corpus_of(c) == "finmy":
            continue
        items = [(qid, v) for (cc, qid), v in D["val"].items() if cc == c]
        if not items:
            continue
        rec = {m: {k: [] for k in ("same_answer", "identifiable", "no_leak", "plausible")} for m in (SONNET, OSS)}
        agree, nboth = [], 0
        for qid, v in items:
            for m in (SONNET, OSS):
                if m in v:
                    for k in rec[m]:
                        rec[m][k].append(int(v[m][k]))
            if SONNET in v and OSS in v:
                a_all = all(v[SONNET].values())
                b_all = all(v[OSS].values())
                agree.append(int(a_all == b_all))
                if a_all and b_all:
                    both_valid.add((c, qid))
                    nboth += 1
        vrows.append([c, len(items)] + [float(np.mean(rec[m][k])) if rec[m][k] else None for m in (SONNET, OSS)
                                        for k in ("same_answer", "identifiable", "no_leak", "plausible")]
                     + [float(np.mean(agree)) if agree else None, nboth])
    _w("v2_t6_validity.csv", ["config", "n"] + [f"{m.split('/')[-1]}_{k}" for m in (SONNET, OSS)
                                                for k in ("same_answer", "identifiable", "no_leak", "plausible")]
       + ["all_four_agreement", "both_valid"], vrows)
    # sensitivity: R2 restricted to items both panel models rate valid and without an oracle flip
    sens = {}
    for c in l2:
        x, y = [], []
        for (cc, qid, a), o in O.items():
            if cc != c or a != "R" or (c, qid) not in both_valid or (c, qid) in flipped:
                continue
            p = O.get((cc, qid, strongest[c]))
            if p:
                x.append(o["correct"])
                y.append(p["correct"])
        if x:
            sens[c] = np.asarray(x, float) - np.asarray(y, float)
    es_, ls_, hs_ = strat_boot(sens)
    H["R2_sensitivity_valid_items"] = {"pooled_diff": es_, "ci": [ls_, hs_],
                                       "n": {k: len(v) for k, v in sens.items()}}
    # ---------------------------------------------------------------- judges: panel agreement + deterministic
    pairs = {(HAIKU, OSS): ([], []), (HAIKU, SONNET): ([], []), (OSS, SONNET): ([], [])}
    maj_h = ([], [])
    det = {"cuad": ([], []), "mhrag": ([], [])}
    for (c, qid, a), o in O.items():
        gid = B.gen_id(c, qid, a)
        j = D["jud"].get(gid, {})
        for (m1, m2), (u, v) in pairs.items():
            if m1 in j and m2 in j:
                u.append(int(j[m1]["correct"]))
                v.append(int(j[m2]["correct"]))
        if o["majority"] is not None and HAIKU in j:
            maj_h[0].append(o["majority"])
            maj_h[1].append(int(j[HAIKU]["correct"]))
        src = B.corpus_of(c)
        if src == "cuad" and o["answered"] and HAIKU in j:
            det["cuad"][0].append(int(_token_f1(o["answer"], o["q"]["answer"]) >= 0.5))
            det["cuad"][1].append(int(j[HAIKU]["correct"]))
        if src == "mhrag" and o["answered"] and HAIKU in j and len(T.tokens(o["q"]["answer"])) <= 5:
            ga = " ".join(T.tokens(o["q"]["answer"]))
            det["mhrag"][0].append(int(bool(ga) and ga in " ".join(T.tokens(o["answer"]))))
            det["mhrag"][1].append(int(j[HAIKU]["correct"]))
    jrows = []
    for (m1, m2), (u, v) in pairs.items():
        if u:
            jrows.append([f"{m1} vs {m2}", len(u), float(np.mean(np.asarray(u) == np.asarray(v))),
                          ST.cohen_kappa(u, v)])
    if maj_h[0]:
        jrows.append(["panel majority vs claude-haiku-5-5", len(maj_h[0]),
                      float(np.mean(np.asarray(maj_h[0]) == np.asarray(maj_h[1]))), ST.cohen_kappa(*maj_h)])
    for k, (u, v) in det.items():
        if u:
            jrows.append([f"deterministic ({'token F1 >= 0.5' if k == 'cuad' else 'exact match, short answers'}, "
                          f"{k}) vs claude-haiku-5-5", len(u), float(np.mean(np.asarray(u) == np.asarray(v))),
                          ST.cohen_kappa(u, v)])
    _w("v2_t7_judges.csv", ["comparison", "n", "agreement", "cohen_kappa"], jrows)
    # ---------------------------------------------------------------- calibration table
    crow = []
    for s, e in sorted(D["cal"].items()):
        for v in B.CARD_VARIANTS:
            ev = e.get(v) or {}
            crow.append([s, v, ev.get("n_dev"), ev.get("tau_high"), ev.get("tau_low"), e.get("route_tau1"),
                         e.get("pdoc_margin")])
    _w("v2_t9_calibration.csv", ["held_out_source", "card_variant", "n_dev_selections", "tau_high", "tau_low",
                                 "router_tau1", "pdoc_margin"], crow)
    # ---------------------------------------------------------------- decisions of R
    drows = []
    for c in B.CONFIGS:
        rs = [o for (cc, _, a), o in O.items() if cc == c and a == "R"]
        if rs:
            n = len(rs)
            drows.append([c, n] + [sum(1 for o in rs if o["decision"] == k) / n for k in ("hard", "soft", "fallback")]
                         + [float(np.mean([o["top1"] or 0 for o in rs]))])
    _w("v2_t10_decisions.csv", ["config", "n", "hard", "soft", "fallback", "top1_acc"], drows)
    # ---------------------------------------------------------------- spend
    spend: dict = {}
    for st in ("route", "resolve", "generate", "judge", "validity"):
        for r in _jl(RUNS / "full" / f"{st}.jsonl"):
            k = (st, r["model"])
            spend[k] = spend.get(k, 0.0) + float(r.get("cost") or 0.0)
    _w("v2_t11_spend.csv", ["stage", "model", "usd"], [[k[0], k[1], v] for k, v in sorted(spend.items())])
    # ---------------------------------------------------------------- hypotheses
    flat_h = {k: (v.get("pass") if isinstance(v, dict) else None) for k, v in H.items()}
    (RESULTS / "v2_hypotheses.json").write_text(json.dumps(_clean(H), indent=1, sort_keys=True) + "\n")
    (RESULTS / "v2_summary.json").write_text(json.dumps({"pass": flat_h, "strongest_L2": strongest}, indent=1,
                                                        sort_keys=True) + "\n")


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if x != x else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def brief_items(ctx) -> list[str]:
    from . import v2a as A
    w = A._warnings()
    head = ["Quality warnings (reported, not gated): " + "; ".join(f"{x['stage']}: {x['msg']}" for x in w)] if w \
        else ["Quality warnings: none."]
    return head + [
        "Read results/v2_summary.json and results/v2_hypotheses.json against R1-R6 and S1-S5 of approach v2.1 "
        "(amendments 10-11); report every hypothesis that did not pass.",
        "Check results/v2_t8_meaning_preservation.csv, results/v2_t6_validity.csv and results/v2_t7_judges.csv "
        "(validation suite) before drafting.",
    ]
