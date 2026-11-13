"""Pilot coverage (rhkit 1.0.4): the pilot must exercise every combination of the full run's dimensions, so
that nothing is met for the first time in the canary or the full run.

A cell's dimensions are model, arm, lane (batch|sync) and every key named in plan.yaml `coverage_dims`
(taken from the cell's `meta`, e.g. library, dataset, length bucket, rung). Pilot and canary selection groups
cells by the whole dimension tuple, so every combination present in the full run is present in the pilot by
construction. For local stages the paper declares `Stage(coverage=fn)` where fn(ctx, mode) returns one dict
of dimensions per unit the stage will process in that mode (e.g. {"library": "netty", "framework": "junit5"}),
or `coverage="derived"` when the stage only aggregates rows of earlier stages. A local stage with neither is a
build error. The report is written to runs/pilot_coverage.json; prep L15 fails on any missing combination and
the pilot rung fails on it before any spend."""
from __future__ import annotations

from .util import RUNS, write_json


def dims_of_cell(ctx, c: dict) -> dict:
    m = ctx.cfg.model(c["model"])
    d = {"model": c["model"], "arm": str(c.get("arm") or "-"), "lane": "batch" if m.get("batch") else "sync"}
    for k in ctx.cfg.plan.get("coverage_dims") or []:
        v = (c.get("meta") or {}).get(k, c.get(k))
        d[k] = "-" if v is None else str(v)
    return d


def key(d: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in d.items()))


def stage_units(ctx, stage, mode: str):
    """List of dimension dicts the stage processes in `mode`, or None when it has nothing to cover."""
    cov = getattr(stage, "coverage", None)
    if cov == "derived":
        return None
    if callable(cov):
        return [dict(u) for u in (cov(ctx, mode) or [])]
    if stage.kind == "llm" and stage.build:
        saved = ctx.mode
        try:
            ctx.mode = mode
            cells = stage.build(ctx)
            if mode != "full":
                cells = ctx.select(cells)
            return [dims_of_cell(ctx, c) for c in cells]
        finally:
            ctx.mode = saved
    raise ValueError(f"stage {stage.name}: local stage declares no coverage (Stage(coverage=fn) or "
                     f"coverage='derived')")


def report(ctx, modes=("pilot", "canary")) -> dict:
    out = {"stages": {}, "missing": 0}
    for st in ctx.paper.STAGES:
        full = stage_units(ctx, st, "full")
        if full is None:
            out["stages"][st.name] = {"coverage": "derived"}
            continue
        fk = {key(d) for d in full}
        entry = {"full_combinations": len(fk), "dims": sorted({k for d in full for k in d})}
        for mode in modes:
            units = stage_units(ctx, st, mode) or []
            mk = {key(d) for d in units}
            missing = sorted(fk - mk)
            entry[mode] = {"combinations": len(mk), "units": len(units),
                           "missing": [dict(t) for t in missing[:50]], "missing_count": len(missing)}
            if mode == "pilot":
                out["missing"] += len(missing)
        out["stages"][st.name] = entry
    write_json(RUNS / "pilot_coverage.json", out)
    return out


def problems(rep: dict) -> list[str]:
    errs = []
    for name, e in rep["stages"].items():
        if e.get("coverage") == "derived":
            continue
        p = e.get("pilot") or {}
        if p.get("missing_count"):
            ex = "; ".join(",".join(f"{k}={v}" for k, v in m.items()) for m in p["missing"][:3])
            errs.append(f"stage {name}: pilot misses {p['missing_count']} of {e['full_combinations']} "
                        f"combinations of {e['dims']} (e.g. {ex})")
        if e.get("full_combinations", 0) == 0:
            errs.append(f"stage {name}: full run has zero units")
    return errs


def table(rep: dict) -> list[str]:
    lines = ["| stage | dims | full combos | pilot combos | canary combos | missing in pilot |",
             "|---|---|---|---|---|---|"]
    for name, e in rep["stages"].items():
        if e.get("coverage") == "derived":
            lines.append(f"| {name} | derived from earlier stages | - | - | - | - |")
            continue
        lines.append(f"| {name} | {', '.join(e['dims'])} | {e['full_combinations']} | "
                     f"{e['pilot']['combinations']} | {e.get('canary', {}).get('combinations', '-')} | "
                     f"{e['pilot']['missing_count']} |")
    return lines
