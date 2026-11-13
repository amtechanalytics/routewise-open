"""Study 1 tests that need the Study 1 stages (index, retrieve, generate, judge) and its models (Sonnet 5.5,
text-embedding-3-small). They ran green for run routewise-20261110T035239Z and return to tests/ in launch 2;
this folder is not collected while the harness runs launch 1 of v2."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _pkg():
    name = "routewise_paper"
    if name in sys.modules:
        return sys.modules[name]
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    spec = importlib.util.spec_from_file_location(name, REPO / "paper" / "__init__.py",
                                                  submodule_search_locations=[str(REPO / "paper")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mod(sub):
    _pkg()
    return importlib.import_module(f"routewise_paper.{sub}")


# --------------------------------------------------------------------------- text

def test_api_embedder_ledger_shards_resume_and_guard():
    import types
    E = _mod("embed")
    from rh import vendors

    class Led:
        def __init__(self):
            self.recorded, self.checked = [], 0

        def estimate(self, m, i, o, batch=False):
            return i * 0.02 / 1e6

        def check(self, m, est, kind="experiment"):
            self.checked += 1

        def record(self, m, usage, **kw):
            self.recorded.append(usage["input"])

    cfg = types.SimpleNamespace(model=lambda mid: {"id": mid, "provider": "openai", "concurrency": 2,
                                                   "tpm_budget": 0, "rpm_budget": 0})
    calls = []

    def fake_http(method, url, headers, body, timeout=600):
        d = json.loads(body)
        calls.append(len(d["input"]))
        assert url.endswith("/embeddings") and d["dimensions"] == E.DIM_API
        out = [{"index": i, "embedding": [1.0 + (len(t) % 7)] + [0.0] * (E.DIM_API - 1)}
               for i, t in enumerate(d["input"])]
        return 200, {"x-ratelimit-remaining-tokens": "999000"}, json.dumps(
            {"data": out, "usage": {"prompt_tokens": 10 * len(d["input"])}}).encode(), 0.01

    old = (vendors._http, vendors._key, E.SHARD, E.MAX_INPUTS)
    tmp = Path(tempfile.mkdtemp(prefix="rw-emb-"))
    os.environ["OPENAI_API_KEY"] = "test"
    try:
        vendors._http = fake_http
        E.SHARD, E.MAX_INPUTS = 10, 4
        led = Led()
        emb = E.OpenAIEmbedder(led, cfg)
        texts = [f"text number {i}" for i in range(25)]
        v = emb.encode(texts, cache_dir=tmp, label="t")
        assert v.shape == (25, E.DIM_API) and abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-6
        assert sum(calls) == 25 and max(calls) <= 4
        assert sum(led.recorded) == 250 and led.checked == len(calls)
        assert len(list(tmp.glob("shard_t_*.npy"))) == 3
        n_before = len(calls)
        v2 = emb.encode(texts, cache_dir=tmp, label="t")          # resume: every shard on disk, no new calls
        assert len(calls) == n_before and np.allclose(v, v2)
        E.GUARD_AFTER_S, E.MAX_HOURS_PER_SOURCE = 0, 1e-9          # guard: projected time over the limit
        try:
            emb.encode([f"other {i}" for i in range(25)], cache_dir=tmp, label="g")
            raise AssertionError("throughput guard did not fire")
        except E.ThroughputTooLow:
            pass
    finally:
        vendors._http, vendors._key, E.SHARD, E.MAX_INPUTS = old
        E.GUARD_AFTER_S, E.MAX_HOURS_PER_SOURCE = 600, 4.0
        os.environ.pop("OPENAI_API_KEY", None)
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- the study ladder on fixtures
_LADDER = r'''
import json, os, sys
from rh import runner, datapins, coverage
from rh.context import Context
from rh.heartbeat import Heartbeat
from rh.util import ensure_dirs
ensure_dirs()
ctx = Context(pins_file="data_pins.fixture.json")
assert not ctx.cfg.errors, ctx.cfg.errors
ctx.hb = Heartbeat()
datapins.verify_all(ctx.cfg.pins)
rep = coverage.report(ctx)
assert not coverage.problems(rep), coverage.problems(rep)
for mode in ("pilot", "canary", "full"):
    runner._stages(ctx, mode)
ctx.mode = "full"
runner._write_plan(ctx)
runner.rung_analysis(ctx)
print(json.dumps({"results": sorted(p.name for p in (ctx.cfg.root / "results").iterdir()),
                  "coverage": {k: v.get("full_combinations") for k, v in rep["stages"].items()}}))
'''


def test_study_ladder_on_fixtures():
    tmp = Path(tempfile.mkdtemp(prefix="rw-study-")) / "repo"
    shutil.copytree(REPO, tmp, ignore=shutil.ignore_patterns(".git", ".venv", "runs", "results", "logs", ".data",
                                                             "data", "__pycache__", ".fake_aws"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("RH_")}
    env.update(RH_ROOT=str(tmp), RH_FAKE_VENDOR="1", RH_FAKE_AWS="1", RH_RUN_ID="study-test",
               RH_FAKE_AWS_DIR=str(tmp / ".fake_aws"))
    r = subprocess.run([sys.executable, "-c", _LADDER], cwd=tmp, env=env, capture_output=True, text=True,
                       timeout=1800)
    assert r.returncode == 0, (r.stdout[-2000:] + r.stderr[-4000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for f in ("t1_configs.csv", "t2_arms.csv", "t3_contrasts.csv", "t4_feature_corr.csv", "t5_loso.csv",
              "t6_policy.csv", "t6_policy_summary.csv", "t7_judges.csv", "t8_replication.csv", "t9_router.csv",
              "audit_sheet_raters.csv", "audit_key.csv", "fig1_deltas.png", "fig2_confusion.png"):
        assert f in out["results"], f
    assert out["coverage"]["generate"] > 0 and out["coverage"]["judge"] > 0
    rows = [json.loads(x) for x in (tmp / "runs" / "full" / "retrieve.jsonl").read_text().splitlines()]
    for x in rows:
        if x["arm"] == "oracle":
            assert x["doc_hit"] == 1
        if x["arm"] == "route" and not x["n_resolved"]:
            assert x["chunks"] == []
    # analysis is reproducible from runs/ alone
    a = {p.name: p.read_bytes() for p in (tmp / "results").iterdir() if p.suffix == ".csv"}
    r2 = subprocess.run([sys.executable, "-c", "from rh.context import Context\nctx=Context(mode='full')\n"
                         "ctx.paper.analyze(ctx)"], cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stderr[-3000:]
    b = {p.name: p.read_bytes() for p in (tmp / "results").iterdir() if p.suffix == ".csv"}
    assert a == b, sorted(k for k in a if a[k] != b.get(k))
    shutil.rmtree(tmp.parent, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {str(e)[:2000]}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
