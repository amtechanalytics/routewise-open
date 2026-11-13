"""rhkit 1.0.4: repair policy signatures, pilot coverage, empty_reply status, heartbeat race, Deflate64 fallback,
git-pin size."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile

import pytest

from rh import coverage, datapins, repair
from rh.exitcodes import CRASH, DataFailed, PILOT_FAIL
from rh.util import ROOT, RUNS, read_json, write_json


# ---------------------------------------------------------------- repair signatures
def test_signature_blanks_run_ids_paths_numbers():
    a = repair.signature({"code": CRASH, "message": "FileNotFoundError: /opt/rh/day0/data/nice.csv run day0-20261106T234802Z"})
    b = repair.signature({"code": CRASH, "message": "FileNotFoundError: /opt/rh/x/data/other.csv run x-20261108T000000Z"})
    assert a == b and a.startswith("1:FileNotFoundError:")
    assert repair.signature({"code": CRASH, "message": "KeyError: 'gold'"}) != repair.signature(
        {"code": CRASH, "message": "TypeError: 'gold'"})


def test_signature_for_pilot_failures_uses_failed_tests():
    s1 = repair.signature({"code": PILOT_FAIL, "message": "pilot tests failed", "failed_tests": ["test_A08_cache::test_cache"]})
    s2 = repair.signature({"code": PILOT_FAIL, "message": "pilot tests failed", "failed_tests": ["test_A14_batch::test_counts"]})
    assert s1 != s2 and s1 == "20:test_A08_cache::test_cache"


def test_transient_classification():
    assert repair.is_transient({"code": CRASH, "message": "ConnectionResetError: peer"})
    assert repair.is_transient(None)
    assert not repair.is_transient({"code": CRASH, "message": "KeyError: 'x'"})
    assert not repair.is_transient({"code": PILOT_FAIL, "message": "x"})


def test_decide_policy():
    lim = {"same_signature_max": 2, "max_per_run": 4, "max_minutes_per_run": 180}
    d = {"events": []}
    assert repair.decide(d, "A", lim)[0]
    repair.record_repair(d, "A", True, "c1", 1, "s")
    assert repair.decide(d, "A", lim)[0]                                  # second look
    repair.record_repair(d, "A", True, "c2", 1, "s")
    ok, why = repair.decide(d, "A", lim)
    assert not ok and "guessing" in why
    assert repair.decide(d, "B", lim)[0]
    repair.record_repair(d, "B", True, "c3", 1, "s")
    repair.record_repair(d, "C", True, "c4", 1, "s")
    ok, why = repair.decide(d, "D", lim)
    assert not ok and "cap" in why


# ---------------------------------------------------------------- coverage
class _Cfg:
    def __init__(self, dims):
        self.plan = {"coverage_dims": dims, "pilot_per_cell": 1, "canary_per_cell": 2}
        self.models = [{"id": "m1", "batch": True}, {"id": "m2"}]

    def model(self, mid):
        return next(m for m in self.models if m["id"] == mid)


class _Ctx:
    def __init__(self, stages, dims=()):
        self.cfg = _Cfg(list(dims))
        self.mode = "full"
        self.paper = type("P", (), {"STAGES": stages})()

    def select(self, cells):
        from rh.context import Context
        return Context.select(self, cells)


def _cells():
    out = []
    for m in ("m1", "m2"):
        for arm in ("zero", "few"):
            for lib in ("netty", "caffeine", "jctools"):
                for i in range(5):
                    out.append({"cell_id": f"{m}|{arm}|{lib}|{i}", "model": m, "arm": arm, "meta": {"library": lib}})
    return out


def test_selection_covers_every_declared_combination():
    from rh.context import Stage
    st = Stage("gen", "llm", build=lambda ctx: _cells(), schema=lambda t: t)
    ctx = _Ctx([st], dims=["library"])
    rep = coverage.report(ctx)
    e = rep["stages"]["gen"]
    assert e["full_combinations"] == 12 and e["pilot"]["combinations"] == 12 and e["pilot"]["units"] == 12
    assert coverage.problems(rep) == []
    ctx2 = _Ctx([st])                                   # library not declared: still covered by model/arm/lane
    assert coverage.report(ctx2)["stages"]["gen"]["full_combinations"] == 4


def test_local_stage_coverage_declared_and_missing_detected():
    from rh.context import Stage

    def cov(ctx, mode):
        libs = ["netty", "caffeine", "jctools", "eclipse"] if mode == "full" else ["jctools"]
        return [{"library": x, "framework": "junit5" if x != "jctools" else "junit4"} for x in libs]
    st = Stage("build", "local", run=lambda ctx: {}, coverage=cov)
    ctx = _Ctx([st])
    rep = coverage.report(ctx)
    probs = coverage.problems(rep)
    assert len(probs) == 1 and "misses 3 of 4" in probs[0] and "framework" in probs[0]
    st2 = Stage("agg", "local", run=lambda ctx: {})
    with pytest.raises(ValueError, match="declares no coverage"):
        coverage.report(_Ctx([st2]))
    st3 = Stage("agg", "local", run=lambda ctx: {}, coverage="derived")
    assert coverage.report(_Ctx([st3]))["stages"]["agg"] == {"coverage": "derived"}


def test_pilot_rung_fails_on_missing_coverage_before_spend(tmp_path):
    dst = tmp_path / ROOT.name
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("runs", "results", "logs", ".data", ".fake_aws",
                                                             "__pycache__", "data"))
    pp = dst / "paper" / "__init__.py"
    pp.write_text(pp.read_text().replace(
        'Stage("aggregate", "local", run=aggregate, gate={"check": _aggregate_check}, coverage="derived"),',
        'Stage("aggregate", "local", run=aggregate, gate={"check": _aggregate_check},\n'
        '          coverage=lambda ctx, mode: [{"unit": u} for u in (["a", "b", "c"] if mode == "full" else ["a"])]),'))
    e = dict(os.environ, RH_ROOT=str(dst), RH_RUN_ID="t", RH_FAKE_AWS_DIR=str(dst / ".fake_aws"))
    r = subprocess.run([sys.executable, "-m", "rh.runner", "--dry-run"], cwd=dst, env=e, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == PILOT_FAIL
    ex = json.loads((dst / "runs" / "exit.json").read_text())
    assert "misses 2 of 3" in ex["message"] and ex["failed_gates"] == ["coverage"]
    led = (dst / "runs" / "ledger.jsonl")
    pilot_calls = [ln for ln in led.read_text().splitlines() if '"mode": "pilot"' in ln] if led.exists() else []
    assert not pilot_calls, "coverage must be checked before any pilot spend"
    assert "Coverage of the full run" in (dst / "runs" / "pilot_report.md").read_text()


# ---------------------------------------------------------------- empty replies
def test_empty_reply_is_its_own_status_and_gate(tmp_path):
    """Every reply of one model empty (compose run 1): status empty_reply, not parse_fail; the ceiling gate fails."""
    dst = tmp_path / ROOT.name
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("runs", "results", "logs", ".data", ".fake_aws",
                                                             "__pycache__", "data"))
    e = dict(os.environ, RH_ROOT=str(dst), RH_RUN_ID="t", RH_FAKE_AWS_DIR=str(dst / ".fake_aws"),
             RH_FAKE_ALWAYS_EMPTY_MODEL="fake-small")
    r = subprocess.run([sys.executable, "-m", "rh.runner", "--dry-run"], cwd=dst, env=e, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == PILOT_FAIL, r.stderr[-1500:]
    rows = [json.loads(x) for x in (dst / "runs" / "pilot" / "label.jsonl").read_text().splitlines()]
    small = [x for x in rows if x["model"] == "fake-small"]
    assert small and all(x["status"] == "empty_reply" for x in small)
    assert all(x["usage"] for x in small), "usage must be kept for empty replies"
    gates = [json.loads(x) for x in (dst / "runs" / "gates.jsonl").read_text().splitlines()]
    g = next(x for x in gates if x["stage"] == "label")
    assert any("empty-reply share" in f for f in g["fails"])
    assert "parse rate" not in " ".join(g["fails"]), "empty replies must not count against the parse rate"


# ---------------------------------------------------------------- heartbeat race
def test_heartbeat_many_threads():
    from rh import heartbeat
    RUNS.mkdir(parents=True, exist_ok=True)
    hb = heartbeat.Heartbeat()
    errors = []

    def work(n):
        try:
            for _ in range(300):
                hb._last_write = 0.0
                hb.progress(lane=f"l{n}")
        except Exception as ex:  # noqa: BLE001
            errors.append(repr(ex))
    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:3]
    assert read_json(heartbeat.PATH)["done"] == 16 * 300


def test_write_json_same_path_from_threads(tmp_path):
    from rh.util import write_json as wj
    p = tmp_path / "x.json"
    errs = []

    def w(i):
        try:
            for k in range(200):
                wj(p, {"i": i, "k": k})
        except Exception as ex:  # noqa: BLE001
            errs.append(repr(ex))
    ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and json.loads(p.read_text())["k"] == 199


# ---------------------------------------------------------------- data pins
def test_zip_deflate64_falls_back_to_unzip(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("tool/bin/run.sh", "x\n")
    arc = tmp_path / "t.zip"
    arc.write_bytes(buf.getvalue())
    real = zipfile.ZipFile.extractall

    def boom(self, *a, **k):
        raise NotImplementedError("compression type 9 (deflate64)")
    monkeypatch.setattr(zipfile.ZipFile, "extractall", boom)
    pin = {"name": "tool", "format": "binary", "url": f"file://{arc}", "record": "r", "file": "t.zip", "title": "t",
           "licence": "MIT", "min_rows": 1, "columns": [], "unpack": "zip",
           "sha256": hashlib.sha256(arc.read_bytes()).hexdigest()}
    if shutil.which("unzip"):
        r = datapins.verify_all([pin])[0]
        assert r["checks"]["unpack_fallback"] == "unzip" and (datapins.path("tool") / "tool" / "bin" / "run.sh").exists()
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(DataFailed, match="unzip"):
        datapins.verify_all([pin])
    monkeypatch.setattr(zipfile.ZipFile, "extractall", real)


def test_git_pin_reports_size(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "build.xml").write_text("<p/>\n")
    (work / "src").mkdir()
    for n in ("A", "B", "C"):
        (work / "src" / f"{n}.java").write_text(f"class {n} {{}}\n")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@x"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=work, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(g + ["commit", "-qm", "c"], cwd=work, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True).stdout.strip()
    bare = tmp_path / "r.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    subprocess.run(["git", "config", "uploadpack.allowReachableSHA1InWant", "true"], cwd=bare, check=True)
    pin = {"name": "subject", "format": "git", "url": f"file://{bare}", "record": sha, "file": "subject", "title": "t",
           "licence": "MIT", "min_rows": 3, "columns": ["build.xml", "src"], "count_glob": "**/*.java"}
    r = datapins.verify_all([pin])[0]
    assert r["ok"] and r["size"] > 0 and r["checks"]["size"] is True


def test_diagnostics_run_before_a_stage_failure(tmp_path):
    dst = tmp_path / ROOT.name
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("runs", "results", "logs", ".data", ".fake_aws",
                                                             "__pycache__", "data"))
    pp = dst / "paper" / "__init__.py"
    src = pp.read_text()
    src = src.replace("def aggregate(ctx)", '''def _diag(ctx, stage):
    from rh.util import RUNS as _R, write_json as _w
    _w(_R / "events.json", {"stage": stage, "note": "partial"})


def aggregate(ctx):
    ctx.register_diagnostics(_diag)
    if ctx.mode == "pilot":
        raise RuntimeError("planted")
    return _aggregate_real(ctx)


def _aggregate_real(ctx)''', 1)
    pp.write_text(src)
    e = dict(os.environ, RH_ROOT=str(dst), RH_RUN_ID="t", RH_FAKE_AWS_DIR=str(dst / ".fake_aws"))
    r = subprocess.run([sys.executable, "-m", "rh.runner", "--dry-run"], cwd=dst, env=e, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == CRASH
    assert json.loads((dst / "runs" / "events.json").read_text())["stage"] == "aggregate"
