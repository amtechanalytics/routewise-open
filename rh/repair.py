"""Repair policy (rhkit 1.0.4). Which failures the repair step may work on, and when to stop.

A failure is reduced to a *signature*: exit code, exception type, the message with run ids, paths, numbers and
hashes blanked, and for gate failures the sorted names of the failed pilot tests or gates. The watchdog keeps
every failure and every repair in runs/repairs.json and decides from that history:

- a signature never seen before always gets a repair (there is no count of "different problems fixed");
- the same signature right after a repair gets `same_signature_max - 1` more tries (default: one), and the
  repair prompt is told that the previous fix changed nothing, so it re-diagnoses instead of re-applying;
- the same signature once more stops the run (the repair step is guessing);
- a signature that comes back after other signatures were fixed (A, B, A) counts as a repeat of A;
- hard caps: `max_per_run` repairs and `max_minutes_per_run` of repair time, then stop.

Failures that cannot be transient (code faults, gate failures, data failures) go straight to the repair step;
only network errors, timeouts and a vanished runner get the plain restart first."""
from __future__ import annotations

import re

from .exitcodes import CRASH, DATA_FAIL, GATE_FAIL, PILOT_FAIL, CANARY_FAIL
from .util import RUNS, read_json, write_json

PATH = RUNS / "repairs.json"

_TRANSIENT = ("ConnectionError", "ConnectionResetError", "RemoteDisconnected", "TimeoutError", "timed out",
              "URLError", "IncompleteRead", "BrokenPipeError", "Temporarily unavailable", "503", "502", "429",
              "MemoryError", "Killed")
_CODE_FAULT = ("TypeError", "KeyError", "NameError", "AttributeError", "ImportError", "ModuleNotFoundError",
               "SyntaxError", "IndexError", "ValueError", "AssertionError", "ZeroDivisionError",
               "UnboundLocalError", "NotImplementedError", "JSONDecodeError", "FileNotFoundError",
               "PermissionError", "IsADirectoryError", "NotADirectoryError")


def normalize(msg: str) -> str:
    m = msg or ""
    m = re.sub(r"[A-Za-z0-9_-]+-\d{8}T\d{6}Z", "<run>", m)           # run ids
    m = re.sub(r"\b[0-9a-f]{12,64}\b", "<hash>", m)
    m = re.sub(r"/[^\s'\"]+", "<path>", m)
    m = re.sub(r"\b\d+(\.\d+)?\b", "<n>", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m[:300]


def signature(exitinfo: dict | None) -> str:
    e = exitinfo or {}
    code = e.get("code")
    if code is None:
        return "vanished"
    msg = str(e.get("message") or "")
    if code in (PILOT_FAIL, CANARY_FAIL, GATE_FAIL, DATA_FAIL):
        failed = e.get("failed_tests") or e.get("failed_gates") or []
        if failed:
            return f"{code}:" + ",".join(sorted(str(x) for x in failed))
        return f"{code}:" + normalize(msg)
    exc = msg.split(":", 1)[0].strip() if ":" in msg else msg[:40]
    return f"{code}:{exc}:{normalize(msg)}"


def is_transient(exitinfo: dict | None) -> bool:
    """Only these get a plain restart before the repair step."""
    e = exitinfo or {}
    code = e.get("code")
    if code is None:
        return True                                   # runner vanished: a restart is the right first move
    if code != CRASH:
        return False                                  # gate/data failures are never transient
    msg = str(e.get("message") or "")
    if any(t in msg for t in _TRANSIENT):
        return True
    if any(msg.startswith(c) for c in _CODE_FAULT):
        return False
    return True                                       # unknown exception: one restart, then repair


def load() -> dict:
    return read_json(PATH, default={"events": []}) or {"events": []}


def save(d: dict) -> None:
    write_json(PATH, d)


def record_failure(d: dict, sig: str, exitinfo: dict | None, rung: str | None) -> None:
    d["events"].append({"kind": "failure", "sig": sig, "rung": rung, "code": (exitinfo or {}).get("code"),
                        "message": str((exitinfo or {}).get("message") or "")[:400]})


def record_repair(d: dict, sig: str, ok: bool, commit: str | None, minutes: float, summary: str) -> None:
    d["events"].append({"kind": "repair", "sig": sig, "ok": ok, "commit": commit, "minutes": round(minutes, 1),
                        "summary": summary[:600]})


def decide(d: dict, sig: str, lim: dict) -> tuple[bool, str]:
    """(allowed, reason). Call after record_failure for this failure."""
    repairs = [e for e in d["events"] if e["kind"] == "repair"]
    if len(repairs) >= int(lim.get("max_per_run", 12)):
        return False, f"{len(repairs)} repairs in this run (cap {lim.get('max_per_run', 12)})"
    spent = sum(e.get("minutes", 0) for e in repairs)
    if spent >= float(lim.get("max_minutes_per_run", 180)):
        return False, f"{spent:.0f} min of repair time in this run (cap {lim.get('max_minutes_per_run', 180)})"
    same = [e for e in repairs if e["sig"] == sig]
    if len(same) >= int(lim.get("same_signature_max", 2)):
        return False, (f"the same failure has already been repaired {len(same)} time(s) without effect "
                       "(the repair step is guessing)")
    return True, ("new failure" if not same else f"same failure as repair {len(same)}; second look")


def prior_attempts(d: dict, sig: str) -> list[dict]:
    return [e for e in d["events"] if e["kind"] == "repair" and e["sig"] == sig]


def history_text(d: dict) -> str:
    lines = []
    for e in d["events"][-24:]:
        if e["kind"] == "failure":
            lines.append(f"- FAILURE  rung={e.get('rung')} code={e.get('code')} sig={e['sig'][:120]}")
        else:
            lines.append(f"- REPAIR   {'ok' if e['ok'] else 'no fix'} commit={e.get('commit')} "
                         f"{e.get('minutes')} min: {e.get('summary', '')[:200]}")
    return "\n".join(lines) or "- none"
