"""A20: the judge panel (Haiku, gpt-oss, Sonnet) and the rewrite-validity panel (Sonnet, gpt-oss) return parsed
replies in the pilot."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2b", reason="launch-2 pilot check (run routewise-20261111T034648Z: PASS); study 3 launch 3a has its own pilot checks (D22, A21)")
from conftest import read_jsonl


def test_panels(runs):
    j = read_jsonl(runs / "pilot" / "judge.jsonl")
    for m in ("claude-haiku-5-5", "openai/gpt-oss-120b", "claude-sonnet-5-5"):
        assert any(r["model"] == m and r["status"] == "ok" for r in j), f"no parsed judge reply from {m}"
    v = read_jsonl(runs / "pilot" / "validity.jsonl")
    for m in ("claude-sonnet-5-5", "openai/gpt-oss-120b"):
        assert any(r["model"] == m and r["status"] == "ok" for r in v), f"no parsed validity reply from {m}"
