"""A21: every model of launch 3a answers in the pilot (gpt-oss rewrites at both attempts, Haiku oracle answers and
judgments, Sonnet and gpt-oss validity votes), the leak check and the oracle retrieval ran, and the screen wrote
its frozen item file, yields and the blind rating sheet with its key."""
import csv
import json

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "") != "v2c", reason="study 3 launch 3a pilot check")

from conftest import read_jsonl  # noqa: E402


def _ok(rows, model):
    return any(r["model"] == model and r["status"] == "ok" for r in rows)


def test_screen_pipeline(runs):
    p = runs / "pilot"
    assert _ok(read_jsonl(p / "rewrite3a.jsonl"), "openai/gpt-oss-120b"), "no parsed first-attempt rewrite"
    assert _ok(read_jsonl(p / "rewrite3b.jsonl"), "openai/gpt-oss-120b"), "no parsed second-attempt rewrite"
    assert _ok(read_jsonl(p / "ogen3.jsonl"), "claude-haiku-5-5"), "no parsed oracle answer"
    assert _ok(read_jsonl(p / "ojudge3.jsonl"), "claude-haiku-5-5"), "no parsed oracle judgment"
    v = read_jsonl(p / "panel3.jsonl")
    for m in ("claude-sonnet-5-5", "openai/gpt-oss-120b"):
        assert _ok(v, m), f"no parsed validity vote from {m}"
    its = read_jsonl(p / "items3.jsonl")
    assert {r["level"] for r in its} >= {"L0", "L1", "L2"}, "leak3 wrote no L1 or L2 items"
    d = json.loads((p / "diag_oret3.json").read_text())
    assert d["items"] >= 1
    s3 = p / "s3"
    fz = json.loads((s3 / "s3_frozen_items.json").read_text())
    assert len(fz["s3_items.jsonl"]["sha256"]) == 64
    with open(s3 / "s3_rating_sheet.csv") as f:
        head = next(csv.reader(f))
    assert head[:4] == ["item", "collection", "original_question", "rewritten_question"]
    assert "screened" not in head, "the rating sheet must not reveal the screen outcome"
    with open(s3 / "s3_rating_key.csv") as f:
        assert "screened" in next(csv.reader(f))
