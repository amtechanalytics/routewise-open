"""D12: the embedding model answers through the API with 512-dimensional unit vectors, is stable across two calls
and ranks a related passage above an unrelated one; the index stage recorded its spend in the ledger."""

import pytest  # noqa: E402

import paper as _paper  # noqa: E402

pytestmark = pytest.mark.skipif(getattr(_paper, "STUDY", "study1") != "study1", reason="Study 1 pilot check (run routewise-20261110T035239Z); v2 has its own pilot checks (D20-D21, A17-A20)")

import json

import numpy as np

from conftest import fake


def test_embedder(cfg, runs):
    from rh.ledger import Ledger
    from paper import embed
    embed._E.clear()
    e = embed.embedder(Ledger(cfg), cfg)
    docs = ["The recommended starting dose of lisinopril is 10 mg once daily.",
            "The team won the championship final by two goals in extra time."]
    a = e.encode(docs)
    b = e.encode(docs)
    assert a.shape == (2, e.dim)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)
    assert float(np.min(np.sum(a * b, axis=1))) > 0.995
    q = e.encode(["What dose of lisinopril should a patient start on?"], query=True)[0]
    assert float(a[0] @ q) > float(a[1] @ q)
    if not fake():
        assert e.dim == 512
        spend = json.loads((runs / "spend.json").read_text())
        assert spend["by_model"].get("text-embedding-3-small", 0) > 0, "index stage recorded no embedding spend"
