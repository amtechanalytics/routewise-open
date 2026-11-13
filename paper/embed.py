"""Sentence embeddings: OpenAI text-embedding-3-small through the API (512 dimensions, unit length), or a
deterministic hashed bag-of-words stand-in in dry runs (RH_FAKE_VENDOR=1).

Every request goes through the harness's rate limiters (token and request buckets from config/models.yaml),
is checked against the run's budget before it is sent, is recorded in the spend ledger and logged in
logs/api_text-embedding-3-small.jsonl (pilot A07 reads the rate-limit headers there). Long passes are saved in
shards, so a restarted stage continues where it stopped; a throughput guard stops the stage early when the
projected time for one source exceeds MAX_HOURS_PER_SOURCE."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import text as T

MODEL_ID = "text-embedding-3-small"
DIM_API = 512
DIM_FAKE = 384
MAX_INPUTS = 256                 # per request (API limit 2,048 inputs / 300k tokens)
MAX_CHARS = 6000                 # per input, far below the 8,191-token input limit
SHARD = 4096
MAX_HOURS_PER_SOURCE = 10.0          # v2: fin-multiyear decoys (~470k chunks) embed in one pass
GUARD_AFTER_S = 600


def fake() -> bool:
    return os.environ.get("RH_FAKE_VENDOR", "") == "1"


class ThroughputTooLow(RuntimeError):
    pass


class FakeEmbedder:
    """Dry-run stand-in. Logs one exchange per call in the same format as the API embedder (pilot A07 reads it)."""
    name = "fake-hash-bow-384"
    dim = DIM_FAKE

    def __init__(self, ledger=None, cfg=None):
        self.cfg = cfg

    def _log(self, n):
        if self.cfg is None:
            return
        from rh import vendors
        try:
            m = self.cfg.model(MODEL_ID)
        except KeyError:
            return
        vendors._log(m, "index", "embed", f"{n} inputs",
                     vendors.Response("", {"input": 0, "output": 0}, False, MODEL_ID, 0.0,
                                      {"x-ratelimit-remaining-tokens": "1000000"}, {}), None)

    def encode(self, texts, query=False, progress=None, cache_dir: Path | None = None, label: str = ""):
        self._log(len(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in T.content_tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
            if progress and i % 2000 == 0:
                progress(i, len(texts))
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n


class OpenAIEmbedder:
    name = f"openai/{MODEL_ID}@{DIM_API}"
    dim = DIM_API

    def __init__(self, ledger, cfg):
        from rh import vendors
        self.v = vendors
        self.ledger = ledger
        self.model = cfg.model(MODEL_ID)
        self.url = vendors.ENDPOINTS["openai"] + "/embeddings"

    def _request(self, batch: list[str]) -> np.ndarray:
        v = self.v
        est_tokens = sum(len(t) for t in batch) // 3 + len(batch)
        est = self.ledger.estimate(MODEL_ID, est_tokens, 0)
        self.ledger.check(MODEL_ID, est)
        v.rpm_bucket(self.model).acquire(1)
        v.bucket(self.model).acquire(est_tokens)
        payload = json.dumps({"model": MODEL_ID, "input": batch, "dimensions": DIM_API,
                              "encoding_format": "float"}).encode()
        headers = dict(v._headers("openai"))

        def once():
            return v._http("POST", self.url, headers, payload, timeout=300)
        status, hdrs, data, lat = v.retry(once, attempts=8, retry_on=(v.RetryableVendorError,))
        d = json.loads(data)
        vecs = [None] * len(batch)
        for e in d["data"]:
            vecs[e["index"]] = e["embedding"]
        if any(x is None for x in vecs):
            raise ValueError("embedding response is missing inputs")
        used = int((d.get("usage") or {}).get("prompt_tokens") or est_tokens)
        v.bucket(self.model).refund(est_tokens - used)
        usage = {"input": used, "output": 0}
        self.ledger.record(MODEL_ID, usage, stage="index", provider="openai")
        v._log(self.model, "index", "embed", f"{len(batch)} inputs",
               v.Response("", usage, False, MODEL_ID, lat, v._rl_headers(hdrs), {}), None)
        arr = np.asarray(vecs, dtype=np.float32)
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return arr / n

    def encode(self, texts, query=False, progress=None, cache_dir: Path | None = None, label: str = ""):
        texts = [(t or " ")[:MAX_CHARS] for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        t0, done_new = time.time(), 0
        from concurrent.futures import ThreadPoolExecutor
        conc = max(1, int(self.model.get("concurrency", 4)))
        for s0 in range(0, len(texts), SHARD):
            shard = texts[s0:s0 + SHARD]
            sp = (cache_dir / f"shard_{label}_{s0:08d}_{len(shard)}.npy") if cache_dir else None
            if sp is not None and sp.exists():
                out[s0:s0 + len(shard)] = np.load(sp)
                if progress:
                    progress(s0 + len(shard), len(texts))
                continue
            parts = [shard[i:i + MAX_INPUTS] for i in range(0, len(shard), MAX_INPUTS)]
            with ThreadPoolExecutor(max_workers=conc) as ex:
                res = list(ex.map(self._request, parts))
            arr = np.concatenate(res, axis=0) if res else np.zeros((0, self.dim), np.float32)
            out[s0:s0 + len(shard)] = arr
            if sp is not None:
                tmp = sp.with_suffix(".tmp.npy")
                np.save(tmp, arr)
                os.replace(tmp, sp)
            done_new += len(shard)
            if progress:
                progress(s0 + len(shard), len(texts))
            el = time.time() - t0
            if el > GUARD_AFTER_S and done_new:
                left = len(texts) - (s0 + len(shard))
                proj_h = (el + el / done_new * left) / 3600
                if proj_h > MAX_HOURS_PER_SOURCE:
                    raise ThroughputTooLow(f"embedding {label}: {done_new / el:.1f} texts/s, projected {proj_h:.1f} h "
                                           f"for {len(texts)} texts > {MAX_HOURS_PER_SOURCE} h")
        return out


_E: dict = {}


def embedder(ledger=None, cfg=None):
    if "e" not in _E:
        if fake():
            _E["e"] = FakeEmbedder(ledger, cfg)
        else:
            if ledger is None or cfg is None:
                raise RuntimeError("the API embedder needs the run's ledger and config (call embed.init(ctx) first)")
            _E["e"] = OpenAIEmbedder(ledger, cfg)
    elif cfg is not None and getattr(_E["e"], "cfg", cfg) is None:
        _E["e"].cfg = cfg        # created earlier without a config (corpora stage); log from now on
    return _E["e"]


def init(ctx) -> None:
    embedder(ctx.ledger, ctx.cfg)
