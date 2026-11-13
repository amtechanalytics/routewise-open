"""Run context and the Stage contract a paper implements.

A paper package (paper/__init__.py) exposes:
  STAGES: list[Stage]                      ordered; kind 'llm' or 'local'
  analyze(ctx) -> None                     writes results/ (tables, figures, audit sheets)
  brief(ctx) -> list[str]                  what needs a human after the run (audits, decisions)
  fake_response(model_id, prefix, item) -> str   deterministic reply used by dry runs
Optional: ctx.register_diagnostics(fn) from a stage, so fn(ctx, stage) writes partial results before a failure.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config as config_mod
from . import vendors
from .heartbeat import Heartbeat
from .ledger import Ledger
from .util import ROOT, RUNS, codename, run_id

MODES = ("pilot", "canary", "full")


@dataclass
class Stage:
    name: str
    kind: str                                   # 'llm' | 'local'
    build: Callable | None = None               # llm: build(ctx) -> list[cell dict]
    schema: Callable | None = None              # llm: schema(text) -> parsed (raise ValueError if invalid)
    run: Callable | None = None                 # local: run(ctx) -> dict of metrics
    gate: dict = field(default_factory=dict)    # min_rows, min_parse_rate, max_duplicate_cells, check(ctx, metrics)
    coverage: Callable | str | None = None      # local: coverage(ctx, mode) -> [dims dict per unit] | "derived"


class Context:
    def __init__(self, mode: str = "full", pins_file: str | None = None, paper_module: str = "paper"):
        self.cfg = config_mod.load(ROOT, pins_file=pins_file)
        self.mode = mode
        self.ledger = Ledger(self.cfg)
        self.run_id = run_id()
        self.codename = self.cfg.codename or codename()
        self.paper = importlib.import_module(paper_module)
        vendors.FAKE_RESPONDER = getattr(self.paper, "fake_response", None)
        self.hb: Heartbeat | None = None
        self._diagnostics: list = []

    # ---- diagnostics written before any failure propagates (1.0.4)
    def register_diagnostics(self, fn) -> None:
        """fn(ctx, stage_name) writes the paper's diagnostic files (events, manifests, partial tables) into runs/.
        Called by the runner when a stage raises, before the backup, so a DEAD results zip carries them."""
        self._diagnostics.append(fn)

    def run_diagnostics(self, stage_name: str) -> None:
        from .util import journal
        for fn in list(self._diagnostics):
            try:
                fn(self, stage_name)
            except Exception as e:  # noqa: BLE001
                journal("DIAG", f"diagnostics writer {getattr(fn, '__name__', fn)} failed: {e}")

    # ---- paths
    def out(self, stage: str) -> Path:
        return RUNS / self.mode / f"{stage}.jsonl"

    def stage_rows(self, stage: str, mode: str | None = None):
        from .util import read_jsonl
        return list(read_jsonl(RUNS / (mode or self.mode) / f"{stage}.jsonl"))

    @property
    def s3_prefix(self) -> str:
        return f"s3://{self.cfg.project['bucket']}/{self.codename}/{self.run_id}"

    # ---- mode subsampling: deterministic, k cells per combination of the coverage dimensions
    # (model, arm, lane and plan.yaml coverage_dims), so the pilot meets every combination of the full run
    def select(self, cells: list[dict]) -> list[dict]:
        if self.mode == "full":
            return cells
        from .coverage import dims_of_cell, key
        k = int(self.cfg.plan["pilot_per_cell" if self.mode == "pilot" else "canary_per_cell"])
        groups: dict[tuple, list] = {}
        for c in cells:
            groups.setdefault(key(dims_of_cell(self, c)), []).append(c)
        out = []
        for _, g in sorted(groups.items()):
            g = sorted(g, key=lambda c: hashlib.sha1(c["cell_id"].encode()).hexdigest())
            out.extend(g[:k])
        return out
