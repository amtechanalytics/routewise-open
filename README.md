# routewise — when does document routing improve retrieval-augmented generation?

Study: across six public document collections (annual 10-K filings, FDA drug labels, commercial contracts,
government help pages, NLP papers, news articles) and fifteen controlled configurations (base, identity-key
masking, document dilution, document merging), it measures when routing a question to its document before
chunk retrieval helps, when it does nothing and when it hurts. It compares flat hybrid retrieval, title-scoped
retrieval, LLM-routed retrieval, routing with fallback, extractor-free rule routing and an oracle; computes
corpus and workload properties before any evaluation; fits a leave-one-source-out predictor of the routing gain;
and evaluates a corpus-aware policy that chooses the retrieval strategy for a collection it has not seen.

Stages: `corpora -> index -> route -> retrieve -> generate -> judge -> aggregate`, then the analysis writes
`results/` (tables t1-t10, figures, a stratified human-rating sheet). Retrieval uses OpenAI text-embedding-3-small
(512 dimensions, through the API) with BM25 and reciprocal-rank fusion. Models measured: claude-haiku-5-5 (router, generator, judge),
claude-sonnet-5-5 (generator, one collection), openai/gpt-oss-120b on Groq (second router, generator and judge
on a subset). Datasets are downloaded from their hosts and verified at run time (`config/data_pins.json`); no
dataset content is stored in this repository.

## Current run: study 3, launch 3a

Study 3 tests ROUTEWISE-R end to end on fresh questions that refer to their document only partially (L1) or by
description (L2), after each question has been screened for meaning preservation. Launch 3a (this harness,
`paper/v2c.py`) builds and screens the questions; launch 3b runs the arms on the accepted items.

`corpora -> index3 -> cand3 -> rewrite3a -> rewrite3b -> leak3 -> oret3 -> ogen3 -> ojudge3 -> panel3 -> screen3`,
then the analysis writes `results/s3_*`.

- Fresh pool: per collection, the questions after the first 400 of the fixed order (never used in study 1 or 2);
  screened collections cuad, fda, mhrag and qasper; fin (10-K filings) gets the leak check only.
- L1: a unique partial reference (contract kind and date; a shortened drug or paper name). L2: the frozen rewrite
  prompt (gpt-oss-120b), given the three facts of the document that differ most from its five nearest documents;
  a second attempt with the next three facts when the first fails the leak check.
- Screen: (a) strict name, answer-leak and uniqueness checks; (b) Sonnet and gpt-oss both rate the rewrite as
  answer-preserving, leak-free and plausible; (c) the oracle (gold documents only) answers both the original and
  the rewrite correctly. Accepted items get a seeded dev/test split (at most 100 dev and 400 test per
  collection and level) and are frozen with their sha256 in `results/s3_frozen_items.json`.
- `results/s3_rating_sheet.csv` is the blind sheet for the human check (150 accepted and 50 rejected items, mixed);
  `results/s3_rating_key.csv` holds the screen outcome and stays with the analysis.

## Study 2, launch 2 (run routewise-20261111T034648Z)

Study 2 asks whether a question that refers to its document only by description can be resolved to the right
document from per-document profile cards, and whether doing so beats equally informed baselines. Launch 1 built
the inputs; they are committed under `frozen/` with their sha256 and checked on load. Launch 2 (this harness,
`paper/v2b.py`, analysis in `paper/analysis_v2b.py`):

`corpora -> edgar -> recheck -> index -> route -> rerank -> resolve -> calibrate -> retrieve -> generate -> judge
-> validity`, then the analysis writes `results/v2_*`.

- Workloads: the 300 test questions per source at three levels of identity explicitness (L0 original, L1 partial
  reference, L2 description; after a strict leak re-check), a 10-K collection with two prior fiscal years per
  company as decoys (`finmy`), and two collections whose questions name no document (`qasper`, `md2d`).
- Arms: `flat`, `meta` (title and metadata added to retrieval), `hdrr` (key-extraction routing, else flat),
  `sac` (summary-augmented chunks), `pdoc` (document retrieval over the profile cards), `pdoc-rr` (the cards
  reranked by a cross-encoder, passages from the top five documents), `R` (ROUTEWISE-R: key router, then a
  resolver over the top ten cards, then scoping by calibrated confidence), its ablations `R-uncal`, `R-noans`,
  `R-title`, `R-shuf`, and `oracle` (passages from the gold documents only).
- Calibration: isotonic regression of the resolver's confidence on dev questions of the other sources
  (leave-one-source-out); thresholds by a fixed precision rule.
- Validation: three judge models on a stratified subset, rule-based scores where the gold answer allows, a
  two-model rewrite-validity panel, and a meaning-preservation check (the oracle arm on original vs rewritten
  questions).

The cross-encoder (`BAAI/bge-reranker-v2-m3`) runs on CPU through sentence-transformers; its revision is recorded
in `runs/<mode>/diag_rerank.json`.

Launch 1 ran from `V2A_STAGES` (`paper/v2a.py`).
Study 1 (the routing study described above) ran from `STUDY1_STAGES` in `paper/__init__.py`; its pilot tests
and its ladder test are kept in `study1/` and return in launch 2.

Study tests: `python -m pytest tests/test_paper_v2c.py tests/test_paper_v2b.py tests/test_paper_v2a.py tests/test_paper_routewise.py -q` (or run each file
with `python`). Fixtures: `python tests/fixtures/routewise/make_fixtures.py` rebuilds the byte-stable fixtures.

## Harness

Runs a declared experiment end to end without supervision: pinned datasets are downloaded and
verified, every model and arm is exercised in a small pilot, a canary checks parse rates and spend
on real volume, and only then the full run executes behind stage gates. Spend is checked before
every call, progress is backed up every five minutes, and a watchdog restarts or repairs a stalled
or failed runner.

## Layout
- `paper/` — the study: stages (cells, schema, gates), analysis, brief
- `rh/` — harness core (ladder, ledger, vendors, data pins, gates, monitor, backup, notifications)
- `config/` — project, models, prices, budget, data pins, gates, plan, compute, required IAM actions
- `pilot/` — tests run on the instance in the pilot rung, one per certificate row
- `tests/` — harness tests (run against the frozen demo study in `tests/demo/`) and the study's own tests in `tests/test_paper_*.py`, which load `paper/` directly
- `certificate/generation_certificate.md` — failure-mode register and how each mode is guarded

## Run ladder
`datacheck -> pilot -> canary -> full -> analysis -> finish`; state in `runs/state.json`.

## Pilot coverage
The pilot must meet every combination the full run will meet, so nothing fails for the first time in the
canary or the full run. A cell's dimensions are model, arm, lane (batch or sync) and every key listed under
`coverage_dims` in `config/plan.yaml` (read from the cell's `meta`: library, dataset, length bucket, rung,
...). Pilot and canary selection takes `pilot_per_cell` / `canary_per_cell` cells per *combination*, so
coverage holds by construction for LLM stages. A local stage declares what it covers with
`Stage(..., coverage=fn)`, where `fn(ctx, mode)` returns one dict of dimensions per unit the stage processes
in that mode (for a system study: every component built, every target platform, every test framework,
every input class), or `coverage="derived"` when it only aggregates earlier stages. `python -m rh.runner
--coverage` prints the report; prep L15 and the pilot rung fail on any missing combination before any spend.

## System and tool studies
Studies that build or run a system (a service, a compiler pass, a pipeline) rather than only call models
are ordinary `local` stages: `run(ctx)` builds, deploys or benchmarks and returns metrics, `coverage` names
the units, `gate` checks them, and `ctx.register_diagnostics(fn)` writes partial results (manifests, logs,
tables) before any failure propagates, so a stopped run still delivers what it computed. Long builds set
`expected_hours` and the instance type in `config/compute.yaml`; GPU instances are optional there.

## Failures and repair
A runner failure is reduced to a signature (exit code, exception type, message with ids and numbers
blanked; for pilot failures the failed test ids). Failures that can be transient (network, timeout, a
vanished runner) get a plain restart first; code faults, gate and data failures go straight to the repair
step. A failure not seen before is always repaired; the same failure again gets one more repair with the
instruction to re-diagnose; the same failure a third time stops the run. Caps: `liveness.repair` in
`config/gates.yaml`. History: `runs/repairs.json`. A reply with no content is status `empty_reply` (kept
with its usage, excluded from the parse rate, gated by `empty_reply_ceiling`).

## Checks the generation sandbox could not run
The chat that builds a harness runs every check its sandbox allows. Any prep row or test it could not run
(for example L04 and L14 when pytest is unavailable) is listed in `certificate/sandbox_gaps.json` as
`[{"id": "L04", "why": "..."}]`. Only environmental reasons qualify (a package cannot be installed, no network, AWS,
vendor keys or GitHub, dataset too large); L02 L03 L05–L08 L10–L13 need none of these and must pass in the
sandbox. A Mac failure in a check not listed here is flagged as unexpected in the report. On the Mac, `bash _tools/launch.sh <codename>` runs the full online prep, has
the local repair step fix code failures under the same signature policy as the instance, records every change in
`<codename>-certification-report.md`, and launches only after PREP PASS.

## Local commands
```
python -m rh.prep --offline       # integrity checks, unit tests, dry run on the fake vendor
python -m rh.runner --dry-run     # the whole ladder on the fake vendor and fake storage
python -m rh.runner --coverage    # pilot-coverage report (what prep L15 checks)
python -m rh.notify print-status  # what the 30-minute status mail would say
```

## Environment variables
| Variable | Meaning |
|---|---|
| `RH_ROOT` | repository root (default: current directory) |
| `RH_RUN_ID` | run identifier (set at launch) |
| `RH_FAKE_VENDOR`, `RH_FAKE_AWS` | dry-run fakes |
| `RH_DATA_DIR` | dataset cache directory |
| `RH_TOPIC_ARN` | notification topic (defaults to `config/project.yaml`) |
| `RH_MAX_HOURS` | wall-clock limit on the instance |
| `RH_EXECUTOR_CMD`, `RH_EXECUTOR_RUNBOOK`, `CLAUDE_CODE_OAUTH_TOKEN` | repair executor (instance only; subscription token) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `XAI_API_KEY` (optional) | provider keys (instance only, from the secret store) |
