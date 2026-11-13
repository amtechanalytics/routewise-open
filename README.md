# ROUTEWISE: resolving implicit document references in retrieval-augmented generation

Replication package for *Resolving implicit document references in retrieval-augmented generation: a cross-domain study of retrieval benefits and benchmark validity*.

It contains the experiment harness, the result tables behind every number, table and figure in the paper, the frozen derived artifacts (profile cards, rewrites, screened Study 3 items), the two human rating sheets, and the analysis scripts.

## Layout
| Path | Contents |
|---|---|
| `paper/` | Harness code: loaders, chunking and indexing, retrieval policies, ROUTEWISE-R (cards, resolver, calibration, decision rule), generation, judging, analysis |
| `config/` | Models, prices, budget, plan and pinned data sources (`data_pins.json`) |
| `tests/` | Unit and pipeline tests |
| `analysis/` | Stand-alone scripts that regenerate the figures and the added checks from `results/` |
| `results/` | Result tables and frozen artifacts (see the map below) |
| `results/human/` | Study 3 rating key and the two raters' sheets |

## Reproduce the figures and checks (no API keys needed)
```
python3 -m pip install matplotlib
python3 analysis/make_figures.py          # writes figures/fig1..fig4 (.pdf, .png)
python3 analysis/grouped_correlation.py   # Study 1 correlation with collection-grouped checks
python3 analysis/rater_agreement.py       # Study 3 validity, inter-rater and rater-vs-panel agreement
```

## Where each result comes from
| Paper item | File(s) in `results/` |
|---|---|
| Collections (Table 3) | `t1_configs.csv` |
| Study 1 correctness and document hit (Table 5) | `t2_arms.csv` |
| Study 1 contrasts, correlations, held-out prediction, policy selection | `t3_contrasts.csv`, `deltas.json`, `t4_feature_corr.csv`, `t5_loso.csv`, `t6_policy*.csv` |
| Study 1 judge and generator robustness | `t7_judges.csv`, `t7_generators.csv`, `t7_aa_replicate.csv`, `t8_replication.csv` |
| Study 1 router resolution and spend | `t9_router.csv`, `t10_spend.csv` |
| Fig. 1 | `t1_configs.csv` + `deltas.json` |
| Study 2 per-arm results (Tables 4, 6, 7, 10; Figs. 2–3) | `v2_t2_arms.csv` |
| Study 2 hypotheses R1–R6, S1–S5 | `v2_hypotheses.json`, `v2_summary.json` |
| Study 2 contrasts, calibration, decisions, judges, meaning preservation, validity sensitivity, spend | `v2_t4_contrasts_L2.csv`, `v2_t9_calibration.csv`, `v2_t10_decisions.csv`, `v2_t7_judges.csv`, `v2_t8_meaning_preservation.csv`, `v2_t6_validity.csv`, `v2_t5_gap_recovered.csv`, `v2_t11_spend.csv` |
| Profile cards and rewrites | `v2a_cards.jsonl`, `v2a_rewrites.jsonl`, `v2a_card_summary.csv`, `v2a_rewrite_summary.csv`, `v2a_frozen_artifacts.json`, `v2a_edgar_manifest.json` |
| Study 3 screening yields and panel agreement (Table 8) | `s3_yields.csv`, `s3_panel_agreement.csv`, `s3_items.jsonl`, `s3_frozen_items.json` |
| Study 3 human check (Table 9, Fig. 4) | `human/s3_rating_key.csv`, `human/s3_rating_sheet_rater1.csv`, `human/s3_rating_sheet_rater2.csv` |

`results/SHA256SUMS` lists a checksum for every file in `results/`.

## Data sources and licences
The document collections are **not** redistributed here. The harness downloads each one from the pinned URL in `config/data_pins.json`:

| Collection | Source (pinned) | Licence |
|---|---|---|
| 10-K filings and questions (FinDER) | https://huggingface.co/datasets/Linq-AI-Research/FinDER/resolve/c4c1b6454aef7f0bb1c37235c7f52ce644642da0/data/train-00000-of-00001.parquet and `10-k.zip` at the same revision | CC BY-NC 4.0 |
| FDA prescription drug labels (FDARxBench) | https://github.com/xiongbetty/FDARxBench.git | Apache-2.0 (repository); U.S. government labels |
| Commercial contracts (CUAD) | https://github.com/TheAtticusProject/cuad/raw/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2/data.zip | CC BY 4.0 |
| News articles (MultiHop-RAG) | https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/corpus.json | ODC-BY |
| NLP research papers (Qasper, test split) | https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/test/0000.parquet | CC BY 4.0 |
| U.S. government help pages (MultiDoc2Dial) | https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip | CC BY 3.0 per the dataset card's licensing section (its header lists Apache-2.0) |
| Earlier-year 10-K filings (fin-multiyear decoys) | SEC EDGAR; filing list in `v2a_edgar_manifest.json` | Public SEC filings |

Derived files keep the attribution and terms of their source collection. Please cite the original datasets when you use them.

**FinDER rows are withheld.** FinDER is licensed for non-commercial use only. To avoid redistributing its text, every row for the 10-K collections (`fin`, `finmy`) was removed from the three text-bearing files:

| File | Rows kept | FinDER rows removed |
|---|---|---|
| `v2a_cards.jsonl` | 2,680 | 1,386 |
| `v2a_rewrites.jsonl` | 3,600 | 1,800 |
| `s3_items.jsonl` | 939 | 738 |

The aggregate tables still report the 10-K results. The checksums in `v2a_frozen_artifacts.json` and `s3_frozen_items.json` refer to the complete frozen files used in the runs, so they do not match these filtered copies; `SHA256SUMS` covers the copies published here. The removed rows can be regenerated from the pinned FinDER release with the harness.

## Human ratings
Two independent raters each rated the same blind sample of 200 Study 3 questions (150 accepted by the screen, 50 rejected) on four yes/no criteria, without seeing each other's ratings.
- `s3_rating_sheet_blank.csv` is the sheet as given to the raters.
- `s3_rating_key.csv` gives each item's screening outcome and the panel models' votes.
- The raters' free-text comments are not published.

## Re-running the experiments
A full re-run calls paid model APIs and needs provider API keys; `config/models.yaml` and `config/prices.yaml` list the models and prices. Studies 1, 2 and 3 cost about USD 16, 35 and 12 in API calls. The harness runs a pilot and a canary stage and enforces a budget ceiling before any full stage. Code paths for later stages that the paper does not report (for example, a confirmatory Study 3 run that was not carried out) remain in `paper/` but have no results here.

The frozen inputs in `frozen/` are the same filtered copies as in `results/` (FinDER rows removed), so the harness's checksum check on `frozen/s3_items.jsonl.gz` will fail on this public copy by design.
