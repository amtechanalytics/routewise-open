"""Builds the tiny, deterministic fixture copies of every pinned source (same layout and field names as the
real files; parquet replaced by jsonl). Run from the repo root: python tests/fixtures/routewise/make_fixtures.py
Outputs are byte-stable (fixed zip timestamps, sorted keys), so their sha256 can be pinned."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
TS = (2026, 11, 8, 0, 0, 0)
BOILER = ("This report contains forward-looking statements that involve risks and uncertainties. Actual results "
          "may differ materially. The company files annual, quarterly and current reports with the Securities and "
          "Exchange Commission. Investors should read the risk factors section carefully before investing. ")


def zwrite(path: Path, files: dict[str, str]) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=TS)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[name])
    path.write_bytes(buf.getvalue())


def dumps(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False)


def fin():
    cos = [("AAPL", "Apple Inc.", "iPhone", "Cupertino", "394.3 billion"),
           ("MSFT", "Microsoft Corporation", "Azure", "Redmond", "211.9 billion"),
           ("KO", "The Coca-Cola Company", "sparkling beverages", "Atlanta", "45.8 billion"),
           ("TJX", "The TJX Companies, Inc.", "off-price apparel", "Framingham", "54.2 billion")]
    files, qrows, ids = {}, [], []
    for i, (tk, name, prod, city, rev) in enumerate(cos):
        facts = [f"{name} reported total net revenue of ${rev} for the fiscal year, driven by {prod}.",
                 f"The headquarters of {name} is located in {city}, where management oversees {prod} strategy.",
                 f"During the year {name} repurchased shares worth {i + 3}.{i} billion dollars and paid dividends.",
                 f"Cybersecurity risk at {name} is overseen by the audit committee with quarterly reviews of {prod} "
                 f"systems.",
                 f"Capital expenditures of {name} reached {i + 1}.{i + 2} billion dollars, mainly for {prod} capacity."]
        body = (f"{name} (Exact name of registrant as specified in its charter) FORM 10-K annual report "
                f"for the fiscal year ended December 31, {2023 + i % 2}. "
                + BOILER * 3 + " ".join(facts) + " " + BOILER * 4)
        files[f"10-K/{tk}.txt"] = body
        qs = [(f"What was {tk}'s total net revenue and what drove it?", facts[0], rev),
              (f"Where is {name} headquartered, ticker {tk}?", facts[1], city),
              (f"How much stock did {tk} repurchase during the year?", facts[2], f"{i + 3}.{i} billion"),
              (f"Who oversees cybersecurity risk at {name}?", facts[3], "the audit committee"),
              (f"What were the capital expenditures of {tk}?", facts[4], f"{i + 1}.{i + 2} billion")]
        for j, (q, ev, ans) in enumerate(qs):
            qid = f"{i:02x}{j:02x}aa{j:02x}"
            ids.append((qid, q))
            qrows.append({"_id": qid, "text": q, "reasoning": False, "category": "Financials", "references": [ev],
                          "answer": ans, "type": "extractive"})
    zwrite(OUT / "finder_10k.zip", files)
    (OUT / "finder_queries.jsonl").write_text("\n".join(dumps(r) for r in qrows) + "\n")
    hd = {}
    for s in range(1, 6):
        b = io.StringIO()
        w = csv.writer(b, lineterminator="\n")
        w.writerow(["query_id", "query"])
        for k, (qid, q) in enumerate(ids):
            if k % 5 == s - 1 or s == 1:
                w.writerow([qid, q])
        hd[f"random_queries_csv/Linq-AI-Research_FinDER-300-0{s}.csv"] = b.getvalue()
    zwrite(OUT / "hdrr_repo.zip", hd)


def fda():
    drugs = [("Lisinopril", "hypertension", "10 mg once daily", "angioedema"),
             ("Metformin Hydrochloride", "type 2 diabetes", "500 mg twice daily", "lactic acidosis"),
             ("Atorvastatin Calcium", "hyperlipidemia", "10 to 20 mg once daily", "myopathy"),
             ("Amoxicillin", "bacterial infections", "500 mg every 12 hours", "anaphylaxis")]
    labels, qa = [], []
    common = ("INDICATIONS AND USAGE. DOSAGE AND ADMINISTRATION. CONTRAINDICATIONS. WARNINGS AND PRECAUTIONS. "
              "Keep out of reach of children. Store at controlled room temperature. ")
    for i, (d, ind, dose, warn) in enumerate(drugs):
        sec = [f"{d} is indicated for the treatment of {ind} in adults.",
               f"The recommended starting dose of {d} is {dose}.",
               f"{d} carries a warning for {warn}; discontinue if it occurs.",
               f"Pregnancy: {d} should be used only if clearly needed; data are limited.",
               f"Renal impairment: adjust the {d} dose based on creatinine clearance.",
               f"Overdosage of {d}: provide supportive care and monitor vital signs."]
        extra = [f"Clinical Studies: Study {i + 1}01 enrolled {200 + 37 * i} patients at {12 + i} sites in North America.",
                 f"Mechanism of Action: {d} was first approved in {1987 + 3 * i} by the Food and Drug Administration.",
                 f"How Supplied: {d} tablets are packaged in bottles of {30 * (i + 1)} with a Child Resistant cap.",
                 f"Manufactured for Northfield Pharma Company in New Jersey, revised {3 + i}/{2019 + i}."]
        raw = (common * 2 + " ".join(f"||PASSAGE_{k:04d}|| {s}" for k, s in enumerate(sec)) + " " + common * 3
               + " " + " ".join(extra))
        sid = f"set-{i:04d}-fixture"
        labels.append({"set_id": sid, "drug_name": d, "label_raw": raw, "chunks": sec})
        qs = [f"What is {d} indicated for?", f"What is the recommended starting dose of {d}?",
              f"What warning does the {d} label carry?", f"What does the {d} label say about pregnancy?",
              f"How should the {d} dose be adjusted in renal impairment?", f"How is an overdose of {d} managed?"]
        for k, q in enumerate(qs):
            qa.append({"task": "factual", "question": q, "answer": sec[k], "set_id": sid, "drug_name": d,
                       "context": [{"doc_chunk_index": k, "section_title": "S", "text": sec[k]}],
                       "qid": f"fx{i}{k}"})
        qa.append({"task": "refusal", "question": f"What is the price of {d}?", "answer": "", "set_id": sid,
                   "drug_name": d, "context": [], "qid": f"fx{i}r"})
    zwrite(OUT / "fdarxbench.zip", {"data/labels/labels.jsonl": "\n".join(dumps(r) for r in labels) + "\n",
                                    "data/qa/qa.jsonl": "\n".join(dumps(r) for r in qa) + "\n",
                                    "data/qa/qa_toy.jsonl": dumps(qa[0]) + "\n"})


def cuad():
    contracts = [("LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT", "Illinois", "five years", "30 days"),
                 ("ACMEROBOTICSINC_01_15_2015-EX-10.2-SUPPLY AGREEMENT", "Delaware", "three years", "60 days"),
                 ("BLUESKYMEDIA_06_30_2018-EX-10.1-LICENSE AGREEMENT", "New York", "two years", "90 days"),
                 ("NORTHWINDFOODSLLC_11_02_2012-EX-10.4-FRANCHISE AGREEMENT", "Texas", "ten years", "45 days")]
    cats = [("Governing Law", "Which state/country's law governs the interpretation of the contract?", 0),
            ("Expiration Date", "On what date will the contract's initial term expire?", 1),
            ("Notice Period To Terminate Renewal", "What is the notice period required to terminate renewal?", 2),
            ("Parties", "The two or more parties who signed the contract", None),
            ("Audit Rights", "Does a party have the right to audit the books of the other party?", 3),
            ("Cap On Liability", "Does the contract cap liability upon breach?", 4),
            ("Insurance", "Is there a requirement for insurance?", 5)]
    boiler = ("WHEREAS the parties wish to enter into this agreement; NOW THEREFORE in consideration of the mutual "
              "covenants the parties agree as follows. Capitalized terms have the meanings set out herein. ")
    out = {"test.json": [], "train_separate_questions.json": []}
    for n, (title, law, term, notice) in enumerate(contracts):
        clauses = [f"This Agreement shall be governed by the laws of the State of {law}.",
                   f"The initial term of this Agreement is {term} from the Effective Date.",
                   f"Either party may terminate renewal by giving {notice} written notice.",
                   f"Licensor may audit the books and records of the other party once per year (contract {n}).",
                   f"Liability of each party is capped at the fees paid in the prior {n + 6} months.",
                   f"Each party shall maintain commercial general liability insurance of {n + 1} million dollars."]
        extra = [f"Schedule A lists {n + 3} products to be sold in the Midwest Region during {2001 + n}.",
                 f"Notices shall be sent to the Chief Financial Officer at {100 + n} Main Street, Springfield.",
                 f"The Distributor shall purchase at least {5000 * (n + 1)} units in each Contract Year."]
        ctx = boiler * 3 + " ".join(clauses) + " " + boiler * 3 + " ".join(extra)
        qas = []
        for c, (cat, det, k) in enumerate(cats):
            ans = [{"text": clauses[k], "answer_start": ctx.find(clauses[k])}] if k is not None else \
                [{"text": "LIME ENERGY CO.", "answer_start": 0}]
            qas.append({"question": f'Highlight the parts (if any) of this contract related to "{cat}" that should '
                                    f"be reviewed by a lawyer. Details: {det}", "id": f"{title}__{cat}",
                        "answers": ans, "is_impossible": False})
        art = {"title": title, "paragraphs": [{"context": ctx, "qas": qas}]}
        out["test.json" if n % 2 == 0 else "train_separate_questions.json"].append(art)
    zwrite(OUT / "cuad.zip", {k: dumps({"version": "aok_v1.0", "data": v}) for k, v in out.items()})


def md2d():
    pages = [("ssa", "Apply for Retirement Benefits | Social Security Administration#1_0",
              ["You can apply for retirement benefits online up to four months before you want them to start.",
               "You must be at least 61 years and 9 months old to apply.",
               "Have your bank account number ready for direct deposit."]),
             ("va", "VA Disability Compensation | Veterans Affairs#2_0",
              ["Disability compensation is a tax-free monthly payment for veterans.",
               "You may be eligible if you have a service-connected condition.",
               "File a claim online or by mail using form 21-526EZ."]),
             ("dmv", "Renew Your Driver License | DMV#3_0",
              ["You can renew your driver license online if it expires within 90 days.",
               "Bring proof of identity if you renew in person.",
               "The renewal fee is 35 dollars for most licenses."]),
             ("studentaid", "Repaying Student Loans | Federal Student Aid#4_0",
              ["Repayment begins six months after you graduate or leave school.",
               "Income-driven repayment plans base payments on your income.",
               "You can postpone payments with deferment or forbearance."])]
    doc_data, dial = {}, {}
    common = "This page explains your options. Contact us if you need help. Sign in to your account to continue. "
    for dom, title, sents in pages:
        text = common + " ".join(sents) + " " + common
        spans = {}
        for k, s in enumerate(sents):
            st = text.find(s)
            spans[str(k + 1)] = {"id_sp": str(k + 1), "tag": "u", "start_sp": st, "end_sp": st + len(s),
                                 "text_sp": s, "title": title, "parent_titles": [], "id_sec": "1",
                                 "start_sec": 0, "text_sec": text, "end_sec": len(text)}
        doc_data.setdefault(dom, {})[title] = {"title": title, "doc_text": text, "spans": spans,
                                               "doc_html_ts": "", "doc_html_raw": ""}
        qs = [("When can I apply?", 1), ("What age do I need to be or what qualifies me?", 2),
              ("What should I have ready or what is the fee?", 3)]
        for k, (q, sp) in enumerate(qs):
            dial.setdefault(dom, []).append({"dial_id": f"{dom}-{k}", "turns": [
                {"turn_id": 1, "role": "user", "da": "query", "utterance": f"{q} ({title.split('|')[0].strip()})",
                 "references": [{"id_sp": str(sp), "label": "precondition", "doc_id": title}]},
                {"turn_id": 2, "role": "agent", "da": "respond", "utterance": sents[sp - 1], "references": []}]})
    zwrite(OUT / "md2d.zip", {"multidoc2dial/multidoc2dial_doc.json": dumps({"doc_data": doc_data}),
                              "multidoc2dial/multidoc2dial_dial_test.json": dumps({"dial_data": dial}),
                              "multidoc2dial/multidoc2dial_dial_validation.json": dumps({"dial_data": {}})})


def qasper():
    papers = [("p1", "Neural Lineage Extraction", "We study lineage extraction with transformers.",
               ["We use the OmniMart dataset with 206 gold edges.", "Our model reaches an F1 of 0.95.",
                "Training took four hours on one GPU."]),
              ("p2", "Dialogue State Tracking with Retrieval", "We propose retrieval for dialogue state tracking.",
               ["Experiments use MultiWOZ 2.1.", "Joint goal accuracy improves by 3 points.",
                "We release code under an MIT licence."]),
              ("p3", "Low-Resource Machine Translation", "We translate between Nepali and English.",
               ["The corpus contains 60 thousand sentence pairs.", "BLEU improves from 7.1 to 9.4.",
                "Back-translation was the most useful technique."]),
              ("p4", "Sarcasm Detection in Tweets", "We detect sarcasm in tweets with context.",
               ["The dataset has 5,000 annotated tweets.", "Context features add 4 points of F1.",
                "Annotators agreed with a kappa of 0.71."])]
    rows = []
    for pid, title, abstract, paras in papers:
        qs = ["What dataset is used?", "What is the main result?", "What else is reported?"]
        rows.append({"id": pid, "title": title, "abstract": abstract,
                     "full_text": {"section_name": ["Experiments"], "paragraphs": [paras]},
                     "qas": {"question": qs, "question_id": [f"{pid}q{k}" for k in range(3)],
                             "answers": [{"answer": [{"unanswerable": False, "extractive_spans": [paras[k]],
                                                      "yes_no": None, "free_form_answer": "",
                                                      "evidence": [paras[k]], "highlighted_evidence": [paras[k]]}],
                                          "annotation_id": ["a"], "worker_id": ["w"]} for k in range(3)]}})
    (OUT / "qasper_test.jsonl").write_text("\n".join(dumps(r) for r in rows) + "\n")


def mhrag():
    arts = [("Chip shortage eases", "TechCrunch", "2023-10-02", "The chip shortage eased as fabs added capacity."),
            ("New phone launched", "TechCrunch", "2023-10-05", "A new phone launched with a faster processor."),
            ("Rates held steady", "Reuters", "2023-11-01", "The central bank held interest rates steady."),
            ("Oil prices climb", "Reuters", "2023-11-03", "Oil prices climbed four percent this week."),
            ("Team wins final", "ESPN", "2023-12-10", "The team won the final by two goals."),
            ("Coach resigns", "ESPN", "2023-12-12", "The coach resigned after the season ended.")]
    corpus = [{"title": t, "author": "staff", "source": s, "published_at": d + "T00:00:00+00:00", "category": "news",
               "url": f"https://example.org/{k}", "body": b + " " + ("Reporting by staff. Editing by desk. " * 30)
               + f" The report cited {k + 2} independent analysts based in New York and San Francisco offices. Shares of "
                 f"Northwind Corp rose {k + 1} percent on {d} after the quarterly filing was released. A spokesperson "
                 f"for Global Partners declined to comment on the story about {t.lower()} this week."}
              for k, (t, s, d, b) in enumerate(arts)]
    qs = []
    for k, (t, s, d, b) in enumerate(arts):
        other = arts[(k + 2) % len(arts)]
        qs.append({"query": f"According to {s}, what happened in the article '{t}', and how does it relate to "
                            f"the report from {other[1]}?", "answer": b, "question_type": "inference_query",
                   "evidence_list": [{"title": t, "author": "staff", "url": "", "source": s, "category": "news",
                                      "published_at": d, "fact": b},
                                     {"title": other[0], "author": "staff", "url": "", "source": other[1],
                                      "category": "news", "published_at": other[2], "fact": other[3]}]})
    qs.append({"query": "Which article mentions aliens?", "answer": "Insufficient information.",
               "question_type": "null_query", "evidence_list": []})
    qs.append({"query": "Did TechCrunch report that the chip shortage eased before Reuters reported rates held?",
               "answer": "Yes", "question_type": "temporal_query",
               "evidence_list": [{"title": arts[0][0], "source": "TechCrunch", "published_at": arts[0][2],
                                  "fact": arts[0][3]},
                                 {"title": arts[2][0], "source": "Reuters", "published_at": arts[2][2],
                                  "fact": arts[2][3]}]})
    (OUT / "mhrag_corpus.json").write_text(dumps(corpus))
    (OUT / "mhrag_queries.json").write_text(dumps(qs))


if __name__ == "__main__":
    fin()
    fda()
    cuad()
    md2d()
    qasper()
    mhrag()
    print("fixtures written to", OUT)
