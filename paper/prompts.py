"""Prompts (byte-stable prefixes; only the item varies), reply schemas and the dry-run stand-ins."""
from __future__ import annotations

import hashlib
import json
import re

from . import text as T

ROUTER_PREFIX = (
    "You route questions to documents in a document collection. The user message describes the collection and "
    "gives one question. Decide whether the question names or clearly identifies one specific document (or a "
    "small set of documents) of that collection, and extract the identifying key exactly as written in the "
    "question.\n"
    "Rules:\n"
    "- entity: the identifying name as it appears in the question (company name or ticker, drug name, contracting "
    "party, page title, paper title, publisher). Copy the words from the question; do not expand abbreviations "
    "and do not guess a name the question does not contain. Use null when the question identifies no document.\n"
    "- qualifier: an additional key that narrows the document, copied from the question (a year, a date, an "
    "agreement type, a version), or null.\n"
    "- confidence: your probability, between 0 and 1, that the entity is correct and identifies the intended "
    "document(s).\n"
    "Answer with one JSON object and nothing else, of the form "
    '{"entity": <string or null>, "qualifier": <string or null>, "confidence": <number between 0 and 1>}. '
    "No markdown, no code fences, no explanation.\n"
)

GEN_PREFIX = (
    "You answer questions about documents using only the numbered context passages in the user message. Each "
    "passage starts with the title of the document it comes from.\n"
    "Rules:\n"
    "- Use only facts stated in the passages. Do not use outside knowledge.\n"
    "- If the passages do not contain the information needed, set found to false and answer exactly "
    '"NOT FOUND".\n'
    "- Keep the answer short and specific: at most 120 words; quote figures, names and dates exactly as "
    "written.\n"
    "- When the question asks about one document and the passages come from several, answer from the passages "
    "of the document the question is about.\n"
    "Answer with one JSON object and nothing else, of the form "
    '{"answer": "<your answer>", "found": <true or false>}. No markdown, no code fences, no explanation.\n'
)

JUDGE_PREFIX = (
    "You grade an answer to a question against a reference answer and the gold evidence it came from. Judge "
    "factual agreement with the reference and the evidence, not style or length.\n"
    "Score:\n"
    "5 = fully correct and complete; 4 = correct, minor omission; 3 = partly correct; 2 = mostly wrong or "
    "missing the key fact; 1 = wrong, unsupported, or NOT FOUND while the evidence contains the answer.\n"
    "correct = true when the score is 4 or 5.\n"
    "category: one of correct, partial, wrong_fact, not_found, wrong_document (the answer is about a different "
    "entity than the question), unsupported (claims not in the evidence).\n"
    "Answer with one JSON object and nothing else, of the form "
    '{"score": <integer 1-5>, "correct": <true or false>, "category": "<category>"}. No markdown, no code '
    "fences, no explanation.\n"
)

CARD_PREFIX = (
    "You write a profile card for one document of a document collection. The user message describes the "
    "collection, gives the document's catalog title and an excerpt of the document (its opening part and a "
    "part from the middle). The card will be used to recognise the document when someone describes it "
    "without naming it.\n"
    "Rules:\n"
    "- name: the document's canonical name (company, drug, contract and parties, article or paper title).\n"
    "- aliases: up to 5 other names the document or its subject is known by (short forms, tickers, brand or "
    "generic names), only if they appear in the excerpt or the title.\n"
    "- doc_type: a short description of the kind of document.\n"
    "- date: the document's date or reporting period as written, or null.\n"
    "- entities: up to 6 parties, organisations, products or people the document is about.\n"
    "- facts: up to 8 distinguishing facts about the document, each with quote = a passage of 15 to 200 "
    "characters copied exactly, character for character, from the excerpt that states the fact.\n"
    "- summary: at most 80 words describing what the document is and what it covers.\n"
    "Use only the excerpt and the title. Answer with one JSON object and nothing else, of the form "
    '{"name": "<string>", "aliases": ["<string>", ...], "doc_type": "<string>", "date": <string or null>, '
    '"entities": ["<string>", ...], "facts": [{"fact": "<string>", "quote": "<exact excerpt passage>"}, ...], '
    '"summary": "<string>"}. No markdown, no code fences, no explanation.\n'
)

REWRITE_PREFIX = (
    "You rewrite a question so that it refers to a document by description instead of by name. The user "
    "message gives a question in which each reference to a document has been replaced by a placeholder such "
    "as \u27e8D1\u27e9, and, for each placeholder, the kind of document and a numbered list of facts taken "
    "from that document.\n"
    "Rules:\n"
    "- Replace each placeholder with a short descriptive phrase (at most 30 words) that identifies the "
    "document through one to three of its listed facts, so that a reader who knows the collection could tell "
    "which document is meant.\n"
    "- Keep every other word of the question unchanged; adjust only articles or prepositions next to the "
    "placeholder so the sentence reads naturally.\n"
    "- Do not use any of the forbidden names, any abbreviation or part of them, or any other name of the "
    "document's subject.\n"
    "- Do not state or hint at the answer; never use the text listed under Do not include.\n"
    "Answer with one JSON object and nothing else, of the form "
    '{"question": "<the rewritten question>", "facts_used": [<fact numbers>]}. No markdown, no code fences, no '
    "explanation.\n"
)

RESOLVER_PREFIX = (
    "You decide which document of a collection a question is about. The user message describes the collection, "
    "gives the question and a numbered list of candidate documents, each described by its profile card. The "
    "question may name its document, refer to it only by description, or concern several documents.\n"
    "Rules:\n"
    "- Choose the candidate or candidates (at most 3) that the question is about, best first. Choose none if no "
    "candidate fits.\n"
    "- For each chosen document give your confidence (0.0 to 1.0) that it is one the question is about.\n"
    "- Choose several documents only when the question compares or combines documents.\n"
    "Answer with one JSON object and nothing else, of the form "
    '{"documents": [{"id": <candidate number>, "confidence": <number>}, ...]}'
    ". Use an empty list when no candidate fits. No markdown, no code fences, no explanation.\n"
)

VALIDITY_PREFIX = (
    "You check a rewritten question. The user message gives an original question about a document, a rewritten "
    "version that should refer to the same document without naming it (or by a shortened name), the reference "
    "answer to the original question, and the title and a short summary of the document.\n"
    "Judge four criteria, each true or false:\n"
    "- same_answer: the reference answer answers the rewritten question exactly as it answers the original.\n"
    "- identifiable: someone who knew the collection could tell from the rewritten question which document is "
    "meant; a description that fits many documents is false.\n"
    "- no_leak: the rewritten question does not reveal the answer and, if it is a description, does not contain "
    "a name of the document or of its subject (company, ticker, drug, party, outlet).\n"
    "- plausible: a real user could ask the question this way.\n"
    "Answer with one JSON object and nothing else, of the form "
    '{"same_answer": <true|false>, "identifiable": <true|false>, "no_leak": <true|false>, "plausible": <true|false>}'
    ". No markdown, no code fences, no explanation.\n"
)

CATEGORIES = ("correct", "partial", "wrong_fact", "not_found", "wrong_document", "unsupported")
MAX_EVIDENCE_CHARS = 2400


def router_item(source_description: str, question: str) -> str:
    return f"Collection: {source_description}\nQuestion: {question}"


def gen_item(question: str, passages: list[tuple[str, str]]) -> str:
    if passages:
        ctx = "\n\n".join(f"[{i + 1}] ({title}) {txt}" for i, (title, txt) in enumerate(passages))
    else:
        ctx = "(no passages were retrieved)"
    return f"Question: {question}\n\nContext passages:\n{ctx}"


def judge_item(question: str, reference: str, evidence: list[str], answer: str) -> str:
    ev, used = [], 0
    for e in evidence:
        e = e.strip()
        if not e:
            continue
        room = MAX_EVIDENCE_CHARS - used
        if room <= 0:
            break
        ev.append(e[:room])
        used += len(ev[-1])
    return (f"Question: {question}\nReference answer: {reference or '(see evidence)'}\nGold evidence:\n- "
            + "\n- ".join(ev) + f"\nAnswer to grade: {answer}")


def _json(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    d = json.loads(m.group(0))
    if not isinstance(d, dict):
        raise ValueError("reply is not a JSON object")
    return d


def parse_router(text: str) -> dict:
    d = _json(text)
    ent, qual = d.get("entity"), d.get("qualifier")
    for k, v in (("entity", ent), ("qualifier", qual)):
        if v is not None and not isinstance(v, str):
            raise ValueError(f"{k} must be a string or null")
    c = float(d.get("confidence"))
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"confidence {c} outside [0,1]")
    return {"entity": (ent or None) and ent.strip() or None, "qualifier": (qual or None) and qual.strip() or None,
            "confidence": c}


def parse_gen(text: str) -> dict:
    d = _json(text)
    a = d.get("answer")
    if not isinstance(a, str):
        raise ValueError("answer must be a string")
    f = d.get("found")
    if not isinstance(f, bool):
        raise ValueError("found must be true or false")
    return {"answer": a.strip(), "found": f}


def parse_judge(text: str) -> dict:
    d = _json(text)
    s = d.get("score")
    if isinstance(s, float) and s.is_integer():
        s = int(s)
    if not isinstance(s, int) or not 1 <= s <= 5:
        raise ValueError(f"score {s!r} not an integer 1-5")
    c = d.get("correct")
    if not isinstance(c, bool):
        raise ValueError("correct must be true or false")
    cat = d.get("category")
    if cat not in CATEGORIES:
        raise ValueError(f"category {cat!r} not in {CATEGORIES}")
    return {"score": s, "correct": c, "category": cat}


def card_item(source_description: str, title: str, excerpt: str) -> str:
    return f"Collection: {source_description}\nCatalog title: {title}\nExcerpt:\n{excerpt}"


def rewrite_item(masked_question: str, blocks: list[tuple[str, str, list[str]]], forbidden: list[str],
                 avoid: str) -> str:
    parts = [f"Question: {masked_question}"]
    for ph, kind, facts in blocks:
        parts.append(f"{ph} is {kind}. Facts:\n" + "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)))
    parts.append("Forbidden names: " + "; ".join(forbidden))
    parts.append("Do not include: " + (avoid or "(nothing specific)"))
    return "\n\n".join(parts)


def _strs(v, n, what=None) -> list[str]:
    """Lenient: keep the string items of a list (or a single string), drop anything else."""
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [x.strip() for x in v if isinstance(x, str) and x.strip()][:n]


def parse_card(text: str) -> dict:
    """Lenient card parser: any JSON object is accepted. Missing or malformed optional fields become empty;
    facts without a quote are dropped; a missing summary is built from the name and the facts. Card quality is
    measured afterwards (grounding in cardcheck) and reported, it never stops the run."""
    d = _json(text)
    name = d.get("name") if isinstance(d.get("name"), str) else ""
    date = d.get("date")
    date = date.strip() if isinstance(date, str) and date.strip() else None
    fs = []
    for f in d.get("facts") if isinstance(d.get("facts"), list) else []:
        if isinstance(f, dict) and isinstance(f.get("quote"), str) and f["quote"].strip():
            fact = f.get("fact") if isinstance(f.get("fact"), str) else f["quote"]
            fs.append({"fact": fact.strip(), "quote": f["quote"].strip()})
        elif isinstance(f, str) and f.strip():
            fs.append({"fact": f.strip(), "quote": f.strip()})
        if len(fs) == 8:
            break
    summ = d.get("summary") if isinstance(d.get("summary"), str) else ""
    if not summ.strip():
        summ = " ".join([name] + [f["fact"] for f in fs])[:600]
    return {"name": name.strip(), "aliases": _strs(d.get("aliases"), 5), "doc_type":
            (d.get("doc_type") if isinstance(d.get("doc_type"), str) else "").strip(), "date": date,
            "entities": _strs(d.get("entities"), 6), "facts": fs, "summary": " ".join(summ.split()[:100]),
            "lenient": sorted(k for k in ("name", "doc_type", "facts", "summary")
                              if not (isinstance(d.get(k), str) and d.get(k).strip()) and
                              not (k == "facts" and isinstance(d.get(k), list) and d.get(k)))}


def parse_rewrite(text: str) -> dict:
    d = _json(text)
    q = d.get("question")
    if not isinstance(q, str) or len(q.strip()) < 8:
        raise ValueError("question must be a non-empty string")
    if "\u27e8" in q or "\u27e9" in q:
        raise ValueError("a placeholder was left in the question")
    used = d.get("facts_used")
    used = [x for x in used if isinstance(x, int)] if isinstance(used, list) else []
    return {"question": " ".join(q.split()), "facts_used": used}


def resolver_item(source_description: str, question: str, cards: list[str]) -> str:
    lst = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(cards)) or "(no candidates)"
    return f"Collection: {source_description}\nQuestion: {question}\nCandidates:\n{lst}"


def parse_resolver(text: str) -> dict:
    """Lenient: {"documents": [{"id", "confidence"}]}; a bare list of ids gets confidence 0.5."""
    d = _json(text)
    docs = d.get("documents")
    if docs is None:
        docs = []
    if not isinstance(docs, list):
        raise ValueError("documents must be a list")
    out = []
    for x in docs[:5]:
        if isinstance(x, dict):
            try:
                i = int(x.get("id"))
                c = float(x.get("confidence", 0.5))
            except (TypeError, ValueError):
                continue
        elif isinstance(x, (int, float)):
            i, c = int(x), 0.5
        else:
            continue
        out.append({"id": i, "confidence": min(1.0, max(0.0, c))})
    return {"documents": out}


def validity_item(original: str, rewritten: str, answer: str, docs: list[str]) -> str:
    return (f"Original question: {original}\nRewritten question: {rewritten}\nReference answer: "
            f"{(answer or '(not given)')[:800]}\nDocument: " + "\n".join(docs))


def parse_validity(text: str) -> dict:
    d = _json(text)
    out = {}
    for k in ("same_answer", "identifiable", "no_leak", "plausible"):
        v = d.get(k)
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        if not isinstance(v, bool):
            raise ValueError(f"{k} must be true or false")
        out[k] = v
    return out


# --------------------------------------------------------------------------- dry-run stand-ins
def fake_response(model_id: str, prefix: str, item: str) -> str:
    h = int(hashlib.sha1((model_id + item).encode()).hexdigest(), 16)
    if prefix == ROUTER_PREFIX:
        q = item.split("Question:", 1)[-1]
        caps = re.findall(r"\b[A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+)*", q)
        ent = max(caps, key=len) if caps else None
        return json.dumps({"entity": ent, "qualifier": None, "confidence": 0.8 if ent else 0.1})
    if prefix == GEN_PREFIX:
        q = item.split("Context passages:", 1)[0]
        ctx = item.split("Context passages:", 1)[-1]
        if "(no passages were retrieved)" in ctx or h % 9 == 0:
            return json.dumps({"answer": "NOT FOUND", "found": False})
        qt = set(T.content_tokens(q))
        best = max(re.split(r"(?<=[.!?])\s+", ctx), key=lambda s: len(qt & set(T.content_tokens(s))))
        return json.dumps({"answer": best[:300], "found": True})
    if prefix == JUDGE_PREFIX:
        ref = item.split("Reference answer:", 1)[-1].split("Answer to grade:", 1)[0]
        ans = item.split("Answer to grade:", 1)[-1]
        cov = T.coverage(ans, ref) if "NOT FOUND" not in ans else 0.0
        s = 5 if cov > 0.5 else 4 if cov > 0.35 else 3 if cov > 0.2 else 1
        cat = "correct" if s >= 4 else "partial" if s == 3 else ("not_found" if "NOT FOUND" in ans else "wrong_fact")
        return json.dumps({"score": s, "correct": s >= 4, "category": cat})
    if prefix == RESOLVER_PREFIX:
        q = item.split("Question:", 1)[-1].split("\n", 1)[0]
        qt = set(re.findall(r"[a-z]{4,}", q.lower()))
        best, bi = -1, None
        for line in item.split("Candidates:\n", 1)[-1].split("\n"):
            m = re.match(r"\[(\d+)\] (.*)", line)
            if m:
                sc = len(qt & set(re.findall(r"[a-z]{4,}", m.group(2).lower())))
                if sc > best:
                    best, bi = sc, int(m.group(1))
        docs = [{"id": bi, "confidence": min(1.0, 0.3 + 0.1 * best)}] if bi else []
        return json.dumps({"documents": docs})
    if prefix == VALIDITY_PREFIX:
        return json.dumps({"same_answer": True, "identifiable": True, "no_leak": "\u27e8" not in item,
                           "plausible": True})
    if prefix == CARD_PREFIX:
        title = item.split("Catalog title:", 1)[-1].split("\n", 1)[0].strip()
        ex = item.split("Excerpt:\n", 1)[-1]
        sents = [x for x in re.split(r"(?<=[.!?])\s+", ex) if 20 <= len(x) <= 190][:8]
        facts = [{"fact": x[:80], "quote": x} for x in sents] or [{"fact": title, "quote": ex[:60]}]
        return json.dumps({"name": title, "aliases": [], "doc_type": "document", "date": None,
                           "entities": [title.split(" ")[0]], "facts": facts, "summary": ex[:300]})
    if prefix == REWRITE_PREFIX:
        q = item.split("Question:", 1)[-1].split("\n\n", 1)[0].strip()
        for k in range(1, 6):
            ph = "\u27e8D%d\u27e9" % k
            if ph in item:
                fact = item.split(ph + " is", 1)[-1].split("1. ", 1)[-1].split("\n", 1)[0]
                words = [w for w in re.findall(r"[a-z]{5,}", fact.lower())][:4]
                q = q.replace(ph, "the document about " + " ".join(words or ["it"]))
        return json.dumps({"question": q, "facts_used": [1]})
    return json.dumps({"echo": item[:40]})
