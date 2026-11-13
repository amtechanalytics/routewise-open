"""Text utilities shared by every source: normalisation, tokens, chunking, alias matching, masking and
the evidence-coverage measure. Pure Python, deterministic, no external dependencies."""
from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be been but by can could did do does for from had has have how if in
into is it its may might more most no not of on or our shall should so such than that the their them then there
these they this those to under upon was we were what when where which while who whom why will with would you your
""".split())
CORP_SUFFIX = frozenset("inc incorporated corp corporation co company ltd limited plc llc lp holdings holding group "
                        "the sa nv ag se".split())


def norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


def tokens(s: str) -> list[str]:
    return _WORD.findall((s or "").lower())


def content_tokens(s: str) -> list[str]:
    return [t for t in tokens(s) if t not in _STOP and len(t) > 1]


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def order_key(qid: str) -> str:
    """Deterministic order used for every subset (sample, sub100, probe): the same everywhere."""
    return stable_hash("routewise|" + qid)


def chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Character windows with overlap, cut back to the last whitespace so words stay whole."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + size)
        if j < n:
            k = text.rfind(" ", i + size // 2, j)
            if k > i:
                j = k
        piece = text[i:j].strip()
        if piece:
            out.append(piece)
        if j >= n:
            break
        nxt = j - overlap
        if nxt > i:
            sp = text.find(" ", nxt, j)
            nxt = sp + 1 if sp != -1 else nxt
        i = max(nxt, i + 1)
    return out


def coverage(evidence: str, text: str) -> float:
    """Share of the evidence's content tokens that appear in text (bag of words)."""
    ev = content_tokens(evidence)
    if not ev:
        return 0.0
    have = set(content_tokens(text))
    return sum(1 for t in ev if t in have) / len(ev)


def evidence_recall(evidences: list[str], retrieved_texts: list[str], thr: float = 0.5) -> float:
    if not evidences:
        return 0.0
    joined = " ".join(retrieved_texts)
    have = set(content_tokens(joined))
    hits = 0
    for e in evidences:
        ev = content_tokens(e)
        if ev and sum(1 for t in ev if t in have) / len(ev) >= thr:
            hits += 1
    return hits / len(evidences)


def company_core(name: str) -> str:
    toks = [t for t in tokens(name) if t not in CORP_SUFFIX]
    return " ".join(toks)


# --------------------------------------------------------------------------- alias matching
def alias_forms(alias: str) -> tuple[str, bool]:
    """(form, case_sensitive). Short all-caps aliases (tickers) are matched case-sensitively as whole tokens."""
    a = (alias or "").strip()
    if re.fullmatch(r"[A-Z][A-Z.\-]{0,5}", a):
        return a, True
    return norm(a), False


def mentions(query: str, alias: str) -> bool:
    form, cs = alias_forms(alias)
    if not form:
        return False
    if cs:
        return re.search(r"(?<![A-Za-z])" + re.escape(form) + r"(?![A-Za-z])", query or "") is not None
    if len(form) < 4:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])", norm(query)) is not None


def mask_aliases(query: str, aliases: list[str], replacement: str) -> str:
    """Replace every mention of any alias (longest first) with the replacement phrase."""
    out = query
    for a in sorted({x for x in aliases if x}, key=len, reverse=True):
        form, cs = alias_forms(a)
        if cs:
            out = re.sub(r"\(?\b" + re.escape(form) + r"\b\)?", replacement, out)
        elif len(form) >= 4:
            words = [re.escape(w) for w in form.split()]
            pat = r"\b" + r"[\W_]*".join(words) + r"(?:'s)?\b"
            out = re.sub(pat, replacement, out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(rf"({re.escape(replacement)}[\s,]*)+{re.escape(replacement)}", replacement, out)
    return out


def fuzzy_score(entity: str, alias: str) -> float:
    """Similarity between an extracted entity string and a catalog alias, in [0, 1]."""
    e, a = norm(entity), norm(alias)
    if not e or not a:
        return 0.0
    if e == a:
        return 1.0
    ec, ac = company_core(e), company_core(a)
    if ec and ec == ac:
        return 0.97
    if len(a) >= 4 and (f" {a} " in f" {e} " or (len(e) >= 4 and f" {e} " in f" {a} ")):
        return 0.9
    if e.replace(" ", "") == a.replace(" ", ""):
        return 0.95
    te, ta = set(e.split()), set(a.split())
    j = len(te & ta) / max(1, len(te | ta))
    return j
