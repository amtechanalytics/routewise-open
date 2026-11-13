"""Study 3: validity of screened questions for two human raters, inter-rater and rater-vs-panel agreement."""
import csv, json, math, os

H = os.path.join(os.path.dirname(__file__), "..", "results", "human")
load = lambda f: {r["item"]: r for r in csv.DictReader(open(os.path.join(H, f), encoding="utf-8-sig"))}
r1, r2, key = load("s3_rating_sheet_rater1.csv"), load("s3_rating_sheet_rater2.csv"), load("s3_rating_key.csv")
C = ["same_answer", "identifiable", "no_leak", "plausible"]
yes = lambda r, c: r[c].strip().lower() == "yes"
valid = lambda r: all(yes(r, c) for c in C)


def kappa(a, b):
    n = len(a); po = sum(p == q for p, q in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


acc = [i for i in key if key[i]["screened"] == "accepted"]; rej = [i for i in key if i not in acc]
for name, f in [("rater 1", lambda i: valid(r1[i])), ("rater 2", lambda i: valid(r2[i])),
                ("both", lambda i: valid(r1[i]) and valid(r2[i])), ("either", lambda i: valid(r1[i]) or valid(r2[i]))]:
    k = sum(f(i) for i in acc); lo, hi = wilson(k, len(acc))
    print(f"{name:8s} accepted valid {k}/{len(acc)}  Wilson [{lo:.3f}, {hi:.3f}]  rejected valid {sum(f(i) for i in rej)}/{len(rej)}")
print("\ninter-rater, all 200 items")
for c in C:
    po, k = kappa([yes(r1[i], c) for i in key], [yes(r2[i], c) for i in key])
    print(f"  {c:12s} agreement {po:.3f}  kappa {k:.2f}")
po, k = kappa([valid(r1[i]) for i in acc], [valid(r2[i]) for i in acc])
print(f"  valid on all four (accepted items) agreement {po:.3f}  kappa {k:.2f}")
print("\nrater vs panel model, items with panel votes")
for rn, rs in (("rater 1", r1), ("rater 2", r2)):
    for m in ("claude-sonnet-5-5", "openai/gpt-oss-120b"):
        out = []
        for c in C:
            a, b = [], []
            for i in key:
                v = json.loads(key[i]["panel_votes"]) if key[i]["panel_votes"] else None
                if v and v.get(m) and c in v[m]:
                    a.append(yes(rs[i], c)); b.append(bool(v[m][c]))
            out.append(f"{c} {kappa(a, b)[1]:.2f}")
        print(f"  {rn} vs {m} (n={len(a)}): " + ", ".join(out))
