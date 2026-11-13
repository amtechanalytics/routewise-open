"""Study 1: mention rate vs routing's evidence-recall gain, with collection-grouped sensitivity checks."""
import csv, itertools, json, os, random
from statistics import mean

R = os.path.join(os.path.dirname(__file__), "..", "results")
cfg = {r["config"]: r for r in csv.DictReader(open(os.path.join(R, "t1_configs.csv")))}
d = json.load(open(os.path.join(R, "deltas.json")))
rows = [(k.split(".")[0], float(cfg[k]["mention_rate"]), d[k]["route"]) for k in d]


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y); mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def exact_p(x, y):
    obs = spearman(x, y); perms = list(itertools.permutations(y))
    return sum(spearman(x, list(q)) >= obs - 1e-12 for q in perms) / len(perms)


x = [r[1] for r in rows]; y = [r[2] for r in rows]
print(f"all {len(rows)} configurations: rho = {spearman(x, y):.2f}")
srcs = sorted({r[0] for r in rows})
idx = {s: [i for i, r in enumerate(rows) if r[0] == s] for s in srcs}
random.seed(1); obs = spearman(x, y); hits = 0; N = 20000
for _ in range(N):
    yp = y[:]
    for ii in idx.values():
        vals = [y[i] for i in ii]; random.shuffle(vals)
        for i, v in zip(ii, vals):
            yp[i] = v
    hits += spearman(x, yp) >= obs - 1e-12
print(f"within-collection permutation p = {hits / N:.4f}")
sx = [mean(r[1] for r in rows if r[0] == s) for s in srcs]; sy = [mean(r[2] for r in rows if r[0] == s) for s in srcs]
print(f"six collection means: rho = {spearman(sx, sy):.2f}, exact one-sided p = {exact_p(sx, sy):.3f}")
bx = [float(cfg[s + ".base"]["mention_rate"]) for s in srcs]; by = [d[s + ".base"]["route"] for s in srcs]
print(f"six base configurations: rho = {spearman(bx, by):.2f}, exact one-sided p = {exact_p(bx, by):.3f}")
for s in srcs:
    xx = [r[1] for r in rows if r[0] != s]; yy = [r[2] for r in rows if r[0] != s]
    print(f"leave out {s}: rho = {spearman(xx, yy):.2f} (n = {len(xx)})")
