"""Statistics used by the analysis; deterministic (fixed seeds), numpy/scipy only."""
from __future__ import annotations

import math

import numpy as np

SEED = 20261108


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def wilcoxon_p(diffs) -> float:
    d = np.asarray([x for x in diffs if x == x], dtype=float)
    d = d[d != 0]
    if len(d) < 1:
        return 1.0
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant counts (b: only A correct, c: only B correct)."""
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * ps[i])
        run = max(run, v)
        adj[i] = run
    return adj


def spearman_perm(x, y, n_perm: int = 10000) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return (float("nan"), float("nan"))
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(n_perm):
        if abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r) - 1e-12:
            cnt += 1
    return (round(r, 4), round((cnt + 1) / (n_perm + 1), 4))


def ridge_fit(X, y, lam: float = 1.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "b": ym}


def ridge_predict(m, X):
    Z = (np.asarray(X, float) - m["mu"]) / m["sd"]
    return Z @ m["w"] + m["b"]


def weighted_kappa(a, b, k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores 1..k."""
    a, b = np.asarray(a, int) - 1, np.asarray(b, int) - 1
    if len(a) == 0:
        return float("nan")
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return round(float(1 - (W * O).sum() / den), 4) if den else float("nan")


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    labs = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labs)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else float("nan")


def cluster_bootstrap(values_by_cluster: dict, n_boot: int = 5000) -> tuple[float, float, float]:
    """Mean over clusters' means with a 95% percentile interval resampling whole clusters."""
    keys = sorted(values_by_cluster)
    means = np.array([np.mean(values_by_cluster[k]) for k in keys], float)
    if len(means) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(SEED)
    boots = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(n_boot)]
    return (round(float(means.mean()), 4), round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))
