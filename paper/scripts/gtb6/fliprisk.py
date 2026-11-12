"""Flip-risk score (second registration, §7): features computed before the model runs, an
L2-penalised logistic regression (C = 1.0) on standardised features, frozen to JSON.

The fit minimises 0.5·‖w‖² + C·Σ logloss with an unpenalised intercept, which is scikit-learn's
LogisticRegression objective, solved by Newton's method (strictly convex, so the optimum is unique).
"""
import math

import numpy as np

FEATURES = ['n_incoming', 'log_min_incoming_rank', 'incoming_above_c0', 'c0_pos_corr',
            'c0_pos_shift', 'one_minus_rbo', 'log_k_eff']
NON_FILE = ('NONE', 'OFF-LIST', 'INVALID')


def features(clean_rank, corr_top5, c0, one_minus_rbo, k_eff):
    top_c = clean_rank[:5]
    incoming = [p for p in corr_top5 if p not in top_c]
    pos = {p: i for i, p in enumerate(clean_rank, 1)}
    min_rank = min((pos.get(p, 50) for p in incoming), default=0)
    c0_corr = corr_top5.index(c0) + 1 if (c0 not in NON_FILE and c0 in corr_top5) else 6
    c0_clean = top_c.index(c0) + 1 if (c0 not in NON_FILE and c0 in top_c) else 6
    above = sum(1 for p in incoming if corr_top5.index(p) + 1 < c0_corr)
    return dict(n_incoming=len(incoming), log_min_incoming_rank=math.log1p(min(min_rank, 50)),
                incoming_above_c0=above, c0_pos_corr=c0_corr, c0_pos_shift=abs(c0_corr - c0_clean),
                one_minus_rbo=float(one_minus_rbo), log_k_eff=math.log1p(max(k_eff or 0, 0)))


def _matrix(rows, names):
    return np.array([[float(r[k]) for k in names] for r in rows])


def fit(rows, label='flip', C=1.0, tol=1e-12, max_iter=100):
    # shortcut: a perfectly separable sample makes the Hessian singular in the (unpenalised)
    # intercept direction and raises LinAlgError; guarded in practice by coef_intervals' class check
    X = _matrix(rows, FEATURES)
    y = np.array([r[label] for r in rows], float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    penalty = np.diag([1.0] * len(FEATURES) + [0.0])
    theta = np.zeros(A.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(A @ theta)))
        grad = penalty @ theta + C * A.T @ (p - y)
        if np.max(np.abs(grad)) < tol:
            break
        hess = penalty + C * (A.T * (p * (1 - p))) @ A
        theta -= np.linalg.solve(hess, grad)
    return dict(features=list(FEATURES), mean=mu.tolist(), scale=sd.tolist(),
                coef=theta[:-1].tolist(), intercept=float(theta[-1]), C=C)


def score(frozen, rows):
    z = (_matrix(rows, frozen['features']) - np.asarray(frozen['mean'])) / np.asarray(frozen['scale'])
    return z @ np.asarray(frozen['coef']) + frozen['intercept']


def coef_intervals(rows, label='flip', n=2000, seed=2026):
    """Exploratory: task-cluster bootstrap 95% intervals for the standardised coefficients."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r['task'], []).append(r)
    tasks = sorted(by)
    boots = []
    for _ in range(n):
        sample = [r for t in rng.choice(tasks, len(tasks)) for r in by[t]]
        if len({r[label] for r in sample}) == 2:
            boots.append(fit(sample, label)['coef'])
    b = np.asarray(boots)
    return {f: (float(np.quantile(b[:, i], .025)), float(np.quantile(b[:, i], .975))) for i, f in enumerate(FEATURES)}
