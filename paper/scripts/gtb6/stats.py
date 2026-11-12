"""Statistics for the second registration (H4-H8). Same conventions as the first:
task-level paired bootstrap, 10,000 resamples, percentile intervals, seed 2026."""
import numpy as np

SEED = 2026
N_BOOT = 10_000


def boot_means(values, n=N_BOOT, seed=SEED):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)


def ci(values, level=0.95, n=N_BOOT, seed=SEED):
    """Mean and percentile bootstrap interval of per-task values."""
    v = np.asarray(values, float)
    if v.size == 0:
        return float('nan'), float('nan'), float('nan')
    b = boot_means(v, n, seed)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(b, a)), float(np.quantile(b, 1 - a))


def one_sided_p(values, n=N_BOOT, seed=SEED, direction='greater'):
    """Share of bootstrap means at or below zero (direction='greater', tests mean > 0)
    or at or above zero (direction='less', tests mean < 0)."""
    if len(values) == 0:
        return float('nan')
    b = boot_means(values, n, seed)
    return float((b <= 0).mean()) if direction == 'greater' else float((b >= 0).mean())


def holm(pvals, alpha=0.05):
    """Holm step-down. Returns a list of reject flags in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, reject = len(pvals), [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def equivalence(values, margin, n=N_BOOT, seed=SEED):
    """Two one-sided tests at alpha = 0.05, decided by the 90% interval lying inside +/- margin."""
    m, lo, hi = ci(values, level=0.90, n=n, seed=seed)
    return dict(mean=m, lo90=lo, hi90=hi, equivalent=bool(lo > -margin and hi < margin))


def h3_reading(p_adj_reject, lo95, hi95, lo90, hi90, margin=0.10):
    """The registered reading rule for a placebo comparison (H3 and its rerun in H4)."""
    if p_adj_reject:
        return 'corruption-specific'
    if hi95 < 0:
        return 'placebo-heavier'
    if lo90 > -margin and hi90 < margin:
        return 'generic'
    return 'inconclusive'


def sign_flip_p(values, n=100_000, seed=SEED):
    """Two-sided sign-flip permutation p for a mean of per-task differences."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n, len(v)))
    return float((np.abs((flips * v).mean(1)) >= abs(v.mean())).mean())


def auc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney statistic (ties count one half)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def auc_task_bootstrap(scores, labels, tasks, n=N_BOOT, seed=SEED):
    """AUC with a task-cluster bootstrap: resample tasks, keep all their cells.
    Returns (auc, lo95, hi95, one-sided p for AUC > 0.5)."""
    s, y, t = np.asarray(scores, float), np.asarray(labels, int), np.asarray(tasks)
    uniq = np.unique(t)
    idx_by_task = {u: np.flatnonzero(t == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by_task[u] for u in pick])
        a = auc(s[ii], y[ii])
        if not np.isnan(a):
            boots.append(a)
    boots = np.asarray(boots)
    return (auc(s, y), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)),
            float((boots <= 0.5).mean()))


def fmt_ci(m, lo, hi, digits=4, signed=True, pct=False):
    if pct:
        return f"{100*m:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
    s = '+' if signed else ''
    return f"{m:{s}.{digits}f} [{lo:{s}.{digits}f}, {hi:{s}.{digits}f}]".replace('-', '−')


def fmt_pts(m, lo=None, hi=None):
    """Percentage points with a sign, e.g. '+1.2' or '+1.2 [−0.4, +2.9]'."""
    out = f"{100*m:+.1f}"
    if lo is not None:
        out += f" [{100*lo:+.1f}, {100*hi:+.1f}]"
    return out.replace('-', '−')
