"""PDHG for the frozen IC8-P / IC9-P / IC9-G objectives."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def shrink(x, threshold, group=False):
    if group:
        length = np.linalg.norm(x, axis=1, keepdims=True)
        return x * np.maximum(0., 1. - np.asarray(threshold) / np.maximum(length, 1e-30))
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.)


def project_ball(v, center, radius):
    difference = v - center
    return center + difference * min(1., radius / max(np.linalg.norm(difference), 1e-30))


def project_observation(v, x, radius, bounds=None):
    projected = project_ball(v, x, radius)
    if bounds is None or np.all((projected >= bounds[0]) & (projected <= bounds[1])):
        return projected
    a, b = np.zeros_like(v), np.zeros_like(v)
    z = v.copy()
    for _ in range(1000):
        old = z.copy()
        y = project_ball(z + a, x, radius)
        a = z + a - y
        z = np.clip(y + b, bounds[0], bounds[1])
        b = y + b - z
        if np.linalg.norm(z - old) <= 1e-12 * max(1., np.linalg.norm(z)):
            break
    return z


def ic_objective(u, p, d, weights, alpha, kind, group=False):
    sparse_cost = np.linalg.norm(p, axis=1).sum() if group else np.abs(p).sum()
    if kind == 'IC8':
        residual = np.sqrt(weights)[:, None] * (d @ u) - p
        return float(alpha * sparse_cost + (1 - alpha) * np.sum(residual ** 2))
    residual = d.T @ (weights[:, None] * (d @ u) - p)
    return float(alpha * sparse_cost + (1 - alpha) * np.linalg.norm(residual))


def solve_ic(x, d, weights, *, alpha=0.5, eta=0.095, kind='IC9', group=False,
             max_iter=30000, tolerance=2e-6):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not group and x.shape[1] != 1:
        raise ValueError('paper IC solves one coordinate at a time')
    weights = np.asarray(weights, dtype=np.float64)
    d = sparse.csr_matrix(d, dtype=np.float64)
    degree = np.asarray(abs(d).T @ weights).ravel()
    unweighted_degree = np.asarray(abs(d).sum(axis=0)).ravel()
    radius = eta * np.linalg.norm(x - x.mean(axis=0))
    bounds = None if group else (float(x.min() - 0.5 * np.ptp(x)), float(x.max() + 0.5 * np.ptp(x)))
    beta = 1 - alpha
    bound_squared = 2 * degree.max() + 1 if kind == 'IC8' else (2 * degree.max()) ** 2 + 2 * unweighted_degree.max()
    tau = sigma = 0.95 / np.sqrt(bound_squared)

    def apply(u, p):
        if kind == 'IC8':
            return np.sqrt(weights)[:, None] * (d @ u) - p
        return d.T @ (weights[:, None] * (d @ u) - p)

    def adjoint(q):
        if kind == 'IC8':
            return d.T @ (np.sqrt(weights)[:, None] * q), -q
        return d.T @ (weights[:, None] * (d @ q)), -(d @ q)

    def dual_prox(q):
        if kind == 'IC8':
            return q / (1. + sigma / (2 * beta))
        return q * min(1., beta / max(np.linalg.norm(q), 1e-30))

    u = x.copy()
    p = np.zeros((d.shape[0], x.shape[1]))
    q = np.zeros((d.shape[0] if kind == 'IC8' else len(x), x.shape[1]))
    u_bar, p_bar = u.copy(), p.copy()
    history = []
    residual = np.inf
    for iteration in range(1, max_iter + 1):
        q = dual_prox(q + sigma * apply(u_bar, p_bar))
        gu, gp = adjoint(q)
        next_u = project_observation(u - tau * gu, x, radius, bounds)
        next_p = shrink(p - tau * gp, tau * alpha, group)
        u_bar, p_bar = 2 * next_u - u, 2 * next_p - p
        u, p = next_u, next_p
        if iteration % 50 == 0 or iteration == max_iter:
            gu, gp = adjoint(q)
            ru = np.linalg.norm(u - project_observation(u - tau * gu, x, radius, bounds)) / tau
            rp = np.linalg.norm(p - shrink(p - tau * gp, tau * alpha, group)) / tau
            rq = np.linalg.norm(q - dual_prox(q + sigma * apply(u, p))) / sigma
            residual = max(ru / (1 + np.linalg.norm(u) + np.linalg.norm(gu)),
                           rp / (1 + np.linalg.norm(p) + np.linalg.norm(gp)),
                           rq / (1 + np.linalg.norm(q) + np.linalg.norm(apply(u, p))))
            feasibility = max(0., np.linalg.norm(u - x) - radius) / max(np.linalg.norm(x), 1e-30)
            if bounds is not None:
                feasibility = max(feasibility, max(0., bounds[0] - u.min(), u.max() - bounds[1]))
            history.append({'iteration': iteration, 'objective': ic_objective(u, p, d, weights, alpha, kind, group),
                            'feasibility': float(feasibility), 'prox_residual': float(residual)})
            if residual < tolerance and feasibility < 1e-9:
                break
    return {'U': u, 'P': p, 'dual': q, 'history': history, 'iterations': iteration,
            'objective': ic_objective(u, p, d, weights, alpha, kind, group),
            'feasibility': float(feasibility), 'stationarity': float(residual),
            'converged': bool(residual < tolerance), 'tau': float(tau), 'sigma': float(sigma),
            'radius': float(radius), 'box': bounds}


def gauge_solve_ic9_group(y, d, weights, **kwargs):
    center = y.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((y - center) ** 2, axis=1)))
    result = solve_ic((y - center) / scale, d, weights, kind='IC9', group=True, **kwargs)
    result['Z'] = center + scale * result['U']
    return result
