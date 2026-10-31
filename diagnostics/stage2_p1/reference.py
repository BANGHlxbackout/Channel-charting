"""Independent CVXPY expressions for the frozen P1 objectives."""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_problem(problem):
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                  tol_feas=1e-9, max_iter=500)
    if problem.status != 'optimal':
        raise ValueError(f'CVXPY reference not optimal: {problem.status}')
    return {'reference_objective': float(problem.value), 'status': problem.status,
            'solver': problem.solver_stats.solver_name, 'iterations': problem.solver_stats.num_iters,
            'solve_time_s': problem.solver_stats.solve_time}


def ic_reference(x, d, weights, alpha, eta, kind, group=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    u, p = cp.Variable(x.shape), cp.Variable((d.shape[0], x.shape[1]))
    radius = eta * np.linalg.norm(x - x.mean(axis=0))
    constraints = [u == x] if radius == 0 else [cp.norm(u - x, 'fro') <= radius]
    if not group:
        constraints += [u >= float(x.min() - 0.5 * np.ptp(x)), u <= float(x.max() + 0.5 * np.ptp(x))]
    sparse_cost = cp.sum(cp.norm(p, 2, axis=1)) if group else cp.sum(cp.abs(p))
    if kind == 'IC8':
        smooth = cp.sum_squares(cp.multiply(np.sqrt(weights)[:, None], d @ u) - p)
    else:
        smooth = cp.norm(d.T @ (cp.multiply(weights[:, None], d @ u) - p), 'fro')
    problem = cp.Problem(cp.Minimize(alpha * sparse_cost + (1 - alpha) * smooth), constraints)
    result = solve_problem(problem)
    return {**result, 'U': u.value, 'P': p.value}


def framelet_reference(x, matrices, degree, nu, group=False):
    u = cp.Variable(x.shape)
    objective = 0.5 * cp.sum(cp.multiply(degree[:, None], cp.square(u - x))) / degree.sum()
    for coefficient, matrix in zip((1., 0.25), matrices[1:]):
        transformed = matrix @ u
        penalty = cp.norm(transformed, 2, axis=1) if group else cp.sum(cp.abs(transformed), axis=1)
        objective += nu * coefficient * cp.sum(cp.multiply(degree, penalty)) / (degree.sum() * 1.25)
    problem = cp.Problem(cp.Minimize(objective))
    return {**solve_problem(problem), 'U': u.value}


def edge_s_reference(u, degree, row_sum):
    n = len(u)
    s = cp.Variable((n, n))
    constraints = [cp.diag(s) == 0]
    if row_sum:
        constraints.append(s @ np.ones(n) == 1)
    objective = cp.sum(cp.abs(s)) + cp.sum(cp.multiply(degree, cp.norm(u - s @ u, 2, axis=1)))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    result = solve_problem(problem)
    return {**result, 'S': s.value, 'E': u - s.value @ u}


def edge_u_reference(x, s, degree, matrices=None, nu=0.):
    u = cp.Variable(x.shape)
    residual = u - s @ u
    objective = cp.sum(cp.abs(s)) + cp.sum(cp.multiply(degree, cp.norm(residual, 2, axis=1)))
    objective += 0.5 * cp.sum(cp.multiply(degree[:, None], cp.square(u - x)))
    if matrices is not None:
        for coefficient, matrix in zip((1., 0.25), matrices[1:]):
            objective += nu * coefficient / 1.25 * cp.sum(cp.multiply(degree[:, None], cp.abs(matrix @ u)))
    problem = cp.Problem(cp.Minimize(objective))
    return {**solve_problem(problem), 'U': u.value}
