"""Exact-discrete-target Framelet node ADMM; every U step uses matrix-free PCG."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from methods.stage2_modern_graph.ic_solver import shrink


def framelet_objective(u, x, operator, nu, group=False):
    degree = operator.degree
    c = operator.highpass_scale_weights
    value = 0.5 * np.sum(degree[:, None] * (u - x) ** 2) / degree.sum()
    for scale, band in zip(c, operator.analysis(u)[1:]):
        penalty = np.linalg.norm(band, axis=1) if group else np.abs(band).sum(axis=1)
        value += nu * scale * np.sum(degree * penalty) / (degree.sum() * sum(c))
    return float(value)


def solve_framelet(x, operator, nu, group=False, *, max_iter=5000, tolerance=2e-7,
                   audit_u_step=False):
    x = np.asarray(x, dtype=np.float64)
    degree = operator.degree
    n = len(x)
    rho = float(degree.mean())
    coefficients = np.array([0., nu / 1.25, 0.25 * nu / 1.25])
    u = x.copy()
    q = operator.analysis(u)
    dual = [np.zeros_like(u) for _ in range(3)]
    linear = LinearOperator((n, n), matvec=lambda v: degree * v + rho * operator.adjoint(operator.analysis(v)), dtype=np.float64)
    preconditioner = LinearOperator((n, n), matvec=lambda v: v / (degree + rho), dtype=np.float64)
    history = []
    pcg_max = 0.
    dense_error_max = 0.
    if audit_u_step:
        matrices = operator.reference_matrices()
        dense_system = np.diag(degree) + rho * sum(t.T @ t for t in matrices)
    for iteration in range(1, max_iter + 1):
        rhs = degree[:, None] * x + rho * operator.adjoint([a - b for a, b in zip(q, dual)])
        linear_error = 0.
        for coordinate in range(x.shape[1]):
            u[:, coordinate], info = cg(linear, rhs[:, coordinate], x0=u[:, coordinate], M=preconditioner,
                                        rtol=1e-11, atol=0., maxiter=2000)
            residual = np.linalg.norm(linear @ u[:, coordinate] - rhs[:, coordinate]) / max(np.linalg.norm(rhs[:, coordinate]), 1e-30)
            linear_error = max(linear_error, residual)
            if info != 0 or residual >= 1e-5:
                raise ValueError(f'Framelet PCG failed: info={info}, residual={residual}')
        pcg_max = max(pcg_max, linear_error)
        if audit_u_step:
            dense_u = np.linalg.solve(dense_system, rhs)
            dense_error_max = max(dense_error_max, np.linalg.norm(u - dense_u) / max(1., np.linalg.norm(dense_u)))
        transformed = operator.analysis(u)
        previous_q = q
        q = [band + lagrange if coefficient == 0 else
             shrink(band + lagrange, coefficient * degree[:, None] / rho, group)
             for band, lagrange, coefficient in zip(transformed, dual, coefficients)]
        dual = [lagrange + band - split for lagrange, band, split in zip(dual, transformed, q)]
        primal = np.sqrt(sum(np.sum((band - split) ** 2) for band, split in zip(transformed, q))) / max(1., np.linalg.norm(u))
        dual_res = rho * np.linalg.norm(operator.adjoint([a - b for a, b in zip(q, previous_q)])) / max(1., np.linalg.norm(degree[:, None] * x))
        stationarity = np.linalg.norm(degree[:, None] * (u - x) + rho * operator.adjoint(dual)) / max(1., np.linalg.norm(degree[:, None] * x))
        if iteration % 10 == 0 or iteration == max_iter or max(primal, dual_res, stationarity) < tolerance:
            history.append({'iteration': iteration, 'objective': framelet_objective(u, x, operator, nu, group),
                'primal_residual': float(primal), 'dual_residual': float(dual_res),
                'stationarity': float(stationarity), 'pcg_relative_residual': float(linear_error)})
        if max(primal, dual_res, stationarity) < tolerance:
            break
    return {'U': u, 'Q': q, 'dual': dual, 'history': history, 'iterations': iteration,
            'objective': framelet_objective(u, x, operator, nu, group), 'primal_residual': float(primal),
            'dual_residual': float(dual_res), 'stationarity': float(stationarity),
            'pcg_max_relative_residual': float(pcg_max), 'rho': rho, 'nu': nu,
            'u_step_dense_relative_error': float(dense_error_max) if audit_u_step else None,
            'converged': bool(max(primal, dual_res, stationarity) < tolerance),
            'lowpass_shrinkage_weight': 0., 'active_bands': list(operator.active_bands)}


def gauge_solve_framelet(y, operator, nu, group=True, **kwargs):
    center = y.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((y - center) ** 2, axis=1)))
    result = solve_framelet((y - center) / scale, operator, nu, group, **kwargs)
    result['Z'] = center + scale * result['U']
    return result
