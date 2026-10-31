"""Dense P1-only ADMM for paper Eq5/Eq17/Eq6; no full-N or sparse-DoT branch."""
from __future__ import annotations

import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, cg

from methods.stage2_modern_graph.ic_solver import shrink


def edge_objective_components(u, s, e, x, degree, operator=None, nu=0., fixed_u=False):
    components = dict(l1_S=float(np.abs(s).sum()), group_E=float(np.sum(degree * np.linalg.norm(e, axis=1))),
                      fidelity=0. if fixed_u else float(0.5 * np.sum(degree[:, None] * (u - x) ** 2)), framelet=0.)
    if operator is not None and nu:
        for coefficient, band in zip((1., 0.25), operator.analysis(u)[1:]):
            components['framelet'] += float(nu / 1.25 * coefficient * np.sum(degree[:, None] * np.abs(band)))
    return components


def edge_objective(u, s, e, x, degree, operator=None, nu=0., fixed_u=False):
    return sum(edge_objective_components(u, s, e, x, degree, operator, nu, fixed_u).values())


def balance_penalty(mu, scaled_dual, primal, dual):
    """Frozen factor-2 rule; preserve the unscaled Lagrange multiplier."""
    next_mu = min(1e4, 2 * mu) if primal > 10 * dual else max(1e-3, mu / 2) if dual > 10 * primal else mu
    return next_mu, scaled_dual * (mu / next_mu)


def solve_edge_small(x, adjacency, *, operator=None, nu=0., fixed_u=False,
                     initialization='author', max_iter=5000, tolerance=2e-5,
                     penalty_strategy='fixed', audit_last_sweep=False, wall_deadline=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n > 256:
        raise ValueError('this implementation is P1 dense smoke only, N<=256')
    if penalty_strategy not in ('fixed', 'residual_balance'):
        raise ValueError('unknown penalty strategy')
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    ones = np.ones((n, 1))
    u = x.copy()
    s = np.zeros((n, n)) if initialization == 'author' else adjacency.toarray() / degree[:, None]
    y = s.copy()
    e = np.zeros_like(x) if initialization == 'author' else u - s @ u
    v1, v3, v4 = np.zeros_like(u), np.zeros((n, 1)), np.zeros_like(s)
    q = operator.analysis(u) if operator is not None else []
    v2 = [np.zeros_like(u) for _ in q]
    mu1, mu3, mu4 = 10., 10., 10.
    mu2 = float(degree.mean())
    history = []
    penalty_history = []
    pcg_max = 0.
    last_sweep = None
    budget_exhausted = False
    for iteration in range(1, max_iter + 1):
        previous_s, previous_e = s.copy(), e.copy()
        previous_u, previous_y, previous_q = u.copy(), y.copy(), q
        if audit_last_sweep:
            last_sweep = dict(U=u.copy(), S=s.copy(), E=e.copy(), Q=[a.copy() for a in q],
                Lambda1=mu1*v1.copy(), Lambda3=mu3*v3.copy(), Lambda4=mu4*v4.copy(),
                Lambda2=[mu2*a.copy() for a in v2], mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4)
        blocks = [np.sqrt(mu1) * u]
        if not fixed_u:
            blocks.append(np.sqrt(mu3) * ones)
        b = np.concatenate(blocks, axis=1)
        rhs_y = mu1 * (u - e + v1) @ u.T + mu4 * (s - v4)
        if not fixed_u:
            rhs_y += mu3 * (ones - v3) @ ones.T
        # Exact Woodbury inverse of mu4 I + mu1 UU^T (+ mu3 11^T).
        y = rhs_y / mu4 - (rhs_y @ b) @ np.linalg.solve(mu4 * np.eye(b.shape[1]) + b.T @ b, b.T) / mu4
        s = shrink(y + v4, 1 / mu4)
        np.fill_diagonal(s, 0.)
        l0 = np.eye(n) - y
        if not fixed_u:
            rhs_u = degree[:, None] * x + mu1 * l0.T @ (e - v1)
            if operator is not None:
                rhs_u += mu2 * operator.adjoint([a - b for a, b in zip(q, v2)])

            def matvec(v):
                result = degree * v + mu1 * l0.T @ (l0 @ v)
                if operator is not None:
                    result += mu2 * operator.adjoint(operator.analysis(v))
                return result

            matrix = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
            diagonal = degree + mu1 * np.sum(l0 ** 2, axis=0) + (mu2 if operator is not None else 0.)
            pre = LinearOperator((n, n), matvec=lambda v: v / diagonal, dtype=np.float64)
            for c in range(2):
                u[:, c], info = cg(matrix, rhs_u[:, c], x0=u[:, c], M=pre, rtol=1e-10, atol=0., maxiter=2000)
                error = np.linalg.norm(matrix @ u[:, c] - rhs_u[:, c]) / max(1e-30, np.linalg.norm(rhs_u[:, c]))
                pcg_max = max(pcg_max, error)
                if info != 0 or error >= 1e-5:
                    raise ValueError(f'Edge U-step failed PCG: {info}, {error}')
        residual_y = l0 @ u
        e = shrink(residual_y + v1, degree[:, None] / mu1, group=True)
        v1 += residual_y - e
        v4 += y - s
        if not fixed_u:
            v3 += y @ ones - ones
        if operator is not None:
            transformed = operator.analysis(u)
            q = [band + lagrange if c == 0 else shrink(band + lagrange, c * degree[:, None] / mu2)
                 for band, lagrange, c in zip(transformed, v2, (0., nu / 1.25, 0.25 * nu / 1.25))]
            v2 = [a + b - c for a, b, c in zip(v2, transformed, q)]
        if iteration % 10 == 0 or iteration == max_iter or iteration == 1:
            lambda1, lambda3, lambda4 = mu1*v1, mu3*v3, mu4*v4
            lambda2 = [mu2*a for a in v2]
            c1, c4 = residual_y-e, y-s
            c3 = np.zeros((n, 1)) if fixed_u else y @ ones-ones
            c2 = [] if operator is None else [a-b for a,b in zip(operator.analysis(u),q)]
            r1 = np.linalg.norm(c1) / max(1., np.linalg.norm(u), np.linalg.norm(y@u), np.linalg.norm(e))
            r3 = np.linalg.norm(c3) / np.sqrt(n)
            r4 = np.linalg.norm(c4) / max(1., np.linalg.norm(y), np.linalg.norm(s))
            # Linearized block dual drifts. With fixed U, s1 is the standard
            # Y/E ADMM dual residual. The row-only s3 is explicitly an image-
            # drift diagnostic, not a canonical two-block ADMM dual residual.
            drift1 = l0 @ (u-previous_u) - (e-previous_e)
            dual1_y = mu1 * (drift1 @ u.T)
            dual1_u = mu1 * l0.T @ ((y-previous_y) @ previous_u + e-previous_e)
            dual1_raw = max(np.linalg.norm(dual1_y), 0. if fixed_u else np.linalg.norm(dual1_u))
            dual1 = dual1_raw / max(1., np.linalg.norm(lambda1@u.T), np.linalg.norm(l0.T@lambda1))
            dual3_raw = 0. if fixed_u else mu3*np.linalg.norm(((y-previous_y)@ones)@ones.T)
            dual3 = dual3_raw / max(1., np.linalg.norm(lambda3@ones.T))
            dual4_raw = mu4*np.linalg.norm(s-previous_s)
            dual4 = dual4_raw / max(1., np.linalg.norm(lambda4))
            dual2_raw = 0. if operator is None else mu2*np.linalg.norm(operator.adjoint([a-b for a,b in zip(q,previous_q)]))
            dual2 = dual2_raw / max(1., 0. if operator is None else np.linalg.norm(operator.adjoint(lambda2)))
            equality = np.linalg.norm(u - s @ u - e) / max(1., np.linalg.norm(u))
            row_sum = 0. if fixed_u else np.max(np.abs(s.sum(axis=1) - 1))
            y_row_max = float(np.max(np.abs(c3)))
            split = np.linalg.norm(y - s) / max(1., np.linalg.norm(s))
            progress = max(np.linalg.norm(s - previous_s) / max(1., np.linalg.norm(s)),
                           np.linalg.norm(e - previous_e) / max(1., np.linalg.norm(e)),
                           np.linalg.norm(u-previous_u) / max(1.,np.linalg.norm(u)),
                           np.linalg.norm(y-previous_y) / max(1.,np.linalg.norm(y)))
            grad_y = -mu1 * v1 @ u.T + mu4 * v4
            if not fixed_u:
                grad_y += mu3 * v3 @ ones.T
            stationarity = np.linalg.norm(grad_y) / max(1., np.linalg.norm(mu1 * v1 @ u.T), np.linalg.norm(mu4 * v4))
            if not fixed_u:
                grad_u = degree[:, None] * (u - x) + mu1 * l0.T @ v1
                if operator is not None:
                    grad_u += mu2 * operator.adjoint(v2)
                stationarity = max(stationarity, np.linalg.norm(grad_u) / max(1., np.linalg.norm(degree[:, None] * x)))
            frame_primal = 0. if operator is None else np.sqrt(sum(np.sum((a - b) ** 2)
                for a, b in zip(operator.analysis(u), q))) / max(1., np.linalg.norm(u))
            prox_s = shrink(s+lambda4,1.)
            np.fill_diagonal(prox_s,0.)
            prox_s_error = np.linalg.norm(s-prox_s)/max(1.,np.linalg.norm(s),np.linalg.norm(lambda4))
            prox_e_error = np.linalg.norm(e-shrink(e+lambda1,degree[:,None],True))/max(1.,np.linalg.norm(e),np.linalg.norm(lambda1))
            prox_q_error = 0.
            if operator is not None:
                prox_q_error = max(np.linalg.norm(a-shrink(a+b,c*degree[:,None]))/max(1.,np.linalg.norm(a),np.linalg.norm(b))
                                   for a,b,c in zip(q,lambda2,(0.,nu/1.25,.25*nu/1.25)))
            stationarity = max(stationarity,prox_s_error,prox_e_error,prox_q_error)
            primal = max(r1,r3,r4,frame_primal)
            dual = max(dual1,dual3,dual4,dual2)
            components = edge_objective_components(u,s,e,x,degree,operator,nu,fixed_u)
            history.append({'iteration': iteration, 'objective': sum(components.values()), **components,
                'equality_residual': float(equality), 'row_sum_residual': float(row_sum),
                'split_residual': float(split), 'frame_primal_residual': float(frame_primal),
                'progress': float(progress), 'stationarity': float(stationarity),
                'self_Y_raw':float(np.linalg.norm(c1)), 'self_Y_relative':float(r1),
                'Y_S_raw':float(np.linalg.norm(c4)), 'Y_S_relative':float(r4),
                'Y_row_raw':float(np.linalg.norm(c3)), 'Y_row_relative':float(r3),'Y_row_max':y_row_max,
                'S_row_raw':0. if fixed_u else float(np.linalg.norm(s@ones-ones)),
                'frame_raw':float(np.sqrt(sum(np.sum(a*a) for a in c2))),
                'primal_residual':float(primal),'dual_residual':float(dual),
                'dual1_raw':float(dual1_raw),'dual1_relative':float(dual1),
                'dual3_raw':float(dual3_raw),'dual3_relative':float(dual3),
                'dual4_raw':float(dual4_raw),'dual4_relative':float(dual4),
                'dual2_raw':float(dual2_raw),'dual2_relative':float(dual2),
                'prox_S':float(prox_s_error),'prox_E':float(prox_e_error),'prox_Q':float(prox_q_error),
                'mu1':mu1,'mu2':mu2,'mu3':mu3,'mu4':mu4,
                'pcg_max_relative_residual': float(pcg_max)})
            stopping_error = max(equality,row_sum,split,frame_primal,progress,stationarity,primal,dual,y_row_max)
            if stopping_error < tolerance:
                break
            if wall_deadline is not None and time.perf_counter() >= wall_deadline:
                budget_exhausted = True
                break
            if penalty_strategy == 'residual_balance' and 100 <= iteration <= 3000 and iteration % 50 == 0:
                for name, r, t in [('mu1',r1,dual1),('mu3',r3,dual3),('mu4',r4,dual4)]:
                    if fixed_u and name=='mu3':
                        continue
                    old_mu = {'mu1':mu1,'mu3':mu3,'mu4':mu4}[name]
                    vector = {'mu1':v1,'mu3':v3,'mu4':v4}[name]
                    new_mu,new_vector=balance_penalty(old_mu,vector,r,t)
                    multiplier_error=float(np.linalg.norm(old_mu*vector-new_mu*new_vector)/max(1.,np.linalg.norm(old_mu*vector)))
                    penalty_history.append(dict(iteration=iteration,penalty=name,old_mu=old_mu,new_mu=new_mu,
                        primal=float(r),dual=float(t),unscaled_multiplier_relative_error=multiplier_error))
                    if name=='mu1':mu1,v1=new_mu,new_vector
                    elif name=='mu3':mu3,v3=new_mu,new_vector
                    else:mu4,v4=new_mu,new_vector
    return {'U': u, 'S': s, 'E': e, 'history': history, 'iterations': iteration,
            'objective': edge_objective(u, s, e, x, degree, operator, nu, fixed_u),
            'feasible_objective': edge_objective(u, s, u - s @ u, x, degree, operator, nu, fixed_u),
            'converged': bool(stopping_error < tolerance), 'stopping_error':float(stopping_error),
            'budget_exhausted':budget_exhausted,
            'equality_residual': float(equality), 'row_sum_residual': float(row_sum),
            'diagonal_max': float(np.max(np.abs(s.diagonal()))), 'stationarity': float(stationarity),
            'pcg_max_relative_residual': float(pcg_max),
            'primal_residual':float(primal),'dual_residual':float(dual),'Y_row_max':y_row_max,
            'self_Y_relative':float(r1),'frame_primal_residual':float(frame_primal),'split_residual':float(split),
            'objective_components':components,'penalty_history':penalty_history,'penalty_strategy':penalty_strategy,
            'Y':y,'Q':q,'Lambda1':mu1*v1,'Lambda3':mu3*v3,'Lambda4':mu4*v4,
            'Lambda2':np.stack([mu2*a for a in v2]) if v2 else np.empty((0,n,x.shape[1])),
            'last_sweep':last_sweep,
            'penalties': {'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4, 'rho': 1.0}}
