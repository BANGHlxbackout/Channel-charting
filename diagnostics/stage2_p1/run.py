"""Frozen P1 mathematical sanity. No P2 scans and no real-data method fitting."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record, incidence_and_laplacian
from diagnostics.stage2_p1.cases import synthetic_case
from diagnostics.stage2_p1.reference import ic_reference, framelet_reference, edge_s_reference, edge_u_reference
from methods.g1_acf_graph_fcf.compare import compare_paths
from methods.stage2_modern_graph.ic_solver import solve_ic, shrink, gauge_solve_ic9_group
from methods.stage2_modern_graph.framelet_operator import FrameletOperator, operator_checks
from methods.stage2_modern_graph.framelet_solver import solve_framelet, gauge_solve_framelet, framelet_objective
from methods.stage2_modern_graph.edge_small import solve_edge_small, edge_objective
from run_adp_siamese import plot_chart

ROOT = Path('results/stage2_modern_graph_20261019/p1_math_sanity')
DELTA = Path('results/stage2_modern_graph_20261019/p0_preflight/delta0.json')
NU = 0.1  # Synthetic math-smoke coefficient only; not a selected B0 parameter.
GATES = []


def emit_gate(family, name, actual, threshold, passed, evidence):
    row = {'family': family, 'gate': name, 'actual': actual, 'threshold': threshold,
           'status': 'PASS' if passed else 'FAIL', 'evidence': str(evidence.resolve())}
    GATES.append(row)
    write_json(ROOT / 'gates.json', GATES)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def save_history(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_native(folder, result, prefix='native'):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    for key in ('Q', 'dual'):
        if key in result and isinstance(result[key], list):
            arrays.update({f'{key}_{i}': value for i, value in enumerate(result[key])})
    np.savez(folder / f'{prefix}_arrays.npz', **arrays)
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)
                and key not in ('Q', 'dual', 'history')}
    write_json(folder / f'{prefix}.json', metadata)
    save_history(folder / f'{prefix}_history.csv', result.get('history', []))


def save_reference(folder, reference, name='reference'):
    arrays = {key: value for key, value in reference.items() if isinstance(value, np.ndarray)}
    np.savez(folder / f'{name}_arrays.npz', **arrays)
    write_json(folder / f'{name}.json', {key: value for key, value in reference.items() if key not in arrays})


def chart_record(folder, name, u, clean, observation, metadata):
    center = observation.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((observation - center) ** 2, axis=1)))
    z = center + scale * u
    np.savez(folder / 'embedding.npz', indices=np.arange(len(z)), positions=clean, embedding=z)
    write_json(folder / 'result.json', {'phase': 'P1_synthetic_math', 'method': name,
               'status': 'computed_awaiting_evaluation', 'num_samples': len(z),
               'scope': 'synthetic coordinates only; not DICHASUS performance', **metadata})
    return folder / 'embedding.npz'


def evaluate_group(paths, folder):
    rows = compare_paths(paths, folder / 'comparison')
    for name, path in paths.items():
        record = json.loads((path.parent / 'result.json').read_text())
        row = rows[name]
        record.update(status='completed' if record.get('converged', True) else 'NONCONVERGED', metrics={key: row[key] for key in ('ct', 'tw', 'ks')},
            localization_errors_after_affine={key: row[key] for key in ('mean_m', 'median_m', 'p90_m', 'p95_m', 'max_m')})
        write_json(path.parent / 'result.json', record)
        saved = np.load(path)
        plot_chart(path.parent / 'chart.png', saved['positions'], saved['embedding'],
                   f"{name}\nCT={row['ct']:.3f}, TW={row['tw']:.3f}, KS={row['ks']:.3f}")


def observation_record(folder, x, clean, observation):
    folder.mkdir(parents=True, exist_ok=True)
    return chart_record(folder, 'Observation', x, clean, observation, {'role': 'synthetic noisy input'})


def normed_gap(value, reference):
    return float(abs(value - reference) / max(abs(reference), 1e-12))


def run_ic(eta):
    family = ROOT / 'ic'
    family.mkdir(exist_ok=False)
    rng = np.random.default_rng(123)
    gradient = rng.normal(size=500) * 3
    alpha, beta = 0.5, 0.5
    p = shrink(gradient, alpha / (2 * beta))
    eliminated = alpha * np.abs(p) + beta * (gradient - p) ** 2
    huber = np.where(np.abs(gradient) <= alpha / (2 * beta), beta * gradient ** 2,
                     alpha * np.abs(gradient) - alpha ** 2 / (4 * beta))
    error = float(np.max(np.abs(eliminated - huber)))
    write_json(family / 'huber.json', {'max_abs_error': error, 'alpha': alpha, 'beta': beta})
    emit_gate('IC', 'IC8_Huber_equality', error, '<=1e-8', error <= 1e-8, family / 'huber.json')
    for n in (16, 32, 64):
        folder = family / f'n{n}'
        folder.mkdir()
        adjacency, x, clean, observation = synthetic_case(n)
        d, weights, *_ = incidence_and_laplacian(adjacency)
        raw_mean = float(weights.mean())
        weights /= raw_mean
        np.savez(folder / 'case.npz', X=x, observation=observation, synthetic_positions=clean,
                 weights_ic=weights, incidence=d.toarray())
        adj_probe = rng.normal(size=(d.shape[0], 2))
        lhs = np.sum((d @ x) * adj_probe)
        rhs = np.sum(x * (d.T @ adj_probe))
        adjoint_error = float(abs(lhs - rhs) / max(1., abs(lhs), abs(rhs)))
        write_json(folder / 'graph_checks.json', {'adjoint_relative_error': adjoint_error, 'original_mean_edge_weight': raw_mean})
        emit_gate('IC', f'n{n}_D_adjoint', adjoint_error, '<1e-8', adjoint_error < 1e-8, folder / 'graph_checks.json')
        paths = {'Observation': observation_record(folder / 'observation', x, clean, observation)}
        for label, kind, group in [('IC8-P', 'IC8', False), ('IC9-P', 'IC9', False), ('IC9-G', 'IC9', True)]:
            method_dir = folder / label.lower()
            method_dir.mkdir()
            values, summaries = [], []
            coordinates = [x] if group else [x[:, c:c + 1] for c in range(2)]
            for coordinate, target in enumerate(coordinates):
                result = solve_ic(target, d, weights, alpha=alpha, eta=eta, kind=kind, group=group)
                save_native(method_dir, result, f'axis{coordinate}_attempt1')
                if not result['converged']:
                    result = solve_ic(target, d, weights, alpha=alpha, eta=eta, kind=kind,
                                      group=group, max_iter=120000)
                    save_native(method_dir, result, f'axis{coordinate}_attempt2')
                reference = ic_reference(target, d, weights, alpha, eta, kind, group)
                save_reference(method_dir, reference, f'axis{coordinate}_reference')
                fixed_reference = ic_reference(target, d, weights, alpha, 0., kind, group)
                save_reference(method_dir, fixed_reference, f'axis{coordinate}_zero_radius_reference')
                gap = normed_gap(result['objective'], reference['reference_objective'])
                stats = {'objective': result['objective'], 'reference_objective': reference['reference_objective'],
                         'relative_objective_gap': gap, 'feasibility': result['feasibility'],
                         'stationarity': result['stationarity'], 'converged': result['converged'],
                         'initial_optimal_aux_objective': fixed_reference['reference_objective'],
                         'objective_decrease': fixed_reference['reference_objective'] - result['objective']}
                summaries.append(stats); values.append(result['U'])
            u = values[0] if group else np.concatenate(values, axis=1)
            summary = {'alpha': alpha, 'eta': eta, 'kind': label, 'coordinate_results': summaries,
                       'max_objective_gap': max(a['relative_objective_gap'] for a in summaries),
                       'max_feasibility': max(a['feasibility'] for a in summaries),
                       'max_stationarity': max(a['stationarity'] for a in summaries)}
            write_json(method_dir / 'checks.json', summary)
            passed = (summary['max_objective_gap'] <= 1e-3 and summary['max_feasibility'] <= 1e-4 and
                      summary['max_stationarity'] <= 5e-3 and all(a['converged'] for a in summaries))
            emit_gate('IC', f'n{n}_{label}_reference', summary,
                      'relative objective<=1e-3; feasibility<=1e-4; stationarity<=5e-3; converged',
                      passed, method_dir / 'checks.json')
            paths[label] = chart_record(method_dir, label, u, clean, observation, summary)
            zero = solve_ic(coordinates[0], d, weights, alpha=alpha, eta=0., kind=kind, group=group)
            zero_error = float(np.max(np.abs(zero['U'] - coordinates[0])))
            zero['zero_output_error'] = zero_error
            save_native(method_dir, zero, 'eta0_native')
            emit_gate('IC', f'n{n}_{label}_eta0', zero_error, '<=1e-8', zero_error <= 1e-8,
                      method_dir / 'eta0_native.json')
        evaluate_group(paths, folder)
    adjacency, x, _, observation = synthetic_case(32)
    d, weights, *_ = incidence_and_laplacian(adjacency)
    weights /= weights.mean()
    base = gauge_solve_ic9_group(observation, d, weights, eta=eta)['Z']
    transforms = [('rotation', np.array([[.6, -.8], [.8, .6]]), 1., np.zeros(2)),
                  ('reflection', np.diag([-1., 1.]), 1., np.zeros(2)),
                  ('translation', np.eye(2), 1., np.array([7., -4.])),
                  ('scale', np.eye(2), 3.7, np.zeros(2))]
    equivariance = {}
    for name, rotation, scale, shift in transforms:
        transformed = gauge_solve_ic9_group(scale * observation @ rotation + shift, d, weights, eta=eta)
        restored = (transformed['Z'] - shift) @ rotation.T / scale
        equivariance[name] = float(np.linalg.norm(restored - base) / np.linalg.norm(base - base.mean(axis=0)))
    write_json(family / 'equivariance.json', equivariance)
    emit_gate('IC', 'IC9-G_geometric_equivariance', equivariance, 'each <1e-3',
              max(equivariance.values()) < 1e-3, family / 'equivariance.json')


def author_operator_error(operator):
    spec = importlib.util.spec_from_file_location('gnn_dot_author_utils', 'references/stage2_p1/gnn_dot/utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # NumPy 2.4 removed trapz; trapezoid is the same quadrature, no source edit.
    with patch.object(np, 'trapz', np.trapezoid, create=True):
        result = module.get_operator(operator.laplacian, [lambda x: np.cos(x / 2), lambda x: np.sin(x / 2)],
                                     operator.n, 2, operator.J, 2)
    references = [result[0, 1].toarray(), result[1, 0].toarray(), result[1, 1].toarray()]
    return max(float(np.linalg.norm(a - b) / max(1e-30, np.linalg.norm(b)))
               for a, b in zip(operator.reference_matrices(), references))


def select_operator(adjacency, folder):
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.diag(1 / np.sqrt(degree))
    maximum = float(np.linalg.eigvalsh(np.eye(len(degree)) - inv @ adjacency.toarray() @ inv)[-1])
    attempts = []
    selected = None
    for order in range(2, 13):
        operator = FrameletOperator(adjacency, order, maximum)
        check = operator_checks(operator, np.random.default_rng(81))
        attempts.append(check)
        if (check['adjoint_relative_error'] < 1e-6 and check['zero_mode_residual'] < 1e-8 and
                max(check['highpass_zero_mode_responses']) <= .02 and check['reconstruction_relative_error'] <= .02):
            selected = operator
            break
    write_json(folder / 'operator_attempts.json', attempts)
    if selected is None:
        emit_gate('Framelet', f'n{len(degree)}_operator', attempts[-1], 'all frozen operator gates by n<=12', False,
                  folder / 'operator_attempts.json')
        return None
    agreement = author_operator_error(selected)
    write_json(folder / 'author_operator_comparison.json', {'relative_error': agreement,
               'active_bands': list(selected.active_bands), 'minimal_n': selected.n,
               'author_commit': '4db39904e4099d82cef686d4732c33150814ece4'})
    emit_gate('Framelet', f'n{len(degree)}_operator', {**attempts[-1], 'author_relative_error': agreement},
              'adjoint<1e-6; zero mode<1e-8; highpass<=.02; reconstruction<=.02; author agreement<=1e-8',
              agreement <= 1e-8, folder / 'operator_attempts.json')
    return selected


def match_group_delta(x, operator, paper_result, same_nu_result):
    target = np.linalg.norm(paper_result['U'] - x) / np.linalg.norm(x - x.mean(axis=0))
    same_delta = np.linalg.norm(same_nu_result['U'] - x) / np.linalg.norm(x - x.mean(axis=0))
    low, high = (NU, NU * 2) if same_delta < target else (0., NU)
    best = (abs(same_delta - target), NU, same_nu_result)
    history = [{'nu': NU, 'delta': float(same_delta), 'cached_same_nu_result': True}]
    for index in range(8):
        value = high if index == 0 and same_delta < target else (low + high) / 2
        result = solve_framelet(x, operator, value, True, audit_u_step=True)
        delta = np.linalg.norm(result['U'] - x) / np.linalg.norm(x - x.mean(axis=0))
        history.append({'nu': value, 'delta': float(delta), 'converged': result['converged']})
        if abs(delta - target) < best[0]:
            best = (abs(delta - target), value, result)
        if delta < target:
            low = value
            if value == high:
                high *= 2
        else:
            high = value
    return best[2], {'target_delta': float(target), 'selected_nu': best[1],
                     'absolute_delta_error': float(best[0]), 'relative_delta_error': float(best[0] / max(target, 1e-30)),
                     'new_solver_calls': 8, 'history': history}


def run_framelet():
    family = ROOT / 'framelet'
    family.mkdir(exist_ok=False)
    selected_nus = {}
    for n in (16, 32, 64):
        folder = family / f'n{n}'
        folder.mkdir()
        adjacency, x, clean, observation = synthetic_case(n)
        operator = select_operator(adjacency, folder)
        if operator is None:
            return False
        paths = {'Observation': observation_record(folder / 'observation', x, clean, observation)}
        paper = solve_framelet(x, operator, NU, False, audit_u_step=True)
        group = solve_framelet(x, operator, NU, True, audit_u_step=True)
        matched, matching = match_group_delta(x, operator, paper, group)
        write_json(folder / 'delta_matching.json', matching)
        for label, result, nu, grouped in [('FN-P', paper, NU, False), ('FN-G-nu', group, NU, True),
                                            ('FN-G-delta', matched, matching['selected_nu'], True)]:
            method_dir = folder / label.lower()
            save_native(method_dir, result, 'attempt1')
            if not result['converged']:
                result = solve_framelet(x, operator, nu, grouped, max_iter=10000, audit_u_step=True)
                save_native(method_dir, result, 'attempt2')
            reference = framelet_reference(x, operator.reference_matrices(), operator.degree, nu, grouped)
            save_reference(method_dir, reference)
            gap = normed_gap(result['objective'], reference['reference_objective'])
            lowpass_error = float(np.max(np.abs(result['Q'][0] - operator.analysis(result['U'])[0])))
            summary = {key: result[key] for key in ('objective', 'primal_residual', 'dual_residual', 'stationarity',
                       'pcg_max_relative_residual', 'u_step_dense_relative_error', 'converged', 'iterations')}
            summary.update(reference_objective=reference['reference_objective'], relative_objective_gap=gap,
                lowpass_constraint_max_abs=lowpass_error, lowpass_shrinkage_weight=0., nu=nu,
                active_bands=list(operator.active_bands), selected_n=operator.n,
                initial_objective=framelet_objective(x, x, operator, nu, grouped),
                delta=float(np.linalg.norm(result['U'] - x) / np.linalg.norm(x - x.mean(axis=0))))
            write_json(method_dir / 'checks.json', summary)
            passed = (gap <= 1e-3 and result['primal_residual'] <= 1e-4 and result['dual_residual'] < 1e-3 and
                      result['stationarity'] <= 5e-3 and result['pcg_max_relative_residual'] < 1e-5 and
                      result['u_step_dense_relative_error'] < 1e-5 and lowpass_error < 1e-8 and result['converged'])
            emit_gate('Framelet', f'n{n}_{label}_reference', summary,
                'objective gap<=1e-3; feasibility<=1e-4; stationarity<=5e-3; primal/dual<1e-3; PCG/dense error<1e-5',
                passed, method_dir / 'checks.json')
            paths[label] = chart_record(method_dir, label, result['U'], clean, observation, summary)
        zero = solve_framelet(x, operator, 0., True, audit_u_step=True)
        zero_error = float(np.max(np.abs(zero['U'] - x)))
        write_json(folder / 'nu_zero.json', {'max_abs_error': zero_error})
        emit_gate('Framelet', f'n{n}_nu0', zero_error, '<=1e-8', zero_error <= 1e-8, folder / 'nu_zero.json')
        selected_nus[n] = (operator, matching['selected_nu'])
        evaluate_group(paths, folder)
    operator, matched_nu = selected_nus[32]
    _, _, _, observation = synthetic_case(32)
    transforms = [('rotation', np.array([[.6, -.8], [.8, .6]]), 1., np.zeros(2)),
                  ('reflection', np.diag([-1., 1.]), 1., np.zeros(2)),
                  ('translation', np.eye(2), 1., np.array([7., -4.])),
                  ('scale', np.eye(2), 3.7, np.zeros(2))]
    for name, nu in [('FN-G-nu', NU), ('FN-G-delta', matched_nu)]:
        base = gauge_solve_framelet(observation, operator, nu)['Z']
        errors = {}
        for transform, rotation, scale, shift in transforms:
            solved = gauge_solve_framelet(scale * observation @ rotation + shift, operator, nu)
            restored = (solved['Z'] - shift) @ rotation.T / scale
            errors[transform] = float(np.linalg.norm(restored - base) / np.linalg.norm(base - base.mean(axis=0)))
        write_json(family / f'{name}_equivariance.json', errors)
        emit_gate('Framelet', name + '_equivariance', errors, 'each <1e-3', max(errors.values()) < 1e-3,
                  family / f'{name}_equivariance.json')
    return True


def run_edge(framelet_allowed):
    family = ROOT / 'edge256'
    family.mkdir(exist_ok=False)
    adjacency, x, clean, observation = synthetic_case(256)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    np.savez(family / 'case.npz', X=x, synthetic_positions=clean, observation=observation,
             adjacency=adjacency.toarray())
    operator = select_operator(adjacency, family) if framelet_allowed else None
    paths = {'Observation': observation_record(family / 'observation', x, clean, observation)}
    if operator is not None:
        node = solve_framelet(x, operator, NU, False)
        folder = family / 'fnode'
        save_native(folder, node)
        reference = framelet_reference(x, operator.reference_matrices(), degree, NU, False)
        save_reference(folder, reference)
        gap = normed_gap(node['objective'], reference['reference_objective'])
        emit_gate('Framelet', 'n256_FNode_small_reference',
                  {'gap': gap, 'primal': node['primal_residual'], 'dual': node['dual_residual']},
                  'gap<=1e-3, primal/dual<1e-3', gap <= 1e-3 and max(node['primal_residual'], node['dual_residual']) < 1e-3,
                  folder / 'reference.json')
        paths['FNode-small'] = chart_record(folder, 'FNode-small', node['U'], clean, observation,
                                           {'nu': NU, 'objective': node['objective'], 'role': 'P1 synthetic node control'})
    results = {}
    for label, fixed_u, frames in [('SEF-P-small', True, None), ('FE-P-small', False, None),
                                   ('DoT-P-small', False, operator)]:
        if label == 'DoT-P-small' and operator is None:
            write_json(family / 'dot_blocked.json', {'status': 'BLOCKED', 'reason': 'Framelet upstream numerical gate failed'})
            continue
        for initialization in (('author',) if fixed_u else ('author', 'prior')):
            case_label = label + '-' + initialization
            folder = family / case_label.lower()
            folder.mkdir()
            started = time.perf_counter()
            result = solve_edge_small(x, adjacency, operator=frames, nu=NU if frames else 0.,
                                      fixed_u=fixed_u, initialization=initialization)
            save_native(folder, result, 'attempt1')
            if not result['converged']:
                result = solve_edge_small(x, adjacency, operator=frames, nu=NU if frames else 0.,
                    fixed_u=fixed_u, initialization=initialization, max_iter=15000)
                save_native(folder, result, 'attempt2')
            sref = edge_s_reference(result['U'], degree, not fixed_u)
            save_reference(folder, sref, 'fixed_U_Sblock_reference')
            native_structure_cost = np.abs(result['S']).sum() + np.sum(degree * np.linalg.norm(result['U'] - result['S'] @ result['U'], axis=1))
            s_gap = normed_gap(native_structure_cost, sref['reference_objective'])
            u_gap, u_ref_objective = None, None
            if not fixed_u:
                uref = edge_u_reference(x, result['S'], degree,
                                       frames.reference_matrices() if frames else None, NU if frames else 0.)
                save_reference(folder, uref, 'fixed_S_Ublock_reference')
                u_ref_objective = uref['reference_objective']
                u_gap = normed_gap(result['feasible_objective'], u_ref_objective)
            prior = adjacency.toarray() / degree[:, None]
            feasible_initial_s = np.zeros_like(prior) if fixed_u else prior
            initial_objective = edge_objective(x, feasible_initial_s, x - feasible_initial_s @ x,
                                               x, degree, frames, NU if frames else 0., fixed_u)
            learned = np.abs(result['S']) > 1e-5
            support = adjacency.toarray() != 0
            summary = {key: result[key] for key in ('objective', 'feasible_objective', 'iterations', 'converged',
                       'equality_residual', 'row_sum_residual', 'diagonal_max', 'stationarity', 'pcg_max_relative_residual', 'penalties')}
            summary.update(fixed_U_Sblock_reference_objective=sref['reference_objective'], Sblock_relative_gap=s_gap,
                fixed_S_Ublock_reference_objective=u_ref_objective, Ublock_relative_gap=u_gap,
                global_joint_reference='N/A: joint Eq17/Eq6 is nonconvex; convex block references only' if not fixed_u else sref['reference_objective'],
                feasible_initial_objective=initial_objective,
                finite_objective_decrease=initial_objective - result['feasible_objective'],
                learned_nonzero_fraction=float(np.mean(learned)),
                learned_support_overlap=float(np.count_nonzero(learned & support) / max(1, np.count_nonzero(learned))),
                signed_coefficient_min=float(result['S'].min()),
                row_sum_min=float(result['S'].sum(axis=1).min()), row_sum_max=float(result['S'].sum(axis=1).max()),
                source_support_delta=float(np.linalg.norm(result['S'] - prior)),
                time_s=time.perf_counter() - started, N=256, initialization=initialization,
                produces_denoised_chart=not fixed_u)
            write_json(folder / 'checks.json', summary)
            constraints_pass = (result['equality_residual'] <= 1e-3 and result['diagonal_max'] <= 1e-3 and
                                (fixed_u or result['row_sum_residual'] <= 1e-3) and np.isfinite(result['objective']))
            emit_gate('Edge', case_label + '_constraints', summary,
                      'diag=0, equality<=1e-3; Eq17/6 row-sum<=1e-3; finite output', constraints_pass, folder / 'checks.json')
            if fixed_u:
                emit_gate('Edge', case_label + '_convex_reference', s_gap, '<=1e-3 relative objective gap',
                          s_gap <= 1e-3, folder / 'fixed_U_Sblock_reference.json')
            else:
                emit_gate('Edge', case_label + '_convergence',
                          {'stationarity': result['stationarity'], 'converged': result['converged'],
                           'Sblock_gap': s_gap, 'Ublock_gap': u_gap},
                          'converged; stationarity<=5e-3; convex block objective gaps<=1e-3',
                          result['converged'] and result['stationarity'] <= 5e-3 and s_gap <= 1e-3 and u_gap <= 1e-3,
                          folder / 'checks.json')
                paths[case_label] = chart_record(folder, case_label, result['U'], clean, observation, summary)
            results[case_label] = result
    stability = {}
    for label in ('FE-P-small', 'DoT-P-small'):
        if label + '-author' in results and label + '-prior' in results:
            a, b = results[label + '-author']['U'], results[label + '-prior']['U']
            stability[label] = float(np.linalg.norm(a - b) / np.linalg.norm(a - a.mean(axis=0)))
    write_json(family / 'initialization_sensitivity.json',
               {'relative_output_differences': stability,
                'scope': '256-node implementation smoke; does not replace P3 512/1024 G4 stability gate',
                'nonconvex_global_optimality': 'not claimed'})
    evaluate_group(paths, family)


def run(family):
    ROOT.mkdir(parents=True, exist_ok=True)
    p0 = json.loads(Path('results/stage2_modern_graph_20261019/p0_preflight/manifest.json').read_text())
    if p0['status'] != 'PASS':
        raise ValueError('P0 must be accepted before P1')
    eta = json.loads(DELTA.read_text())['delta0']
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        GATES.extend(json.loads((ROOT / 'gates.json').read_text()))
    else:
        protected = [file_record(Path('AGENTS.md')), file_record(DELTA),
                     file_record(Path('results/stage2_modern_graph_20261019/p0_preflight/manifest.json'))]
        manifest = {'phase': 'P1', 'status': 'running', 'scope': 'small-graph math only; STOP before P2A',
            'eta': eta, 'alpha': 0.5, 'synthetic_nu': NU, 'synthetic_sizes': [16, 32, 64], 'edge_smoke_size': 256,
            'reference': {'cvxpy': cp.__version__, 'solver': 'CLARABEL', 'installed_solvers': cp.installed_solvers()},
            'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()},
            'protected_inputs': protected,
            'initial_git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
            'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
            'commands': [], 'families': {}}
    manifest['commands'].append([sys.executable, '-m', 'diagnostics.stage2_p1.run', '--family', family])
    manifest['code'] = [file_record(path) for folder in ('methods/stage2_modern_graph', 'diagnostics/stage2_p1')
                        for path in Path(folder).glob('*.py')]
    manifest['author_sources'] = json.loads(Path('references/stage2_p1/gnn_dot/source_manifest.json').read_text())
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        with threadpool_limits(1):
            if family in ('ic', 'all'):
                run_ic(eta)
                manifest['families']['IC'] = 'FAIL' if any(r['family'] == 'IC' and r['status'] == 'FAIL' for r in GATES) else 'PASS'
                write_json(manifest_path, manifest)
            if family in ('framelet', 'all'):
                run_framelet()
                manifest['families']['Framelet'] = 'FAIL' if any(r['family'] == 'Framelet' and r['status'] == 'FAIL' for r in GATES) else 'PASS'
                write_json(manifest_path, manifest)
            if family in ('edge', 'all'):
                run_edge(manifest['families'].get('Framelet') == 'PASS')
                manifest['families']['Edge'] = 'FAIL' if any(r['family'] == 'Edge' and r['status'] == 'FAIL' for r in GATES) else 'PASS'
                write_json(manifest_path, manifest)
        for item in manifest['protected_inputs']:
            if file_record(Path(item['path'])) != item:
                raise ValueError('protected P0 input or AGENTS changed')
        manifest['status'] = 'completed' if len(manifest['families']) == 3 else 'partial'
        manifest.setdefault('run_times_s', []).append(time.perf_counter() - started)
        manifest['stop'] = 'P1 only; P2A awaits user review'
        manifest['protected_inputs_unchanged'] = True
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status='BLOCKER', error_type=type(error).__name__, error=str(error))
        write_json(manifest_path, manifest)
        raise
    finally:
        if GATES:
            with (ROOT / 'gates.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(GATES[0]))
                writer.writeheader()
                for row in GATES:
                    writer.writerow({**row, 'actual': json.dumps(row['actual'], ensure_ascii=False)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('ic', 'framelet', 'edge', 'all'), default='all')
    run(parser.parse_args().family)
