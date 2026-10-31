"""P0 only: immutable-input audit, G128 incidence, Jacobi200, and same-target CG."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
import torch

from common_metrics import write_json
from methods.g1_acf_graph_fcf.compare import compare_paths
from run_adp_siamese import plot_chart
from run_diffusion_refined_charting import (
    build_local_scaled_graph, compute_neighbor_indices, solve_graph_harmonic_refinement,
)

N = 20997
PARAMETERS = dict(top_k=512, graph_k=128, lambda_smooth=0.15, iterations=200)
DISTANCE = Path('results/adp_old_fused/stride1_n20997_seed10')
SUBSET = Path('results/g1_acf_graph_fcf/comparison/metric_subset_seed10.npy')
INPUTS = {
    'B0': Path('results/adp_old_siamese/stride1_n20997_seed10/embedding.npz'),
    'G1-Identity': Path('results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/embedding.npz'),
    'G1-Graph': Path('results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz'),
}
HISTORY = Path('results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json')
CG_CONFIG = dict(rtol=1e-10, atol=0.0, maxiter=5000, preconditioner='inverse diagonal of I+lambda L')


def file_record(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'sha256': hasher.hexdigest()}


def csr_record(matrix) -> dict:
    matrix = matrix.tocsr(copy=True)
    matrix.sort_indices()
    hasher = hashlib.sha256()
    for array in (np.asarray(matrix.shape, dtype=np.int64), matrix.indptr, matrix.indices, matrix.data):
        hasher.update(str(array.dtype).encode())
        hasher.update(np.ascontiguousarray(array).tobytes())
    return {'shape': list(matrix.shape), 'nnz': int(matrix.nnz),
            'data_dtype': str(matrix.dtype), 'sha256': hasher.hexdigest()}


def stats(array: np.ndarray) -> dict:
    return {'min': float(np.min(array)), 'median': float(np.median(array)),
            'mean': float(np.mean(array)), 'max': float(np.max(array))}


def incidence_and_laplacian(adjacency):
    """One -1/+1 row per strictly upper-triangular edge; original weights unchanged."""
    graph = adjacency.astype(np.float64).tocsr()
    graph.sort_indices()
    upper = sparse.triu(graph, k=1).tocoo()
    m, n = upper.nnz, graph.shape[0]
    rows = np.repeat(np.arange(m, dtype=np.int64), 2)
    columns = np.stack((upper.row, upper.col), axis=1).reshape(-1)
    incidence = sparse.coo_matrix((np.tile([-1., 1.], m), (rows, columns)), shape=(m, n)).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree, format='csr') - graph
    return incidence, upper.data, upper.row, upper.col, laplacian.tocsr(), degree


def solve_same_target_cg(laplacian, y: np.ndarray, strength: float):
    """1/2||Z-Y||² + strength/2 ||sqrt(W)DZ||²; solve after label-free gauge."""
    y = np.asarray(y, dtype=np.float64)
    center = y.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((y - center) ** 2, axis=1))))
    if scale <= 0:
        raise ValueError('zero centered chart scale')
    x = (y - center) / scale
    system = sparse.eye(len(y), format='csr') + strength * laplacian
    preconditioner = sparse.diags(1.0 / system.diagonal(), format='csr')
    u = np.empty_like(x)
    histories = []
    infos = []
    for coordinate in range(2):
        residuals = []
        rhs = x[:, coordinate]

        def callback(current):
            residuals.append(float(np.linalg.norm(system @ current - rhs) / np.linalg.norm(rhs)))

        u[:, coordinate], info = cg(system, rhs, x0=x[:, coordinate],
            M=preconditioner, rtol=CG_CONFIG['rtol'], atol=CG_CONFIG['atol'],
            maxiter=CG_CONFIG['maxiter'], callback=callback)
        histories.append(residuals)
        infos.append(int(info))
    z = center + scale * u
    normalized_residuals = np.linalg.norm(system @ u - x, axis=0) / np.linalg.norm(x, axis=0)
    raw_residuals = np.linalg.norm(system @ z - y, axis=0) / np.linalg.norm(y, axis=0)
    return z, {'center': center, 'scale': scale, 'X': x, 'U': u,
               'info': infos, 'iterations': [len(h) for h in histories],
               'residual_history': histories, 'normalized_relative_residuals': normalized_residuals.tolist(),
               'raw_relative_residuals': raw_residuals.tolist()}


def run(args: argparse.Namespace) -> int:
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    command = [sys.executable, '-m', 'diagnostics.stage2_p0.preflight',
               '--out-dir', str(output), '--device', args.device]
    command_text = subprocess.list2cmdline(command)
    gates = []

    def gate(name, passed, actual, threshold, evidence):
        gates.append({'item': name, 'status': 'PASS' if passed else 'FAIL',
                      'actual': actual, 'threshold': threshold,
                      'evidence': str((output / evidence).resolve()), 'command': command_text})
        write_json(output / 'G0.json', gates)
        if not passed:
            raise ValueError(f'G0 BLOCKER {name}: {actual}; required {threshold}')

    def finish_gates():
        with (output / 'G0.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['item', 'status', 'actual', 'threshold', 'evidence', 'command'])
            writer.writeheader()
            for row in gates:
                writer.writerow({**row, 'actual': json.dumps(row['actual'], ensure_ascii=False)})

    manifest = {'phase': 'P0', 'status': 'running', 'parameters': PARAMETERS, 'cg_config': CG_CONFIG,
                'command': command, 'scope': 'P0 only; stop for user review; no P1 or new regularizers',
                'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                                'torch': str(torch.__version__), 'platform': platform.platform()},
                'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'initial_git_status': subprocess.check_output(['git', 'status', '--short'], text=True)}
    write_json(output / 'manifest.json', manifest)
    (output / 'run_command.txt').write_text(command_text, encoding='utf-8')
    try:
        protected_paths = [Path('AGENTS.md'), DISTANCE / 'geodesic.npy', DISTANCE / 'distance_data.npz',
                           DISTANCE / 'result.json', SUBSET, HISTORY,
                           Path('run_diffusion_refined_charting.py'), Path('common_metrics.py'),
                           Path('run_adp_siamese.py'), Path('methods/g1_acf_graph_fcf/compare.py')]
        for label, path in INPUTS.items():
            protected_paths.extend([path, path.parent / 'result.json'])
            if label != 'B0':
                protected_paths.extend(path.parent / name for name in ('run_config.json', 'model_best.pt',
                    'graph_metadata.json', 'feature_metadata.json'))
        # Protect existing distance caches too; only fresh caches under output may be written.
        protected_paths.extend(path for path in DISTANCE.iterdir() if path.is_file())
        protected_paths = list(dict.fromkeys(protected_paths))
        records = [file_record(path) for path in protected_paths]
        manifest['protected_inputs'] = records
        manifest['distance_directory_listing'] = sorted(path.name for path in DISTANCE.iterdir())
        manifest['code'] = [file_record(Path(__file__)), file_record(Path('tests/test_stage2_p0.py'))]
        manifest['task_path'] = str(Path('docs/tasks/Stage2_2026-10-19.md').resolve())
        write_json(output / 'manifest.json', manifest)
        charts = {name: np.load(path) for name, path in INPUTS.items()}
        distance_data = np.load(DISTANCE / 'distance_data.npz')
        indices, positions = distance_data['indices'], distance_data['positions']
        inputs_check = {}
        for name, chart in charts.items():
            y = chart['embedding']
            inputs_check[name] = {'path': str(INPUTS[name].resolve()), 'shape': list(y.shape),
                'dtype': str(y.dtype), 'N': len(y), 'indices_equal': bool(np.array_equal(chart['indices'], indices)),
                'positions_equal': bool(np.array_equal(chart['positions'], positions)),
                'finite': bool(np.isfinite(y).all()), 'std': np.std(y, axis=0).tolist()}
        geo = np.load(DISTANCE / 'geodesic.npy', mmap_mode='r')
        finite = True
        off_diagonal_zero_count = -len(geo)
        for start in range(0, len(geo), 256):
            block = geo[start:start + 256]
            finite &= bool(np.isfinite(block).all() and (block >= 0).all())
            off_diagonal_zero_count += int(np.count_nonzero(block == 0))
        inputs_check['D_geo'] = {'shape': list(geo.shape), 'dtype': str(geo.dtype), 'finite_nonnegative': finite,
                                'zero_diagonal': bool(np.all(geo.diagonal() == 0)),
                                'off_diagonal_zero_count': off_diagonal_zero_count}
        subset = np.load(SUBSET)
        expected_subset = np.asarray(random.Random(10).sample(range(N), int(N * 0.2)), dtype=np.int64)
        inputs_check['evaluation_subset'] = {'size': len(subset), 'matches_original_seed10_rule': bool(np.array_equal(subset, expected_subset))}
        write_json(output / 'input_checks.json', inputs_check)
        source_meta = json.loads((DISTANCE / 'result.json').read_text())
        lineage = {'distance_run': source_meta, 'charts': {name: json.loads((path.parent / 'result.json').read_text())
                                                        for name, path in INPUTS.items()}}
        write_json(output / 'input_provenance.json', lineage)
        gate('N_and_shapes', len(indices) == N and geo.shape == (N, N) and
             all(value['shape'] == [N, 2] for key, value in inputs_check.items() if key in INPUTS),
             {'N': len(indices), 'embedding_shapes': {name: inputs_check[name]['shape'] for name in INPUTS}},
             'N=20997; all embeddings=(20997,2); D_geo=(20997,20997)', 'input_checks.json')
        gate('indices_and_positions', all(inputs_check[name]['indices_equal'] and inputs_check[name]['positions_equal'] for name in INPUTS),
             {name: {key: inputs_check[name][key] for key in ('indices_equal', 'positions_equal')} for name in INPUTS},
             'all arrays elementwise identical to distance_data', 'input_checks.json')
        gate('finite_noncollapsed_inputs', finite and np.isfinite(positions).all() and all(
             inputs_check[name]['finite'] and max(inputs_check[name]['std']) > 0 for name in INPUTS),
             {name: {'finite': inputs_check[name]['finite'], 'std': inputs_check[name]['std']} for name in INPUTS},
             'finite embeddings/D_geo/positions; nonzero embedding std', 'input_checks.json')
        gate('original_topk_self_precondition', inputs_check['D_geo']['zero_diagonal'] and off_diagonal_zero_count == 0,
             inputs_check['D_geo'], 'zero diagonal; no non-self zero ties before original topk[:,1:]', 'input_checks.json')
        gate('evaluation_subset', np.array_equal(subset, expected_subset), inputs_check['evaluation_subset'],
             'same fixed 4199 seed10 evaluation indices', 'input_checks.json')
        historical = json.loads(HISTORY.read_text())
        write_json(output / 'historical_reference.json', {'path': str(HISTORY.resolve()), 'record': historical})
        if any(historical[key] != PARAMETERS[key] for key in PARAMETERS):
            raise ValueError('historical baseline parameter mismatch')
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            manifest['environment']['gpu'] = torch.cuda.get_device_name(device)
        cache = output / 'graph'
        cache.mkdir()
        neighbors = compute_neighbor_indices(geo, PARAMETERS['top_k'], device, cache / 'geodesic_top512.npy')
        graph = build_local_scaled_graph(geo, neighbors, PARAMETERS['graph_k']).tocsr()
        graph.sort_indices()
        construction_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        asymmetry = graph.astype(np.float64) - graph.T.astype(np.float64)
        asym_max = float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0
        incidence, weights, edge_i, edge_j, laplacian, degree = incidence_and_laplacian(graph)
        degree_counts = np.diff(graph.indptr)
        random_probe = np.random.default_rng(10).standard_normal((N, 4))
        incidence_matvec = incidence.T @ (weights[:, None] * (incidence @ random_probe))
        csr_matvec = laplacian @ random_probe
        matvec_relative = float(np.linalg.norm(incidence_matvec - csr_matvec) / np.linalg.norm(csr_matvec))
        l1_max = float(np.max(np.abs(laplacian @ np.ones(N))))
        sparse.save_npz(cache / 'G128.npz', graph)
        sparse.save_npz(cache / 'incidence_D.npz', incidence)
        sparse.save_npz(cache / 'L_csr_float64.npz', laplacian)
        np.savez(cache / 'edges_and_degrees.npz', edge_i=edge_i, edge_j=edge_j, edge_weights=weights,
                 weighted_degree=degree, unweighted_degree=degree_counts)
        graph_info = {'G128': csr_record(graph), 'incidence_D': csr_record(incidence),
            'L_csr': csr_record(laplacian), 'undirected_edges': len(weights),
            'symmetric_nnz_entries': graph.nnz, 'symmetry_max_abs': asym_max,
            'self_loops': int(np.count_nonzero(graph.diagonal())), 'weights': stats(weights),
            'weighted_degree': stats(degree), 'unweighted_degree': stats(degree_counts),
            'connected_components': int(connected_components(graph, directed=False)[0]),
            'incidence_row_nnz_unique': np.unique(np.diff(incidence.indptr)).tolist(),
            'incidence_row_sum_max_abs': float(np.max(np.abs(np.asarray(incidence.sum(axis=1))))),
            'incidence_values_unique': np.unique(incidence.data).tolist(),
            'matvec_float64_relative_error': matvec_relative, 'L_times_one_max_abs': l1_max,
            'probe_seed': 10, 'probe_columns': 4, 'construction_peak_vram_bytes': construction_peak}
        write_json(output / 'graph_checks.json', graph_info)
        gate('G128_structure', asym_max == 0 and graph_info['self_loops'] == 0 and
             weights.min() > 0 and np.isfinite(weights).all() and degree.min() > 0 and
             degree_counts.min() >= 128 and graph.nnz == 2 * len(weights) and
             np.all(np.diff(incidence.indptr) == 2) and np.all(edge_i < edge_j),
             graph_info, 'symmetric; no self loops; finite positive weights/degrees; each undirected edge once in D', 'graph_checks.json')
        gate('DTWD_matches_CSR_L', matvec_relative < 1e-6, matvec_relative, '<1e-6 (float64 relative matvec)', 'graph_checks.json')
        gate('L_times_one', l1_max < 1e-8, l1_max, '<1e-8 max absolute', 'graph_checks.json')
        b0 = charts['B0']['embedding']
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        z_j = solve_graph_harmonic_refinement(b0, graph.tocoo(), 0.15, 200, device)
        j_seconds = time.perf_counter() - begin
        j_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        begin = time.perf_counter()
        z_cg, cg_info = solve_same_target_cg(laplacian, b0, 0.15)
        cg_seconds = time.perf_counter() - begin
        np.savez(output / 'chart_gauge.npz', center=cg_info['center'], scale=cg_info['scale'],
                 X=cg_info['X'], U_cg=cg_info['U'], U_j=(z_j - cg_info['center']) / cg_info['scale'])
        system = sparse.eye(N, format='csr') + 0.15 * laplacian
        j_relative = float(np.linalg.norm(system @ z_j.astype(np.float64) - b0) / np.linalg.norm(b0))
        solver_info = {'target': '(I+0.15*L)Z=Y; equivalently normalized (I+0.15*L)U=X',
            'objective': '0.5||U-X||_F^2 + 0.15/2 ||sqrt(W) D U||_F^2; each undirected edge once',
            'jacobi': {'dtype': 'float32, original function unchanged', 'iterations': 200,
                       'relative_raw_linear_residual': j_relative, 'time_s': j_seconds, 'peak_vram_bytes': j_peak},
            'cg': {**CG_CONFIG, 'dtype': 'float64', 'info': cg_info['info'], 'iterations': cg_info['iterations'],
                   'normalized_relative_residuals': cg_info['normalized_relative_residuals'],
                   'raw_relative_residuals': cg_info['raw_relative_residuals'], 'time_s': cg_seconds},
            'cg_minus_jacobi_raw': {'max_abs': float(np.max(np.abs(z_cg - z_j))),
                'rms_per_node': float(np.sqrt(np.mean(np.sum((z_cg - z_j) ** 2, axis=1))))}}
        write_json(output / 'solver_checks.json', solver_info)
        write_json(output / 'cg_residual_history.json', cg_info['residual_history'])
        residual_max = max(cg_info['normalized_relative_residuals'] + cg_info['raw_relative_residuals'])
        gate('S2_CG_linear_residual', all(info == 0 for info in cg_info['info']) and residual_max <= 1e-7,
             {'max_relative_residual': residual_max, 'solver_info': cg_info['info']}, '<=1e-7; CG exits converged', 'solver_checks.json')
        numerator = float(np.linalg.norm(z_j.astype(np.float64) - b0))
        denominator = float(np.linalg.norm(b0.astype(np.float64) - cg_info['center']))
        delta = numerator / denominator
        delta_info = {'delta0': delta, 'numerator_frobenius': numerator, 'denominator_centered_frobenius': denominator,
            'formula': '||Z_B0+S2-J-Y_B0||_F / ||Y_B0-mean(Y_B0)||_F',
            'anchor': 'original Jacobi200 output, not CG or GT-selected output',
            'input_B0': file_record(INPUTS['B0']), 'parameters': PARAMETERS,
            'jacobi_output': str((output / 's2_j/embedding.npz').resolve()),
            'normalized_delta0': float(np.linalg.norm((z_j - cg_info['center']) / cg_info['scale'] - cg_info['X']) /
                                       np.linalg.norm(cg_info['X'])),
            'c': cg_info['center'].tolist(), 'a': cg_info['scale']}
        paths = {'B0': INPUTS['B0']}
        for label, folder_name, z, timing in [('B0+S2-J', 's2_j', z_j, j_seconds),
                                              ('B0+S2-CG', 's2_cg', z_cg, cg_seconds)]:
            folder = output / folder_name
            folder.mkdir()
            np.savez(folder / 'embedding.npz', indices=indices, positions=positions, embedding=z)
            record = {'status': 'computed_awaiting_evaluation', 'phase': 'P0', 'method': label,
                'num_samples': N, 'seed': 10, **PARAMETERS, 'graph_hash': graph_info['G128']['sha256'],
                'source_embedding': str(INPUTS['B0'].resolve()), 'postprocess_time_s': timing,
                'solver': solver_info['jacobi' if folder_name == 's2_j' else 'cg']}
            if folder_name == 's2_cg':
                record.update(jacobi_reference_iterations=200, iterations=cg_info['iterations'])
            write_json(folder / 'result.json', record)
            write_json(folder / 'run_config.json', {'phase': 'P0', **PARAMETERS, 'command': command,
                       'solver_kind': 'CG' if folder_name == 's2_cg' else 'Jacobi',
                       'iterations': cg_info['iterations'] if folder_name == 's2_cg' else 200,
                       'cg_config': CG_CONFIG if folder_name == 's2_cg' else None, 'device': str(device)})
            paths[label] = folder / 'embedding.npz'
        delta_info['jacobi_output_hash'] = file_record(paths['B0+S2-J'])['sha256']
        write_json(output / 'delta0.json', delta_info)
        gate('delta0_auditable', np.isfinite(delta) and delta >= 0 and denominator > 0 and
             abs(delta - delta_info['normalized_delta0']) < 1e-12,
             delta_info, 'finite scalar; saved raw-J anchor, input/output SHA256, formula and gauge', 'delta0.json')
        results = compare_paths(paths, output / 'comparison', seed=10, subset_path=SUBSET)
        for label in ('B0+S2-J', 'B0+S2-CG'):
            path = paths[label].parent
            row = results[label]
            record = json.loads((path / 'result.json').read_text())
            record.update(status='completed', metrics={key: row[key] for key in ('ct', 'tw', 'ks')},
                          localization_errors_after_affine={key: row[key] for key in
                            ('mean_m', 'median_m', 'p90_m', 'p95_m', 'max_m')})
            write_json(path / 'result.json', record)
            z = z_j if label.endswith('-J') else z_cg
            plot_chart(path / 'chart.png', positions, z,
                       f"{label}\nCT={row['ct']:.3f}, TW={row['tw']:.3f}, KS={row['ks']:.3f}")
        history_diff = {key: abs(results['B0+S2-J'][key] - historical[source][original])
                        for key, source, original in [('ks', 'metrics', 'ks'),
                            ('mean_m', 'localization_errors_after_affine', 'mean_m'),
                            ('p95_m', 'localization_errors_after_affine', 'p95_m')]}
        write_json(output / 'historical_comparison.json', {'history': str(HISTORY.resolve()), 'absolute_differences': history_diff})
        for key, threshold in [('ks', 2e-4), ('mean_m', 2e-3), ('p95_m', 3e-3)]:
            gate('Jacobi_history_' + key, history_diff[key] <= threshold, history_diff[key],
                 f'absolute difference <= {threshold}', 'historical_comparison.json')
        after = [file_record(path) for path in protected_paths]
        unchanged = after == records and sorted(path.name for path in DISTANCE.iterdir()) == manifest['distance_directory_listing']
        write_json(output / 'immutability_checks.json', {'unchanged': unchanged, 'before': records, 'after': after})
        gate('source_inputs_and_AGENTS_unchanged', unchanged, {'unchanged': unchanged, 'files': len(records)},
             'all SHA256 equal; no source distance-run entries added', 'immutability_checks.json')
        manifest.update(status='PASS', gate='G0 PASS; STOP before P1', total_time_s=time.perf_counter() - started,
                        results=results, delta0=delta, output_hashes=[file_record(path) for path in
                          (cache / 'G128.npz', cache / 'incidence_D.npz', cache / 'L_csr_float64.npz',
                           cache / 'edges_and_degrees.npz', output / 'delta0.json', *[paths[k] for k in ('B0+S2-J', 'B0+S2-CG')])])
        write_json(output / 'manifest.json', manifest)
        finish_gates()
        print(json.dumps({'status': manifest['status'], 'delta0': delta, 'graph': graph_info,
                          'solver': solver_info, 'results': results}, indent=2), flush=True)
        return 0
    except Exception as error:
        manifest.update(status='BLOCKER', error_type=type(error).__name__, error=str(error),
                        total_time_s=time.perf_counter() - started, gate='STOP; no P1')
        write_json(output / 'manifest.json', manifest)
        finish_gates()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('results/stage2_modern_graph_20261019/p0_preflight'))
    parser.add_argument('--device', default='cuda:0')
    sys.exit(run(parser.parse_args()))
