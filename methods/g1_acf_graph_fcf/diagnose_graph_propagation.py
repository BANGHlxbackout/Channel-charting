"""E2: frozen-checkpoint adjacency sensitivity, without training or GT inputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common_metrics import write_json
from methods.g1_acf_graph_fcf.graph_utils import (
    audit_shortest_paths, feature_cache, file_identity, identity_adjacency,
    propagation_adjacency, rebuild_directed_graph, validation_pairs,
)
from methods.g1_acf_graph_fcf.run import GraphFCFWithExistingACF
from methods.g1_acf_graph_fcf.run_stage2_comparison import (
    DISTANCE, INITIAL, check_protected_inputs, digest, finish_evaluation, load_initial_charts,
)
from run_adp_siamese import siamese_loss


def distribution(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def state_digest(model: torch.nn.Module) -> str:
    return digest(*(value.detach().cpu().numpy() for value in model.state_dict().values()))


@torch.no_grad()
def probe_forward(model: GraphFCFWithExistingACF, features, adjacency,
                  chunk_size: int | None) -> tuple[np.ndarray, dict]:
    if model.training:
        raise ValueError("probe requires model.eval()")
    arrays = {}
    handles = []
    for layer_index, layer in enumerate(model.convs):
        context = {}

        def capture_input(module, inputs, context=context):
            context['h'] = inputs[0]

        def capture_message(module, inputs, context=context):
            context['message'] = inputs[0]

        def capture_delta(module, inputs, output, context=context):
            context['delta'] = output

        def capture_layer(module, inputs, output, context=context, layer_index=layer_index):
            h, message, delta = context['h'], context['message'], context['delta']
            h_norm = torch.linalg.vector_norm(h, dim=1)
            denominator = h_norm.clamp_min(1e-12)
            fields = {
                'h_norm': h_norm,
                'message_norm': torch.linalg.vector_norm(message, dim=1),
                'delta_norm': torch.linalg.vector_norm(delta, dim=1),
                'delta_over_h': torch.linalg.vector_norm(delta, dim=1) / denominator,
                'message_minus_self_over_h': torch.linalg.vector_norm(message - h, dim=1) / denominator,
            }
            for key, value in fields.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"nonfinite layer {layer_index} {key}")
                arrays[f'layer{layer_index}_{key}'] = value.cpu().numpy()
            context.clear()

        handles.extend([layer.register_forward_pre_hook(capture_input),
                        layer.lin.register_forward_pre_hook(capture_message),
                        layer.drop.register_forward_hook(capture_delta),
                        layer.register_forward_hook(capture_layer)])
    try:
        output = model(features, adjacency, chunk_size).cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return output, arrays


def run(args: argparse.Namespace) -> None:
    e1 = json.loads((args.out_root / 'e1_stage2/manifest.json').read_text())
    if e1['status'] != 'completed':
        raise ValueError('E1 must be completed and saved before E2')
    output = args.out_root / 'e2_propagation'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    charts, indices, positions = load_initial_charts()
    fused = np.load(DISTANCE / 'fused.npy', mmap_mode='r')
    geo = np.load(DISTANCE / 'geodesic.npy', mmap_mode='r')
    graph_result = json.loads((INITIAL['G1-Graph'].parent / 'result.json').read_text())
    graph_meta = json.loads((INITIAL['G1-Graph'].parent / 'graph_metadata.json').read_text())
    directed = rebuild_directed_graph(fused, graph_result['neighbors'])
    path_audit = audit_shortest_paths(directed, geo)
    graph_adj, affinity_info = propagation_adjacency(directed, fused)
    if affinity_info != graph_meta['affinity']:
        raise ValueError('GCN propagation graph differs from original metadata')
    graph_adj = graph_adj.to(device)
    identity_adj = identity_adjacency(len(indices)).to(device)
    feat_meta = json.loads((INITIAL['G1-Graph'].parent / 'feature_metadata.json').read_text())
    cache_path = Path(feat_meta['cache_path'])
    if not cache_path.exists() or not cache_path.with_suffix('.json').exists():
        raise FileNotFoundError('original ACF feature cache required; do not regenerate')
    data_path = Path(graph_result['dataset'])
    h = np.load(data_path / 'H.npy', mmap_mode='r').reshape(len(indices), 4, 8, 13)
    features, feature_info = feature_cache(data_path, indices, h, cache_path.parent, 128, device)
    if feature_info['h_file'] != feat_meta['h_file']:
        raise ValueError('CSI identity differs from original feature metadata')
    val_left, val_right, val_keys = validation_pairs(len(indices), 200000, 100010)
    pair_hash = digest(val_keys)
    for label in ('G1-Graph', 'G1-Identity'):
        original = json.loads((INITIAL[label].parent / 'result.json').read_text())
        if pair_hash != original['validation_pair_hash']:
            raise ValueError('validation pair hash differs from training record')
    pair_info = {'hash': pair_hash, 'count': len(val_keys),
                 'left_index_distribution': distribution(val_left),
                 'right_index_distribution': distribution(val_right),
                 'limitation': 'original sorted-key truncation biases pairs toward smaller indices; reproduced unchanged'}
    write_json(output / 'validation_pairs.json', pair_info)
    np.save(output / 'validation_keys.npy', val_keys)
    left = torch.from_numpy(val_left).to(device)
    right = torch.from_numpy(val_right).to(device)
    target = torch.from_numpy(np.asarray(geo[val_left, val_right]).astype(np.float32)).to(device)
    manifest = {'status': 'computing', 'command': [sys.executable, *sys.argv],
                'path_audit': path_audit, 'affinity': affinity_info, 'validation_pairs': pair_info,
                'native_tolerance': {'atol': 1e-4, 'rtol': 1e-4}, 'probes': {}}
    write_json(output / 'manifest.json', manifest)
    models = {}
    for mode in ('graph', 'identity'):
        folder = INITIAL['G1-Graph' if mode == 'graph' else 'G1-Identity'].parent
        config = json.loads((folder / 'run_config.json').read_text())
        model = GraphFCFWithExistingACF(input_dim=config['input_dim'], hidden=config['hidden_dim'],
                                       layers=config['num_layers'], dropout=config['dropout']).to(device)
        model.load_state_dict(torch.load(folder / 'model_best.pt', map_location=device, weights_only=True))
        model.eval()
        models[mode] = model
    saved_outputs = {}
    paths = {}
    layer_rows = []

    def run_one(key: str, label: str, mode: str, adj: torch.Tensor, native_label: str | None):
        model = models[mode]
        before = state_digest(model)
        torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        y, arrays = probe_forward(model, features, adj, 128)
        elapsed = time.perf_counter() - begin
        if before != state_digest(model):
            raise ValueError('frozen parameters or BatchNorm buffers changed')
        with torch.no_grad():
            prediction = torch.from_numpy(y).to(device)
            stress = float(siamese_loss(prediction[left], prediction[right], target,
                                        graph_result['margin']).item())
        info = {'checkpoint_mode': mode, 'stress': stress, 'inference_time_s': elapsed,
                'peak_vram_bytes': torch.cuda.max_memory_allocated(device),
                'state_unchanged': True, 'layer_statistics': {}}
        if native_label is not None:
            expected = charts[native_label]['embedding']
            info['native_embedding_max_abs_diff'] = float(np.max(np.abs(y - expected)))
            if not np.allclose(y, expected, atol=1e-4, rtol=1e-4):
                raise ValueError(f'{label} fails native checkpoint reproduction')
            old = json.loads((INITIAL[native_label].parent / 'result.json').read_text())
            info['stored_validation_stress'] = old['best_validation_loss']
            if not np.isclose(stress, old['best_validation_loss'], atol=1e-4, rtol=1e-5):
                raise ValueError(f'{label} fails validation stress reproduction')
        for field, array in arrays.items():
            if field.endswith('_over_h'):
                stats = distribution(array)
                info['layer_statistics'][field] = stats
                layer_rows.append({'probe': label, 'quantity': field, **stats})
        folder = output / key
        folder.mkdir()
        np.savez(folder / 'embedding.npz', indices=indices, positions=positions, embedding=y)
        np.savez(folder / 'layer_values.npz', indices=indices, **arrays)
        checkpoint = INITIAL['G1-Graph' if mode == 'graph' else 'G1-Identity'].parent / 'model_best.pt'
        write_json(folder / 'result.json', {'status': 'computed_awaiting_evaluation',
                   'phase': 'frozen_checkpoint_diagnostic', 'method': label,
                   'source_checkpoint': str(checkpoint.resolve()), 'num_samples': len(indices),
                   'seed': 10, 'protocol': 'post-training adjacency sensitivity; not retraining', **info})
        write_json(folder / 'run_config.json', {'checkpoint': str(checkpoint.resolve()),
                   'adjacency': 'graph' if key == 'graph_at_graph' else 'identity',
                   'eval': True, 'grad_enabled': False, 'feature_chunk_size': 128,
                   'validation_pair_hash': pair_hash, 'margin': graph_result['margin']})
        saved_outputs[key] = y
        paths[label] = folder / 'embedding.npz'
        manifest['probes'][label] = info
        write_json(output / 'manifest.json', manifest)

    # Both original models must reproduce before applying the shifted adjacency.
    run_one('graph_at_graph', 'Graph ckpt @ Graph', 'graph', graph_adj, 'G1-Graph')
    run_one('identity_at_identity', 'Identity ckpt @ I', 'identity', identity_adj, 'G1-Identity')
    run_one('graph_at_identity', 'Graph ckpt @ I', 'graph', identity_adj, None)
    change = np.linalg.norm(saved_outputs['graph_at_graph'] - saved_outputs['graph_at_identity'], axis=1)
    native = saved_outputs['graph_at_graph']
    manifest['graph_checkpoint_raw_displacement'] = {**distribution(change), 'unit': 'raw chart coordinates',
        'relative_centered_rms': float(np.linalg.norm(saved_outputs['graph_at_graph'] -
                                                      saved_outputs['graph_at_identity']) /
                                       np.linalg.norm(native - native.mean(axis=0)))}
    normal_stress = manifest['probes']['Graph ckpt @ Graph']['stress']
    changed_stress = manifest['probes']['Graph ckpt @ I']['stress']
    manifest['stress_change'] = {'absolute': changed_stress - normal_stress,
                                 'relative': changed_stress / normal_stress - 1}
    np.savez(output / 'embedding_changes.npz', indices=indices, raw_displacement=change)
    with (output / 'layer_statistics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['probe', 'quantity', 'mean', 'median', 'p10', 'p90', 'p95', 'max'])
        writer.writeheader()
        writer.writerows(layer_rows)
    ordered = {label: paths[label] for label in ('Graph ckpt @ Graph', 'Graph ckpt @ I', 'Identity ckpt @ I')}
    manifest['chart_evaluation'] = finish_evaluation(ordered, output)
    check_protected_inputs(args.out_root)
    manifest.update(status='completed', total_time_s=time.perf_counter() - started,
                    source_inputs_unchanged=True, optional_identity_at_graph='not executed')
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', type=Path, default=Path('results/g1_acf_graph_fcf_followup'))
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())
