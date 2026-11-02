# Graph-Diffusion Channel Charting

An independent experimental line for Channel Charting. It does not replace or reinterpret the historical G1-ACF experiment.

## Research question

Can graph diffusion over an ADP-similarity graph improve the learned channel chart? Does learning edge-wise diffusion strengths help beyond fixed heat diffusion or ordinary GCN message passing?

Three encoders share the same geodesic pair-distance training objective:

| Model | Propagation | Purpose |
|---|---|---|
| `gcn` | Three fixed graph-convolution layers | Controlled GNN baseline, close in spirit to the 2026 Channel Charting graph paper |
| `heat` | Fixed graph heat equation, solved with RK4 | Test whether multi-scale heat diffusion helps |
| `adaptive` | Edge-conditioned anisotropic diffusion with a fidelity term | Test whether learned conductance can suppress unreliable propagation |

The GNN experiment is separate from Transformer experiments. No Transformer or Q/K/V attention is used.

## Why retain the geodesic training loss?

All three models use the repository's existing geodesic Siamese distance loss. This first comparison changes the representation architecture while keeping the training objective fixed.

The 2026 paper *Rethinking Channel Charting: A Graph Perspective* introduces a position-diffusion / Laplacian-smoothness formulation. It is useful theoretical motivation, but a standalone Laplacian-smoothness objective has a constant embedding as a minimizer on a connected graph. This implementation therefore does not use that term alone as the training objective.

Ground-truth positions are not used for graph construction, training, validation, or checkpoint selection. They are used only for final CT/TW/KS and affine-aligned localization evaluation.

## Graph construction

- Symmetrized kNN support from either ADP dissimilarity or fused ADP/time dissimilarity.
- Locally scaled heat-kernel affinity: `w_ij = exp(-d_ij^2 / (sigma_i sigma_j))`.
- `--edge-attr-mode adp` gives the adaptive gate ADP edge distance only, providing a controlled physical-graph setting.
- `--edge-attr-mode multiview` additionally exposes fused distance and timestamp gap to the adaptive gate.
- The graph remains sparse because the intended manifold prior is local. Dense connectivity is not assumed to be better.

## Models

### Fixed heat diffusion

The hidden state follows `dH/dt = -L_sym H` and is integrated with a differentiable RK4 solver. The chart decoder receives the initial representation and three diffusion-time snapshots.

### Adaptive anisotropic diffusion

The learned conductance is the product of the base graph affinity and a feature-conditioned edge gate. The evolution is

`dH/dt = alpha (P_theta(H) H - H) + beta (H0 - H)`.

The first term is nonlinear graph diffusion. The second is a fidelity term that retains information from the initial encoded CSI and reduces uncontrolled oversmoothing. Positive diffusion rates are bounded for stable RK4 integration. The decoder fuses the initial state and multiple diffusion scales.

This is GRAND-inspired graph neural diffusion, not an exact reproduction of GRAND's attention parameterisation.

## Run

First create the normal ADP/fused/geodesic distance run using run_adp_fused.py. Then pass that run directory to this script.

Example distance directory (adjust if the actual run has a different name):

`results/adp_fused/stride1_n20997_seed10`

Run the three model comparisons with the same data, graph, seed, edge attributes, and loss:

```bash
python -m methods.grand_diffusion_charting.run --distance-run results/adp_fused/stride1_n20997_seed10 --model gcn --features raw --graph-source adp --edge-attr-mode adp --seed 10

python -m methods.grand_diffusion_charting.run --distance-run results/adp_fused/stride1_n20997_seed10 --model heat --features raw --graph-source adp --edge-attr-mode adp --seed 10

python -m methods.grand_diffusion_charting.run --distance-run results/adp_fused/stride1_n20997_seed10 --model adaptive --features raw --graph-source adp --edge-attr-mode adp --seed 10
```

Then test multi-view edge information and the ACF feature branch:

```bash
python -m methods.grand_diffusion_charting.run --distance-run results/adp_fused/stride1_n20997_seed10 --model adaptive --features raw --graph-source adp --edge-attr-mode multiview --seed 10

python -m methods.grand_diffusion_charting.run --distance-run results/adp_fused/stride1_n20997_seed10 --model adaptive --features acf --feature-mode stream --graph-source adp --edge-attr-mode adp --seed 10
```

To compare the fused graph, change `--graph-source adp` to `--graph-source fused`. To assess seed sensitivity, repeat with seeds 11 and 12. Keep all other settings identical when comparing models.

Useful parameters:

- `--neighbors 30`: kNN degree.
- `--diffusion-steps 8`: RK4 steps.
- `--diffusion-time 1.0`: total diffusion time.
- `--hidden-dim 256`: latent width.
- `--max-steps 2000`: maximum training steps.
- `--pairs-per-step 65536`: geodesic pairs per update.

## Outputs

Each run writes chart.png, embedding.npz, result.json, training_history.json, graph_metadata.json, model_best.pt, and model_last.pt.

The evaluation subset is matched to the existing G1 evaluation protocol to make CT/TW/KS comparisons consistent.

## Validation status

The graph builder and all three model variants passed local synthetic forward/backward tests, including a short end-to-end synthetic training run. The real DICHASUS dataset has not been run in this environment; no real-data improvement is claimed yet.

## References

1. Y. Jin, Y. Zhao, D. Hao, and G. Fodor, "Rethinking Channel Charting: A Graph Perspective," arXiv:2609.06204, 2026.
2. B. P. Chamberlain et al., "GRAND: Graph Neural Diffusion," ICML, 2021.
3. Y. Li et al., "A Generalized Neural Diffusion Framework on Graphs," AAAI, 2024.
4. P. Stephan, F. Euchner, and S. ten Brink, "Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting," 2023.
5. DICHASUS, "Dissimilarity Metric-Based Channel Charting" tutorial.
