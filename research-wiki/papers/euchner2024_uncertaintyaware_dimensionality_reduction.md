---
type: paper
node_id: paper:euchner2024_uncertaintyaware_dimensionality_reduction
title: "Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss"
authors: ["Florian Euchner", "Phillip Stephan", "Stephan ten Brink"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2412.01715"
  doi: null
  s2: null
tags: ["channel-charting", "measured", "loss"]
added: 2026-10-28T11:05:30Z
---

# Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss

## One-line thesis
Measured DICHASUS study of geodesic and uncertainty-aware losses; its path-based target is separate from current G1.

## Problem / Gap
Examines mismatch between a Euclidean chart-distance loss and graph-geodesic pseudo-distances on nonconvex measured channel manifolds; models uncertainty in CSI/time-derived distances. [Primary text](https://arxiv.org/html/2412.01715), Sections IV–VIII.

## Method
Uses a batch-wise charting architecture to make many outputs available to a path-based chart geodesic loss. It saves graph shortest-path predecessors, sums chart segment lengths along those paths, and uses a schedule of path sub-sampling to avoid poor optimization from random initialization. A separate uncertainty model maps observed dissimilarities to conditional physical-distance distributions and weights a geodesic loss by modeled path variance; acceleration is another separate term.

## Key Results
Measured DICHASUS cf0x with four 2×4 arrays and 13 CIR taps. In its full-array ablation, path geodesic loss MAE 0.340 m, with acceleration 0.318 m, uncertainty-only 0.339 m; the authors explicitly caution that hyperparameters matter. Single-array results show larger differences. These numbers are not comparable to our 20,997-point G1 protocol.

## Assumptions
Path variance relies on conditional Gaussian models and covariance simplifications: ADP edge errors treated independent, time-derived errors conservatively correlated. Shortest paths are fixed from the graph; the path-based objective needs predecessor information and a training schedule.

## Limitations / Failure Modes
This is an already published, separate loss program, not a new graph-propagation mechanism. Directly dropping its path/uncertainty/acceleration terms into G1 would change several factors and require new assumptions and memory. The old `geodesic_stage1_loss.md` plan is not active for the current task.

## Reusable Ingredients
Use as a warning that pseudo-distance fitting and physical geometry can differ, and as prior art when assessing any proposed reliability-weighted loss. Keep changes to graph propagation distinguishable from objective changes.

## Open Questions
Is there a GT-free edge or path reliability signal on this exact measured subset that predicts the observed target mismatch after matching distance and region? Current local-error correlations are weak.

## Claims
Paper reports gains for its own batch-wise/path/uncertainty setup. It does not show that generic Laplacian smoothing or ADP-only propagation improves our G1 Graph versus Identity control.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Close prior art for target and uncertainty claims; do not treat as an instruction to rerun the rejected historical loss plan. Any new loss must identify a distinct problem and controlled effect on the frozen G1 setting.

## Abstract (original)

> Channel Charting is a dimensionality reduction technique that learns to reconstruct a low-dimensional, physically interpretable map of the radio environment by taking advantage of similarity relationships found in high-dimensional channel state information. One particular family of Channel Charting methods relies on pseudo-distances between measured CSI datapoints, computed using dissimilarity metrics. We suggest several techniques to improve the performance of dissimilarity metric-based Channel Charting. For one, we address an issue related to a discrepancy between Euclidean distances and geodesic distances that occurs when applying dissimilarity metric-based Channel Charting to datasets with nonconvex low-dimensional structure. Furthermore, we incorporate the uncertainty of dissimilarities into the learning process by modeling dissimilarities not as deterministic quantities, but as probability distributions. Our framework facilitates the combination of multiple dissimilarity metrics in a consistent manner. Additionally, latent space dynamics like constrained acceleration due to physical inertia are easily taken into account thanks to changes in the training procedure. We demonstrate the achieved performance improvements for localization applications on a measured channel dataset

