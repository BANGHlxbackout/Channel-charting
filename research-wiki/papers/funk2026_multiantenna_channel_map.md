---
type: paper
node_id: paper:funk2026_multiantenna_channel_map
title: "Multiantenna Channel Map Prediction with Missing Location Information using Contrastive Learning and Graph Neural Networks"
authors: ["Wilson Funk", "Seung-Jun Kim"]
year: 2026
venue: "ICASSP 2026"
external_ids:
  arxiv: null
  doi: "10.1109/ICASSP55912.2026.11462797"
  s2: null
tags: ["gnn", "radio-map", "contrastive"]
added: 2026-10-28T11:06:00Z
---

# Multiantenna Channel Map Prediction with Missing Location Information using Contrastive Learning and Graph Neural Networks

## One-line thesis
Graph construction from CSI/location embeddings supports radio-map interpolation; the target is not channel-chart coordinate quality.

## Problem / Gap
Interpolates a multiantenna statistical-CSI radio map when some measurements lack location tags. [Publisher record](https://ieeexplore.ieee.org/abstract/document/11462797/), ICASSP 2026 DOI [10.1109/ICASSP55912.2026.11462797](https://doi.org/10.1109/ICASSP55912.2026.11462797).

## Method
Publisher abstract describes contrastive metric embeddings linking CSI and available locations, a proximity graph over labeled/unlabeled measurements, and GNN-based map interpolation. Full manuscript method and experiments were not accessible in this pass; details beyond the abstract remain unverified.

## Key Results
Publisher abstract reports numerical effectiveness for graph construction and map estimation; no detailed measured-versus-simulated data protocol or chart CT/TW/KS is established here.

## Assumptions
The task includes some location labels and potentially local building-map patches, unlike G1's GT-free training.

## Limitations / Failure Modes
Its output is a radio map rather than UE channel-chart coordinates, and the graph may use location information. It is not evidence that its GNN can be transferred to GT-free G1.

## Reusable Ingredients
Only a high-level example that graph quality and task semantics must be checked separately; no implementation detail imported.

## Open Questions
Obtain the complete author manuscript if this work becomes a serious comparator; otherwise leave it as metadata-level context.

## Claims
No method-performance claim for G1 is supported by the accessible publisher abstract.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Peripheral GNN reference; exclude from primary method selection unless a full text and a clear GT-free charting link are established.

