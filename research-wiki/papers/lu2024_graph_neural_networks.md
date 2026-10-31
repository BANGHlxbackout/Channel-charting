---
type: paper
node_id: paper:lu2024_graph_neural_networks
title: "Graph Neural Networks for Wireless Networks: Graph Representation, Architecture and Evaluation"
authors: ["Yang Lu", "Yuhang Li", "Ruichen Zhang", "Wei Chen", "Bo Ai", "Dusit Niyato"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2404.11858"
  doi: null
  s2: null
tags: ["gnn", "graph-architecture", "survey"]
added: 2026-10-28T11:05:37Z
---

# Graph Neural Networks for Wireless Networks: Graph Representation, Architecture and Evaluation

## One-line thesis
Wireless GNN survey discusses graph representation and architecture, with resource allocation rather than charting as its task.

## Problem / Gap
Survey of how graph representation, aggregation, attention, and residual updates affect wireless-network optimization. [Primary text](https://arxiv.org/html/2404.11858), Sections II–V; IEEE Wireless Communications 2025 DOI [10.1109/MWC.006.2400131](https://doi.org/10.1109/MWC.006.2400131).

## Method
Contrasts isomorphic and heterogeneous wireless graphs, then reviews vanilla message passing, multi-head attention, residual connections, and task-specific constraints. Its case study concerns resource allocation/beamforming, not channel-chart coordinates.

## Key Results
Reports application-specific comparisons among GCN, GAT, and residual/attention variants on wireless optimization metrics. These results do not identify an optimal graph architecture for measured channel charting.

## Assumptions
Graph nodes and edges encode the wireless optimization system, such as users, links, interference, or antennas. These semantic graph objects differ from G1's UE measurement-neighbor graph.

## Limitations / Failure Modes
A survey/case-study overview cannot justify adding attention or a third propagation layer solely because they are popular. G1 already has a residual path; greater depth can enlarge exposure to incorrect neighbors.

## Reusable Ingredients
Distinguish graph representation quality from message/update architecture. If testing depth or attention, use matched Identity controls and change only one architecture factor.

## Open Questions
Is G1's lack of incremental propagation gain caused by unreliable edge semantics or by its update operator? Requires measured-data diagnosis, not transfer from resource-allocation benchmarks.

## Claims
Paper's architectural discussion is background, not primary evidence of G1 chart improvement.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Graph architecture reference with clear task mismatch; useful for designing controls and failure checks.

## Abstract (original)

> Graph neural networks (GNNs) have been regarded as the basic model to facilitate deep learning (DL) to revolutionize resource allocation in wireless networks. GNN-based models are shown to be able to learn the structural information about graphs representing the wireless networks to adapt to the time-varying channel state information and dynamics of network topology. This article aims to provide a comprehensive overview of applying GNNs to optimize wireless networks via answering three fundamental questions, i.e., how to input the wireless network data into GNNs, how to improve the performance of GNNs, and how to evaluate GNNs. Particularly, two graph representations are given to transform wireless network parameters into graph-structured data. Then, we focus on the architecture design of the GNN-based models via introducing the basic message passing as well as model improvement methods including multi-head attention mechanism and residual structure. At last, we give task-oriented evaluation metrics for DL-enabled wireless resource allocation. We also highlight certain challenges and potential research directions for the application of GNNs in wireless networks.

