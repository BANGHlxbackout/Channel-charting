---
type: paper
node_id: paper:stephan2023_angledelay_profilebased_timestampaided
title: "Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting"
authors: ["Phillip Stephan", "Florian Euchner", "Stephan ten Brink"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2308.09539"
  doi: null
  s2: null
tags: ["channel-charting", "measured", "adp"]
added: 2026-10-28T11:05:28Z
---

# Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting

## One-line thesis
Measured DICHASUS ADP and time dissimilarity fusion underlies the frozen Stage-1 graph.

## Problem / Gap
Builds CSI/time dissimilarities that preserve local and global geometry on measured distributed MIMO channel data. [Primary text](https://arxiv.org/html/2308.09539), Sections III, V, VII; journal DOI [10.1109/TCOMM.2024.3386571](https://doi.org/10.1109/TCOMM.2024.3386571).

## Method
Combines ADP dissimilarity and scaled timestamp separation by a pairwise minimum, constructs a k=20 nearest-neighbor graph, then uses shortest-path geodesic dissimilarities. Compares classical embeddings and Siamese/triplet forward charting; the Siamese branch uses CSI-derived features and matches chart distances to pseudo-distances.

## Key Results
On measured DICHASUS cf02/cf03/cf04, four 2×4 arrays, 1.272 GHz and 50 MHz, the paper's every-fifth-sample training set has 16,797 points. Its geodesic fused metric beats the listed raw ADP/time metrics on CT/TW/KS in that protocol. Parametric methods are evaluated on a distinct 16,797-point feature set; this differs from our same-node transductive G1 evaluation.

## Assumptions
The time scaling separates moving and stationary regimes; the authors note approximately constant transmitter velocity may fail on other datasets. Short local dissimilarities are assumed more reliable than direct long-range dissimilarities.

## Limitations / Failure Modes
Our 20,997-point indexing, exact fused cache, loss, and evaluation subset are not identical to the paper's 16,797-point train/test protocol. A good geodesic fused metric does not imply every propagation edge is useful to a GCN.

## Reusable Ingredients
Preserve original directed k=20 fused graph and its shortest-path target when testing G1 architecture effects; inspect time-versus-ADP edge source as a diagnostic, not as an automatic deletion rule.

## Open Questions
Which input-only edge properties predict whether message passing helps once the same fused geodesic target already supervises the model?

## Claims
Paper evidence supports its fusion metric on its measured protocol. Current G1 Graph-versus-Identity evidence is separate and does not inherit a graph-propagation claim.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Closest source for the inherited preprocessing and distance graph. Relevant to graph provenance and data assumptions, not evidence that a GCN or any new loss will improve current G1.

## Abstract (original)

> Channel charting is a self-supervised learning technique whose objective is to reconstruct a map of the radio environment, called channel chart, by taking advantage of similarity relationships in high-dimensional channel state information. We provide an overview of processing steps and evaluation methods for channel charting and propose a novel dissimilarity metric that takes into account angular-domain information as well as a novel deep learning-based metric. Furthermore, we suggest a method to fuse dissimilarity metrics such that both the time at which channels were measured as well as similarities in channel state information can be taken into consideration while learning a channel chart. By applying both classical and deep learning-based manifold learning to a dataset containing sub-6GHz distributed massive MIMO channel measurements, we show that our metrics outperform previously proposed dissimilarity measures. The results indicate that the new metrics improve channel charting performance, even under non-line-of-sight conditions.

