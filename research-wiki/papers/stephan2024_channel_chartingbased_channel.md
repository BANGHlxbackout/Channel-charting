---
type: paper
node_id: paper:stephan2024_channel_chartingbased_channel
title: "Channel Charting-Based Channel Prediction on Real-World Distributed Massive MIMO CSI"
authors: ["Phillip Stephan", "Florian Euchner", "Stephan ten Brink"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2410.11486"
  doi: null
  s2: null
tags: ["channel-charting", "measured", "application"]
added: 2026-10-28T11:05:33Z
---

# Channel Charting-Based Channel Prediction on Real-World Distributed Massive MIMO CSI

## One-line thesis
Real-world channel prediction uses channel charts as a representation for downstream prediction.

## Problem / Gap
Uses a measured-data channel chart as a coordinate system for predicting future CSI and mitigating channel aging. [Primary text](https://arxiv.org/html/2410.11486), Sections V–VI.

## Method
First trains an ACF-feature Siamese forward charting function with fused ADP/time geodesic pseudo-distances. Then extrapolates recent latent positions and predicts CSI using chart-neighbor lookup or Delaunay-based interpolation. The predictive stage is downstream of chart construction, not GCN propagation.

## Key Results
On real distributed massive-MIMO measurements, reports chart CT/TW about 0.997/0.997 and KS 0.082 on its training chart, and evaluates channel prediction by achievable sum rate against stale CSI and Wiener prediction. Performance gains depend on prediction horizon; short horizons can favor stale CSI.

## Assumptions
The chart is spatially coherent enough for latent extrapolation and interpolation; prediction uses a temporal memory and a train/pred set separation.

## Limitations / Failure Modes
Its primary outcome is future CSI quality, not a controlled comparison of chart-graph propagation. Direct performance numbers do not share our sample and evaluation protocol.

## Reusable Ingredients
Confirms the established ACF plus fused-geodesic Siamese pipeline as relevant measured-data context; no need to import its downstream predictor into Stage 1.

## Open Questions
Would a graph mechanism that improves G1's own chart geometry also improve downstream tasks? This paper does not answer the controlled G1 question.

## Claims
Paper supports horizon-dependent CSI prediction from its learned chart, not a universal graph-message benefit.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Useful measured-data comparator and protocol reminder; not a method candidate for the current bounded G1 mechanism test.

## Abstract (original)

> Distributed massive MIMO is considered a key advancement for improving the performance of next-generation wireless telecommunication systems. However, its efficacy in scenarios involving user mobility is limited due to channel aging. To address this challenge, channel prediction techniques are investigated to forecast future channel state information (CSI) based on previous estimates. We propose a new channel prediction method based on channel charting, a self-supervised learning technique that reconstructs a physically meaningful latent representation of the radio environment using similarity relationships between CSI samples. The concept of inertia within a channel chart allows for predictive radio resource management tasks through the latent space. We demonstrate that channel charting can be used to predict future CSI by exploiting spatial relationships between known estimates that are embedded in the channel chart. Our method is validated on a real-world distributed massive MIMO dataset, and compared to a Wiener predictor and the outdated CSI in terms of achievable sum rate.

