---
type: paper
node_id: paper:yassine2023_optimizing_multicarrier_multiantenna
title: "Optimizing Multicarrier Multiantenna Systems for LoS Channel Charting"
authors: ["Taha Yassine", "Luc Le Magoarou", "Matthieu Crussière", "Stephane Paquelet"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2310.03762"
  doi: null
  s2: null
tags: ["channel-charting", "graph-quality", "ray-traced"]
added: 2026-10-28T11:16:50Z
---

# Optimizing Multicarrier Multiantenna Systems for LoS Channel Charting

## One-line thesis
LoS phase-insensitive channel distance ambiguities can corrupt neighborhood graphs; its mechanism differs from measured ADP/time fusion.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Channel charting (CC) consists in learning a mapping between the space of raw channel observations, made available from pilot-based channel estimation in multicarrier multiantenna system, and a low-dimensional space where close points correspond to channels of user equipments (UEs) close spatially. Among the different methods of learning this mapping, some rely on a distance measure between channel vectors. Such a distance should reliably reflect the local spatial neighborhoods of the UEs. The recently proposed phase-insensitive (PI) distance exhibits good properties in this regards, but suffers from ambiguities due to both its periodic and oscillatory aspects, making users far away from each other appear closer in some cases. In this paper, a thorough theoretical analysis of the said distance and its limitations is provided, giving insights on how they can be mitigated. Guidelines for designing systems capable of learning quality charts are consequently derived. Experimental validation is then conducted on synthetic and realistic data in different scenarios.

