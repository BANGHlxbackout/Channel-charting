---
type: paper
node_id: paper:taner2024_channel_charting_realworld
title: "Channel Charting in Real-World Coordinates with Distributed MIMO"
authors: ["Sueda Taner", "Victoria Palhares", "Christoph Studer"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2406.13722"
  doi: null
  s2: null
tags: ["channel-charting", "measured", "loss"]
added: 2026-10-28T11:05:32Z
---

# Channel Charting in Real-World Coordinates with Distributed MIMO

## One-line thesis
Distributed-MIMO charting uses AP-side bilateration and LoS constraints, not a GCN propagation test.

## Problem / Gap
Learns chart coordinates tied to known infrastructure positions without UE labels, addressing the arbitrary-coordinate limitation of ordinary channel charts. [Primary text](https://arxiv.org/html/2406.13722), Sections III–V; published TWC 2025 DOI [10.1109/TWC.2025.3559723](https://doi.org/10.1109/TWC.2025.3559723).

## Method
Uses AP receive-power comparisons to form bilateration constraints only for estimated LoS AP pairs, and LoS bounding-box constraints. Combines these with a self-supervised chart loss; it does not use GCN message passing. The power threshold and AP pair margin govern which weak geometry constraints are admitted.

## Key Results
Evaluates simulated outdoor, simulated indoor, and measured DICHASUS factory data. In the measured scenario, uses the first three trajectories, every fourth sample, 32×1024 CSI, 13 selected taps, and manually chosen LoS power threshold. This is a different sample/protocol from our G1 comparison, even though both draw on DICHASUS.

## Assumptions
Known AP positions and usable AP-side power/LoS classification are required. A received-power ordering is informative only for correctly identified LoS APs; the paper discusses the false-pair-versus-coverage tradeoff.

## Limitations / Failure Modes
Direct adoption would add infrastructure geometry and threshold choices, changing the current task's inputs and research question. It does not establish an effect of neighborhood propagation under fixed ACF and geodesic stress.

## Reusable Ingredients
Its measured-data warning about conditional reliability of radio-derived constraints is relevant when judging any proposed edge-selection rule. Keep AP information out unless explicitly included in a separate study.

## Open Questions
Does our existing frozen feature set contain a relation-quality signal that can identify harmful propagation edges without UE GT or AP-label assumptions?

## Claims
Paper demonstrates its AP-informed losses in its reported protocols; those results are not comparable numerically to G1.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Measured-data context and prior art for constraint reliability, not a direct candidate under the present fixed-preprocessing/ACF scope.

## Abstract (original)

> Channel charting is an emerging self-supervised method that maps channel-state information (CSI) to a low-dimensional latent space (the channel chart) that represents pseudo-positions of user equipments (UEs). While channel charts preserve local geometry, i.e., nearby UEs are nearby in the channel chart (and vice versa), the pseudo-positions are in arbitrary coordinates and global geometry is typically not preserved. In order to embed channel charts in real-world coordinates, we first propose a bilateration loss for distributed multiple-input multiple-output (D-MIMO) wireless systems in which only the access point (AP) positions are known. The idea behind this loss is to compare the received power at pairs of APs to determine whether a UE should be placed closer to one AP or the other in the channel chart. We then propose a line-of-sight (LoS) bounding-box loss that places the UE in a predefined LoS area of each AP that is estimated to have a LoS path to the UE. We demonstrate the efficacy of combining both of these loss functions with neural-network-based channel charting using ray-tracing-based and measurement-based channel vectors. Our proposed approach outperforms several baselines and maintains the self-supervised nature of channel charting as it neither relies on geometrical propagation models nor on any ground-truth UE position information.

