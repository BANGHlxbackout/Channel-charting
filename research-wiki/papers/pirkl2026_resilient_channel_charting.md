---
type: paper
node_id: paper:pirkl2026_resilient_channel_charting
title: "Resilient Channel Charting Under Varying Radio Link Availability"
authors: ["Jonas Pirkl", "Jonathan Ott", "Maximilian Stahlke", "George Yammine", "Tobias Feigl", "Christopher Mutschler"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2602.04704"
  doi: null
  s2: null
tags: ["channel-charting", "measured", "architecture"]
added: 2026-10-28T11:05:35Z
---

# Resilient Channel Charting Under Varying Radio Link Availability

## One-line thesis
AdaPos handles varying radio-link availability with variable-input fusion on real data.

## Problem / Gap
Addresses variable antenna availability in channel charting, rather than graph construction or message passing between UE samples. [Primary text](https://arxiv.org/html/2602.04704), Sections III–V.

## Method
AdaPos uses a shared 1D-ResNet signal encoder and a transformer combiner with antenna identifiers and attention over available radio links. Uses Siamese pseudo-distance training; random-N antenna masking is its key robustness strategy.

## Key Results
On measured DICHASUS, trains on cf02/cf04 (60,507 samples) and tests on cf03 (23,478), with 4×8 antenna elements and fused time/ADP geodesic distance. Reports robustness under missing individual antennas or whole arrays. Also studies a separate Fraunhofer dataset.

## Assumptions
Input radio links may disappear, and a variable-input architecture is needed. Training protocol varies available antenna subsets.

## Limitations / Failure Modes
Current G1 fixes the full ACF features and does not study missing antennas, so adopting AdaPos would change the feature encoder and problem. Its attention is over antenna signals, not neighbors in the G1 sample graph.

## Reusable Ingredients
Shows that antenna-configuration robustness is a separate research question and that real DICHASUS protocol splits matter; no direct G1 graph edge rule follows.

## Open Questions
Could measured graph context help when antennas are missing? Outside the present fixed-input study.

## Claims
Paper supports variable-link robustness under its measured train/test protocol, not a G1 Graph-versus-Identity conclusion.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Context for real CSI model design and protocol; exclude from the single-change G1 experiment.

## Abstract (original)

> Channel charting (CC) has become a key technology for RF-based localization, enabling unsupervised radio fingerprinting, even in non line of sight scenarios, with a minimum of reference position labels. However, most CC models assume fixed-size inputs, such as a constant number of antennas or channel measurements. In practical systems, antennas may fail, signals may be blocked, or antenna sets may change during handovers, making fixed-input architectures fragile. Existing radio-fingerprinting approaches address this by training separate models for each antenna configuration, but the resulting training effort scales prohibitively with the array size. We propose Adaptive Positioning (AdaPos), a CC architecture that natively handles variable numbers of channel measurements. AdaPos combines convolutional feature extraction with a transformer-based encoder using learnable antenna identifiers and self-attention to fuse arbitrary subsets of CSI inputs. Experiments on two public real-world datasets (SISO and MIMO) show that AdaPos maintains state-of-the-art accuracy under missing-antenna conditions and replaces roughly 57 configuration-specific models with a single unified model. With AdaPos and our novel training strategies, we provide resilience to both individual antenna failures and full-array outages.

