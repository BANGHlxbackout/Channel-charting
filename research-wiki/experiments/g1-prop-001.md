---
type: experiment
node_id: exp:g1-prop-001
title: "Frozen G1 graph adjacency sensitivity"
idea_id: ""
verdict: yes
confidence: medium
date: "2026-10-17"
hardware: "RTX 3050 Laptop GPU 4 GB"
duration: "38.80 s inference and evaluation; no training"
provenance: "results/g1_acf_graph_fcf_followup/e2_propagation/manifest.json; EXP-G1-PROP-001"
added: 2026-10-25T07:09:13Z
tags: ["channel-charting", "diagnostic", "graph-fcf", "transductive", "sensitivity"]
---

# Frozen G1 graph adjacency sensitivity

**verdict:** `yes`  ·  **confidence:** `medium`

## Metrics
| Frozen checkpoint / adjacency | validation stress |
|---|---:|
| Graph @ Graph | 1.872658 |
| Graph @ I | 3.208952 |
| Identity @ I | 1.862524 |

Graph @ I increased stress by 71.36%; raw embedding displacement median 12.967083 chart units. Graph-native layer 1/2 delta-over-h medians: 0.859600, 0.597705. Source: [E2 manifest](../../results/g1_acf_graph_fcf_followup/e2_propagation/manifest.json), [layer statistics](../../results/g1_acf_graph_fcf_followup/e2_propagation/layer_statistics.csv), [chart evaluation](../../results/g1_acf_graph_fcf_followup/e2_propagation/comparison/comparison.csv).

## Reasoning
Question: is the already trained Graph model sensitive to its fused adjacency? Native Graph/Identity checkpoints reproduced saved embeddings and validation stress before the swap. The frozen Graph checkpoint changed materially when A_graph was replaced by I, so the model does depend on the graph. That dependence did not produce a net geometric advantage over the separately trained Identity control. The swap changes the trained input distribution and is only a post-training sensitivity probe; it cannot uniquely establish supervision redundancy, residual masking, or bad neighbor relations. The 200,000 validation pairs retain the original sorted-key index bias. Details: [follow-up report](../../results/g1_acf_graph_fcf_followup/REPORT.md).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

