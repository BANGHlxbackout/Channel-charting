---
type: experiment
node_id: exp:g1-acf-full-001
title: "G1-ACF fused graph versus identity propagation"
idea_id: ""
verdict: no
confidence: medium
date: "2026-10-16"
hardware: "RTX 3050 Laptop GPU 4 GB"
duration: "Graph 6224.64 s; Identity 5292.06 s; combined 3.199 GPU h"
provenance: "results/g1_acf_graph_fcf/REPORT.md; EXP-G1-ACF-FULL-001"
added: 2026-10-25T07:09:13Z
tags: ["channel-charting", "stage1", "graph-fcf", "transductive", "negative"]
---

# G1-ACF fused graph versus identity propagation

**verdict:** `no`  ·  **confidence:** `medium`

## Metrics
| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 |
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 |
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 |

Source: [common comparison CSV](../../results/g1_acf_graph_fcf/comparison/comparison.csv); [Graph result JSON](../../results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json); [Identity result JSON](../../results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json).

## Reasoning
Research question: does fused-graph neighborhood propagation improve the measured DICHASUS chart beyond the identical G1 network with identity adjacency? Graph and Identity shared N=20,997, seed=10, 26,624-D ACF, model size, loss, optimizer, pair generator and evaluation; only propagation adjacency changed. The G1 package modestly improved B0, but Graph was worse than trained Identity on ct, tw, mean_m, median_m, p90_m, p95_m and max_m; only ks improved by about 0.000062. Verdict no applies to incremental propagation benefit in this frozen single-seed transductive run, not to all GNNs or the entire G1 direction. GT was used after training only for evaluation; pair holdout does not establish unseen-node generalization. The fixed validation-pair sampler sorts keys before truncation, biasing it toward smaller indices. Full methods and boundaries: [G1 report](../../results/g1_acf_graph_fcf/REPORT.md) and [experiment log](../../docs/research/experiment_log.md).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

