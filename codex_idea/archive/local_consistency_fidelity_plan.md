# Local Consistency-Aware Fidelity Refinement Plan

## 0. Motivation

Current graph-harmonic diffusion refinement is globally applied:

```math
\min_Z \|Z - Y_0\|_F^2 + \lambda \sum_{ij} W_{ij}^{phys}\|Z_i - Z_j\|_2^2
```

which is equivalent to:

```math
(I + \lambda L_{phys})Z = Y_0
```

This refinement is effective overall, but diagnostics show that it has two different effects:

1. It repairs bad / outlier points in the initial chart `Y0`.
2. It can damage points that were already accurate before refinement.

The key problem is that the original method applies the same graph refinement strength to every node. It does not know which nodes in `Y0` are already locally reliable and should be protected, and which nodes conflict with the physical graph and should be repaired.

Therefore, the next method should move from:

```text
Global graph refinement
```

to:

```text
Selective / local-consistency-aware graph refinement
```

The method must not use UE positions, localization errors, or affine-aligned errors to construct the refinement weights. UE positions are allowed only for post-hoc diagnostics and final evaluation.

---

## 1. Revised Core Hypothesis

Do not frame the first executable module as a symmetric "good point vs bad point" classifier.

The proxy diagnostics show an asymmetric situation:

```text
repair-needed / outlier-like nodes:
    have usable unsupervised signal from B_i and r_i.

protect-worthy / already-good nodes:
    are hard to identify from Y0-physical graph self-consistency alone.
```

Therefore, the first executable module should not claim:

```text
identify good nodes -> strongly protect them
identify bad nodes -> repair them
```

Instead, it should claim a narrower and more defensible goal:

```text
identify high-confidence repair-needed nodes -> selectively relax fidelity there
all remaining nodes -> leave near the baseline fidelity, not declare them "good"
```

This changes the method from:

```text
two-sided adaptive fidelity
```

to:

```text
one-sided repair-gated adaptive fidelity
```

Important refinement after Codex diagnostic:

```text
The goal is not to make refinement stronger.
The goal is to preserve the baseline's outlier repair while reducing damage to already-good local regions.
```

The current diagnostic result shows:

```text
init_error < 0.2 m:
    73.63% are worsened by baseline diffusion.

init_error >= 1.0 m:
    only 0.63% are worsened and the mean error drops by 0.619 m.
```

Therefore, a useful first adaptive fidelity method should concentrate extra repair freedom on high-confidence repair-needed regions without globally weakening the baseline refinement.

---

## 2. Proposed Objective

Replace the uniform fidelity term:

```math
\|Z - Y_0\|_F^2
```

with a node-adaptive fidelity term:

```math
\min_Z
\sum_i a_i \|Z_i - Y_{0,i}\|_2^2
+
\lambda \sum_{ij} W_{ij}^{phys}\|Z_i - Z_j\|_2^2
```

where:

```math
A = \operatorname{diag}(a_1, \ldots, a_N)
```

The linear system becomes:

```math
(A + \lambda L_{phys})Z = A Y_0
```

Interpretation:

```text
a_i > 1: explicitly protect node i, keep it close to Y0_i
a_i < 1: allow node i to move more, so graph diffusion can repair it
a_i = 1: same as baseline fidelity strength
```

Important correction from the earlier design:

The full two-sided method would require `a_i > 1` for reliably identified protect-worthy nodes. However, the revised proxy diagnostic did not find a protection proxy strong enough to justify this.

Therefore, the next executable version should be one-sided:

```text
repair-gated first version:
    a_i in [a_min, 1]

two-sided extension only if a future protection proxy passes:
    a_i in [a_min, a_max], with a_max > 1
```

First repair-gated range:

```text
conservative: a_i in [0.75, 1.0]
stress-test:  a_i in [0.50, 1.0]
```

Do not median-normalize the one-sided weights if most nodes stay at `a_i = 1`.

Reason:

```text
Median normalization can accidentally turn non-repair nodes into a_i > 1 protection nodes.
That would reintroduce an unvalidated protection mechanism.
```

The one-sided version must be compared against lower-lambda and constant-weaker-fidelity controls.

```text
If the gain can be reproduced by lowering lambda or setting a constant a_i < 1, the module is not adding real selectivity.
```

---

## 3. Unsupervised Proxy Scores

The method should first compute several node-level proxies using only `Y0` and physical graph information.

Allowed inputs:

```text
Y0 initial embedding
W_phys
geodesic_topK neighbors
geodesic distance d_geo
physical graph local scale sigma if available
```

Forbidden inputs for score construction:

```text
UE position
localization error
affine-aligned error
refined error delta
post-hoc labels
```

### 3.1 Protection Proxy: Failed / Deferred Hypothesis

Original diagnostic proxy:

```math
A_i^{nbr}
=
\frac{|N_{phys}(i) \cap N_{Y0}(i)|}{k}
```

where:

```text
N_phys(i): top-k physical/geodesic neighbors of node i
N_Y0(i): top-k Euclidean neighbors of node i in initial chart Y0
```

This raw top-k overlap should now be treated as a baseline diagnostic proxy, not as the final protection score.

Reason:

```text
The first proxy diagnostic found that A_nbr is weak for identifying protect-worthy points:
    AUC(good_initial) is close to random.
    AUC(strong_worsened) is close to random.

The next diagnostic tried stricter protection candidates:

```text
C_mutual
C_rank_weighted
C_multi_k_stability
C_prot = mutual * rank * stability
C_prot_anti
C_prot_minus_repair_z
```

Result:

```text
No protection proxy passed the strict gate.
Best candidate C_prot_minus_repair_z:
    AUC(good_initial) = 0.5470
    AUC(strong_worsened) = 0.5335
    Spearman(error_delta) = 0.1377
```

Therefore, the missing piece is not simply a better top-k overlap formula. The deeper issue is:

```text
"good point" is weakly observable from Y0-physical graph self-consistency alone.
```

The following protection proxy definitions are kept as diagnostic records and future ablation candidates. They should not be used as the first adaptive-fidelity driver.

Use the following candidate protection score:

```math
C_i^{prot}
=
C_i^{mutual}
\cdot
C_i^{rank}
\cdot
C_i^{stab}
\cdot
C_i^{anti}
```

#### 3.1.1 Mutual Neighborhood Support

```math
C_i^{mutual}
=
\frac{1}{k}
\sum_{j\in N_k^{phys}(i)}
\mathbf{1}[j\in N_k^{Y0}(i)]
\mathbf{1}[i\in N_k^{Y0}(j)].
```

This is stricter than raw overlap. A physical neighbor only supports protection if the relation is also mutual in the current chart.

#### 3.1.2 Rank-Weighted Physical Neighbor Support

```math
C_i^{rank}
=
\frac{1}{k}
\sum_{j\in N_k^{phys}(i)}
\exp\left(-\frac{\operatorname{rank}_{Y0}(i,j)}{\tau_r}\right).
```

Raw overlap loses information by asking only whether a physical neighbor is inside top-k. Rank-weighted support checks whether physical neighbors are actually near the front of the `Y0` neighbor list.

#### 3.1.3 Multi-k Stability

```math
C_i^{stab}
=
\min_{k'\in\{16,32,64\}}
\frac{|N_{k'}^{phys}(i)\cap N_{k'}^{Y0}(i)|}{k'}.
```

A node should not be strongly protected if it only looks consistent at one arbitrary neighborhood size.

#### 3.1.4 Anti-Repair Gate

```math
C_i^{anti}
=
\exp\left(
-z([B_i]_+)-\rho z(r_i)
\right).
```

This gate prevents a node from being protected when the repair proxies say it is likely inconsistent:

```text
[B_i]_+ high:
    physical neighbors are stretched in Y0, which suggests tearing or outlier behavior.

r_i high:
    Y0_i has high graph smoothness residual.
```

This does not mean `r_i` is used alone. It only weakens protection when combined with rank / mutual / stability evidence.

First diagnostic candidates:

```text
1. raw A_nbr only: baseline proxy
2. mutual only
3. rank-weighted only
4. multi-k stability only
5. C_prot = mutual * rank * stability
6. C_prot_anti = mutual * rank * stability * anti-repair
```

The diagnostic goal was to verify whether the revised protection proxy improves identification of:

```text
good_initial nodes
strong_worsened nodes
```

It did not pass. Therefore, do not use `C_prot` to set `a_i > 1` in the next experiment.

---

### 3.2 Local Physical-Neighbor Scale / Stress Score

Codex previously proposed:

```math
s_i = \operatorname{median}_{j \in N_{phys}(i)} \|Y_{0,i} - Y_{0,j}\|
```

and compared it with the global median. This is useful but too coarse, because it ignores local physical density.

An overly local normalization can be problematic. If `eta_i` is fitted freely per node, it can absorb exactly the local scale anomaly that the score is supposed to detect.

The first diagnostic version should therefore use a robust global scale calibration:

```math
\eta
=
\operatorname{median}_{(i,j)\in E_{phys}}
\frac{
\|Y_{0,i}-Y_{0,j}\|+\epsilon
}{
d_{ij}^{geo}+\epsilon
}.
```

Then compute:

```math
B_i
=
\operatorname{median}_{j \in N_{phys}(i)}
\left[
\log
\frac{
\|Y_{0,i} - Y_{0,j}\| + \epsilon
}{
\eta(d_{ij}^{geo}+\epsilon)
}
\right]
```

This makes `B_i` answer a concrete question:

```text
Are node i's physical neighbors farther or closer in Y0 than the global physical-to-chart scale predicts?
```

Simpler first version:

```math
B_i
=
\log
\frac{
\operatorname{median}_{j \in N_{phys}(i)}\|Y_{0,i} - Y_{0,j}\| + \epsilon
}{
\operatorname{median}_{j \in N_{phys}(i)}d_{ij}^{geo} + \epsilon
}
```

then robust-normalize `B_i` across all nodes.

Use the simple version only as an ablation. The preferred first diagnostic is the globally calibrated `eta` version above.

Interpretation:

```text
B_i very positive:
    physical neighbors are far apart in Y0.
    This may indicate tearing, outlier behavior, or under-smoothed local geometry.
    Node may need repair.

B_i very negative:
    physical neighbors are already compressed in Y0.
    Stronger diffusion may over-compress this region.
    Node should be protected or at least not strongly repaired.
```

This score is important because diagnostics show that diffusion refinement strongly compresses local physical-neighbor lengths. The next method should avoid unnecessary additional compression in already-compact regions.

---

### 3.3 Smoothness Residual Score

Definition:

```math
r_i
=
\left\|
Y_{0,i}
-
\frac{\sum_j W_{ij}^{phys}Y_{0,j}}{\sum_j W_{ij}^{phys}}
\right\|_2
```

Interpretation:

```text
r_i high:
    Y0_i deviates from the weighted physical-neighbor average.
    Node may conflict with physical smoothness prior.

r_i low:
    Y0_i is already smooth on the physical graph.
```

Important warning:

This score must not be used alone. Earlier alpha-only experiments failed because high graph residual was incorrectly interpreted as “Y0 is wrong and should be smoothed more.”

High residual can come from at least three causes:

```text
1. Y0 coordinate noise
2. wrong or unreliable physical graph edges
3. real local geometry / curvature / high-frequency structure
```

Therefore, `r_i` should only be an auxiliary factor. It should be combined with neighborhood agreement and scale stress.

---

### 3.4 Baseline Diffusion Displacement

Definition:

```math
d_i^{move} = \|Z_i^{base} - Y_{0,i}\|_2
```

where `Z_base` is the baseline diffusion-refined embedding.

Interpretation:

```text
d_i^{move} high:
    baseline solver strongly wants to move node i.

d_i^{move} low:
    baseline solver does not modify node i much.
```

Important warning:

This score is informative for diagnostics but should not enter the first version of the method. Diagnostics show that large movement often repairs outliers, but some initially accurate points are also moved and damaged.

So first use `d_i^{move}` only to analyze whether the proposed proxy scores explain baseline movement and error delta.

---

### 3.5 Repair Confidence Score

The current usable signal is repair need, not protection confidence.

Define a repair score using only unsupervised quantities:

```math
R_i
=
z([B_i]_+)
+
\rho z(r_i)
```

where:

```text
[B_i]_+:
    positive physical-neighbor scale stress.
    High value means physical neighbors are stretched in Y0.

r_i:
    graph smoothness residual.
    High value means Y0_i disagrees with the physical graph average.

z(.):
    robust normalization, for example median/IQR normalization with clipping.
```

Convert it to a sparse repair confidence:

```math
q_i
=
\sigma(\alpha(R_i-\tau)).
```

or, for the first experiment, use a quantile gate:

```math
q_i
=
\begin{cases}
1, & R_i \ge Q_p(R)\\
0, & R_i < Q_p(R)
\end{cases}
```

with candidate repair fractions:

```text
p in {0.80, 0.85, 0.90}
```

Interpretation:

```text
q_i high:
    high-confidence repair-needed node.

q_i low:
    no strong repair evidence.
    This does not mean the node is good.
```

This distinction is central. The complement of high repair confidence should be treated as:

```text
uncertain / no-extra-repair
```

not as:

```text
protect-worthy / definitely good
```

---

## 4. Recommended First Diagnostic Step

Do not directly implement adaptive refinement first.

First compute the proxy scores and evaluate whether they can predict which points are damaged or improved by baseline refinement.

Use UE positions only for diagnostic labels, not for score construction.

Diagnostic labels:

```text
init_error_i
error_delta_i = refined_error_i - init_error_i
worsened_i = error_delta_i > 0
strong_worsened_i = error_delta_i > 0.05
strongly_improved_i = error_delta_i < -0.05
good_initial_i = init_error_i < 0.2
outlier_initial_i = init_error_i >= 1.0
```

For each score, report:

```text
1. Spearman correlation with init_error
2. Spearman correlation with error_delta
3. AUC for predicting strong_worsened_i
4. AUC for predicting outlier_initial_i and strongly_improved_i
5. Mean score in init_error buckets:
   - init_error < 0.2 m
   - 0.2 m <= init_error < 0.5 m
   - 0.5 m <= init_error < 1.0 m
   - init_error >= 1.0 m
6. Top-20% enrichment for good_initial / outlier_initial / strongly_improved
7. Scatter plots:
   - score vs error_delta
   - score vs init_error
```

The protection proxy scores could not distinguish strong-worsened / initially-good points strongly enough.

Therefore, the next module should not require a protection-proxy pass. It should require a repair-confidence pass.

Hard gate before implementing repair-gated adaptive refinement:

```text
Repair-proxy gate:

1. A repair proxy based on [B_i]_+ and/or r_i must keep:
   - AUC(outlier_initial) >= 0.70, or
   - |Spearman(error_delta)| >= 0.20 with the expected negative sign.

2. Top repair-confidence nodes should be enriched for outliers / strong improvements:
   - top 20% R_i outlier_initial fraction >= 1.5x global outlier_initial fraction,
     or top 20% R_i strongly_improved fraction >= 1.2x global strongly_improved fraction.

3. The repair gate must be sparse enough:
   - first experiments should activate at most 20% of nodes.

4. Low repair confidence must not be interpreted as protection confidence.
   - q_i low -> a_i = 1, not a_i > 1.
```

If these gates fail:

```text
Do not implement adaptive fidelity.
Return to repair-proxy design instead.
```

---

## 5. Candidate Fidelity Weight Construction

The next fidelity construction should be one-sided.

Do not use:

```math
a_i = 1 + \gamma C_i^{prot}
```

because `C_prot` did not pass the protection gate.

Instead use repair confidence:

```math
R_i
=
z([B_i]_+)
+
\rho z(r_i)
```

and map it to:

```math
a_i
=
1-\delta q_i.
```

where:

```text
q_i in [0,1]:
    high-confidence repair gate.

delta:
    maximum fidelity relaxation.

a_i:
    clipped to [a_min, 1].
```

First implementation:

```math
q_i
=
\mathbf{1}[R_i \ge Q_{0.85}(R)]
```

```math
a_i
=
\begin{cases}
a_{min}, & q_i=1\\
1, & q_i=0
\end{cases}
```

with:

```text
a_min = 0.75
```

This means:

```text
top 15% repair-confidence nodes:
    allow more diffusion repair.

remaining 85% nodes:
    keep the baseline fidelity.
    do not declare them good.
    do not set a_i > 1.
```

Second implementation, only if the hard gate version is too coarse:

```math
q_i
=
\sigma(\alpha(R_i-\tau))
```

```math
a_i
=
\operatorname{clip}(1-\delta q_i, a_{min}, 1)
```

Use:

```text
alpha in {2, 4}
tau = Q_0.80(R), Q_0.85(R), or Q_0.90(R)
delta in {0.15, 0.25, 0.50}
a_min in {0.75, 0.50}
```

Important:

```text
Do not median-normalize one-sided a_i.
Do not use a_i > 1 in this version.
Do not tune tau/delta by looking for the best final test metric only.
Always compare against constant-a and lower-lambda controls.
```

---

## 6. Why This Differs From Failed Alpha-Only

Failed alpha-only logic:

```text
L_phys Y0 residual high
-> assume Y0 is wrong
-> reduce fidelity
-> stronger graph smoothing
```

This failed because residual only indicates disagreement, not which side is correct.

New logic:

```text
Y0 has high repair confidence from scale stress / residual
-> locally relax fidelity

Y0 does not have high repair confidence
-> keep baseline fidelity
-> do not claim the node is good
```

The new method is not based on a single smoothness residual and does not globally increase smoothing. It uses a sparse repair gate so that only high-confidence repair-needed regions get more freedom to move.

This is still risky:

```text
If it behaves like lowering lambda or setting a constant a_i < 1,
then the module is not selective enough.
```

Therefore shuffled-gate, constant-a, and lower-lambda controls are mandatory.

---

## 7. Minimum Experiment Groups

If diagnostics justify implementation, run the following experiments:

### A. Baseline

```math
A = I
```

Equivalent to the original graph-harmonic refinement.

### B. Hard repair-gated fidelity

Use:

```math
q_i=\mathbf{1}[R_i \ge Q_p(R)]
```

```math
a_i =
\begin{cases}
a_{min}, & q_i=1\\
1, & q_i=0
\end{cases}
```

with:

```text
p in {0.80, 0.85, 0.90}
a_min in {0.75, 0.50}
```

This is the first real adaptive-fidelity candidate.

### C. Soft repair-gated fidelity

Use:

```math
q_i = \sigma(\alpha(R_i-\tau))
```

```math
a_i = \operatorname{clip}(1-\delta q_i, a_{min}, 1)
```

This is tested only after the hard gate version shows a useful trend.

### D. Repair-score ablations

```math
R_i = z([B_i]_+)
```

```math
R_i = z(r_i)
```

```math
R_i = z([B_i]_+) + \rho z(r_i)
```

Purpose: determine whether scale stress, smoothness residual, or their combination is responsible.

### E. Failed-protection controls

Optionally test raw `A_nbr` and `C_prot` only as controls:

```text
They should not be presented as the main method unless a later diagnostic contradicts the current failure.
```

### F. Random shuffled fidelity

Shuffle `q_i` or `a_i` across nodes while preserving the same distribution.

Purpose: verify that improvement comes from node-specific consistency, not just changing global fidelity distribution.

### G. Constant stronger fidelity baseline

Use:

```math
A = a I
```

with fixed constants:

```text
a = 1.5
a = 2.0
a = 3.0
```

Purpose: verify that improvement is not simply caused by globally protecting all nodes.

### H. Constant weaker fidelity baseline

Use:

```math
A = a I
```

with a constant `a < 1`.

Use fixed constants:

```text
a = 0.5
a = 0.75
```

Purpose: verify that improvement is not simply caused by globally allowing more repair.

### I. Lower-lambda baseline

Use the original baseline with smaller global lambda values.

Purpose:

```text
Verify that repair-gated fidelity is not equivalent to simply weakening diffusion globally.
```

### J. Conservative-range vs wide-range repair gate

Run the same adaptive score with:

```text
conservative: a_i in [0.75, 1.0]
wide:         a_i in [0.50, 1.0]
```

Purpose:

```text
Check whether gains come from the repair-confidence signal or from an aggressive relaxation range.
```

---

## 8. Success Criteria

The adaptive fidelity method is worth continuing only if:

```text
1. It improves the baseline, not only matches it.
2. It reduces unnecessary damage or does not increase it.
3. It preserves the baseline's large-outlier repair ability.
4. It outperforms shuffled fidelity weights.
5. It outperforms constant weaker-fidelity baselines.
6. It does not win only by effectively changing the global lambda.
```

Strong pass:

```text
mean error improves by >= 2% relative to baseline
P95 error improves by >= 2% relative to baseline
max error does not increase
KS does not increase
strong-worsened fraction does not increase
good_initial worsened fraction does not increase
init_error >= 1.0 m slice keeps at least 90% of the baseline mean-error improvement
adaptive method beats shuffled-a_i by >= 1% mean error or >= 2% P95 error
adaptive method beats the best constant-weaker-A baseline by >= 1% mean error or >= 2% P95 error
adaptive method beats the best lower-lambda baseline by >= 1% mean error or >= 2% P95 error
```

Acceptable pass:

```text
mean error improves by >= 1% relative to baseline
or mean error is not worse by more than 0.3% while P95 improves by >= 3%
strong-worsened fraction does not increase by more than 3% relative
good_initial worsened fraction does not increase by more than 3% relative
CT/TW/KS do not degrade beyond normal run-to-run noise
init_error >= 1.0 m slice keeps at least 90% of the baseline mean-error improvement
adaptive method beats shuffled-a_i, the best constant-weaker-A baseline, and the best lower-lambda baseline on at least one main metric
```

Failure:

```text
adaptive fidelity does not beat baseline
or does not beat shuffled a_i
or does not beat constant weaker a_i baselines
or does not beat lower-lambda baselines
or loses more than 10% of baseline outlier repair
or only works in the wide [0.5, 1.0] range but not in the conservative [0.75, 1.0] range
or the gain can be reproduced by changing lambda or using constant A=cI
or improvement appears only in mean error while P95/max/KS degrade
```

---

## 9. Codex Implementation Instructions

Completed diagnostic task:

```text
diagnose_local_consistency_proxies.py has been extended with C_prot candidates and strict gates.
```

Required diagnostic script:

```text
diagnose_local_consistency_proxies.py
```

Inputs:

```text
--data-path
--distance-run
--init-embedding
--refined-embedding
--neighbor-k
--physical-k
--out-dir
```

Outputs:

```text
local_consistency_proxy_diagnostics.json
local_consistency_proxy_diagnostics.md
node_proxy_scores.csv
score_vs_error_delta plots
score_vs_init_error plots
AUC / Spearman summary table
```

The revised diagnostic report has produced the following conclusion:

```text
Protection proxy:
    failed strict gate.

Repair proxy:
    passed repair gate.
```

Therefore, the next implementation target is not protection-driven adaptive fidelity.

Implement:

```text
run_repair_gated_fidelity_refinement.py
```

This script should implement one-sided repair-gated fidelity only.

The next repair-gated refinement experiments should explicitly compare:

```text
1. Does repair-gated fidelity beat baseline?
2. Does it beat shuffled q_i?
3. Does it beat constant weaker fidelity?
4. Does it beat lower-lambda baseline?
5. Does it preserve baseline outlier repair?
6. Does it avoid increasing strong-worsened / good_initial damage?
```

---

## 10. Current Recommendation

The useful part of Codex's proposal is the proxy design:

```text
raw A_i: neighborhood agreement as a weak baseline/control proxy
C_prot: failed/deferred protection proxy built from mutual support, rank support, multi-k stability, and anti-repair gating
B_i: local physical-neighbor scale/stress
R_i: repair confidence built from [B_i]_+ and r_i
r_i: smoothness residual as repair auxiliary
d_i: baseline movement for diagnostics only
```

The part that must be changed is the fidelity direction.

Do not use in the next implementation:

```math
a_i > 1
```

because the current protection proxy is not reliable enough to justify strong protection.

Use instead:

```math
a_i = 1-\delta q_i,\quad a_i\in[a_{min},1]
```

without median normalization.

This better matches the diagnostic finding:

```text
The baseline diffusion repairs outliers but can damage initially accurate points.
The next module should first test whether high-confidence repair gating can improve selectivity without claiming to identify good points.
```

## 11. Integrated Recommendation After Review

This plan is executable only as a two-stage program:

```text
Stage A: proxy diagnosis.
Stage B: repair-gated adaptive fidelity, because repair gate passed and protection gate failed.
```

Do not execute the two-sided protection + repair design first.

Recommended order:

```text
1. Implement hard repair-gated fidelity:
   q_i = 1[R_i >= Q_p(R)], a_i in {a_min, 1}.
2. Test p in {0.80, 0.85, 0.90} and a_min in {0.75, 0.50}.
3. Compare against baseline, shuffled q_i, constant weaker A, and lower-lambda baselines.
4. If hard gate is promising, test soft sigmoid q_i.
5. Keep C_prot as a diagnostic/control only unless a future proxy diagnostic contradicts the current failure.
```

The central research question is:

```text
Can high-confidence repair-needed nodes be selectively relaxed without globally weakening diffusion?
```

If the answer is no, then repair-gated fidelity is equivalent to parameter tuning and should be stopped.
