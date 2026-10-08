# SC² 边级实验事后分析

**范围。**沿用已固定的 Stage-2 图、SC² 选边掩码、seed-10 分层随机掩码和三份保存的最终图谱。本节没有重算测地距离、训练 FCF、重新选边、改变权重或重跑优化。现有 CG-A 是全样本 transductive 协议，并无独立 held-out CSI 测试集。真实 UE 位置只用于离线评价和选取展示案例，不参与分数、掩码或调参。

**分数和实际改动。**对原始 D_geo 的有向 top-128 局部样本对，先比较 Y0 距离与 D_geo 距离的对数差，减去全体对数差的中位数，再取绝对值 r。r 不超过 median(r)+MAD(r)=0.626941 时记一阶兼容 B_ij=1。对 B 作对称并集；边的二阶分数 T_ij=B_ij×sum_k(B_ik B_kj)，即两端共同拥有多少个兼容的第三点。原 Stage-2 图有 1,544,167 条无向边，其中 1,108,622 条 B=1。SC² 方案用 seed-10 固定打破同分并列，在这些边中选 T 最低的 110,862 条（原图边数的 7.18%），把其 W_ij 乘 0.5，其余边和所有节点 fidelity 不变。随机对照在十个 D_geo 距离层里各选同样数量的 B=1 边，同样乘 0.5。准确顺序和公式见 methods/sc2_stage2_edge_consensus/run.py；固定配置见 config.json。

**选中了什么边。**SC² 和随机各选 110,862 条，十个 D_geo 层的数量完全一致；两组所选边 D_geo 中位数约 41.4，原 W 中位数分别为 0.401、0.409。SC² 选边的共同支持数中位数为 11，随机为 27。仅作事后检查：在每个 D_geo 层的 B=1 边中，把物理端点距离前 10% 标为“跨度偏大”，SC² 选边有 28.1% 落入该组，随机有 10.1%；所选边的物理距离中位数分别为 0.582 m 和 0.442 m。这个 GT 标签说明评分能富集物理跨度大的边，**不能证明这些边对 Stage-2 一定有害**。逐层比较见 [选边诊断图](posthoc_selected_edge_diagnostic.png)，机器明细见 posthoc_edge_and_sample_audit.json。

**为什么总结果没赢随机。**每张图谱都按原评价代码各自做全样本 affine 对齐，再逐点与 GT 比误差；对齐本身也会影响逐点差值，因此不能把变化归因于某条边。对原 baseline 逐点仿射误差比较，SC² 方案修好 10,527 点、伤到 10,470 点，修复和损害的总量分别为 128.04 m、112.77 m；seed-10 随机方案修好 11,181 点、伤到 9,816 点，总量分别为 64.88 m、43.83 m。SC² 对单点的改动更剧烈，正负影响都大，净平均改善 0.000727 m；随机的改动较温和，净平均改善 0.001003 m。SC² 降权集中在 19,582 个样本附近，随机触及 20,898 个；受影响最多的 10% 样本承接的选边端点比例为 28.1% 对 18.0%，每点被选中边数的 P95 为 28 对 18。这与[位置改善损害图](posthoc_repair_harm_map.png)中 SC² 大片蓝红区域相符，是机制线索，并非因果证明。

**改善和变差发生在哪里。**按真实位置分成 4×4 个事后网格，SC² 方案在右上约 (-0.59,-5.89 m) 的 535 个点平均改善 0.0185 m，在左下约 (-8.95,-8.61 m) 的 501 个点改善 0.0221 m；但在中左约 (-5.98,-7.34 m) 的 1,492 个点平均变差 0.0084 m，在中部约 (-2.74,-5.82 m) 的 1,163 个点变差 0.0082 m。随机方案在这些区域的变化通常较小。网格只描述结果位置，不能由此判断是哪条边造成。

**具体样本与连接。**事后挑出的样本 8876（物理位置约 -10.99,-6.71 m）在 SC² 方案下误差从 0.1501 降到 0.0799 m，周围有 24 条选中边；样本 8337（约 -9.51,-5.74 m）则从 0.2479 升到 0.3111 m，周围有 30 条选中边。一条展示用的低支持边 3183–14204，T=9、物理端点距离 2.398 m；另一条 14876–15269，T=11、距离 0.498 m。前者两端样本在完整 SC² 掩码下改善，后者两端样本之一变差。**边和样本示例是看完 GT 后挑出来说明现象的；完整掩码同时改变十万余条边，不能把某个样本的改变归因于上述单条边。**原始编号、坐标、r、D_geo、W、端点误差与逐点数组已保存。

**对不同天线视角的判断。**原始 CSI H.npy 为 20,997×32×13，现有加载代码把 32 个天线重排为 4 组、每组 8 个，AP_pos.npy 有 4 个位置。因此可以进一步核验“同一候选边是否在多个阵列视角都得到支持”。目前保存的 ADP 距离是合并视角后的结果，并没有现成的四组独立边分数。本轮不添加多视角规则；下一步先核对每组阵列的独立距离定义、尺度和缺失情况，再设计不使用 GT 选边的相互核验对照。

---
# SC²-inspired Stage-2 edge-consensus test

This experiment scores candidate Stage-2 graph edges, not nodes. Node fidelity remains 1 in every variant.

## Fixed setup

N=20997; seed=10; original W k=128; lambda=0.15; Jacobi iterations=200.
First-order compatibility threshold=median+MAD=0.626941; B=1 on 1,108,622 of 1,544,167 undirected W edges.
Bottom 110,862 eligible edges were attenuated by 0.5. Uniform solver max coordinate gap to saved Z=0.000366211.

## Does common support flag suspicious edges?

Offline suspect means a GT endpoint separation in the top 10% within its D_geo-distance bin. GT did not enter edge scores or masks.
Suspect rate, low-score versus high-score edge quintile, pooled across D_geo bins: 0.2460 versus 0.0277.
Spearman(score, physical endpoint distance)=-0.5438; first-order residual comparison=0.0141.

## Fixed five-way refinement comparison

| Variant | CT | TW | KS | Mean m | Median m | P90 m | P95 m | Max m | Total W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558674 | 0.634640 | 1.141431 | 1727021.7 |
| sc2 | 0.998215 | 0.998226 | 0.064982 | 0.318021 | 0.293037 | 0.558251 | 0.636267 | 1.128056 | 1680276.5 |
| first_order | 0.998216 | 0.998224 | 0.064990 | 0.318652 | 0.293098 | 0.558974 | 0.635685 | 1.135260 | 1663700.5 |
| shuffled | 0.998225 | 0.998232 | 0.064874 | 0.317745 | 0.291641 | 0.557478 | 0.633300 | 1.135774 | 1679327.6 |
| uniform_mass | 0.998221 | 0.998225 | 0.065015 | 0.318606 | 0.292580 | 0.559256 | 0.634479 | 1.138186 | 1680276.6 |

SC² mean change versus original baseline: -0.000727 m; P95 change: +0.001628 m.
Frozen support criterion met: False.

## Interpretation boundary

This tests a specific edge-consensus transfer, not the full SC²-PCR registration pipeline. The relationship between D_geo and Y0 is not a rigid-transform invariant, and Y0 was trained using D_geo. The offline GT suspect label is an imperfect proxy for a graph edge's causal effect on refinement.
No Stage-1 artifact was changed. No node reliability or adaptive fidelity was used. All inputs on D: were read-only.

Machine outputs: result.json and embeddings.npz. The reproducible edge_scores_and_masks.npz is generated locally by run.py and is not included in the repository.

## Follow-up: what actually caused the improvement?

The original SC²-ranked mask did beat the unmodified advisor baseline in mean error (0.318021 vs 0.318748 m), but its P95 increased (0.636267 vs 0.634640 m). A single distance-stratified random mask on first-order-compatible edges was better (mean 0.317745 m). These are small improvements on the same transductive chart, not evidence that the SC² score caused the gain.

We fixed the number of attenuated edges in each D_geo decile, the 0.5 attenuation, graph, Y0, lambda=0.15 and 200 iterations. Repeating seeds 10..29 gave:

| Random candidate pool | 20-seed median mean error (m) | range (m) | median P95 (m) | seeds beating original baseline |
|---|---:|---:|---:|---:|
| B=1, first-order compatible | 0.317755 | 0.317688–0.317833 | 0.633538 | 20/20 |
| All graph edges | 0.317847 | 0.317797–0.317925 | 0.633682 | 20/20 |
| B=0, first-order incompatible | 0.318247 | 0.318211–0.318296 | 0.633872 | 20/20 |

Every one of the 20 B=1 random masks beat the fixed SC²-ranked result on mean error. Every B=1 random result was also below its paired all-edge random result. Thus the repeatable benefit is **small random attenuation of graph constraints, especially while leaving the first-order-incompatible edges intact**. It is not a demonstrated benefit from the paper's second-order ranking. The B=0 result is consistent with, but does not prove, the idea that disagreement edges are useful correction constraints. The mass differences among random pools are under 0.06% of total original W and are much too small to explain the full mean gap by the retained lambda sweep alone.

The fixed SC² mask improved 10,527 points and worsened 10,470 points relative to uniform, with 128.04 m summed repair and 112.77 m summed harm. The seed-10 B=1 random mask improved 11,181 and worsened 9,816 points, with 64.88 m summed repair and 43.83 m summed harm. SC² makes larger changes in both directions; random attenuation yields a smaller but more consistently positive balance. Both sets of point errors use each chart's own full-data affine alignment, exactly as the baseline evaluation does.

The retained original-graph lambda sweep has its lowest recorded mean at lambda=0.1344: 0.318396 m. The same fixed B=1 random masks with lambda=0.1344 yielded median mean 0.317687 m (range 0.317621–0.317760; 20/20 below the original-W lambda=0.1344 result). Its median P95 was 0.637157 m, almost identical to the original-W lambda=0.1344 P95 of 0.637179 m and above the lambda=0.15 baseline P95 of 0.634640 m. This follow-up is on the same data and is not an independent test or a new lambda search.

The 200-step solver's equation residual RMS was about 0.000387 chart units for original W and 0.000377 for the seed-10 random graph, versus Y0 RMS about 165 chart units. Both are already small, so the observed localization gain is unlikely to be merely faster Jacobi convergence.

## Paper-transfer judgment

[SC²-PCR](https://arxiv.org/html/2203.14453) uses a rigid-distance invariant between two point clouds to identify mutually compatible candidate correspondences, then selects consensus sets and estimates a rigid transformation. Here sample identity is known, D_geo and Y0 are not rigidly related, and Y0 was trained against D_geo. Our edge score reproduces the hard first-order compatibility times common-witness count, but attenuation of W is our own intervention, not the paper's estimator. The score's strong offline association with physically long edges (24.6% suspect in low-score versus 2.8% in high-score quintiles, matched by D_geo decile) did not translate into the strongest refinement. This result does not validate or refute SC²-PCR itself.

The repeatable baseline improvement is currently a graph-constraint perturbation finding. It merits a separate mechanism and generalization study before being called a research contribution. No Stage-1 graph, D_geo, FCF or node fidelity was changed.

Machine evidence: [result.json](result.json), [complete_metrics_followup.json](complete_metrics_followup.json), and [posthoc_edge_and_sample_audit.json](posthoc_edge_and_sample_audit.json).

All 20 seeds in **each** of the three random pools also had P95 below the original lambda=0.15 P95 (0.634640 m). Thus the improvement is not confined to mean error. The eligible-only random mask downweighted 110,862 of 1,544,167 undirected W edges (7.18% of all edges), reducing total graph weight by about 2.76%; the equal-total-weight uniform scaling control achieved only 0.318606 m mean, so uniform weakening alone did not match the random mask's 0.317755 m median.

Exact commands, run from this worktree:
- D:\anaconda3\envs\diffusion_charting\python.exe -u methods\sc2_stage2_edge_consensus\run.py
- D:\anaconda3\envs\diffusion_charting\python.exe -u methods\sc2_stage2_edge_consensus\random_followup.py --pool eligible
- D:\anaconda3\envs\diffusion_charting\python.exe -u methods\sc2_stage2_edge_consensus\random_followup.py --pool all
- D:\anaconda3\envs\diffusion_charting\python.exe -u methods\sc2_stage2_edge_consensus\random_followup.py --pool incompatible
- D:\anaconda3\envs\diffusion_charting\python.exe -u methods\sc2_stage2_edge_consensus\best_lambda_followup.py

All results here are transductive and evaluated on the same original chart. Repeated edge-mask seeds measure perturbation stability, not generalization to new CSI or new environments. The positive random-attenuation finding needs an independent protocol before a paper claim.

## Complete metrics (same evaluation protocol)

Higher CT/TW is better; lower KS and all errors are better. Original and fixed-mask rows are single runs. Random rows are medians across the pre-fixed 20 seeds (10–29); all per-seed values and min/max ranges are in [complete_metrics_followup.json](complete_metrics_followup.json). CT/TW/KS use the same seed-10 subset of 4,199 samples; affine errors use all 20,997 samples.

| Method | CT ↑ | TW ↑ | KS ↓ | Mean m ↓ | Median m ↓ | P90 m ↓ | P95 m ↓ | Max m ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original W, lambda 0.15 | 0.998219593 | 0.998222527 | 0.065064576 | 0.318748 | 0.292917 | 0.558674 | 0.634640 | 1.141431 |
| SC²-ranked edges, lambda 0.15 | 0.998215170 | 0.998226069 | 0.064982254 | 0.318021 | 0.293037 | 0.558251 | 0.636267 | 1.128056 |
| First-order-ranked edges, lambda 0.15 | 0.998215597 | 0.998223504 | 0.064990387 | 0.318652 | 0.293098 | 0.558974 | 0.635685 | 1.135260 |
| Compatible-edge random, lambda 0.15 | 0.998224484 | 0.998231849 | 0.064876288 | **0.317755** | **0.291713** | **0.557865** | **0.633537** | 1.135059 |
| All-edge random, lambda 0.15 | 0.998225826 | 0.998232877 | 0.064887008 | 0.317847 | 0.291831 | 0.557988 | 0.633682 | 1.135528 |
| Incompatible-edge random, lambda 0.15 | 0.998230276 | 0.998232060 | 0.064929313 | 0.318247 | 0.292145 | 0.558305 | 0.633871 | 1.138161 |
| Original W, lambda 0.1344 (retained) | 0.998223398 | 0.998232226 | 0.064921521 | 0.318396 | 0.292453 | 0.559025 | 0.637179 | 1.128971 |
| Compatible-edge random, lambda 0.1344 | 0.998225966 | 0.998240580 | 0.064804183 | **0.317687** | **0.291537** | 0.559515 | 0.637157 | **1.123354** |

Against the original lambda=0.15 baseline, compatible-edge and all-edge random controls improved all eight reported metrics in all 20 runs. Incompatible-edge random improved CT/TW/KS/mean/median/P95 in all 20, P90 in 18, and max in 19. SC²-ranked attenuation improved mean, P90, max, TW and KS, but worsened CT, median and P95. The first-order-ranked variant improved mean, max, TW and KS, but worsened CT, median, P90 and P95.

Against the stronger original-W lambda=0.1344 result, compatible-edge random improved CT/TW/KS/mean/median/max in all 20 runs; P95 improved in 12/20, while P90 worsened in all 20. Against the original lambda=0.15 result, its P90 and P95 were worse in all 20. Thus the lower mean at lambda=0.1344 carries a tail-error tradeoff.

This complete-metric rerun used the exact fixed masks from the earlier follow-ups. Across the 80 repeated solves, the largest change from saved follow-up mean error was 5.96e-8 m and from P95 was 1.97e-6 m, consistent with CUDA accumulation variation. No mask or evaluation subset was changed.

## Representative final Channel Chart

The 20-run median is a summary of metrics, not an embedding. Within the compatible-edge random group at lambda=0.15, seed 18 is closest to the median mean error: recorded 0.317754328 m versus the group median 0.317754909 m. Its fixed mask and 200-step solve were reconstructed; the reproduced mean and P95 matched the saved per-seed record within 1e-5 m. [channel_chart_seed18_vs_baseline.png](channel_chart_seed18_vs_baseline.png) shows physical positions, original Stage-2, and this representative seed-18 final chart on identical aligned axes and colors. [compatible_random_seed18_channel_chart.npz](compatible_random_seed18_channel_chart.npz) saves its final coordinates and sample indices. The visual difference from baseline is subtle, consistent with the roughly 1 mm mean-error change.
