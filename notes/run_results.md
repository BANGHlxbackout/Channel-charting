# Reproduction Run Results

This file separates runnable baselines from strict paper reproductions. A
result is "strict" only if the paper-native dataset, full method, full training
schedule, and reported metrics have all been matched.

## Distributed MIMO P2

- Dataset: DICHASUS `cf0x`, local `channel-charting-distributed-mimo-p2-fix/data1`
- Run: `CC_DATA_PATH=data1 python3 main.py`
- Test metrics:
  - KS: `0.234`
  - TW: `0.929`
  - CT: `0.936`
  - mean / median / 95th / max error: `1.33 / 1.18 / 2.80 / 7.88 m`

## Triplet Baseline on DICHASUS

- Status: unified-dataset algorithmic baseline, not a strict reproduction of
  the JSAC 2021 commercial 4G dataset experiment.
- Dataset: same DICHASUS `data1`
- Run: `CC_LAMBDA_B=0 CC_LAMBDA_BOX=0 CC_NUM_EPOCHS=100 python3 main.py`
- Test metrics:
  - KS: `0.270`
  - TW: `0.944`
  - CT: `0.945`
  - mean / median / 95th / max error: `1.82 / 1.71 / 3.43 / 8.18 m`
- Saved model:
  - `channel-charting-distributed-mimo-p2-fix/results/network_params/your_runID-triplet-Tc5-Mt0.8-1tperanch-bs100-lr0.001-100epochs-hidden832-fereim-schpar50.pth`

### Triplet Paper-Formula PyTorch Reimplementation

- Script: `repro_common/run_triplet_paper.py`
- Status: independent PyTorch implementation of the JSAC triplet principle on
  the unified DICHASUS dataset. The private commercial 4G dataset from the
  paper is still not publicly available.
- Implemented:
  - beamspace + tap-autocorrelation + log-magnitude feature preprocessing,
    adapted to DICHASUS `data1`
  - time-based triplet construction from Eq. (14)
  - margin triplet loss from Eq. (12), with log-sum-exp Eq. (13) available
  - dense charting network with BN/ReLU blocks
- Dataset: DICHASUS `data1`, all `16681` samples, P2 seed-10 train/test split
- Run: `python3 repro_common/run_triplet_paper.py --epochs 40 --batches-per-epoch 100 --batch-size 1000 --out-dir repro_common/results/triplet_paper_dichasus`
- Note: The paper uses `Tc = 0.2 s`; the local sparse DICHASUS data has
  median spacing about `0.24 s`, so this unified-dataset run uses `Tc = 5 s`.
- Test metrics:
  - KS: `0.2879`
  - TW: `0.9289`
  - CT: `0.9270`
  - affine mean / median / 90th / 95th / max error:
    `1.81 / 1.51 / 3.45 / 4.30 / 14.61 m`
- Output:
  - `repro_common/results/triplet_paper_dichasus/n16681_tc5_tfinf_margin_seed10/result.json`
  - `repro_common/results/triplet_paper_dichasus/n16681_tc5_tfinf_margin_seed10/model.pt`
  - `repro_common/results/triplet_paper_dichasus/n16681_tc5_tfinf_margin_seed10/chart_test.png`

#### Triplet Paper-Formula Aligned Network Run

- Run: `python3 repro_common/run_triplet_paper.py --epochs 100 --batches-per-epoch 100 --batch-size 2000 --out-dir repro_common/results/triplet_paper_dichasus_aligned`
- Alignment changes:
  - dense network widths changed to the paper Fig. 5 structure:
    `1024,512,256,128,64 -> 2`
  - margin triplet loss, `M = 1`
  - 100 training epochs; `2000` triplets per optimization batch
- Dataset adaptation:
  - the paper's raw CSI yields `1024` real coefficients after beamspace and
    frequency-autocorrelation preprocessing
  - DICHASUS `data1` has four 8-antenna arrays and 13 delay taps, yielding
    `416` log-magnitude autocorrelation coefficients with the same invariance
    principle
  - the paper uses `Tc = 0.2 s`; sparse DICHASUS uses `Tc = 5 s`, giving
    about `30.95` close candidates per anchor
- Metrics:
  - all samples: CT/TW/KS `0.9640 / 0.9608 / 0.2260`
  - train split: CT/TW/KS `0.9727 / 0.9692 / 0.2164`
  - test split: CT/TW/KS `0.9303 / 0.9284 / 0.2612`
  - all-sample affine mean / median / 95th / max error:
    `1.84 / 1.74 / 3.66 / 11.11 m`
- Paper reference for Dataset 1 triplets-margin:
  - CT/TW/KS `0.977 / 0.967 / 0.194`
- Output:
  - `repro_common/results/triplet_paper_dichasus_aligned/n16681_tc5_tfinf_margin_seed10/result.json`
  - `repro_common/results/triplet_paper_dichasus_aligned/n16681_tc5_tfinf_margin_seed10/model.pt`
  - `repro_common/results/triplet_paper_dichasus_aligned/n16681_tc5_tfinf_margin_seed10/chart_train.png`
  - `repro_common/results/triplet_paper_dichasus_aligned/n16681_tc5_tfinf_margin_seed10/chart_test.png`

#### Triplet Paper-Formula Raw-Frequency Run

- Run: `python3 repro_common/run_triplet_paper.py --feature-source raw_frequency --epochs 100 --batches-per-epoch 100 --batch-size 2000 --train-metrics-max-samples 5000 --all-metrics-max-samples 5000 --out-dir repro_common/results/triplet_paper_raw_frequency`
- This is the closest local implementation to the paper's preprocessing:
  - parse raw 1024-subcarrier DICHASUS TFRecords
  - apply STO/CPO calibration
  - order antennas as 4 panels with 2x4 elements
  - apply 2D FFT over the antenna grid
  - compute frequency-domain autocorrelation
  - keep lags `0,4,8,...,60`
  - take `log(abs(correlation))`
- Data-geometry note:
  - the JSAC paper has a dual-polarized `4x8` array and therefore reports
    `1024` real coefficients
  - DICHASUS has four `2x4` panels without a polarization axis, so the same
    formula gives `512` coefficients
- Training:
  - margin triplet loss, `M = 1`
  - 100 epochs
  - 100 batches per epoch
  - 2000 triplets per batch
  - dense network widths `1024,512,256,128,64 -> 2`
- Metrics:
  - all-sample 5000-point metric subset: CT/TW/KS `0.9854 / 0.9876 / 0.2116`
  - train 5000-point metric subset: CT/TW/KS `0.9857 / 0.9879 / 0.2121`
  - exact test split: CT/TW/KS `0.9835 / 0.9860 / 0.2160`
  - test affine mean / median / 90th / 95th / max error:
    `0.60 / 0.49 / 1.13 / 1.48 / 3.32 m`
- Paper reference for Dataset 1 triplets-margin:
  - CT/TW/KS `0.977 / 0.967 / 0.194`
- Output:
  - `repro_common/cache/triplet_raw_frequency_features.npz`
  - `repro_common/results/triplet_paper_raw_frequency/n16681_tc5_tfinf_margin_seed10/result.json`
  - `repro_common/results/triplet_paper_raw_frequency/n16681_tc5_tfinf_margin_seed10/model.pt`
  - `repro_common/results/triplet_paper_raw_frequency/n16681_tc5_tfinf_margin_seed10/chart_train.png`
  - `repro_common/results/triplet_paper_raw_frequency/n16681_tc5_tfinf_margin_seed10/chart_test.png`

## ADP / Timestamp Fused Metric on DICHASUS

- Status: strict paper-native reproduction completed against the public
  DICHASUS notebook protocol.
- Dataset: DICHASUS dataset version 2.0 files `cf02`, `cf03`, `cf04`, sparse
  downloaded from persistent IDs `/8`, `/9`, `/10`.
- Sampling protocol: concatenate files and keep `full_dataset.shard(4, 0)`,
  giving `20997` samples.
- Preprocessing: STO/CPO calibration, IFFT to time domain, taps `507:520`,
  fixed four-array antenna ordering from the notebook.
- Classical fused-distance run:
  - script: `repro_common/run_adp_fused.py`
  - method: ADP + timestamp fused metric + 20-NN geodesic + metric MDS
  - gamma: `12.9833336`
  - MDS samples: `2100` (`nth_reduction=10`)
  - CT/TW/KS: `0.9917 / 0.9934 / 0.0907`
  - affine mean / median / 95th / max error:
    `0.458 / 0.385 / 0.961 / 8.897 m`
- Siamese FCF run:
  - script: `repro_common/run_adp_siamese.py`
  - feature layer: CSI sample autocorrelation
  - network: `1024,512,256,128,64 -> 2`
  - schedule: public notebook learning rates and batch sizes, 8 sessions
  - CT/TW/KS on one-fifth random subset: `0.9957 / 0.9960 / 0.0841`
  - affine mean / median / 95th / max error:
    `0.409 / 0.371 / 0.854 / 7.986 m`
- Output:
  - `repro_common/datasets/adp_old_shard4`
  - `repro_common/results/adp_old_fused/stride1_n20997_seed10/result.json`
  - `repro_common/results/adp_old_siamese/stride1_n20997_seed10/result.json`
  - `repro_common/results/adp_old_siamese/stride1_n20997_seed10/model.pt`

### New Method: Graph-Harmonic Diffusion Refined Charting

- Status: new self-supervised idea tested on the same ADP-native DICHASUS data.
- Script: `repro_common/run_diffusion_refined_charting.py`
- Main idea: estimate a Laplace-Beltrami graph from ADP/time geodesic
  neighbors, then solve `(I + lambda L) Z = Y0` to denoise the Siamese FCF
  chart without using UE positions.
- Run:
  - `python3 repro_common/run_diffusion_refined_charting.py --data-path repro_common/datasets/adp_old_shard4 --distance-run repro_common/results/adp_old_fused/stride1_n20997_seed10 --init-embedding repro_common/results/adp_old_siamese/stride1_n20997_seed10/embedding.npz --out-dir repro_common/results/diffusion_refined_charting --stride 1 --graph-k 128 --lambda-smooth 0.15 --iterations 200`
- Result:
  - baseline ADP/time Siamese FCF CT/TW/KS:
    `0.9957 / 0.9960 / 0.0841`
  - diffusion-refined CT/TW/KS:
    `0.9982 / 0.9982 / 0.0651`
  - baseline affine mean / median / 95th / max error:
    `0.409 / 0.371 / 0.854 / 7.986 m`
  - diffusion-refined affine mean / median / 95th / max error:
    `0.319 / 0.293 / 0.635 / 1.142 m`
  - focused search note: `graph_k=128, lambda=0.15` is best among tested
    settings by KS, mean error, and P95 error; `graph_k=96, lambda=0.2` has
    slightly higher CT/TW but worse KS and localization error.
- Output:
  - `repro_common/results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json`
  - `repro_common/results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz`
  - `repro_common/notes/new_idea_diffusion_refined_channel_charting.md`

### ADP Siamese FCF Diagnostic

- Script: `run_adp_siamese.py`
- Target distances: `results/adp_fused/stride1_n4000_seed10/geodesic.npy`
- Training schedule: public notebook schedule, 8 sessions, 150000 random pairs/session
- Result on P2 `data1` first 4000 samples:
  - CT: `0.6273`
  - TW: `0.5723`
  - KS: `0.6010`
  - affine mean / median / 95th / max error: `3.60 / 3.25 / 7.25 / 9.41 m`
- Status: failed diagnostic. The PyTorch implementation runs and the loss
  decreases, but this run does not reproduce the high-quality Siamese chart.
  Next required fix is to run on the ADP-native `shard(4,0)` data and re-check
  target-distance scaling/training stability against the notebook.

## Indoor Robust Global Channel Charting, UWB Native Dataset

- Status: paper-native UWB pipeline using the official code and dataset, with
  limited-epoch training. The official repository sets `epochs=20000`; this run
  was stopped after convergence around epoch 234.
- Dataset: Fraunhofer UWB dataset
- Official source: `external/cc_tmcn`
- Preprocessed tensors:
  - training: `(18034, 6, 200)`
  - test: `(3382, 6, 200)`
- Distance matrix:
  - NN: `0.0098 s`
  - graph: `19.53 s`
  - shortest path: `31.57 s`
- Training:
  - official config sets `epochs=20000`, so the run was manually stopped after epoch 234 once loss stabilized near `0.038`
  - limited-epoch wrapper added: `run_cc_tmcn_limited.py`
- Evaluation:
  - train CT/TW/CE90: `0.997144 / 0.996673 / 1.38 m`
  - test CT/TW/CE90: `0.996701 / 0.996286 / 1.38 m`
- Reports:
  - `repro_common/results/cc_tmcn_uwb/reports/uwb_train.pdf`
  - `repro_common/results/cc_tmcn_uwb/reports/uwb_test.pdf`

## Indoor Robust Global Channel Charting, 5G Native Dataset

- Status: paper-native 5G pipeline using the official code and dataset, with
  limited-epoch training via `run_cc_tmcn_limited.py`.
- Dataset: Fraunhofer 5G dataset
- Download:
  - `repro_common/datasets/fraunhofer/5G_dataset.zip`
  - extracted to `repro_common/datasets/fraunhofer/5g`
- Preprocessed tensors:
  - training: `(18717, 6, 49)`
  - test: `(15722, 6, 49)`
- Distance matrix:
  - NN: `0.0046 s`
  - graph: `5.13 s`
  - shortest path: `36.95 s`
- Training:
  - script: `run_cc_tmcn_limited.py`
  - model: `Model_5G`
  - epochs: `250`
  - final displayed loss: about `0.066`
- Evaluation:
  - train CT/TW/CE90: `0.983432 / 0.981356 / 2.33 m`
  - test CT/TW/CE90: `0.980293 / 0.978531 / 2.55 m`
- Reports:
  - `repro_common/results/cc_tmcn_5g/reports/5g_train.pdf`
  - `repro_common/results/cc_tmcn_5g/reports/5g_test.pdf`

## Model-Based Approaches to Channel Charting

- Script: `repro_common/run_model_based.py`
- Status: Python reimplementation of the MATLAB simulator and the paper's
  model-based algorithms on the 3D LoS scenario.
- Implemented:
  - public ChannelCharting 2048-UE / 32-antenna / VIP LoS scenario
  - spherical-wave LoS channel, SNR 0 dB
  - official PCA feature extraction ported from `channel_charting.m`
  - MUSIC AoA, ISQ, 256-label LR, MUSIC/MUSIC (MM), and rotate-and-sum (RS)
- Run: `python3 repro_common/run_model_based.py --u 2048 --skip-feature-mds --out-dir repro_common/results/model_based_los_u2048`
- Metrics at `k = 0.05N = 102`:
  - PCA official features: CT/TW/KS `0.7628 / 0.6844 / 0.6358`
  - ISQ + MUSIC angle: CT/TW/KS `0.9816 / 0.9702 / 0.2268`
  - LR + MUSIC angle: CT/TW/KS `0.9953 / 0.9915 / 0.1496`
  - MM: CT/TW/KS `1.0000 / 1.0000 / 0.0057`
  - RS: CT/TW/KS `1.0000 / 1.0000 / 0.0057`
- Output:
  - `repro_common/results/model_based_los_u2048/result.json`
  - `repro_common/results/model_based_los_u2048/*.png`

### Model-Based Ideal LoS Table-Alignment Run

- Run: `python3 repro_common/run_model_based.py --u 2048 --snr-db inf --skip-feature-mds --out-dir repro_common/results/model_based_los_u2048_ideal`
- Rationale: the paper's main Tables II, III, and V use the nominal LoS,
  QLoS, and QNLoS simulations; the explicit noise degradation case is reported
  separately in Table VIII. The ideal LoS run is therefore the closest
  Python-side comparison to the main LoS table entries.
- Metrics at `k = 102`:
  - PCA official features: CT/TW/KS `0.9244 / 0.8548 / 0.4292`
  - ISQ + MUSIC angle: CT/TW/KS `0.9959 / 0.9956 / 0.1477`
  - LR + MUSIC angle: CT/TW/KS `0.9990 / 0.9987 / 0.0787`
  - MM 32 subcarriers: CT/TW/KS `1.0000 / 1.0000 / 0.0054`
  - RS 32 subcarriers: CT/TW/KS `1.0000 / 1.0000 / 0.0054`
- Paper LoS table references:
  - Table II PCA CT/TW `0.9288 / 0.8603`
  - Table II LR CT/TW `0.9968 / 0.9930`
  - Table II ISQ CT/TW `0.9940 / 0.9885`
  - Table III MM 32sc CT/TW `1.0000 / 0.9998`
  - Table V RS 32sc CT/TW `0.9998 / 0.9998`
- Output:
  - `repro_common/results/model_based_los_u2048_ideal/result.json`
  - `repro_common/results/model_based_los_u2048_ideal/*.png`

## Pending

- Triplet JSAC 2021: strict native reproduction is blocked unless the original
  commercial 4G dataset/code is obtained from the authors; unified DICHASUS
  formula reproduction is now implemented and run.
- Fraunhofer 5G dataset: downloaded and run with the official pipeline.
