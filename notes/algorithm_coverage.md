# Algorithm Coverage Audit

This note tracks whether each paper's algorithmic pieces are implemented in
the local reproduction code, and whether the result is a strict paper-native
reproduction or an algorithmic reproduction on the shared DICHASUS data.

## 1. Angle-Delay Profile / Timestamp-Aided Dissimilarity

Implemented:

- ADP dissimilarity matrix.
- Timestamp dissimilarity.
- Histogram-based scaling factor `gamma`.
- Fused metric `min(D_ADP, gamma * D_time)`.
- kNN graph and shortest-path geodesic metric.
- MDS embedding.
- Siamese FCF architecture and geodesic-distance regression loss.
- CT/TW/KS and affine localization-error evaluation.

Current limitation:

- The strict run is a PyTorch port of the TensorFlow notebook. The algorithmic
  pieces, data version, shard protocol, feature layer, loss, and training
  schedule are matched, but bitwise-identical TensorFlow numerics are not
  expected.

Status:

- Classical fused-distance path: implemented and run on ADP-native data.
- Neural FCF path: implemented and run on ADP-native data.
- Strict paper-native result: completed with Siamese CT/TW/KS
  `0.9957 / 0.9960 / 0.0841`.

## 2. Channel Charting in Real-World Coordinates With Distributed MIMO

Implemented:

- Official released code path is present locally.
- DICHASUS cf0x preprocessing from TFRecords.
- STO/CPO calibration, time-domain taps `507:520`, and 4-panel antenna order.
- Full-delay AP receive-power extraction for P2 weak losses.
- Paper P2 AP-pair preflight check for `M_p = 0,3,6`.
- Timestamp triplet loss.
- AP bilateration triplet loss.
- LoS bounding-box loss.
- Seed-10 80/20 train/test split and Table-IV style metrics.

Current limitation:

- The default local run uses `reim` features and calibrated `M_t=M_b=0.8`,
  because the public preprocessing path with literal `abs` and margin `0.5`
  under-scales the chart. The code can still run the literal setting for a
  diagnostic, but the corrected run is the one matching Table IV.

Status:

- Strict measured-indoor P2 reproduction: implemented and matched within the
  expected numerical tolerance.

## 3. Indoor Localization With Robust Global Channel Charting

Implemented:

- Official code is present locally.
- Fraunhofer UWB native dataset pipeline.
- Fraunhofer 5G native dataset pipeline.
- Burst grouping, CIR alignment/windowing/normalization through official code.
- Cityblock local distances, kNN graph, shortest-path geodesic distances.
- UWB and 5G CNN Siamese models through official code.
- CT/TW/CE90 evaluation and report generation.

Current limitation:

- The official config uses very high epoch counts. Local runs use limited
  epochs and stop after convergence instead of exactly matching every official
  long training schedule.

Status:

- Paper-native UWB and 5G pipelines: implemented and run.

## 4. Model-Based Approaches to Channel Charting

Implemented:

- Python port of the public ChannelCharting LoS simulator.
- 2048 UE, 32 ULA antennas, VIP shape, carrier frequency and bandwidth from
  the paper.
- Official PCA feature extraction ported from `channel_charting.m`.
- MUSIC angle estimation.
- ISQ range estimate.
- 256-label LR range estimate.
- MUSIC/MUSIC range estimate over 32 subcarriers.
- Rotate-and-sum angle/range estimate.
- CT/TW/KS evaluation at `k = 102`.

Current limitation:

- The Python script currently targets the LoS table-alignment path.
- The paper's QLoS/QNLoS multi-subcarrier model and JM comparison are not fully
  reproduced in Python. The downloaded public `.mat` files contain only
  single-carrier `H` for the old baseline simulator, so they are not sufficient
  for strict MM/RS QLoS/QNLoS table reproduction.

Status:

- LoS Table-II/III/V model-based algorithms: implemented and numerically
  aligned.
- Full paper including QLoS/QNLoS/JM: partial.

## 5. Triplet-Based Wireless Channel Charting

Implemented:

- Timestamp triplet set from Eq. (14).
- Margin triplet loss from Eq. (12).
- Log-sum-exp triplet loss from Eq. (13) as a selectable option.
- Dense network with widths matching Fig. 5:
  `1024,512,256,128,64 -> 2`.
- Paper-style raw-frequency preprocessing on DICHASUS:
  STO/CPO calibration, 2D antenna FFT, frequency autocorrelation, lags
  `0,4,...,60`, and `log(abs(.))`.
- CT/TW/KS and affine localization-error evaluation.

Current limitation:

- The original commercial Huawei 4G Massive MIMO datasets are not available in
  the local resources, so strict native Dataset-1/Dataset-2 reproduction is
  blocked.
- DICHASUS has no dual-polarization axis. Applying the same feature formula
  gives `512` real coefficients instead of the paper's `1024`.
- Curriculum learning, meta-sample CNN extension, and semi-supervised extension
  from later sections are not the default run; the core self-supervised triplet
  method is the one implemented and validated.

Status:

- Core self-supervised triplet algorithm: implemented with the closest local
  preprocessing path and strong metric alignment.
- Strict native paper result: blocked by private dataset availability.
