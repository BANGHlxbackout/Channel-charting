# Channel Charting Reference Reproduction Plan

## Shared Baseline

- Primary shared dataset: `channel-charting-distributed-mimo-p2-fix/data1`
- Source dataset: DICHASUS `dichasus-cf0x`, files `cf02`, `cf03`, `cf04`
- Current local tensors:
  - `H.npy`: `(16681, 32, 13)` complex CSI taps
  - `UE_pos.npy`: `(16681, 3)` ground-truth positions
  - `timestamps.npy`: `(16681,)`
  - `AP_pos.npy`: `(4, 3)`
- Standard split: reuse seed `10` and 80/20 train/test where a paper does not prescribe another split.
- Standard metrics:
  - topology: CT, TW, KS
  - localization after affine alignment: mean, median, 95th percentile, max error

## Paper-Specific Reproduction Targets

| Paper | Native data/code status | Shared DICHASUS baseline | Reproduction target |
| --- | --- | --- | --- |
| Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting | Public DICHASUS tutorial notebook downloaded as `external/DissimilarityChannelCharting.ipynb`; paper uses `dichasus-cf0x` old version | Directly compatible. Reshape `H` to `(N, 4, 8, 13)` using the tutorial antenna assignments | Reproduce ADP, timestamp, fused, geodesic-fused metrics with Isomap/MDS and Siamese/PyTorch equivalent |
| Indoor Localization With Robust Global Channel Charting | Official source downloaded as `external/cc_tmcn`; Fraunhofer provides UWB and 5G HDF5 datasets | Not directly equivalent, because the method expects multi-anchor CIR plus ToF/TDoA | First reproduce on paper-native UWB/5G data; then add a DICHASUS adapter for the geodesic CIR-distance baseline if meaningful |
| Model-Based Approaches to Channel Charting | Official dependency downloaded as `external/ChannelCharting`; original paper is MATLAB/QuaDRiGa/model simulation oriented | Not a natural fit for measured DICHASUS, because the proposed algorithms estimate geometry from a channel model | Reproduce the simulator baseline; port the model-based ISQ/LR/MUSIC/RS logic to Python where practical |
| Triplet-Based Wireless Channel Charting: Architecture and Experiments | Original commercial 4G massive-MIMO dataset/code does not appear publicly downloadable from the paper or indexed public resources | Compatible as an algorithmic reproduction using timestamps and CSI | Implement timestamp-triplet PyTorch baseline on DICHASUS and compare with the already-run P2 triplet result |

## Current Local Resource Status

- `common_dichasus.py`: shared loader for DICHASUS `data1`, with P2-compatible seed-10 train/test split and `(N, 4, 8, 13)` antenna-array view.
- `external/cc_tmcn`: official code for the robust global channel charting paper.
- `external/ChannelCharting`: official simple MATLAB simulator used by model-based work.
- `external/DissimilarityChannelCharting.ipynb`: official DICHASUS tutorial code for fused metric / Siamese channel charting.
- `channel-charting-distributed-mimo-p2-fix/data1`: already preprocessed DICHASUS cf0x data.

## Next Execution Order

1. Reproduce ADP/fused-metric paper on the existing DICHASUS data, converting TensorFlow notebook logic to Python 3 + PyTorch where needed.
2. Implement Triplet JSAC 2021 timestamp-triplet baseline on the same DICHASUS split.
3. Download Fraunhofer UWB and 5G datasets and run the official `cc_tmcn` pipeline.
4. Run or port the model-based simulator, keeping it as a paper-native simulation rather than forcing it onto measured DICHASUS data.
