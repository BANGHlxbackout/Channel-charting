# Strict Reproduction Checklist

This checklist is the acceptance standard for a strict reproduction. A paper is
not considered complete until every required item is checked.

## 1. Angle-Delay / Timestamp-Aided Dissimilarity Metrics

Paper target:

- Dataset: DICHASUS `dichasus-cf0x`, old version, files `cf02`, `cf03`, `cf04`.
- Sampling protocol: concatenate files, then use `full_dataset.shard(4, 0)`.
- CSI preprocessing:
  - parse complex frequency-domain CSI
  - apply reference transmitter STO/CPO calibration
  - IFFT to time domain
  - keep taps `507:520`
  - order antennas into `(4 arrays, 8 antennas, 13 taps)`
- Dissimilarity metrics:
  - ADP / angle-delay profile metric
  - timestamp metric
  - fused metric `min(D_ADP, gamma * D_time)`
  - geodesic fused metric by kNN graph + shortest paths
- Manifold learning / FCF:
  - classical MDS/Isomap-style embedding from geodesic distances
  - Siamese forward charting function
  - feature engineering by CSI sample autocorrelations
  - Siamese loss `mean((||f(x_i)-f(x_j)|| - D_ij)^2 / (D_ij + margin))`
  - learning-rate and batch-size schedule from the public notebook
- Evaluation:
  - CT, TW, KS on a random one-fifth subset with `k=5%`
  - affine alignment and absolute localization error

Current status:

- Completed: ADP-native DICHASUS version 2.0 `cf02/cf03/cf04` sparse dataset
  downloaded and preprocessed with `full_dataset.shard(4,0)`.
- Completed: ADP/fused/geodesic/MDS path on all `20997` samples.
- Completed: PyTorch Siamese FCF with the public notebook architecture, loss,
  feature-engineering layer, and 8-session training schedule.
- Result:
  - MDS CT/TW/KS: `0.9917 / 0.9934 / 0.0907`
  - Siamese FCF CT/TW/KS: `0.9957 / 0.9960 / 0.0841`
  - Siamese affine mean / median / 95th error:
    `0.409 / 0.371 / 0.854 m`
- Remaining caveat: the implementation is a PyTorch port rather than executing
  the original TensorFlow notebook verbatim, so tiny numerical differences are
  expected from initialization, optimizer, and random-pair sampling.

## 2. Triplet-Based Wireless Channel Charting

Paper target:

- Native data: commercial Huawei 4G massive-MIMO dataset.
- Block structure:
  - CSI preprocessing / feature vector construction
  - DNN embedding model
  - timestamp-based triplet construction
  - triplet loss with close/far temporal samples
  - optional curriculum learning and semi-supervised extensions
- Evaluation:
  - trustworthiness / continuity / neighborhood metrics reported in paper
  - visual channel chart comparisons against PCA/autoencoder/Siamese baselines

Current status:

- Native dataset is not publicly available in local resources.
- Implemented: standalone PyTorch core triplet method with Eq. (12), Eq. (13),
  Eq. (14), Fig. 5 network widths, and raw-frequency preprocessing matching
  the paper formula as closely as DICHASUS geometry allows.
- Best local run: `repro_common/results/triplet_paper_raw_frequency`.
- Missing for strict completion: original commercial Huawei 4G dataset; later
  curriculum/meta-sample/semi-supervised extensions are not the default run.

## 3. Channel Charting in Real-World Coordinates With Distributed MIMO

Paper target:

- Dataset: DICHASUS `cf0x` measured indoor distributed MIMO setup.
- Preprocessing:
  - parse TFRecords
  - apply reference transmitter STO/CPO calibration
  - IFFT to time-domain CSI
  - keep taps `507:520`
  - preserve full-delay AP powers for weak losses
- Algorithms:
  - timestamp triplet loss
  - AP-side bilateration loss from received-power ordering
  - LoS bounding-box loss
  - feature extractor and FC network from the released code
- Evaluation:
  - AP-pair statistics for the measured-indoor P2 protocol
  - train/test split, CT, TW, KS, and metric localization error

Current status:

- Implemented with the released code and local fixes for full-delay AP power.
- `check_p2_protocol.py` validates the paper AP-pair statistics before
  reporting Table-IV style P2 runs.
- Seed-10 P2 result matches the README/table target within expected tolerance.
- Caveat: the corrected local reproduction uses `reim` features and
  calibrated margins to match the public raw-data path; literal `abs`/`0.5`
  remains available as a diagnostic.

## 4. Indoor Localization With Robust Global Channel Charting

Paper target:

- Native datasets: Fraunhofer UWB and 5G HDF5 datasets.
- Preprocessing:
  - group anchor records into bursts
  - align CIRs using ToF/TDoA offsets
  - crop/pad to configured window
  - normalize train/test consistently
- Distance matrix:
  - cityblock local CIR distance
  - kNN graph with `n_neighbors=5`
  - shortest-path geodesic distances
  - normalize by 90th percentile
- Siamese model:
  - UWB CNN and 5G CNN architectures
  - pairwise distance regression loss
- Evaluation:
  - CT/TW
  - affine transform to ground-truth coordinates
  - CE90 and MAE as in paper tables

Current status:

- UWB native pipeline runs with official code and native data.
- 5G native pipeline runs with official code and native data.
- Remaining caveat: local runs use limited epochs/stopped-after-convergence
  wrappers instead of every full official `epochs=20000` schedule.

## 5. Model-Based Approaches to Channel Charting

Paper target:

- Native simulator: MATLAB channel charting simulator plus model-based additions.
- Channel models:
  - LoS / Quadriga LoS / Quadriga NLoS where applicable
  - ULA geometry and generated UE locations
- Algorithms:
  - PCA, Sammon mapping, autoencoder baselines
  - MUSIC-based angle estimation
  - ISQ, LR, MM/JM, RS model-based charting algorithms
- Evaluation:
  - CT/TW and visual charts on the paper scenarios
  - runtime/complexity comparison

Current status:

- Official baseline simulator downloaded.
- Implemented: Python LoS simulator and model-based algorithms ISQ, LR, MM,
  and RS. LoS Table-II/III/V values align with the paper.
- Missing for full strict completion: QLoS/QNLoS multi-subcarrier tables and JM
  comparison. The downloaded public `.mat` files contain only single-carrier
  baseline channels, which are insufficient for strict MM/RS QLoS/QNLoS.
