# Hilti-Trimble-Oxford SLAM Challenge 2026 — Visual-Inertial SLAM

A dual-fisheye visual-inertial SLAM system developed for the
Hilti-Trimble-Oxford SLAM Challenge 2026. The H2 configuration combines temporal
visual tracking, IMU preintegration, sliding-window visual-inertial optimization,
a marginalization prior, loop closure, and a rigid pose graph.

Landmarks use C_quality information weighting, source/prior quarantine, and
B3 global quality-aware admission with a native camera1 floor and protected
prior IDs. The geometry horizon uses 12 source states and 11 retained states
(approximately 1.67 seconds at the normal keyframe cadence). Camera information
weights are fixed at 1.0 / 0.5, with an active landmark budget of 240. Frame
removal remains FIFO. Startup retry and late-start recovery are retained.

## Build and dependencies

Linux, CMake 3.16+, a C++14-capable compiler, Python 3.8+, Eigen3, OpenCV **3.4**,
Sophus, g2o with CSparse, Pangolin, glog, SQLite3, OpenMP, and the OpenGL support
libraries are required. GoogleTest is required for the synthetic unit checks.
See [build instructions and dependency revisions](docs/BUILD.md).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$DEPENDENCY_PREFIX" -DG2O_ROOT="$DEPENDENCY_PREFIX"
cmake --build build --parallel 2
python3 scripts/run_h2.py verify --build-dir build
```

The source builds H2 directly; no prebuilt estimator binaries or runtime
interposition libraries are required. The runner preserves the fixed internal
OpenMP, BLAS, and OpenCV environment settings.

## Dataset setup and run

Obtain bags and the original calibration assets from the
[official dataset](https://hilti-challenge.com/dataset-2026) and
[official tools/configuration repository](https://github.com/Hilti-Research/hilti-trimble-slam-challenge-2026).
Keep the data outside this source tree. Supply the official `mask_cam0.png`
and `mask_cam1.png` masks (the `hilti_mask_cam*.png` filenames are also accepted) and the compatible ORB FBOW vocabulary separately.
No dataset, GT, masks, vocabulary, or generated trajectory is distributed here.
The reader accesses the original ROS2 SQLite bag directly; ROS installation
and bag conversion are not required.

```sh
python3 scripts/run_h2.py run \
  --bag /path/to/sequence/rosbag \
  --vocabulary /path/to/orb_vocab.fbow \
  --mask-dir /path/to/official/masks \
  --sequence floor_2_2025-12-03_run_1 \
  --output-prefix outputs/floor_2/run \
  --build-dir build
```

`--config` accepts a portable H2 JSON configuration. The runner rejects changes
to the declared H2 configuration and existing output prefixes. It generates a
camera YAML beside the outputs with only the external mask paths substituted;
calibration numbers are unchanged. No ground truth is read.

Native body states and keyframe records are written beside the output prefix.
The exporter writes local/global camera-0 trajectories into
`<output-prefix>_trajectories/{local,global}/<sequence>.txt`:

```text
timestamp tx ty tz qx qy qz qw
```

Timestamps are seconds on the original camera clock, translation is metric,
and quaternion order is `qx qy qz qw`. The camera convention is
`T_gc0 = T_gb * inverse(T_cam0_body)`. Exported trajectories have no GT alignment
or scale correction. Uninitialized poses are explicitly omitted.

## Results

Verified official Hilti server result for H1: **1643.9 / 2500**.
No official H2 server score is claimed here.

## Repository structure

```text
app/          Sequence runner and camera-0 trajectory exporter
cmake/        Required dependency discovery modules
config/       H2 definition and calibrated sensor/backend templates
include/      Public C++ interfaces
src/          Tracking, IMU, backend, admission, initialization, and pose graph
scripts/      Portable H2 runtime entrypoint
tests/       Synthetic horizon, admission, and prior-protection checks
third_party/  FBOW inference source and its license
LICENSES/     Preserved license texts and calibration attribution
docs/         Build and licensing information
```

## License and acknowledgements

Built on the SLAM-book framework by Xiang Gao and contributors, with Eigen,
OpenCV, Sophus, g2o, Pangolin, glog, SQLite, SuiteSparse/CSparse, and FBOW.
Calibration and dataset credit belongs to Hilti, Trimble, Oxford, and the
[dataset authors](LICENSES/Hilti-calibration.txt).

This tree has mixed existing notices. See [LICENSE](LICENSE) and
[the licensing audit](docs/LICENSING.md), including unresolved grants for
unmarked H2 files. No blanket permissive license or redistribution clearance
is asserted for those files.
