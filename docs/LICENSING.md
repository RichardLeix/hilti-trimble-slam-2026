# Licensing and attribution

## Preserved notices

* The SLAM-book framework is attributed to Xiang Gao and contributors. Its
  [upstream license](https://github.com/gaoxiang12/slambook2/blob/master/LICENSE)
  is MIT; the full notice is preserved in `LICENSES/slambook2-MIT.txt`.
* `src/frame.cpp`, `src/map.cpp`, and `src/mappoint.cpp` contain explicit
  GPL-3.0-or-later headers with unresolved template copyright-holder/email
  placeholders. Those headers and the full GPL-3.0 text are retained. They
  are not silently replaced by the upstream MIT notice. Their precise
  provenance and intended license relationship require maintainer confirmation.
* `third_party/fbow/` retains the existing MIT notice, contributor attribution,
  and its upstream revision. No FBOW binaries or vocabulary are bundled.
* `cmake/FindGlog.cmake` retains the full Google/Ceres BSD-style notice and its
  public upstream author contact. These are intentional public attributions.
* The calibration templates derive from Hilti-Research's released calibration;
  see `LICENSES/Hilti-calibration.txt`. Camera masks, vocabulary, bags, and GT
  must be obtained separately under the respective authors' terms.
* Unbundled dependencies retain their own licenses: Eigen, OpenCV, Sophus,
  g2o, Pangolin, glog, SQLite, SuiteSparse/CSparse, OpenMP, OpenGL support
  libraries, and GoogleTest. Linking does not relicense those projects.

## Unresolved grants

The files below have no self-contained explicit license grant in the source
export. Some framework files are apparently SLAM-book-derived; many H2 files
are project-specific. Author/date comments alone are not a license grant.
This inventory also includes the new build/wrapper/test scaffolding so that no
unmarked file is presented as automatically cleared. Public-release preparation
has preserved these files and has not invented a copyright owner or a new
blanket license. Confirm ownership and select compatible terms before public
redistribution. The three ambiguous GPL-header files above require separate
provenance confirmation.

- `CMakeLists.txt`
- `app/export_hilti_submission.cpp`
- `app/run_hilti_sequence.cpp`
- `cmake/FindCSparse.cmake`
- `cmake/FindG2O.cmake`
- `include/myslam/algorithm.h`
- `include/myslam/backend.h`
- `include/myslam/bearing_geometry.h`
- `include/myslam/camera.h`
- `include/myslam/common_include.h`
- `include/myslam/config.h`
- `include/myslam/dataset.h`
- `include/myslam/eucm_camera.h`
- `include/myslam/feature.h`
- `include/myslam/frame.h`
- `include/myslam/frontend.h`
- `include/myslam/g2o_types.h`
- `include/myslam/global_correction.h`
- `include/myslam/global_trajectory.h`
- `include/myslam/hilti_backend.h`
- `include/myslam/hilti_camera_information.h`
- `include/myslam/hilti_dataset.h`
- `include/myslam/hilti_initializer.h`
- `include/myslam/hilti_landmark_quarantine.h`
- `include/myslam/hilti_landmark_weighting.h`
- `include/myslam/hilti_local_map.h`
- `include/myslam/hilti_rolling_shutter.h`
- `include/myslam/hilti_runtime.h`
- `include/myslam/hilti_submission.h`
- `include/myslam/hilti_temporal_tracker.h`
- `include/myslam/hilti_vi_linearization.h`
- `include/myslam/imu_preintegration.h`
- `include/myslam/keyframe_database.h`
- `include/myslam/loop_closing.h`
- `include/myslam/map.h`
- `include/myslam/mappoint.h`
- `include/myslam/marginalization.h`
- `include/myslam/nav_state.h`
- `include/myslam/pose_graph.h`
- `include/myslam/sensor_packet.h`
- `include/myslam/startup_visual_trajectory.h`
- `include/myslam/viewer.h`
- `include/myslam/vio_initializer.h`
- `scripts/run_h2.py`
- `src/admission/camera_floor.cpp`
- `src/admission/camera_floor.h`
- `src/admission/quality.cpp`
- `src/admission/quality.h`
- `src/admission/snapshot.cpp`
- `src/backend.cpp`
- `src/bearing_geometry.cpp`
- `src/camera.cpp`
- `src/config.cpp`
- `src/eucm_camera.cpp`
- `src/feature.cpp`
- `src/frontend.cpp`
- `src/global_correction.cpp`
- `src/global_trajectory.cpp`
- `src/hilti_backend.cpp`
- `src/hilti_dataset.cpp`
- `src/hilti_frontend.cpp`
- `src/hilti_image_epochs.cpp`
- `src/hilti_initializer.cpp`
- `src/hilti_landmark_quarantine.cpp`
- `src/hilti_landmark_weighting.cpp`
- `src/hilti_local_map.cpp`
- `src/hilti_rolling_shutter.cpp`
- `src/hilti_submission.cpp`
- `src/hilti_temporal_tracker.cpp`
- `src/hilti_vi_linearization.cpp`
- `src/hilti_vi_optimizer.cpp`
- `src/imu_preintegration.cpp`
- `src/initialization/metric_alignment.cpp`
- `src/initialization/metric_alignment.h`
- `src/initialization/staged_metric_initializer.cpp`
- `src/initialization/staged_metric_initializer.h`
- `src/initialization/startup_visual_bootstrap.cpp`
- `src/initialization/startup_visual_bootstrap.h`
- `src/initialization/uncertainty_metric_initializer.cpp`
- `src/initialization/uncertainty_metric_initializer.h`
- `src/initialization/visual_bootstrap_uncertainty.cpp`
- `src/initialization/visual_bootstrap_uncertainty.h`
- `src/keyframe_database.cpp`
- `src/loop_closing.cpp`
- `src/loop_geometry.cpp`
- `src/marginalization.cpp`
- `src/nav_state.cpp`
- `src/pose_graph.cpp`
- `src/startup_visual_trajectory.cpp`
- `src/viewer.cpp`
- `src/vio_initializer.cpp`
- `tests/admission.cpp`
- `tests/horizon.cpp`
- `tests/synthetic_window.h`
