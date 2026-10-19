#ifndef MYSLAM_STARTUP_VISUAL_TRAJECTORY_H
#define MYSLAM_STARTUP_VISUAL_TRAJECTORY_H

#include "myslam/vio_initializer.h"

namespace myslam {

// Visual-only, temporary structure and bearing-pose fitting for initialization.
// Uses ONE camera's persistent tracks. No IMU values or metric motion enter sign
// selection. The first camera defines the world; a long anchor baseline fixes
// one arbitrary visual unit. Fits points/poses only inside this startup window.
// Per-pair signs are tested against structure triangulated without that pair's
// observations. Unsupported/ambiguous pairs prevent publishing a full window.
VioVisualWindow BuildStartupVisualTrajectory(
    const std::vector<HiltiTrackingResult>& tracks,
    const std::vector<BearingMotionResult>& motions, const Camera& camera,
    const ImuCalibration& calibration,
    const VioInitializerOptions& options = VioInitializerOptions());

}  // namespace myslam
#endif
