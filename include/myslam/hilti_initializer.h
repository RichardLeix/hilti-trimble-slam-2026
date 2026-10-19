#ifndef MYSLAM_HILTI_INITIALIZER_H
#define MYSLAM_HILTI_INITIALIZER_H
#include "myslam/vio_initializer.h"
#include <ostream>

namespace myslam {
// Single-camera robust visual BA followed by the validated full visual+IMU
// covariance alignment. No GT, stereo overlap, persistent map or fallback input.
VioInitializationResult InitializeHiltiStartup(const std::vector<HiltiTrackingResult>& tracks,
    const std::vector<BearingMotionResult>& motions, const Camera& camera,
    const std::vector<ImuSample>& samples, const ImuCalibration& calibration,
    const VioInitializerOptions& options = VioInitializerOptions());
void WriteHiltiInitialization(std::ostream& out, const VioInitializationResult& result);
}
#endif
