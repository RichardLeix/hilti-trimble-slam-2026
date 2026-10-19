#ifndef MYSLAM_HILTI_ROLLING_SHUTTER_H
#define MYSLAM_HILTI_ROLLING_SHUTTER_H
#include "myslam/hilti_backend.h"

namespace myslam {
// Validated Hilti convention, not verified hardware timestamp semantics.
// Image headers are assumed to refer to the center row; readout is top to bottom.
constexpr double HiltiReadoutSeconds = 0.03105122184753418;
constexpr double HiltiTimestampReferenceRow = 0.5;
constexpr int HiltiReadoutImageHeight = 1440;
double HiltiRowTimeSeconds(double image_y);
Vec3 HiltiInterpolateGyro(const std::vector<ImuSample>& samples, int64_t timestamp_ns);
SE3 HiltiRowCameraWorldPose(const HiltiKeyframeState& frame, int camera,
    const HiltiCameras& cameras, const Vec3& gravity_w, double image_y);
// Row is recovered from the measured EUCM bearing, exactly as in the offline model.
SE3 HiltiVisualCameraWorldPose(const HiltiKeyframeState& frame, int camera,
    const Vec3& bearing, const HiltiCameras& cameras, const Vec3& gravity_w,
    const HiltiBackendOptions& options);
// Configuration documents the fixed validated model; alternative references and
// readout settings are rejected, rather than exposing another tuning mechanism.
HiltiBackendOptions LoadHiltiBackendOptions(const std::string& path);
} // namespace myslam
#endif
