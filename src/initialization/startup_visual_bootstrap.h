#ifndef STARTUP_VISUAL_BOOTSTRAP_EXPERIMENT_H
#define STARTUP_VISUAL_BOOTSTRAP_EXPERIMENT_H
#include "myslam/startup_visual_trajectory.h"
namespace myslam {
struct BootstrapObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t frame=0;
    Vec3 ray=Vec3::Zero(); // Actual EUCM oriented unit bearing used in refinement.
};
struct BootstrapLandmark {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t id=0;
    Vec3 point=Vec3::Zero(); // Arbitrary visual world units; optimized, not retriangulated.
    std::vector<BootstrapObservation,Eigen::aligned_allocator<BootstrapObservation>> observations;
};
// Diagnostic-only. Poses are T_camera_visualWorld; translations are arbitrary
// visual units. No IMU samples, ground truth, runtime state or Map are accepted.
struct BootstrapPairAudit {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t frame = 0, shared = 0, pair_positive = 0, pair_opposite_positive = 0;
    size_t multiview_shared = 0, multiview_positive = 0, heldout_shared = 0;
    Vec3 pair_axis = Vec3::Constant(NAN), reconstructed_direction = Vec3::Constant(NAN);
    bool input_translation_valid = false, pose_available = false;
    double epipolar_rms = NAN, rotation_fit_rms = NAN, rotation_change = NAN;
    double accumulated_rotation_change = NAN, axis_difference = NAN;
    double sign_confidence = NAN, heldout_vote = NAN, heldout_margin = NAN;
    double heldout_residual = NAN, heldout_positive_fraction = NAN;
    double parallax = NAN, depth_condition_median = NAN, depth_condition_max = NAN;
    double baseline = NAN, baseline_ratio = NAN, bearing_median = NAN;
    double positive_fraction = NAN, pose_condition = NAN;
};
struct BootstrapAttempt {
    size_t anchor = 0, provisional_pose_failures = 0, tracks = 0, weak_tracks = 0;
    bool valid = false;
    double anchor_parallax = NAN, rms = NAN, condition = NAN;
    std::string reason;
};
struct BootstrapResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VioVisualWindow window;
    std::vector<SE3, Eigen::aligned_allocator<SE3>> poses;
    // Read-only snapshot for offline covariance. Never published to runtime Map.
    std::vector<BootstrapLandmark,Eigen::aligned_allocator<BootstrapLandmark>> landmarks;
    std::vector<BootstrapPairAudit, Eigen::aligned_allocator<BootstrapPairAudit>> pairs;
    std::vector<BootstrapAttempt> attempts;
    size_t audited_pose_count = 0, weak_tracks = 0;
    double track_parallax_median = NAN, track_condition_median = NAN;
    double cloud_planarity_ratio = NAN, trimmed_planarity_ratio = NAN;
    double trimmed_linearity_ratio = NAN, max_pose_median = NAN;
    double positive_fraction = NAN, runtime_ms = 0;
    std::string audit_scope;
};
BootstrapResult AuditCurrentStartup(const std::vector<HiltiTrackingResult>& frames,
    const std::vector<BearingMotionResult>& motions, const Camera& camera,
    const ImuCalibration& calibration, const VioInitializerOptions& options = VioInitializerOptions());
BootstrapResult BuildMultiFrameStartup(const std::vector<HiltiTrackingResult>& frames,
    const std::vector<BearingMotionResult>& motions, const Camera& camera,
    const ImuCalibration& calibration, const VioInitializerOptions& options = VioInitializerOptions());
}
#endif
