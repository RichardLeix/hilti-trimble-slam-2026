#ifndef MYSLAM_VIO_INITIALIZER_H
#define MYSLAM_VIO_INITIALIZER_H

#include "myslam/bearing_geometry.h"
#include "myslam/nav_state.h"
#include <limits>

namespace myslam {

struct VioVisualFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t timestamp_ns = 0;  // Derived IMU clock; raw image times are untouched.
    SO3 rotation_wb;           // IMU/body to the visual reference world.
    Vec3 camera_position = Vec3::Zero();  // Camera centre in ONE common visual scale.
};
using VioVisualFrames = std::vector<VioVisualFrame, Eigen::aligned_allocator<VioVisualFrame>>;
using VioImuIntervals = std::vector<ImuPreintegration, Eigen::aligned_allocator<ImuPreintegration>>;
using VioNavStates = std::vector<NavState, Eigen::aligned_allocator<NavState>>;

enum class StartupPairStatus { Accepted, Ambiguous, Rejected };
const char* StartupPairStatusName(StartupPairStatus status);
struct StartupPairDiagnostic {
    size_t pair = 0;  // One-based consecutive interval.
    // Relative to input t / unsigned essential axis. Without either, use the
    // reconstructed axis with its largest-magnitude component positive.
    int selected_sign = 0;  // Zero if undecided.
    StartupPairStatus status = StartupPairStatus::Ambiguous;
    size_t supporting_tracks = 0;
    size_t positive_rays = 0, opposite_positive_rays = 0;
    double residual_rad = std::numeric_limits<double>::quiet_NaN();
    double opposite_residual_rad = std::numeric_limits<double>::quiet_NaN();
    double sign_vote_fraction = 0;
    double residual_margin_rad = 0;
    double parallax_rad = 0;
    double baseline = 0;  // Arbitrary visual units.
    double continuity_score = 0;  // Adjacent velocity change, diagnostic only.
    std::string reason = "not evaluated";
};

struct VioVisualWindow {
    VioVisualFrames frames;
    bool positions_in_common_scale = false;
    bool stationary_visual_evidence = false;
    size_t reconstruction_tracks = 0;
    double visual_condition = std::numeric_limits<double>::infinity();
    double bearing_rms_rad = std::numeric_limits<double>::quiet_NaN();
    std::string translation_failure;
    size_t anchor_frame = 0;
    size_t rejected_tracks = 0;
    std::vector<StartupPairDiagnostic> pair_diagnostics;
};

struct VioInitializerOptions {
    size_t min_frames = 10;
    double min_duration_s = 0.5;
    double gravity_magnitude = 9.81;  // Known magnitude; direction is estimated.
    double max_gravity_magnitude_error = 0.5;
    double min_acceleration_excitation = 0.25;  // RMS variation, m/s^2.
    double max_condition = 1e5;  // SVD after column normalization.
    double max_visual_condition = 1e4;
    double max_rotation_rms_rad = 0.01;
    double max_rotation_error_rad = 0.025;
    double max_gyro_bias = 0.2;  // rad/s.
    double max_accel_bias = 1.0;  // m/s^2.
    double max_position_rms_m = 0.02;
    double max_velocity_rms_m_s = 0.15;
    double max_bearing_rms_rad = 0.005;
    double stationary_bearing_step_rad = 0.001;
    double stationary_rotation_rad = 0.005;
    double stationary_accel_std = 0.1;  // Norm of component standard deviations.
    double stationary_gyro_std = 0.005;
};

enum class VioInitializationStatus {
    Success,
    StationaryPartial,  // Rest assumption: gravity/gyro/v=0 available, scale unavailable.
    InsufficientData,
    InvalidInput,
    RotationInconsistent,
    VisualStructureUnavailable,
    InsufficientExcitation,
    IllConditioned,
    ImplausibleSolution,
    AlignmentInconsistent,
    UncertaintyInvalid
};
const char* VioInitializationStatusName(VioInitializationStatus status);

struct StartupUncertaintyDiagnostics {
    bool combined = false, visual_covariance_valid = false, converged = false;
    int gauge_dimensions = 0, visual_rank = 0, visual_columns = 0;
    int deficient_landmarks = 0, residual_dof = 0, reweighting_iterations = 0;
    size_t anchor_frame = 0;
    double bearing_variance = NAN, visual_condition = INFINITY;
    double gauge_null_leakage = NAN, anchor_constraint_error = NAN;
    double covariance_condition = INFINITY, normalized_covariance_condition = INFINITY;
    double cross_interval_covariance_norm = NAN, whitening_error = NAN;
    double relative_scale_step = NAN, scale_std_unit_noise = NAN;
    double scale_std_residual_scaled = NAN, velocity_scale_correlation = NAN;
    double position_visual_variance = NAN, position_imu_variance = NAN;
    double velocity_visual_variance = NAN, velocity_imu_variance = NAN;
};

struct VioInitializationResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VioInitializationStatus status = VioInitializationStatus::InsufficientData;
    std::string message;
    std::string path = "undetermined";
    size_t frames_used = 0;
    Vec3 gravity_w = Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
    double scale = std::numeric_limits<double>::quiet_NaN();  // metres/visual unit.
    Vec3 initial_velocity_w = Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
    ImuBias bias;  // Unestimated accel bias is an EXPLICIT zero prior.
    bool gyro_bias_estimated = false, accel_bias_estimated = false;
    double duration_s = 0;
    double acceleration_excitation = 0;
    double accel_std = 0, gyro_std = 0;
    double gyro_condition = std::numeric_limits<double>::infinity();
    double alignment_condition = std::numeric_limits<double>::infinity();
    double accel_bias_condition = std::numeric_limits<double>::infinity();
    int alignment_rank = 0, alignment_columns = 0;
    double rotation_rms_rad = std::numeric_limits<double>::quiet_NaN();
    double max_rotation_error_rad = std::numeric_limits<double>::quiet_NaN();
    double position_rms_m = std::numeric_limits<double>::quiet_NaN();
    double velocity_rms_m_s = std::numeric_limits<double>::quiet_NaN();
    VioNavStates states;  // Published only for complete Success; first body origin is zero.
    StartupUncertaintyDiagnostics uncertainty;
    bool Success() const { return status == VioInitializationStatus::Success; }
};

// Standalone startup-window reconstruction. Uses the existing per-pair R/t and
// persistent bearing tracks for ONE camera. First camera pose defines the world;
// A long-baseline shared structure fixes ONE ARBITRARY VISUAL UNIT, not one metre.
// Signs/relative magnitudes follow multi-frame bearing-pose/structure consistency.
// Temporary points are discarded; no persistent Map or stereo overlap is used.
VioVisualWindow MakeVioVisualWindow(const std::vector<HiltiTrackingResult>& frames,
                                   const std::vector<BearingMotionResult>& motions,
                                   const Camera& camera, const ImuCalibration& calibration,
                                   const VioInitializerOptions& options = VioInitializerOptions());

class VioInitializer {
   public:
    explicit VioInitializer(const VioInitializerOptions& options = VioInitializerOptions());

    // The supplied preintegrations connect consecutive frames at zero bias.
    // Raw samples are retained to re-integrate at the estimated fixed biases.
    // T_camera_body maps IMU points into this camera, including its metric lever arm.
    // Dynamic alignment estimates common metric scale, world gravity direction,
    // world velocities and gyro bias; accel bias is fitted only if observable.
    // Stationary initialization is partial: v=0 is a rest assumption, accel bias
    // remains a zero prior, and Success() is false because scale is unobservable.
    // Unknown/failed metric estimates remain NaN; scale is never forced to one.
    VioInitializationResult Initialize(const VioVisualWindow& window,
                                        const VioImuIntervals& intervals,
                                        const std::vector<ImuSample>& samples,
                                        const ImuCalibration& calibration,
                                        const SE3& T_camera_body) const;

    // Same gyro stage and validation as Initialize; stops BEFORE metric/bias
    // alignment. gyro_bias_estimated reports success; no NavStates are published.
    VioInitializationResult EstimateGyroBias(const VioVisualWindow& window,
        const VioImuIntervals& intervals, const std::vector<ImuSample>& samples,
        const ImuCalibration& calibration, const SE3& T_camera_body) const;

   private:
    VioInitializationResult InitializeStages(const VioVisualWindow& window,
        const VioImuIntervals& intervals, const std::vector<ImuSample>& samples,
        const ImuCalibration& calibration, const SE3& T_camera_body, bool gyro_only) const;
    VioInitializerOptions options_;
};

}  // namespace myslam
#endif
