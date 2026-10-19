#ifndef MYSLAM_HILTI_RUNTIME_H
#define MYSLAM_HILTI_RUNTIME_H

#include "myslam/hilti_dataset.h"
#include "myslam/vio_initializer.h"

namespace myslam {
struct HiltiBackendResult;
struct GlobalNavigation;

struct HiltiImageEpoch {
    int64_t timestamp_ns = 0;  // cam0 exposure on calibrated IMU clock.
    std::array<int64_t, 2> camera_imu_timestamps_ns{{0, 0}};
    std::array<SensorPacket, 2> cameras;  // Original raw timestamps/JPEG bytes.
    // Brackets previous reference epoch, this reference epoch and both exposures.
    // Includes boundary neighbors for interpolation; never extrapolates.
    std::vector<ImuSample> imu_samples;
};

// This Hilti run has exactly synchronized RAW image timestamps. Missing/mismatched
// images fail explicitly rather than pairing nearby but different exposures.
class HiltiImageEpochReader {
   public:
    HiltiImageEpochReader(HiltiDataset& dataset, const ImuCalibration& calibration);
    bool Next(HiltiImageEpoch& epoch);
    size_t EpochsRead() const { return epochs_read_; }
   private:
    HiltiDataset& dataset_;
    ImuCalibration calibration_;
    std::vector<ImuSample> samples_;
    std::array<SensorPacket, 2> pending_;
    std::array<bool, 2> have_{{false, false}};
    bool have_previous_ = false;
    int64_t previous_time_ = 0, previous_raw_ = 0;
    size_t epochs_read_ = 0;
};

enum class HiltiFrontendStatus { Buffering, Initialized, Tracking, PredictionOnly, InitializationFailed };
const char* HiltiFrontendStatusName(HiltiFrontendStatus status);

struct HiltiFrontendOptions {
    size_t startup_frames = 60;
    // Validated Hilti production retry policy. Frame indices are zero based, so
    // 299 permits the last fixed window 240-299.
    bool startup_retry_enabled = true;
    size_t startup_retry_stride = 10;
    size_t startup_max_end_frame = 299;
    // Transactional late-start recovery. The accepted initializer remains the
    // sole metric anchor; buffered states are reconstructed only from tracks,
    // calibrated IMU propagation and fixed accepted metric landmarks.
    bool startup_backfill_enabled = false;
    double startup_backfill_start_s = 5.0;
    HiltiTrackingOptions tracking;
    VioInitializerOptions initialization;  // Existing defaults, unchanged.
    size_t min_visual_inliers = 20;
    double max_bearing_error_rad = 0.005;
    double max_pose_condition = 1e4;
    double max_correction_m = 0.15;
    double max_correction_rad = 0.1;
    // Explicit diagnostic opt-in; never an automatic failure fallback.
    bool diagnostic_legacy_startup = false;
};
HiltiFrontendOptions LoadHiltiFrontendOptions(const std::string& path);

struct HiltiStartupAttempt {
    size_t attempt = 0, start_frame = 0, end_frame = 0;
    int64_t raw_cam0_start_ns = 0, raw_cam0_end_ns = 0;
    int64_t imu_sample_start_ns = 0, imu_sample_end_ns = 0;
    size_t imu_samples = 0;
    VioInitializationStatus status = VioInitializationStatus::InsufficientData;
    bool recoverable = false;
    std::string reason;
};

enum class HiltiStartupRecoveryMode {
    InitializerAnchor,
    VisualRefined,
    ImuPropagatedOnly
};
const char* HiltiStartupRecoveryModeName(HiltiStartupRecoveryMode mode);

struct HiltiStartupRecoveryFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t frame_index = 0;
    int64_t raw_cam0_timestamp_ns = 0, reference_timestamp_ns = 0;
    bool initializer_overlap = false;
    HiltiStartupRecoveryMode mode = HiltiStartupRecoveryMode::ImuPropagatedOnly;
    size_t metric_landmark_observations = 0;
    double pose_only_residual_rad = NAN;
    double pose_condition = INFINITY;
    double imu_forward_position_error_m = NAN;
    double imu_forward_rotation_error_rad = NAN;
    double imu_forward_velocity_error_m_s = NAN;
    double initializer_position_disagreement_m = NAN;
    double initializer_rotation_disagreement_rad = NAN;
    double initializer_velocity_disagreement_m_s = NAN;
    double initializer_bias_disagreement = NAN;
    NavState state;
};
using HiltiStartupRecoveryFrames =
    std::vector<HiltiStartupRecoveryFrame, Eigen::aligned_allocator<HiltiStartupRecoveryFrame>>;

struct HiltiStartupRecovery {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool attempted = false, accepted = false;
    size_t first_eligible_frame = 0, anchor_frame = 0, initialization_end_frame = 0;
    size_t eligible_buffered_frames = 0, eligible_published_frames = 0;
    size_t visual_refined_frames = 0, imu_only_frames = 0;
    size_t overlap_frames = 0, timestamp_mismatches = 0, duplicate_frames = 0;
    size_t nonfinite_states = 0;
    double eligible_coverage_percent = 0;
    double overlap_max_position_disagreement_m = NAN;
    double overlap_max_rotation_disagreement_rad = NAN;
    double overlap_max_velocity_disagreement_m_s = NAN;
    double overlap_max_bias_disagreement = NAN;
    std::string reason;
    // Contains actual pre-window backfill frames followed by a diagnostic-only
    // shadow reconstruction of the accepted initializer overlap.
    HiltiStartupRecoveryFrames frames;
};

// Immutable one-shot publication from the accepted initializer. These are the
// actual metric-aligned states; no interpolation or backwards pose fill occurs.
struct HiltiStartupPublication {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t start_frame = 0, end_frame = 0;
    std::vector<int64_t> raw_cam0_timestamps_ns;
    std::vector<int64_t> reference_timestamps_ns;
    VioNavStates states;
    std::shared_ptr<const HiltiStartupRecovery> recovery;
};

// Fixed frontend reference, local to ONE camera's persistent track namespace.
// No runtime landmark variables or Map insertion; only the CURRENT body pose is fitted.
struct HiltiPoseObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int camera_id = 0;
    uint64_t feature_id = 0;
    Vec3 point_w = Vec3::Zero();
    Vec3 bearing = Vec3::Zero();
};
using HiltiPoseObservations = std::vector<HiltiPoseObservation, Eigen::aligned_allocator<HiltiPoseObservation>>;
using HiltiCameras = std::array<std::shared_ptr<const EucmCamera>, 2>;

struct HiltiPoseCorrection {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool accepted = false;
    NavState state;  // Prediction is retained if correction is rejected.
    std::array<size_t, 2> observations{{0, 0}}, inliers{{0, 0}};
    double bearing_rms_rad = std::numeric_limits<double>::quiet_NaN();
    double condition = std::numeric_limits<double>::infinity();
    double correction_m = 0, correction_rad = 0;
    std::string reason;
};

// Pose-only spherical fit at both calibrated exposure times. Velocity/biases and
// reference points are fixed. Gravity is the initialized world-frame gravity.
HiltiPoseCorrection RefineHiltiBodyPose(const NavState& prediction, const HiltiImageEpoch& epoch,
    const HiltiPoseObservations& observations, const HiltiCameras& cameras,
    const ImuCalibration& calibration, const Vec3& gravity_w, const HiltiFrontendOptions& options);

struct HiltiFrontendResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Frame::Ptr frame;
    HiltiFrontendStatus status = HiltiFrontendStatus::Buffering;
    bool has_prediction = false;
    NavState prediction;  // Read ONLY when has_prediction=true.
    HiltiPoseCorrection visual;
    // Populated only when HILTI_CAMERA_CONSISTENCY_AUDIT is set. These
    // independent fits are never published or fed back into the estimator.
    std::array<HiltiPoseCorrection, 2> camera_only_visual;
    std::array<HiltiTrackingResult, 2> tracking;
    bool initialized_now = false;
    size_t epochs = 0;
    std::array<size_t, 2> reference_points{{0, 0}};
    std::string reason;
    uint64_t backend_version = 0;
    bool backend_job_submitted = false;
    size_t backend_window_size = 0, archived_landmarks = 0, archived_observations = 0;
    std::shared_ptr<const HiltiBackendResult> backend_update; // Whole result, never partial state fields.
    std::shared_ptr<const GlobalNavigation> global_navigation; // Optional output only; local frame/state remain authoritative.
    std::shared_ptr<const HiltiStartupPublication> startup_publication; // Present exactly once, on acceptance.
};

class HiltiFrontendRuntime;  // Frontend's isolated runtime implementation.
}  // namespace myslam
#endif
