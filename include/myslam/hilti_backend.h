#ifndef MYSLAM_HILTI_BACKEND_H
#define MYSLAM_HILTI_BACKEND_H

#include "myslam/hilti_runtime.h"
#include "myslam/marginalization.h"
#include <chrono>
#include <deque>
#include <functional>
#include <limits>

namespace myslam {
template<class T> using HiltiAlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;
using HiltiLandmarkId = std::pair<int, uint64_t>; // Camera-local track ID; no stereo association.
class HiltiLandmarkQuarantine;

struct HiltiBackendOptions {
    // Production Hilti model. False is retained for explicit GS regression fixtures.
    // The center-row physical header reference is still unverified; no adaptive selection.
    bool center_row_rolling_shutter = true;
    bool enable_marginalization = true;
    // H2 enables this source/prior quarantine in its fixed configuration.
    bool experimental_landmark_quarantine_B = false;
    std::string landmark_quarantine_log_path;
    double marginalization_rank_tolerance = 1e-10;
    // The H2 configuration loader overrides the raw native window default to 11.
    size_t window_size = 7, keyframe_stride = 5, max_landmarks = 240;
    size_t min_landmarks = 20, max_iterations = 10;
    double bearing_sigma_rad = 0.002, visual_huber_sigma = 2.5;
    double max_initial_bearing_rad = 0.02;
    double max_landmark_bearing_rad = 0.005;
    double velocity_prior_sigma = 0.5, gyro_prior_sigma = 0.02, accel_prior_sigma = 0.2;
    double max_pose_update_m = 0.5, max_rotation_update_rad = 0.2;
    double max_velocity_update = 1.0, max_gyro_update = 0.03, max_accel_update = 0.5;
};

struct HiltiKeyframeState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t timestamp_ns = 0;
    NavState state; // T_wb, world velocity, biases in IMU axes.
    std::array<int64_t, 2> exposure_timestamps_ns{{0,0}};
    // Deltas between reference and exposure, linearized at state.bias. A zero
    // exposure interval has dt=0. The sign comes from exposure_timestamps_ns.
    std::array<ImuPreintegration, 2> exposure_deltas;
    // Immutable raw gyro interpolated at each calibrated nominal camera exposure.
    // Bias is subtracted by the row-time factor at its current linearization state.
    std::array<Vec3, 2> exposure_angular_velocity{{
        Vec3::Constant(std::numeric_limits<double>::quiet_NaN()),
        Vec3::Constant(std::numeric_limits<double>::quiet_NaN())}};
};
struct HiltiBearingObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t timestamp_ns = 0;
    Vec3 bearing = Vec3::Zero(); // EUCM-unprojected unit ray, camera axes.
};
struct HiltiLandmark {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    HiltiLandmarkId id;
    Vec3 position_w = Vec3::Zero();
    bool initialized = false;
    bool usable_for_tracking = false; // Recent, geometrically validated local point; archive is independent.
    HiltiAlignedVector<HiltiBearingObservation> observations; // Archive; never erased on window eviction.
};

struct HiltiBackendJob {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t version = 0, base_version = 0;
    HiltiAlignedVector<HiltiKeyframeState> keyframes; // Strict chronological order.
    HiltiAlignedVector<HiltiLandmark> landmarks; // Only local observations in job.
    HiltiAlignedVector<ImuPreintegration> intervals; // n-1 consecutive keyframe factors.
    HiltiCameras cameras;
    Vec3 gravity_w = Vec3::Zero(); // Fixed initialized gravity, not estimated here.
    ImuCalibration calibration;
    std::shared_ptr<const MarginalizationPrior> prior;
    // Quarantined previous graph plus pending keyframes/IMU, before eviction.
    // The worker marginalizes this snapshot and then optimizes the active seven.
    std::shared_ptr<const HiltiBackendJob> marginalization_source;
    bool has_initial_anchor = false;
    NavState initial_anchor; // Original first-window velocity/bias prior mean.
};
using HiltiObservationId=std::pair<HiltiLandmarkId,int64_t>;
struct HiltiBackendDiagnostics {
    bool accepted = false;
    size_t window_size = 0, landmarks = 0, visual_observations = 0, iterations = 0;
    std::array<size_t,2> camera_observations{{0,0}};
    double visual_before_rad = 0, visual_after_rad = 0;
    double imu_before = 0, imu_after = 0; // RMS of 9 whitened motion components.
    double bias_before = 0, bias_after = 0; // RMS of 6 whitened bias RW components.
    double cost_before = 0, cost_after = 0, runtime_ms = 0;
    double pose_update_m = 0, rotation_update_rad = 0, velocity_update = 0;
    double gyro_update = 0, accel_update = 0; // Maximum norms over all keyframes.
    std::string reason;
    size_t marginalizations = 0, marginalized_landmarks = 0;
    size_t prior_dimension = 0, prior_rank = 0;
    double prior_condition = 0, marginalization_ms = 0;
};
struct HiltiBackendResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t version = 0, base_version = 0;
    HiltiAlignedVector<HiltiKeyframeState> keyframes;
    HiltiAlignedVector<HiltiLandmark> landmarks;
    HiltiBackendDiagnostics diagnostics;
    std::shared_ptr<const MarginalizationPrior> prior;
    std::shared_ptr<const HiltiBackendJob> factor_snapshot; // Accepted, optimized graph; source pointer cleared.
    std::set<HiltiObservationId> absorbed_observations;
    std::set<HiltiLandmarkId> retired_landmarks;
};

// Residual order [right rotation, delta_v, delta_p, bg_j-bg_i, ba_j-ba_i].
// Motion components and bias RW are whitened JOINTLY using the existing 15x15
// preintegration covariance, retaining its cross terms without counting RW twice.
Eigen::Matrix<double,15,1> HiltiImuResidual(const NavState& i, const NavState& j,
    const ImuPreintegration& delta, const Vec3& gravity_w);
SE3 HiltiCameraWorldPose(const HiltiKeyframeState& frame, int camera,
    const HiltiCameras& cameras, const Vec3& gravity_w);
Vec3 HiltiBearingResidual(const HiltiKeyframeState& frame, const Vec3& point_w,
    int camera, const Vec3& bearing, const HiltiCameras& cameras, const Vec3& gravity_w);
HiltiBackendResult OptimizeHiltiWindow(const HiltiBackendJob& job, const HiltiBackendOptions& options,
    HiltiLandmarkQuarantine* quarantine = nullptr);
std::shared_ptr<const MarginalizationPrior> MarginalizeHiltiGraph(const HiltiBackendJob& source,
    const HiltiBackendJob& retained, const HiltiBackendOptions& options,
    std::set<HiltiObservationId>& absorbed, std::set<HiltiLandmarkId>& retired);

// Separate from legacy Backend/Map. One immutable job in flight; caller can keep
// tracking, then create its next snapshot after consuming this result. Worker
// never writes Frames, frontend states, landmarks, or the KITTI map.
class HiltiBackend {
   public:
    explicit HiltiBackend(const HiltiBackendOptions& options = HiltiBackendOptions());
    ~HiltiBackend();
    HiltiBackend(const HiltiBackend&) = delete;
    HiltiBackend& operator=(const HiltiBackend&) = delete;
    bool Submit(const HiltiBackendJob& job); // Deep copies, rejects busy/stopped/stale submission.
    std::shared_ptr<const HiltiBackendResult> TakeResult();
    bool Busy() const;
    bool WaitUntilIdle(std::chrono::milliseconds timeout);
    void Stop(); // Drains accepted work, predicate wake, idempotent join.
    const HiltiBackendOptions& Options() const { return options_; }
   private:
    void Loop();
    HiltiBackendOptions options_;
    std::unique_ptr<HiltiLandmarkQuarantine> quarantine_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_, idle_;
    bool stopping_ = false, working_ = false;
    uint64_t last_version_ = 0;
    std::shared_ptr<const HiltiBackendJob> pending_;
    std::shared_ptr<const HiltiBackendResult> result_;
    std::thread worker_;
};
} // namespace myslam
#endif
