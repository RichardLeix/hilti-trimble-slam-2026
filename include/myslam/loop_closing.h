#ifndef MYSLAM_LOOP_CLOSING_H
#define MYSLAM_LOOP_CLOSING_H

#include "myslam/keyframe_database.h"
#include <deque>

namespace myslam {
struct PoseGraphInput;
struct LoopClosingOptions {
    std::string vocabulary_path;
    uint64_t minimum_keyframe_separation = 90;
    double minimum_time_separation_s = 15;
    size_t maximum_candidates = 5;
    double minimum_score = 0.035;
    int orb_features_per_camera = 1000;
    int max_hamming_distance = 64;
    double descriptor_ratio = 0.8;
    size_t minimum_inliers = 24;
    double minimum_inlier_ratio = 0.3;
    double inlier_threshold_rad = 0.008;
    double maximum_rms_rad = 0.004;
    double maximum_pose_condition = 1e8;
    size_t ransac_iterations = 600;
    double ransac_confidence = 0.999;
    uint32_t random_seed = 20261006;
};
LoopClosingOptions LoadLoopClosingOptions(const std::string& path,
    const std::string& vocabulary_override = "");
void ValidateLoopClosingOptions(const LoopClosingOptions& options);

struct LoopDiagnostic {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t query_keyframe = 0, candidate_keyframe = 0;
    double score = 0, temporal_separation_s = 0;
    uint64_t keyframe_separation = 0;
    size_t raw_matches = 0, metric_matches = 0, geometric_inliers = 0;
    std::array<size_t,2> camera_inliers{{0,0}};
    double relative_pose_rms_rad = std::numeric_limits<double>::quiet_NaN();
    double pose_condition = std::numeric_limits<double>::infinity();
    bool verification_attempted = false, accepted = false;
    SE3 relative_body_transform; // T_query_candidate: p_bq = T_bq_bc * p_bc.
    std::string reason;
};
struct LoopConstraint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t query_keyframe = 0, candidate_keyframe = 0;
    int64_t query_timestamp_ns = 0, candidate_timestamp_ns = 0;
    SE3 relative_body_transform; // Metric SE(3), same direction as diagnostic.
    size_t inliers = 0;
    double bearing_rms_rad = 0;
};
struct LoopStatistics {
    size_t keyframes = 0, queries = 0, candidates = 0, geometrically_verified = 0;
    size_t accepted_constraints = 0, failed_inputs = 0;
};

// Independent of both absolute poses: hypotheses use only archived metric
// points, EUCM unit rays and body/camera calibration. No pose-prior gate.
LoopDiagnostic VerifyLoopGeometry(const LoopKeyframe& query,
    const LoopKeyframe& candidate, const LoopClosingOptions& options);

struct LoopDescriptorMatch {
    int query_camera = 0, candidate_camera = 0;
    size_t query_row = 0, candidate_row = 0;
};
std::vector<LoopDescriptorMatch> MatchLoopDescriptors(const LoopKeyframe& query,
    const LoopKeyframe& candidate, const LoopClosingOptions& options);

struct LoopInput {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    LoopKeyframe frame;
    std::array<SensorPacket,2> images;
    std::array<HiltiTrackingResult,2> tracks;
    HiltiAlignedVector<HiltiLandmark> landmarks; // Frozen accepted local solution.
};

// A Hilti-only worker. It owns no Frame, HiltiLocalMap, backend or trajectory.
// Submit/Stop drain every accepted immutable input; accepted constraints are
// retained here for future consumers, never applied to navigation or landmarks.
class LoopClosing {
   public:
    LoopClosing(const HiltiCameras& cameras, const LoopClosingOptions& options);
    ~LoopClosing();
    LoopClosing(const LoopClosing&) = delete;
    LoopClosing& operator=(const LoopClosing&) = delete;
    bool Submit(const LoopInput& input);
    bool SubmitPrepared(const LoopKeyframe& frame); // Descriptor database/testing path.
    bool WaitUntilIdle(std::chrono::milliseconds timeout);
    void Stop();
    LoopStatistics Statistics() const;
    HiltiAlignedVector<LoopDiagnostic> Diagnostics() const;
    HiltiAlignedVector<LoopConstraint> Constraints() const;
    const KeyframeDatabase& Database() const { return database_; }
    // Atomic completed-query prefix. A current in-flight query is excluded.
    // Poses remain local T_wb, and the worker never changes archive geometry.
    PoseGraphInput GraphSnapshot() const;
   private:
    void Run();
    LoopKeyframe Extract(const LoopInput& input) const;
    void EnrichGeometry(LoopKeyframe& current);
    void Process(LoopKeyframe frame);
    HiltiCameras cameras_;
    std::array<cv::Mat,2> valid_masks_;
    LoopClosingOptions options_;
    KeyframeDatabase database_;
    struct Work {
        std::shared_ptr<const LoopInput> raw;
        std::shared_ptr<const LoopKeyframe> prepared;
    };
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_, idle_;
    std::deque<Work> queue_;
    bool stopping_ = false, working_ = false, have_id_ = false;
    uint64_t last_id_ = 0;
    LoopStatistics statistics_;
    HiltiAlignedVector<LoopDiagnostic> diagnostics_;
    HiltiAlignedVector<LoopConstraint> constraints_;
    std::thread worker_;
};
} // namespace myslam
#endif
