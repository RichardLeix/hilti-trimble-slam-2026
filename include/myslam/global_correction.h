#ifndef MYSLAM_GLOBAL_CORRECTION_H
#define MYSLAM_GLOBAL_CORRECTION_H

#include "myslam/pose_graph.h"

namespace myslam {
struct GlobalCorrectionOptions {
    double live_smoothing_seconds = 0; // Output only. Zero publishes the exact T_gw.
};
struct GlobalCorrectionBatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t version = 0, graph_generation = 0;
    int64_t publication_timestamp_ns = 0;
    bool has_solution = false;
    uint64_t anchor_keyframe = 0;
    int64_t anchor_timestamp_ns = 0;
    SE3 local_to_global; // T_gw = T_gb(anchor) * T_wb(anchor)^-1.
    HiltiAlignedVector<PoseGraphPose> local_keyframes; // Frozen T_wb input.
    HiltiAlignedVector<PoseGraphPose> global_keyframes; // EXACT optimized T_gb history.
    SE3 live_start_correction;
    double live_smoothing_seconds = 0;
    SE3 LiveCorrection(int64_t timestamp_ns) const;
    const PoseGraphPose* FindGlobalKeyframe(uint64_t keyframe_id) const;
};
// Deliberately distinct from NavState: reported values cannot be silently fed
// back through a NavState-based local solver API.
struct GlobalNavigation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t timestamp_ns = 0;
    uint64_t correction_version = 0;
    bool smoothed = false;
    SE3 body_pose_global; // T_gb = T_gw * T_wb, body -> global.
    Vec3 velocity_global = Vec3::Zero(), gravity_global = Vec3::Zero();
    ImuBias bias_body; // Sensor axes: copied exactly, NEVER rotated/translated.
    std::shared_ptr<const GlobalCorrectionBatch> correction; // Same immutable batch as all fields.
};
GlobalNavigation ProjectGlobalNavigation(const std::shared_ptr<const GlobalCorrectionBatch>& batch,
    const NavState& local, const Vec3& gravity_w, int64_t timestamp_ns, bool smooth_live = false);

// No estimator, Frame, Map, IMU interval or marginalization-prior handles.
// Serializes publishers; readers atomically load ONE owned immutable batch.
class GlobalCorrection {
   public:
    explicit GlobalCorrection(const GlobalCorrectionOptions& options = GlobalCorrectionOptions());
    std::shared_ptr<const GlobalCorrectionBatch> Snapshot() const;
    // Returns false for failed/stale solutions. Invalid/incomplete history throws
    // without changing the current batch. Generation/time must be monotonic.
    bool Publish(const PoseGraphInput& local, const PoseGraphResult& global,
        uint64_t graph_generation, int64_t publication_timestamp_ns);
    GlobalNavigation Project(const NavState& local, const Vec3& gravity_w,
        int64_t timestamp_ns, bool smooth_live = false) const;
   private:
    GlobalCorrectionOptions options_;
    mutable std::mutex publication_mutex_;
    std::shared_ptr<const GlobalCorrectionBatch> current_;
};
} // namespace myslam
#endif
