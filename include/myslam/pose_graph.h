#ifndef MYSLAM_POSE_GRAPH_H
#define MYSLAM_POSE_GRAPH_H

#include "myslam/loop_closing.h"
#include <iosfwd>

namespace myslam {
struct PoseGraphOptions {
    size_t maximum_iterations = 50;
    // Per-edge variance = floor^2 + diffusion^2 * dt + (relative * motion)^2.
    double odometry_translation_floor_m = 0.005;
    double odometry_translation_diffusion = 0.03; // m / sqrt(s)
    double odometry_translation_relative = 0.02;
    double odometry_rotation_floor_rad = 0.001;
    double odometry_rotation_diffusion = 0.02; // rad / sqrt(s)
    double odometry_rotation_relative = 0.02;
    double loop_translation_sigma_m = 0.10, loop_rotation_sigma_rad = 0.03;
    double loop_min_translation_sigma_m = 0.05, loop_max_translation_sigma_m = 0.50;
    double loop_min_rotation_sigma_rad = 0.01, loop_max_rotation_sigma_rad = 0.15;
    double loop_reference_inliers = 80, loop_max_effective_inliers = 200;
    double loop_reference_bearing_rms_rad = 0.002, loop_reference_condition = 10;
    double loop_correlation_seconds = 2.0;
    double loop_huber_delta = 3.54846; // sqrt(chi-square(6), 95%).
    double innovation_chi2_limit = 22.4577; // chi-square(6), 99.9%.
    double postfit_chi2_limit = 22.4577;
};
PoseGraphOptions LoadPoseGraphOptions(const std::string& path);
void ValidatePoseGraphOptions(const PoseGraphOptions& options);

struct PoseGraphPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t keyframe_id = 0;
    int64_t timestamp_ns = 0;
    SE3 body_pose; // T_wb: maps this IMU body to world.
};
struct PoseGraphLoop {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t query_keyframe = 0, candidate_keyframe = 0;
    SE3 relative_body_transform; // T_query_candidate, candidate body -> query body.
    size_t inliers = 0;
    double bearing_rms_rad = 0, pose_condition = 1;
};
struct PoseGraphInput {
    HiltiAlignedVector<PoseGraphPose> poses;
    HiltiAlignedVector<PoseGraphLoop> loops;
};
// Take an owned, pose-only snapshot AFTER LoopClosing::Stop(). Query diagnostics
// enumerate archive IDs (including queries with no candidate). Never writes DB.
PoseGraphInput SnapshotPoseGraph(const KeyframeDatabase& database,
    const HiltiAlignedVector<LoopDiagnostic>& diagnostics,
    const HiltiAlignedVector<LoopConstraint>& constraints);
void WritePoseGraphInput(std::ostream& stream, const PoseGraphInput& input);
PoseGraphInput ReadPoseGraphInput(std::istream& stream);

// Same convention for EVERY edge: Z_ij = T_wb_i^-1 * T_wb_j maps j -> i.
// Loop edge i=query, j=candidate: use T_query_candidate WITHOUT inversion.
// Residual [translation tangent (m), rotation tangent (rad)] =
// Log(Z_ij^-1 * T_wb_i^-1 * T_wb_j). Updates are left SE(3) exponentials.
Vec6 PoseGraphResidual(const SE3& measurement, const SE3& from, const SE3& to);
SE3 PoseGraphApplyUpdate(const SE3& pose, const Vec6& increment);
void LinearizePoseGraphEdge(const SE3& measurement, const SE3& from, const SE3& to,
    Mat66& from_jacobian, Mat66& to_jacobian);
Mat66 PoseGraphOdometryCovariance(const PoseGraphOptions& options, double dt,
    const SE3& relative_pose);
Mat66 PoseGraphLoopCovariance(const PoseGraphOptions& options,
    const PoseGraphLoop& loop, size_t correlated_loops = 1);

struct PoseGraphLoopDiagnostic {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t query_keyframe = 0, candidate_keyframe = 0;
    bool retained = false;
    size_t correlated_loops = 1;
    double translation_sigma_m = 0, rotation_sigma_rad = 0;
    double innovation_chi2 = 0, initial_chi2 = 0, final_chi2 = 0;
    double initial_translation_residual_m = 0, final_translation_residual_m = 0;
    double initial_rotation_residual_rad = 0, final_rotation_residual_rad = 0;
    double postfit_cycle_chi2 = 0;
    std::string reason;
};
struct PoseGraphResult {
    // Separate output. No handles to runtime navigation, Map, or local states.
    HiltiAlignedVector<PoseGraphPose> poses;
    HiltiAlignedVector<PoseGraphLoopDiagnostic> loops;
    bool success = false;
    std::string reason;
    size_t vertices = 0, sequential_edges = 0, loop_edges = 0;
    size_t retained_loop_edges = 0, rejected_loop_edges = 0;
    size_t optimizer_iterations = 0, optimizer_passes = 0;
    // Robust objective for ALL original loops, including rejected ones.
    double initial_total_error = 0, final_total_error = 0;
    // Comparable objective for the final retained graph, before / after.
    double initial_retained_error = 0, final_retained_error = 0;
    double odometry_error = 0, maximum_odometry_edge_chi2 = 0;
    double maximum_translation_correction_m = 0, maximum_rotation_correction_rad = 0;
    double endpoint_translation_correction_m = 0, endpoint_rotation_correction_rad = 0;
    double runtime_ms = 0;
};
class PoseGraph {
   public:
    explicit PoseGraph(const PoseGraphOptions& options = PoseGraphOptions());
    PoseGraphResult Optimize(const PoseGraphInput& input) const;
   private:
    PoseGraphOptions options_;
};
} // namespace myslam
#endif
