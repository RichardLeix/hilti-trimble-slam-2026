#ifndef VISUAL_BOOTSTRAP_UNCERTAINTY_EXPERIMENT_H
#define VISUAL_BOOTSTRAP_UNCERTAINTY_EXPERIMENT_H
#include "startup_visual_bootstrap.h"

namespace myslam {
namespace visual_uncertainty {
struct Estimate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool usable = false;
    std::string reason;
    int pose_columns = 0, conditional_columns = 0, rank = 0;
    int landmark_rank = 0, deficient_landmarks = 0, observations = 0, dof = 0;
    int gauge_dimensions = 7; // Six first-pose coordinates and one visual unit.
    double bearing_variance = NAN, weighted_sse = NAN, condition = INFINITY;
    double scale_null_leakage = NAN, anchor_constraint_error = NAN;
    // Undamped landmark-marginalized information, before fixing the scale unit.
    MatXX schur_information;
    // Conditional covariance of LEFT T_camera_world twists [rho,phi], poses 1..N-1.
    // Gauge zeros are definitions, not a claim of physical world/metre certainty.
    MatXX pose_covariance, null_directions;
    std::vector<uint64_t> deficient_landmark_ids;
    std::vector<Vec3,Eigen::aligned_allocator<Vec3>> landmark_null_directions;
    VecX normalized_eigenvalues, anchor_constraint;
    std::vector<Mat33,Eigen::aligned_allocator<Mat33>> center_covariances;
};
Estimate Compute(const BootstrapResult& bootstrap, double bearing_limit = .005);
// Full cross-pose covariance is retained in the relative camera center.
Mat33 RelativePositionCovariance(const Estimate& estimate,
    const std::vector<SE3,Eigen::aligned_allocator<SE3>>& poses, size_t i, size_t j);
MatXX CenterJacobian(const std::vector<SE3,Eigen::aligned_allocator<SE3>>& poses, size_t i);
} // namespace visual_uncertainty
} // namespace myslam
#endif
