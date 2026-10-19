#ifndef METRIC_INITIALIZER_AUDIT_EXPERIMENT_H
#define METRIC_INITIALIZER_AUDIT_EXPERIMENT_H
#include "myslam/vio_initializer.h"
namespace myslam {
namespace metric_audit {
enum class Variant { ZeroAccelBias, BaselineAccelBias, FixedGravityZeroAccelBias, Full };
struct System {
    MatXX a;
    VecX b;
    int velocity_columns = 0, gravity_column = -1, scale_column = -1, accel_column = -1;
};
struct Fit {
    VecX x, column_norms, singular_values, raw_singular_values;
    MatXX covariance_unit_noise;
    MatXX normalized_covariance;
    VecX profiled_normal_eigenvalues;
    MatXX profiled_normal, profiled_covariance;
    int rank = 0;
    double condition = INFINITY, weighted_rms = NAN, residual_variance = NAN;
};
struct Result {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool success = false, accel_bias_fitted = false;
    std::string reason;
    System initial_system;
    Fit initial_fit;
    double candidate_scale = NAN, candidate_final_scale = NAN, scale = NAN;
    Vec3 candidate_gravity = Vec3::Constant(NAN), gravity = Vec3::Constant(NAN);
    Vec3 candidate_accel_bias = Vec3::Constant(NAN), candidate_final_accel_bias = Vec3::Constant(NAN);
    ImuBias bias;
    double rotation_rms = NAN, excitation = NAN;
    double final_condition = NAN, linear_final_rms = NAN, position_rms = NAN, velocity_rms = NAN;
    VioNavStates states;
};
// Weighted production equations; deltas have already been reintegrated at the
// fixed gyro bias and at zero (D/A/C) or baseline (B) accelerometer bias.
System BuildSystem(const VioVisualFrames& frames, const VioImuIntervals& deltas,
                   const SE3& T_camera_body, bool fit_accel_bias);
Fit SolveAndDiagnose(const MatXX& a, const VecX& b, int nuisance_velocity_columns);
// Independent physical residual, for sign/frame and finite-difference tests.
VecX PhysicalResidual(const VioVisualFrames& frames, const VioImuIntervals& deltas,
                     const SE3& T_camera_body, const VecX& velocities,
                     const Vec3& gravity, double scale, const Vec3& accel_bias);
Result Run(const VioVisualWindow& visual, const std::vector<ImuSample>& samples,
           const ImuCalibration& calibration, const SE3& T_camera_body,
           const Vec3& fitted_gyro_bias, Variant variant,
           const Vec3& baseline_accel_bias, const Vec3& baseline_gravity_in_window,
           const VioInitializerOptions& options = VioInitializerOptions());
} // namespace metric_audit
} // namespace myslam
#endif
