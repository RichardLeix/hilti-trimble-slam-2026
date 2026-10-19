#ifndef STAGED_METRIC_INITIALIZER_EXPERIMENT_H
#define STAGED_METRIC_INITIALIZER_EXPERIMENT_H

#include "metric_alignment.h"
#include <array>

namespace myslam {
namespace staged_metric {
enum class Weighting { Heuristic, ImuCovariance };
enum class FinalStage { FixedAccelBias, ReleasedAccelBias };
using Mat66 = Eigen::Matrix<double, 6, 6>;

struct Stage {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool success = false;
    std::string reason = "not reached";
    int number = 0, iterations = 0, velocity_columns = 0, gravity_dimensions = 0;
    int scale_column = -1, bias_column = -1;
    double scale = NAN, weighted_data_rms = NAN, heuristic_rms = NAN;
    double position_rms = NAN, velocity_rms = NAN, prior_cost = 0;
    double remaining_gravity_step = NAN;
    Vec3 gravity = Vec3::Constant(NAN), accel_bias = Vec3::Constant(NAN);
    VecX velocities;
    MatXX gravity_basis;
    // Data-only and prior-augmented linearizations, both at the final gravity.
    MatXX data_a, posterior_a;
    metric_audit::Fit data_fit, posterior_fit;
    VioNavStates states; // Diagnostic states published only after unchanged gates.
};
struct Result {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool success = false;
    std::string reason;
    double bias_prior_sigma = NAN, rotation_rms = NAN, excitation = NAN;
    Vec3 gyro_bias = Vec3::Zero();
    std::array<Stage, 3> stages;
};

// Covariance of [position prediction-target, velocity target-prediction].
// Retains p/v cross covariance, BEGIN-body axes -> world, opposite residual signs.
Mat66 ResidualCovariance(const ImuPreintegration& delta, const SO3& rotation_wb);
Mat66 CovarianceWhitening(const Mat66& covariance); // W C W^T = I; no noise floor.
double BiasPriorSigma(const ImuCalibration& calibration, double duration,
                      const VioInitializerOptions& options = VioInitializerOptions());
metric_audit::System WeightedSystem(const VioVisualFrames& frames,
                                   const VioImuIntervals& corrected,
                                   const SE3& T_camera_body, bool release_bias,
                                   const ImuCalibration& calibration, Weighting weighting);
// Gyro bias is the unchanged native gyro solve, shared across both weightings.
Result Run(const VioVisualWindow& visual, const std::vector<ImuSample>& samples,
           const ImuCalibration& calibration, const SE3& T_camera_body,
           const Vec3& native_gyro_bias, Weighting weighting,
           const VioInitializerOptions& options = VioInitializerOptions(),
           FinalStage final_stage = FinalStage::ReleasedAccelBias,
           const metric_audit::System* fixed_bias_system_override = nullptr,
           bool fast_offline_covariance_solver = false);
// Physical-coordinate correlations, with tangent covariance mapped to world g.
double Correlation(const Stage& stage, bool posterior, bool gravity_bias);
} // namespace staged_metric
} // namespace myslam
#endif
