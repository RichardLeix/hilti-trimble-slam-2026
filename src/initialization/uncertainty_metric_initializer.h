#ifndef UNCERTAINTY_METRIC_INITIALIZER_EXPERIMENT_H
#define UNCERTAINTY_METRIC_INITIALIZER_EXPERIMENT_H
#include "visual_bootstrap_uncertainty.h"
#include "staged_metric_initializer.h"
namespace myslam {
namespace uncertainty_metric {
struct Covariance {
    MatXX imu, visual, combined, whitening;
    double raw_condition=NAN, normalized_condition=NAN;
    double position_visual_variance=NAN, position_imu_variance=NAN;
    double velocity_visual_variance=NAN, velocity_imu_variance=NAN;
    double cross_interval_norm=NAN;
};
struct Result {
    staged_metric::Result metric;
    Covariance covariance;
    bool converged=false;
    int reweighting_iterations=0;
    double relative_scale_step=NAN;
};
MatXX VisualResidualJacobian(const BootstrapResult& visual,const VioImuIntervals& deltas,
                            const SE3& T_camera_body,double scale);
Covariance BuildCovariance(const BootstrapResult& visual,const visual_uncertainty::Estimate& uncertainty,
    const VioImuIntervals& deltas,const SE3& T_camera_body,double scale);
metric_audit::System RawFixedBiasSystem(const VioVisualFrames& frames,const VioImuIntervals& deltas,
                                      const SE3& T_camera_body);
Result Run(const BootstrapResult& visual,const visual_uncertainty::Estimate& uncertainty,
    const std::vector<ImuSample>& samples,const ImuCalibration& calibration,
    const SE3& T_camera_body,const Vec3& native_gyro_bias,double seed_scale,
    const VioInitializerOptions& options=VioInitializerOptions());
} // namespace uncertainty_metric
} // namespace myslam
#endif
