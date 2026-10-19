#ifndef MYSLAM_IMU_PREINTEGRATION_H
#define MYSLAM_IMU_PREINTEGRATION_H

#include "myslam/common_include.h"
#include "myslam/sensor_packet.h"

namespace myslam {

// Kalibr continuous-time noise strengths, NOT per-sample standard deviations.
struct ImuCalibration {
    double gyroscope_noise_density = 0;      // (rad/s)/sqrt(Hz).
    double accelerometer_noise_density = 0;  // (m/s^2)/sqrt(Hz).
    double gyroscope_random_walk = 0;        // (rad/s^2)/sqrt(Hz).
    double accelerometer_random_walk = 0;    // (m/s^3)/sqrt(Hz).
    double update_rate_hz = 1000;            // Metadata; integration uses actual dt.
    double time_offset_s = 0;                // Reference IMU: must be zero.
    std::array<double, 2> camera_time_offsets_s{{0, 0}};
    std::string topic = "/imu/data_raw";
};

// Loads noise parameters and camera shifts from the official Kalibr YAML files.
// The Hilti reference IMU has identity intrinsics/extrinsics. Unsupported models,
// nonidentity IMU calibration and nonzero reference-IMU time_offset are rejected.
ImuCalibration LoadImuCalibration(const std::string& imu_yaml,
                                  const std::string& camera_yaml);

// Kalibr: t_imu = t_camera + timeshift_cam_imu. Round the shift to nearest ns.
// Returns a derived interval timestamp; does not alter recorded sensor samples.
int64_t CameraTimeOnImuClock(const ImuCalibration& calibration, int camera_id,
                             int64_t camera_timestamp_ns);

struct ImuBias {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Vec3 gyroscope = Vec3::Zero();      // rad/s, in sensor axes.
    Vec3 accelerometer = Vec3::Zero();  // m/s^2, in sensor axes.
};

using ImuCovariance = Eigen::Matrix<double, 15, 15>;
using ImuBiasJacobian = Eigen::Matrix<double, 9, 6>;

struct ImuPreintegration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t begin_timestamp_ns = 0, end_timestamp_ns = 0;  // IMU clock.
    double dt = 0;  // Seconds.
    size_t integration_steps = 0;
    bool interpolated_begin = false, interpolated_end = false;
    ImuBias bias;  // Fixed bias at which deltas/Jacobians are evaluated.

    // Delta_R = R_world_begin^T R_world_end: maps END IMU vectors to BEGIN IMU.
    SO3 delta_rotation;
    // Integral of bias-corrected SPECIFIC FORCE, expressed in BEGIN IMU axes.
    // No gravity compensation and no initial velocity are included.
    // These are preintegrated deltas, not world-frame velocity/position estimates.
    Vec3 delta_velocity = Vec3::Zero();  // m/s.
    Vec3 delta_position = Vec3::Zero();  // m.

    // First-order continuous-noise discretization, zero initial uncertainty.
    // Order: [right rotation error, delta_v, delta_p, gyro bias error, accel bias
    // error], 3 axes each. Rotation error is in END IMU axes; v/p in BEGIN axes.
    // Bias random walks are included, without estimating/updating the mean bias.
    ImuCovariance covariance = ImuCovariance::Zero();

    // Rows [right rotation error, delta_v, delta_p]; columns [gyro bias, accel
    // bias]. For small db: R(b+db) ~= R(b)*Exp(J_R*db), dv/dp += J_v/J_p*db.
    ImuBiasJacobian bias_jacobian = ImuBiasJacobian::Zero();
};

// Read-only, isolated midpoint integration of ordered, finite raw IMU samples.
// Subtract the supplied fixed bias BEFORE integration. SO(3) exponential updates
// rotation; midpoint orientation rotates specific force into the initial frame.
// Both boundaries are linearly interpolated from bracketing samples, without
// extrapolation. Uses all internal sample times; begin must be strictly < end.
// Measurement covariance uses sigma_density^2/dt; bias increments use sigma_rw^2*dt.
// Covariance is the local continuous-noise approximation, not a model of the
// temporal correlation induced by reusing/interpolating stored sample endpoints.
// Throws on invalid calibration/bias/data, ordering, or missing boundary coverage.
ImuPreintegration PreintegrateImu(const std::vector<ImuSample>& samples,
                                 int64_t begin_timestamp_ns, int64_t end_timestamp_ns,
                                 const ImuCalibration& calibration,
                                 const ImuBias& bias = ImuBias());

}  // namespace myslam

#endif  // MYSLAM_IMU_PREINTEGRATION_H
