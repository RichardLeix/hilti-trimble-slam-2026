#ifndef MYSLAM_NAV_STATE_H
#define MYSLAM_NAV_STATE_H

#include "myslam/imu_preintegration.h"

namespace myslam {

// Body b is the IMU sensor frame, with the official identity IMU calibration.
// T_wb maps body points to world: p_w = rotation_wb * p_b + position_wb.
// The world origin/orientation and gravity are supplied by the caller.
struct NavState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    SO3 rotation_wb;                      // Body-to-world orientation, identity by default.
    Vec3 position_wb = Vec3::Zero();       // IMU origin in world, metres.
    Vec3 velocity_w = Vec3::Zero();        // World-frame velocity, m/s.
    ImuBias bias;                         // Gyro (rad/s), accel (m/s^2), in IMU axes.

    SE3 Pose() const { return SE3(rotation_wb, position_wb); }
};

// Isolated mean-state propagation using deltas expressed in BEGIN IMU axes:
// R_j = R_i * DeltaR
// v_j = v_i + gravity_w*dt + R_i*DeltaV
// p_j = p_i + v_i*dt + 0.5*gravity_w*dt^2 + R_i*DeltaP
// gravity_w is a constant world-frame acceleration over this interval. For a
// world with +z up, a typical supplied value is (0, 0, -9.81) m/s^2. Specific
// force in the preintegration is already bias-corrected; gravity is added here.
// Both biases are preserved, without estimation, random sampling, or rotation.
// The preintegration must use exactly initial.bias; mismatches require fresh
// preintegration and are rejected rather than implicitly applying corrections.
// The caller must connect intervals consecutively; NavState has no timestamp.
// Does not propagate a state covariance or fuse any measurements.
// Throws for invalid state/deltas/gravity, interval metadata, or bias mismatch.
NavState PropagateNavState(const NavState& initial, const ImuPreintegration& delta,
                           const Vec3& gravity_w);

}  // namespace myslam

#endif  // MYSLAM_NAV_STATE_H
