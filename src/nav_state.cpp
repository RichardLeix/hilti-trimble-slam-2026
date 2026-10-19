#include "myslam/nav_state.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace myslam {
namespace {

bool ValidRotation(const SO3& rotation) {
    const auto& quaternion = rotation.unit_quaternion();
    return quaternion.coeffs().allFinite() && std::abs(quaternion.squaredNorm() - 1) <= 1e-12;
}

bool ValidBias(const ImuBias& bias) {
    return bias.gyroscope.allFinite() && bias.accelerometer.allFinite();
}

}  // namespace

NavState PropagateNavState(const NavState& initial, const ImuPreintegration& delta,
                           const Vec3& gravity_w) {
    if (!ValidRotation(initial.rotation_wb) || !initial.position_wb.allFinite() ||
        !initial.velocity_w.allFinite() || !ValidBias(initial.bias) || !gravity_w.allFinite() ||
        !ValidRotation(delta.delta_rotation) || !delta.delta_velocity.allFinite() ||
        !delta.delta_position.allFinite() || !ValidBias(delta.bias)) {
        throw std::invalid_argument("navigation propagation requires finite states, deltas and gravity");
    }
    if (!std::isfinite(delta.dt) || delta.dt <= 0 ||
        delta.end_timestamp_ns <= delta.begin_timestamp_ns) {
        throw std::invalid_argument("navigation propagation requires a positive IMU interval");
    }
    const double interval_dt = double((static_cast<long double>(delta.end_timestamp_ns) -
                                      static_cast<long double>(delta.begin_timestamp_ns)) * 1e-9L);
    if (std::abs(delta.dt - interval_dt) > 1e-12 * std::max(1.0, interval_dt)) {
        throw std::invalid_argument("preintegration dt disagrees with its timestamp interval");
    }
    if ((initial.bias.gyroscope.array() != delta.bias.gyroscope.array()).any() ||
        (initial.bias.accelerometer.array() != delta.bias.accelerometer.array()).any()) {
        throw std::invalid_argument("preintegration bias must match the navigation state's bias");
    }

    NavState result;
    result.rotation_wb = initial.rotation_wb * delta.delta_rotation;
    result.velocity_w = initial.velocity_w + gravity_w * delta.dt +
                        initial.rotation_wb * delta.delta_velocity;
    result.position_wb = initial.position_wb + initial.velocity_w * delta.dt +
                         0.5 * gravity_w * delta.dt * delta.dt +
                         initial.rotation_wb * delta.delta_position;
    result.bias = initial.bias;
    if (!ValidRotation(result.rotation_wb) || !result.velocity_w.allFinite() ||
        !result.position_wb.allFinite()) {
        throw std::overflow_error("nonfinite navigation propagation result");
    }
    return result;
}

}  // namespace myslam
