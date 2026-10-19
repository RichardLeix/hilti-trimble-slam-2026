#include "myslam/vio_initializer.h"
#include "myslam/startup_visual_trajectory.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace myslam {
namespace {
double Seconds(int64_t a, int64_t b) {
    return double((static_cast<long double>(b) - static_cast<long double>(a)) * 1e-9L);
}
bool ValidRotation(const SO3& r) {
    const auto& q = r.unit_quaternion();
    return q.coeffs().allFinite() && std::abs(q.squaredNorm() - 1) < 1e-10;
}
void CheckOptions(const VioInitializerOptions& o) {
    if (o.min_frames < 4) throw std::invalid_argument("initializer needs at least four frames");
    for (double value : {o.min_duration_s, o.gravity_magnitude, o.max_gravity_magnitude_error,
         o.min_acceleration_excitation, o.max_condition, o.max_visual_condition,
         o.max_rotation_rms_rad, o.max_rotation_error_rad, o.max_gyro_bias, o.max_accel_bias,
         o.max_position_rms_m, o.max_velocity_rms_m_s, o.max_bearing_rms_rad,
         o.stationary_bearing_step_rad, o.stationary_rotation_rad,
         o.stationary_accel_std, o.stationary_gyro_std}) {
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("invalid initializer option");
    }
    if (o.max_condition <= 1 || o.max_visual_condition <= 1) {
        throw std::invalid_argument("condition limits must exceed one");
    }
}
struct LinearSolution {
    VecX x;
    int rank = 0;
    double condition = std::numeric_limits<double>::infinity();
    bool Usable(int columns, double limit) const {
        return rank == columns && condition <= limit && x.allFinite();
    }
};
LinearSolution Solve(const MatXX& a, const VecX& b) {
    if (!a.allFinite() || !b.allFinite()) throw std::invalid_argument("nonfinite initialization system");
    VecX norms(a.cols());
    for (int i = 0; i < a.cols(); ++i) norms[i] = std::max(a.col(i).norm(), 1e-15);
    const MatXX normalized = a * norms.cwiseInverse().asDiagonal();
    Eigen::JacobiSVD<MatXX> svd(normalized, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(1e-8);
    LinearSolution result;
    result.rank = int(svd.rank());
    const auto& values = svd.singularValues();
    if (values.size() == a.cols() && values[values.size() - 1] > 1e-15) {
        result.condition = values[0] / values[values.size() - 1];
    }
    result.x = svd.solve(b).cwiseQuotient(norms);
    return result;
}
VioImuIntervals Reintegrate(const VioVisualFrames& frames, const std::vector<ImuSample>& samples,
                            const ImuCalibration& calibration, const ImuBias& bias) {
    VioImuIntervals result;
    for (size_t i = 1; i < frames.size(); ++i) {
        result.push_back(PreintegrateImu(samples, frames[i - 1].timestamp_ns,
                                        frames[i].timestamp_ns, calibration, bias));
    }
    return result;
}
Eigen::Matrix<double, 3, 2> GravityBasis(const Vec3& gravity) {
    const Vec3 direction = gravity.normalized();
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = direction.unitOrthogonal();
    basis.col(1) = direction.cross(basis.col(0));
    return basis;
}
}  // namespace

const char* VioInitializationStatusName(VioInitializationStatus status) {
    switch (status) {
        case VioInitializationStatus::Success: return "success";
        case VioInitializationStatus::StationaryPartial: return "stationary_partial_scale_unobservable";
        case VioInitializationStatus::InsufficientData: return "insufficient_data";
        case VioInitializationStatus::InvalidInput: return "invalid_input";
        case VioInitializationStatus::RotationInconsistent: return "visual_imu_rotation_inconsistent";
        case VioInitializationStatus::VisualStructureUnavailable: return "visual_structure_unavailable";
        case VioInitializationStatus::InsufficientExcitation: return "insufficient_excitation";
        case VioInitializationStatus::IllConditioned: return "ill_conditioned";
        case VioInitializationStatus::ImplausibleSolution: return "implausible_solution";
        case VioInitializationStatus::AlignmentInconsistent: return "alignment_inconsistent";
        case VioInitializationStatus::UncertaintyInvalid: return "initializer_uncertainty_invalid";
    }
    return "unknown";
}

VioVisualWindow MakeVioVisualWindow(const std::vector<HiltiTrackingResult>& tracks,
                                   const std::vector<BearingMotionResult>& motions,
                                   const Camera& camera, const ImuCalibration& calibration,
                                   const VioInitializerOptions& options) {
    CheckOptions(options);
    return BuildStartupVisualTrajectory(tracks, motions, camera, calibration, options);
}

VioInitializer::VioInitializer(const VioInitializerOptions& options) : options_(options) {
    CheckOptions(options_);
}

VioInitializationResult VioInitializer::Initialize(const VioVisualWindow& window,
                                                    const VioImuIntervals& intervals,
                                                    const std::vector<ImuSample>& samples,
                                                    const ImuCalibration& calibration,
                                                    const SE3& T_camera_body) const {
    return InitializeStages(window, intervals, samples, calibration, T_camera_body, false);
}

VioInitializationResult VioInitializer::EstimateGyroBias(const VioVisualWindow& window,
        const VioImuIntervals& intervals, const std::vector<ImuSample>& samples,
        const ImuCalibration& calibration, const SE3& T_camera_body) const {
    return InitializeStages(window, intervals, samples, calibration, T_camera_body, true);
}

VioInitializationResult VioInitializer::InitializeStages(const VioVisualWindow& window,
        const VioImuIntervals& intervals, const std::vector<ImuSample>& samples,
        const ImuCalibration& calibration, const SE3& T_camera_body, bool gyro_only) const {
    VioInitializationResult result;
    result.frames_used = window.frames.size();
    auto fail = [&](VioInitializationStatus status, const std::string& message) {
        result.status = status;
        result.message = message;
        return result;
    };
    const auto& frames = window.frames;
    const size_t n = frames.size();
    if (n < options_.min_frames) {
        return fail(VioInitializationStatus::InsufficientData, window.translation_failure.empty() ?
                    "too few consecutive frames" : window.translation_failure);
    }
    try {
        if (intervals.size() + 1 != n || !ValidRotation(T_camera_body.so3()) ||
            !T_camera_body.translation().allFinite()) {
            return fail(VioInitializationStatus::InvalidInput, "interval count or camera-IMU extrinsics invalid");
        }
        for (size_t i = 0; i < n; ++i) {
            if (!ValidRotation(frames[i].rotation_wb) ||
                (window.positions_in_common_scale && !frames[i].camera_position.allFinite())) {
                return fail(VioInitializationStatus::InvalidInput, "nonfinite visual pose");
            }
            if (i) {
                const auto& delta = intervals[i - 1];
                if (frames[i].timestamp_ns <= frames[i - 1].timestamp_ns ||
                    delta.begin_timestamp_ns != frames[i - 1].timestamp_ns ||
                    delta.end_timestamp_ns != frames[i].timestamp_ns || !std::isfinite(delta.dt) ||
                    std::abs(delta.dt - Seconds(frames[i - 1].timestamp_ns, frames[i].timestamp_ns)) > 1e-12 ||
                    !ValidRotation(delta.delta_rotation) || !delta.delta_velocity.allFinite() ||
                    !delta.delta_position.allFinite() || !delta.bias_jacobian.allFinite() ||
                    !delta.bias.gyroscope.allFinite() || !delta.bias.accelerometer.allFinite() ||
                    delta.bias.gyroscope.norm() != 0 || delta.bias.accelerometer.norm() != 0) {
                    return fail(VioInitializationStatus::InvalidInput, "inconsistent timestamps/deltas or nonzero starting bias");
                }
            }
        }
        result.duration_s = Seconds(frames.front().timestamp_ns, frames.back().timestamp_ns);
        if (result.duration_s < options_.min_duration_s) {
            return fail(VioInitializationStatus::InsufficientData, "startup window is too short");
        }

        // Right rotation residual and the existing analytical gyro-bias Jacobian.
        ImuBias bias;
        auto corrected = intervals;
        for (int iteration = 0; iteration < 6; ++iteration) {
            MatXX a(3 * (n - 1), 3);
            VecX b(3 * (n - 1));
            for (size_t i = 0; i + 1 < n; ++i) {
                const SO3 visual_delta = frames[i].rotation_wb.inverse() * frames[i + 1].rotation_wb;
                const double weight = 1 / std::sqrt(corrected[i].dt);
                a.block<3, 3>(3 * i, 0) = weight * corrected[i].bias_jacobian.block<3, 3>(0, 0);
                b.segment<3>(3 * i) = weight * (corrected[i].delta_rotation.inverse() * visual_delta).log();
            }
            const auto fit = Solve(a, b);
            result.gyro_condition = fit.condition;
            if (!fit.Usable(3, options_.max_condition)) {
                return fail(VioInitializationStatus::IllConditioned, "gyro bias Jacobian lacks rank/conditioning");
            }
            bias.gyroscope += fit.x;
            if (bias.gyroscope.norm() > options_.max_gyro_bias) {
                return fail(VioInitializationStatus::ImplausibleSolution, "gyro bias exceeds configured limit");
            }
            corrected = Reintegrate(frames, samples, calibration, bias);
            if (fit.x.norm() < 1e-9) break;
        }
        double rotation_squared = 0;
        result.max_rotation_error_rad = 0;
        for (size_t i = 0; i + 1 < n; ++i) {
            const SO3 visual_delta = frames[i].rotation_wb.inverse() * frames[i + 1].rotation_wb;
            const double error = (corrected[i].delta_rotation.inverse() * visual_delta).log().norm();
            rotation_squared += error * error;
            result.max_rotation_error_rad = std::max(result.max_rotation_error_rad, error);
        }
        result.rotation_rms_rad = std::sqrt(rotation_squared / (n - 1));
        if (result.rotation_rms_rad > options_.max_rotation_rms_rad ||
            result.max_rotation_error_rad > options_.max_rotation_error_rad) {
            return fail(VioInitializationStatus::RotationInconsistent, "no constant gyro bias reconciles visual/IMU rotations");
        }
        result.bias = bias;
        result.gyro_bias_estimated = true;
        if (gyro_only) {
            result.message = "gyro bias estimated; metric alignment not requested";
            return result;
        }

        Vec3 mean_accel = Vec3::Zero(), mean_gyro = Vec3::Zero();
        Vec3 accel_squared = Vec3::Zero(), gyro_squared = Vec3::Zero();
        size_t sample_count = 0;
        for (const auto& sample : samples) {
            if (sample.timestamp_ns < frames.front().timestamp_ns || sample.timestamp_ns > frames.back().timestamp_ns) continue;
            const Vec3 a(sample.linear_acceleration[0], sample.linear_acceleration[1], sample.linear_acceleration[2]);
            const Vec3 w = Vec3(sample.angular_velocity[0], sample.angular_velocity[1], sample.angular_velocity[2]) - bias.gyroscope;
            mean_accel += a; mean_gyro += w;
            accel_squared += a.cwiseProduct(a); gyro_squared += w.cwiseProduct(w);
            ++sample_count;
        }
        if (!sample_count) return fail(VioInitializationStatus::InsufficientData, "no internal IMU samples");
        mean_accel /= sample_count; mean_gyro /= sample_count;
        result.accel_std = std::sqrt((accel_squared / sample_count - mean_accel.cwiseProduct(mean_accel)).cwiseMax(0).sum());
        result.gyro_std = std::sqrt((gyro_squared / sample_count - mean_gyro.cwiseProduct(mean_gyro)).cwiseMax(0).sum());
        Vec3 average_world_force = Vec3::Zero();
        std::vector<Vec3, Eigen::aligned_allocator<Vec3>> forces;
        for (size_t i = 0; i + 1 < n; ++i) {
            forces.push_back(frames[i].rotation_wb * corrected[i].delta_velocity / corrected[i].dt);
            average_world_force += forces.back();
        }
        average_world_force /= forces.size();
        double excitation_squared = 0;
        for (const auto& force : forces) excitation_squared += (force - average_world_force).squaredNorm();
        result.acceleration_excitation = std::sqrt(excitation_squared / forces.size());

        double rotation_span = 0, visual_motion = 0;
        for (const auto& frame : frames) {
            rotation_span = std::max(rotation_span, (frames.front().rotation_wb.inverse() * frame.rotation_wb).log().norm());
            visual_motion = std::max(visual_motion, (frame.camera_position - frames.front().camera_position).norm());
        }
        const bool stationary = window.stationary_visual_evidence && rotation_span <= options_.stationary_rotation_rad &&
            (!window.positions_in_common_scale || visual_motion < 1e-5) &&
            result.accel_std <= options_.stationary_accel_std && result.gyro_std <= options_.stationary_gyro_std &&
            mean_gyro.norm() < 0.01 && std::abs(mean_accel.norm() - options_.gravity_magnitude) <= options_.max_gravity_magnitude_error;
        if (stationary) {
            result.path = "stationary";
            result.gravity_w = -(frames.front().rotation_wb * mean_accel.normalized()) * options_.gravity_magnitude;
            result.initial_velocity_w = Vec3::Zero();  // Explicit rest assumption.
            return fail(VioInitializationStatus::StationaryPartial,
                "rest initialization assumes zero accel bias and velocity; metric scale and accel bias are unobservable");
        }
        result.path = "dynamic";
        if (!window.positions_in_common_scale) {
            return fail(VioInitializationStatus::VisualStructureUnavailable, window.translation_failure.empty() ?
                "independent unit translations do not provide a common visual scale" : window.translation_failure);
        }
        if (result.acceleration_excitation < options_.min_acceleration_excitation) {
            return fail(VioInitializationStatus::InsufficientExcitation, "world specific force has insufficient variation for scale");
        }

        const Vec3 p_body_camera = T_camera_body.inverse().translation();
        const int gravity_column = int(3 * n), scale_column = gravity_column + 3;
        auto make_system = [&](bool fit_accel_bias) {
            const int columns = scale_column + 1 + (fit_accel_bias ? 3 : 0);
            MatXX a = MatXX::Zero(6 * (n - 1), columns);
            VecX b = VecX::Zero(6 * (n - 1));
            for (size_t i = 0; i + 1 < n; ++i) {
                const double h = corrected[i].dt;
                const Mat33 ri = frames[i].rotation_wb.matrix(), rj = frames[i + 1].rotation_wb.matrix();
                const Vec3 dc = frames[i + 1].camera_position - frames[i].camera_position;
                const int row = int(6 * i);
                a.block<3, 3>(row, 3 * i) = h * Mat33::Identity();
                a.block<3, 3>(row, gravity_column) = 0.5 * h * h * Mat33::Identity();
                a.block<3, 1>(row, scale_column) = -dc;
                b.segment<3>(row) = -ri * corrected[i].delta_position - (rj - ri) * p_body_camera;
                a.block<3, 3>(row + 3, 3 * i) = -Mat33::Identity();
                a.block<3, 3>(row + 3, 3 * (i + 1)) = Mat33::Identity();
                a.block<3, 3>(row + 3, gravity_column) = -h * Mat33::Identity();
                b.segment<3>(row + 3) = ri * corrected[i].delta_velocity;
                if (fit_accel_bias) {
                    a.block<3, 3>(row, scale_column + 1) = ri * corrected[i].bias_jacobian.block<3, 3>(6, 3);
                    a.block<3, 3>(row + 3, scale_column + 1) = -ri * corrected[i].bias_jacobian.block<3, 3>(3, 3);
                }
                // Position/dt and velocity equations have m/s units; sqrt(dt)
                // gives comparable continuous-noise weighting across intervals.
                a.middleRows(row, 3) /= h;
                b.segment<3>(row) /= h;
                a.middleRows(row, 6) /= std::sqrt(h);
                b.segment(row, 6) /= std::sqrt(h);
            }
            return std::make_pair(a, b);
        };
        auto full = make_system(true);
        auto fit = Solve(full.first, full.second);
        result.accel_bias_condition = fit.condition;
        const bool estimate_accel_bias = fit.Usable(full.first.cols(), options_.max_condition);
        auto system = estimate_accel_bias ? std::move(full) : make_system(false);
        if (!estimate_accel_bias) fit = Solve(system.first, system.second);
        result.alignment_condition = fit.condition;
        result.alignment_rank = fit.rank;
        result.alignment_columns = int(system.first.cols());
        if (!fit.Usable(system.first.cols(), options_.max_condition)) {
            return fail(VioInitializationStatus::IllConditioned, "gravity/velocity/scale alignment lacks rank/conditioning");
        }
        Vec3 gravity = fit.x.segment<3>(gravity_column);
        if (!gravity.allFinite() || std::abs(gravity.norm() - options_.gravity_magnitude) > options_.max_gravity_magnitude_error) {
            return fail(VioInitializationStatus::ImplausibleSolution, "unconstrained gravity magnitude is inconsistent with calibration");
        }
        gravity = gravity.normalized() * options_.gravity_magnitude;
        // Refine the gravity direction on its two-dimensional tangent plane.
        for (int iteration = 0; iteration < 6; ++iteration) {
            const auto basis = GravityBasis(gravity);
            MatXX a(system.first.rows(), system.first.cols() - 1);
            a.leftCols(gravity_column) = system.first.leftCols(gravity_column);
            a.middleCols(gravity_column, 2) = system.first.middleCols(gravity_column, 3) * basis;
            a.rightCols(system.first.cols() - scale_column) = system.first.rightCols(system.first.cols() - scale_column);
            const VecX b = system.second - system.first.middleCols(gravity_column, 3) * gravity;
            const auto refined = Solve(a, b);
            result.alignment_condition = refined.condition;
            result.alignment_rank = refined.rank;
            result.alignment_columns = int(a.cols());
            if (!refined.Usable(a.cols(), options_.max_condition)) {
                return fail(VioInitializationStatus::IllConditioned, "gravity tangent alignment lacks conditioning");
            }
            const Vec3 change = basis * refined.x.segment<2>(gravity_column);
            gravity = (gravity + change).normalized() * options_.gravity_magnitude;
            if (change.norm() < 1e-8) break;
        }
        // Solve velocities/scale/bias once with the final fixed gravity direction.
        MatXX fixed(system.first.rows(), system.first.cols() - 3);
        fixed.leftCols(gravity_column) = system.first.leftCols(gravity_column);
        fixed.rightCols(system.first.cols() - scale_column) = system.first.rightCols(system.first.cols() - scale_column);
        const VecX fixed_rhs = system.second - system.first.middleCols(gravity_column, 3) * gravity;
        const auto final = Solve(fixed, fixed_rhs);
        if (!final.Usable(fixed.cols(), options_.max_condition)) {
            return fail(VioInitializationStatus::IllConditioned, "final velocity/scale solve lacks conditioning");
        }
        const double scale = final.x[gravity_column];
        if (estimate_accel_bias) bias.accelerometer = final.x.segment<3>(gravity_column + 1);
        if (!std::isfinite(scale) || scale <= 1e-6 || !bias.accelerometer.allFinite() ||
            bias.accelerometer.norm() > options_.max_accel_bias) {
            return fail(VioInitializationStatus::ImplausibleSolution, "nonpositive scale or implausible accelerometer bias");
        }
        corrected = Reintegrate(frames, samples, calibration, bias);
        VioNavStates states;
        for (size_t i = 0; i < n; ++i) {
            NavState state;
            state.rotation_wb = frames[i].rotation_wb;
            state.position_wb = scale * (frames[i].camera_position - frames.front().camera_position) -
                (frames[i].rotation_wb.matrix() - frames.front().rotation_wb.matrix()) * p_body_camera;
            state.velocity_w = final.x.segment<3>(3 * i);
            state.bias = bias;
            states.push_back(state);
        }
        double position_squared = 0, velocity_squared = 0;
        for (size_t i = 0; i + 1 < n; ++i) {
            const auto predicted = PropagateNavState(states[i], corrected[i], gravity);
            position_squared += (predicted.position_wb - states[i + 1].position_wb).squaredNorm();
            velocity_squared += (predicted.velocity_w - states[i + 1].velocity_w).squaredNorm();
        }
        result.position_rms_m = std::sqrt(position_squared / (n - 1));
        result.velocity_rms_m_s = std::sqrt(velocity_squared / (n - 1));
        if (result.position_rms_m > options_.max_position_rms_m || result.velocity_rms_m_s > options_.max_velocity_rms_m_s) {
            return fail(VioInitializationStatus::AlignmentInconsistent, "visual trajectory and corrected IMU translation disagree");
        }
        result.gravity_w = gravity;
        result.scale = scale;
        result.initial_velocity_w = states.front().velocity_w;
        result.bias = bias;
        result.accel_bias_estimated = estimate_accel_bias;
        result.states = std::move(states);
        result.status = VioInitializationStatus::Success;
        result.message = estimate_accel_bias ? "dynamic metric initialization, both biases observable" :
            "dynamic metric initialization, accelerometer bias held at explicit zero prior (unobservable)";
        return result;
    } catch (const std::exception& error) {
        return fail(VioInitializationStatus::InvalidInput, error.what());
    }
}

}  // namespace myslam
