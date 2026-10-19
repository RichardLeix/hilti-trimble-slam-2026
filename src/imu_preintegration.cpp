#include "myslam/imu_preintegration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace myslam {
namespace {

double Scalar(const cv::FileNode& node, const std::string& name) {
    const auto value = node[name];
    if (value.empty() || (!value.isInt() && !value.isReal())) {
        throw std::runtime_error("missing/non-numeric calibration field: " + name);
    }
    const double result = double(value);
    if (!std::isfinite(result)) throw std::runtime_error("nonfinite calibration field: " + name);
    return result;
}

void CheckMatrix(const cv::FileNode& node, const std::string& name, int size, bool identity) {
    const auto matrix = node[name];
    if (!matrix.isSeq() || int(matrix.size()) != size) {
        throw std::runtime_error("invalid IMU calibration matrix: " + name);
    }
    for (int r = 0; r < size; ++r) {
        if (!matrix[r].isSeq() || int(matrix[r].size()) != size) {
            throw std::runtime_error("invalid IMU calibration matrix row: " + name);
        }
        for (int c = 0; c < size; ++c) {
            const auto entry = matrix[r][c];
            if (!entry.isReal() && !entry.isInt()) throw std::runtime_error("non-numeric IMU matrix");
            const double expected = identity && r == c ? 1 : 0;
            if (!std::isfinite(double(entry)) || std::abs(double(entry) - expected) > 1e-12) {
                throw std::runtime_error("nonidentity IMU intrinsics/extrinsics are unsupported: " + name);
            }
        }
    }
}

void ValidateCalibration(const ImuCalibration& calibration) {
    for (double sigma : {calibration.gyroscope_noise_density, calibration.accelerometer_noise_density,
                         calibration.gyroscope_random_walk, calibration.accelerometer_random_walk}) {
        if (!std::isfinite(sigma) || sigma < 0) throw std::invalid_argument("invalid IMU noise strength");
    }
    if (!std::isfinite(calibration.update_rate_hz) || calibration.update_rate_hz <= 0 ||
        calibration.time_offset_s != 0) {
        throw std::invalid_argument("invalid IMU rate or unsupported reference-IMU time_offset");
    }
    for (double shift : calibration.camera_time_offsets_s) {
        if (!std::isfinite(shift)) throw std::invalid_argument("nonfinite camera-IMU time shift");
    }
}

double Seconds(int64_t begin, int64_t end) {
    // Subtract integers before conversion to double; long double avoids int64
    // overflow for extreme inputs and preserves nanoseconds at Hilti epoch values.
    return double((static_cast<long double>(end) - static_cast<long double>(begin)) * 1e-9L);
}

Mat33 Skew(const Vec3& v) {
    Mat33 result;
    result << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return result;
}

Mat33 RightJacobian(const Vec3& phi) {
    const double angle = phi.norm();
    const Mat33 skew = Skew(phi);
    if (angle < 1e-6) return Mat33::Identity() - 0.5 * skew + (1.0 / 6) * skew * skew;
    return Mat33::Identity() - (1 - std::cos(angle)) / (angle * angle) * skew +
           (angle - std::sin(angle)) / (angle * angle * angle) * skew * skew;
}

struct Measurement {
    int64_t timestamp_ns;
    Vec3 gyro, accel;
};

Measurement RawMeasurement(const ImuSample& sample) {
    return {sample.timestamp_ns,
            Vec3(sample.angular_velocity[0], sample.angular_velocity[1], sample.angular_velocity[2]),
            Vec3(sample.linear_acceleration[0], sample.linear_acceleration[1], sample.linear_acceleration[2])};
}

Measurement At(const std::vector<ImuSample>& samples, int64_t timestamp, bool& interpolated) {
    const auto after = std::lower_bound(samples.begin(), samples.end(), timestamp,
        [](const ImuSample& sample, int64_t time) { return sample.timestamp_ns < time; });
    if (after != samples.end() && after->timestamp_ns == timestamp) {
        interpolated = false;
        return RawMeasurement(*after);
    }
    if (after == samples.begin() || after == samples.end()) {
        throw std::out_of_range("IMU samples must bracket both interval boundaries; no extrapolation");
    }
    const auto left = RawMeasurement(*(after - 1)), right = RawMeasurement(*after);
    const double fraction = Seconds(left.timestamp_ns, timestamp) /
                            Seconds(left.timestamp_ns, right.timestamp_ns);
    interpolated = true;
    return {timestamp, (1 - fraction) * left.gyro + fraction * right.gyro,
            (1 - fraction) * left.accel + fraction * right.accel};
}
}  // namespace

ImuCalibration LoadImuCalibration(const std::string& imu_yaml,
                                  const std::string& camera_yaml) {
    cv::FileStorage imu(imu_yaml, cv::FileStorage::READ);
    cv::FileStorage camera(camera_yaml, cv::FileStorage::READ);
    if (!imu.isOpened() || !camera.isOpened()) throw std::runtime_error("cannot open IMU/camera calibration");
    const auto node = imu["imu0"];
    if (!node.isMap() || !node["model"].isString() || std::string(node["model"]) != "kalibr" ||
        !node["rostopic"].isString() || std::string(node["rostopic"]) != "/imu/data_raw") {
        throw std::runtime_error("expected Hilti imu0, model: kalibr, rostopic: /imu/data_raw");
    }
    ImuCalibration result;
    result.gyroscope_noise_density = Scalar(node, "gyroscope_noise_density");
    result.accelerometer_noise_density = Scalar(node, "accelerometer_noise_density");
    result.gyroscope_random_walk = Scalar(node, "gyroscope_random_walk");
    result.accelerometer_random_walk = Scalar(node, "accelerometer_random_walk");
    result.update_rate_hz = Scalar(node, "update_rate");
    result.time_offset_s = Scalar(node, "time_offset");
    result.topic = std::string(node["rostopic"]);
    CheckMatrix(node, "T_i_b", 4, true);
    for (const char* name : {"Tw", "Ta", "R_IMUtoGYRO", "R_IMUtoACC"}) CheckMatrix(node, name, 3, true);
    CheckMatrix(node, "Tg", 3, false);
    for (size_t i = 0; i < result.camera_time_offsets_s.size(); ++i) {
        const auto cam = camera["cam" + std::to_string(i)];
        if (!cam.isMap()) throw std::runtime_error("missing camera time calibration");
        result.camera_time_offsets_s[i] = Scalar(cam, "timeshift_cam_imu");
    }
    ValidateCalibration(result);
    return result;
}

int64_t CameraTimeOnImuClock(const ImuCalibration& calibration, int camera_id,
                             int64_t camera_timestamp_ns) {
    ValidateCalibration(calibration);
    const double shift = calibration.camera_time_offsets_s.at(static_cast<size_t>(camera_id));
    const long double offset = std::round(static_cast<long double>(shift) * 1e9L);
    const long double timestamp = static_cast<long double>(camera_timestamp_ns) + offset;
    if (timestamp < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        timestamp > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("camera-IMU timestamp shift overflow");
    }
    return static_cast<int64_t>(timestamp);
}

ImuPreintegration PreintegrateImu(const std::vector<ImuSample>& samples,
                                 int64_t begin_timestamp_ns, int64_t end_timestamp_ns,
                                 const ImuCalibration& calibration, const ImuBias& bias) {
    ValidateCalibration(calibration);
    if (!bias.gyroscope.allFinite() || !bias.accelerometer.allFinite()) {
        throw std::invalid_argument("IMU bias must be finite");
    }
    if (begin_timestamp_ns >= end_timestamp_ns || samples.size() < 2) {
        throw std::invalid_argument("positive IMU interval and at least two samples are required");
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i && samples[i].timestamp_ns <= samples[i - 1].timestamp_ns) {
            throw std::invalid_argument("IMU sample timestamps must be strictly increasing");
        }
        if (samples[i].frame_id != samples.front().frame_id) {
            throw std::invalid_argument("cannot mix IMU coordinate frames");
        }
        for (double value : samples[i].angular_velocity) {
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite gyroscope measurement");
        }
        for (double value : samples[i].linear_acceleration) {
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite accelerometer measurement");
        }
    }
    ImuPreintegration result;
    result.begin_timestamp_ns = begin_timestamp_ns;
    result.end_timestamp_ns = end_timestamp_ns;
    result.dt = Seconds(begin_timestamp_ns, end_timestamp_ns);
    result.bias = bias;
    std::vector<Measurement> nodes;
    nodes.push_back(At(samples, begin_timestamp_ns, result.interpolated_begin));
    auto internal = std::upper_bound(samples.begin(), samples.end(), begin_timestamp_ns,
        [](int64_t timestamp, const ImuSample& sample) { return timestamp < sample.timestamp_ns; });
    while (internal != samples.end() && internal->timestamp_ns < end_timestamp_ns) {
        nodes.push_back(RawMeasurement(*internal++));
    }
    nodes.push_back(At(samples, end_timestamp_ns, result.interpolated_end));
    result.integration_steps = nodes.size() - 1;
    for (size_t i = 1; i < nodes.size(); ++i) {
        const double h = Seconds(nodes[i - 1].timestamp_ns, nodes[i].timestamp_ns);
        const Vec3 gyro = 0.5 * nodes[i - 1].gyro + 0.5 * nodes[i].gyro - bias.gyroscope;
        const Vec3 force = 0.5 * nodes[i - 1].accel + 0.5 * nodes[i].accel - bias.accelerometer;
        const Vec3 phi = gyro * h;
        if (!phi.allFinite() || !std::isfinite(phi.norm()) || !force.allFinite()) {
            throw std::runtime_error("IMU corrected measurement overflow");
        }
        const SO3 increment = SO3::exp(phi), half_increment = SO3::exp(0.5 * phi);
        const Mat33 midpoint_rotation = (result.delta_rotation * half_increment).matrix();
        const Vec3 acceleration = midpoint_rotation * force;

        // Linearize the SAME midpoint update with right-multiplicative rotation
        // error. dtheta_mid = H^T*dtheta - Jr(phi/2)*h/2*(db_g + n_g).
        const Mat33 dtheta = -midpoint_rotation * Skew(force) * half_increment.matrix().transpose();
        const Mat33 dgyro = midpoint_rotation * Skew(force) * RightJacobian(0.5 * phi) * (0.5 * h);
        const Mat33 daccel = -midpoint_rotation;
        ImuCovariance f = ImuCovariance::Identity();
        f.block<3, 3>(0, 0) = increment.matrix().transpose();
        f.block<3, 3>(0, 9) = -RightJacobian(phi) * h;
        f.block<3, 3>(3, 0) = dtheta * h;
        f.block<3, 3>(3, 9) = dgyro * h;
        f.block<3, 3>(3, 12) = daccel * h;
        f.block<3, 3>(6, 0) = 0.5 * dtheta * h * h;
        f.block<3, 3>(6, 3) = Mat33::Identity() * h;
        f.block<3, 3>(6, 9) = 0.5 * dgyro * h * h;
        f.block<3, 3>(6, 12) = 0.5 * daccel * h * h;
        result.bias_jacobian = (f.block<9, 9>(0, 0) * result.bias_jacobian + f.block<9, 6>(0, 9)).eval();

        Eigen::Matrix<double, 15, 12> g = Eigen::Matrix<double, 15, 12>::Zero();
        g.block<3, 3>(0, 0) = f.block<3, 3>(0, 9);
        g.block<3, 3>(3, 0) = dgyro * h;
        g.block<3, 3>(6, 0) = 0.5 * dgyro * h * h;
        g.block<3, 3>(3, 3) = daccel * h;
        g.block<3, 3>(6, 3) = 0.5 * daccel * h * h;
        g.block<3, 3>(9, 6) = Mat33::Identity();
        g.block<3, 3>(12, 9) = Mat33::Identity();
        Eigen::Matrix<double, 12, 1> variances;
        variances.segment<3>(0).setConstant(std::pow(calibration.gyroscope_noise_density, 2) / h);
        variances.segment<3>(3).setConstant(std::pow(calibration.accelerometer_noise_density, 2) / h);
        variances.segment<3>(6).setConstant(std::pow(calibration.gyroscope_random_walk, 2) * h);
        variances.segment<3>(9).setConstant(std::pow(calibration.accelerometer_random_walk, 2) * h);
        const ImuCovariance fp = f * result.covariance;
        ImuCovariance covariance = fp * f.transpose();
        const Eigen::Matrix<double, 15, 12> gq = g * variances.asDiagonal();
        covariance.noalias() += gq * g.transpose();
        result.covariance = (0.5 * (covariance + covariance.transpose())).eval();

        result.delta_position += result.delta_velocity * h + 0.5 * acceleration * h * h;
        result.delta_velocity += acceleration * h;
        result.delta_rotation = result.delta_rotation * increment;
    }
    if (!result.delta_velocity.allFinite() || !result.delta_position.allFinite() ||
        !result.bias_jacobian.allFinite() || !result.covariance.allFinite()) {
        throw std::runtime_error("nonfinite IMU preintegration result");
    }
    return result;
}

}  // namespace myslam
