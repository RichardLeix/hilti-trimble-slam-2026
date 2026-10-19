#include "myslam/frontend.h"
#include "myslam/hilti_local_map.h"
#include "myslam/loop_closing.h"
#include "myslam/global_trajectory.h"
#include "myslam/hilti_initializer.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>

namespace myslam {
namespace {
Mat33 Cross(const Vec3& v) {
    Mat33 m; m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0; return m;
}
double Angle(const Vec3& a, const Vec3& b) {
    return std::atan2(a.cross(b).norm(), a.dot(b));
}
void CheckOptions(const HiltiFrontendOptions& o) {
    if (o.startup_frames < o.initialization.min_frames || o.startup_frames > 200 || o.min_visual_inliers < 6)
        throw std::invalid_argument("invalid Hilti startup/visual support limits");
    if (!o.startup_retry_stride || o.startup_max_end_frame + 1 < o.startup_frames)
        throw std::invalid_argument("invalid Hilti startup retry stride/horizon");
    if (!std::isfinite(o.startup_backfill_start_s) || o.startup_backfill_start_s < 0)
        throw std::invalid_argument("invalid Hilti startup backfill boundary");
    for (double x : {o.max_bearing_error_rad, o.max_pose_condition, o.max_correction_m, o.max_correction_rad})
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("invalid Hilti pose correction limit");
    if (o.max_pose_condition <= 1) throw std::invalid_argument("invalid Hilti pose condition limit");
    VioInitializer initializer(o.initialization); // Reuse existing option validation, unchanged.
}
bool RecoverableInitializationFailure(VioInitializationStatus status) {
    switch (status) {
        case VioInitializationStatus::Success:
        case VioInitializationStatus::InvalidInput:
            return false;
        case VioInitializationStatus::StationaryPartial:
        case VioInitializationStatus::InsufficientData:
        case VioInitializationStatus::RotationInconsistent:
        case VioInitializationStatus::VisualStructureUnavailable:
        case VioInitializationStatus::InsufficientExcitation:
        case VioInitializationStatus::IllConditioned:
        case VioInitializationStatus::ImplausibleSolution:
        case VioInitializationStatus::AlignmentInconsistent:
        case VioInitializationStatus::UncertaintyInvalid:
            return true;
    }
    return false;
}
bool Finite(const NavState& state) {
    return state.Pose().matrix().allFinite() && state.velocity_w.allFinite() &&
        state.bias.gyroscope.allFinite() && state.bias.accelerometer.allFinite();
}
NavState BackwardPropagateNavState(const NavState& next, const ImuPreintegration& delta,
                                   const Vec3& gravity_w) {
    if (!Finite(next) || !delta.delta_rotation.matrix().allFinite() ||
        !delta.delta_velocity.allFinite() || !delta.delta_position.allFinite() ||
        !gravity_w.allFinite() || !std::isfinite(delta.dt) || delta.dt <= 0 ||
        delta.end_timestamp_ns <= delta.begin_timestamp_ns ||
        delta.bias.gyroscope != next.bias.gyroscope ||
        delta.bias.accelerometer != next.bias.accelerometer)
        throw std::invalid_argument("backward navigation propagation requires a finite consecutive interval");
    NavState state;
    state.rotation_wb = next.rotation_wb * delta.delta_rotation.inverse();
    state.velocity_w = next.velocity_w - gravity_w * delta.dt -
        state.rotation_wb * delta.delta_velocity;
    state.position_wb = next.position_wb - state.velocity_w * delta.dt -
        0.5 * gravity_w * delta.dt * delta.dt - state.rotation_wb * delta.delta_position;
    state.bias = next.bias;
    if (!Finite(state)) throw std::overflow_error("nonfinite backward navigation state");
    return state;
}
double BiasDifference(const ImuBias& a, const ImuBias& b) {
    return std::max((a.gyroscope-b.gyroscope).norm(), (a.accelerometer-b.accelerometer).norm());
}
void CheckEpoch(const HiltiImageEpoch& e, const ImuCalibration& calibration) {
    if (e.cameras[0].stream != SensorStream::Cam0 || e.cameras[1].stream != SensorStream::Cam1 ||
        e.cameras[0].timestamp_ns != e.cameras[1].timestamp_ns || e.timestamp_ns != e.camera_imu_timestamps_ns[0])
        throw std::invalid_argument("Hilti epoch has incorrect camera IDs, synchronization or reference clock");
    for (int k = 0; k < 2; ++k)
        if (e.camera_imu_timestamps_ns[k] != CameraTimeOnImuClock(calibration, k, e.cameras[k].timestamp_ns))
            throw std::invalid_argument("Hilti exposure does not match calibrated IMU clock");
    if (e.imu_samples.size() < 2) throw std::invalid_argument("Hilti epoch lacks IMU boundary samples");
    for (size_t i = 0; i < e.imu_samples.size(); ++i) {
        const auto& s = e.imu_samples[i];
        if (i && s.timestamp_ns <= e.imu_samples[i - 1].timestamp_ns)
            throw std::invalid_argument("epoch IMU timestamps are not strictly increasing");
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(s.angular_velocity[k]) || !std::isfinite(s.linear_acceleration[k]))
                throw std::invalid_argument("nonfinite epoch IMU measurement");
    }
    const int64_t first = std::min(e.camera_imu_timestamps_ns[0], e.camera_imu_timestamps_ns[1]);
    const int64_t last = std::max(e.camera_imu_timestamps_ns[0], e.camera_imu_timestamps_ns[1]);
    if (e.imu_samples.front().timestamp_ns > first || e.imu_samples.back().timestamp_ns < last)
        throw std::invalid_argument("IMU does not bracket camera exposure times");
}

struct Exposure {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double dt = 0;
    SO3 rotation; Vec3 displacement = Vec3::Zero();
    // At exposure: R=R_ref*rotation;
    // p=p_ref+v_ref*dt+0.5*g*dt^2+R_ref*displacement.
};
Exposure AtExposure(const std::vector<ImuSample>& samples, int64_t reference, int64_t exposure,
                    const ImuCalibration& calibration, const ImuBias& bias) {
    Exposure e;
    e.dt = double((static_cast<long double>(exposure) - reference) * 1e-9L);
    if (exposure > reference) {
        const auto d = PreintegrateImu(samples, reference, exposure, calibration, bias);
        e.rotation = d.delta_rotation; e.displacement = d.delta_position;
    } else if (exposure < reference) {
        const auto d = PreintegrateImu(samples, exposure, reference, calibration, bias);
        e.rotation = d.delta_rotation.inverse();
        e.displacement = e.rotation * (d.delta_velocity * d.dt - d.delta_position);
    }
    return e;
}
SE3 ExposurePose(const NavState& state, const Exposure& e, const Vec3& gravity) {
    return SE3(state.rotation_wb * e.rotation, state.position_wb + state.velocity_w * e.dt +
        0.5 * gravity * e.dt * e.dt + state.rotation_wb * e.displacement);
}
struct PreparedObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int camera;
    Vec3 point, ray;
    SE3 T_camera_reference_body;
};
using Prepared = std::vector<PreparedObservation, Eigen::aligned_allocator<PreparedObservation>>;
double Cost(const SE3& T_bw, const Prepared& observations, double huber) {
    double total = 0;
    for (const auto& o : observations) {
        const Vec3 p = o.T_camera_reference_body * (T_bw * o.point);
        const double r = p.norm() > 1e-12 ? (p.normalized() - o.ray).norm() : 2;
        total += r < huber ? r * r : huber * (2 * r - huber);
    }
    return total;
}
using PoseJacobian = Eigen::Matrix<double, 3, 6>;
PoseJacobian Jacobian(const SE3& T_bw, const PreparedObservation& o, Vec3& p) {
    const Vec3 body = T_bw * o.point;
    p = o.T_camera_reference_body * body;
    const Vec3 u = p.normalized();
    PoseJacobian body_j; body_j.leftCols<3>() = Mat33::Identity(); body_j.rightCols<3>() = -Cross(body);
    return (Mat33::Identity() - u * u.transpose()) / p.norm() *
        o.T_camera_reference_body.rotationMatrix() * body_j;
}

struct ReferenceObservation { EIGEN_MAKE_ALIGNED_OPERATOR_NEW; SE3 pose; Vec3 ray; };
using ReferenceTrack = std::vector<ReferenceObservation, Eigen::aligned_allocator<ReferenceObservation>>;
using ReferenceMap = std::map<uint64_t, Vec3, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Vec3>>>;
bool MakeReference(const ReferenceTrack& observations, Vec3& point, double limit, double condition_limit) {
    if (observations.size() < 3) return false;
    Mat33 h = Mat33::Zero(); Vec3 rhs = Vec3::Zero();
    for (const auto& o : observations) {
        const Vec3 ray = o.pose.so3() * o.ray;
        const Mat33 a = Mat33::Identity() - ray * ray.transpose();
        h += a; rhs += a * o.pose.translation();
    }
    const Eigen::SelfAdjointEigenSolver<Mat33> eigen(h);
    if (eigen.info() != Eigen::Success) return false;
    const Vec3 values = eigen.eigenvalues();
    if (values[0] <= 1e-12 || std::sqrt(values[2] / values[0]) > condition_limit) return false;
    point = h.ldlt().solve(rhs);
    for (int iteration = 0; iteration < 8; ++iteration) {
        h.setZero(); rhs.setZero();
        for (const auto& o : observations) {
            const SE3 T_cw = o.pose.inverse(); const Vec3 p = T_cw * point;
            if (p.norm() < 1e-10) return false;
            const Vec3 u = p.normalized(), r = u - o.ray;
            const Mat33 j = (Mat33::Identity() - u * u.transpose()) / p.norm() * T_cw.rotationMatrix();
            const double w = std::min(1.0, 0.5 * limit / std::max(r.norm(), 1e-12));
            h += w * j.transpose() * j; rhs -= w * j.transpose() * r;
        }
        const Vec3 step = h.ldlt().solve(rhs);
        if (!step.allFinite()) return false;
        point += step;
        if (step.norm() < 1e-9) break;
    }
    size_t good = 0; double error = 0;
    for (const auto& o : observations) {
        const Vec3 p = o.pose.inverse() * point;
        const double r = p.norm() > 1e-10 && p.dot(o.ray) > 0 ? Angle(p.normalized(), o.ray) : M_PI;
        if (r <= limit) { ++good; error += r * r; }
    }
    return good >= 0.8 * observations.size() && good >= 3 && std::sqrt(error / good) <= limit;
}
}  // namespace

const char* HiltiFrontendStatusName(HiltiFrontendStatus status) {
    switch (status) {
        case HiltiFrontendStatus::Buffering: return "buffering";
        case HiltiFrontendStatus::Initialized: return "initialized";
        case HiltiFrontendStatus::Tracking: return "tracking";
        case HiltiFrontendStatus::PredictionOnly: return "prediction_only";
        case HiltiFrontendStatus::InitializationFailed: return "initialization_failed";
    }
    return "unknown";
}

const char* HiltiStartupRecoveryModeName(HiltiStartupRecoveryMode mode) {
    switch (mode) {
        case HiltiStartupRecoveryMode::InitializerAnchor: return "initializer_anchor";
        case HiltiStartupRecoveryMode::VisualRefined: return "visual_refined";
        case HiltiStartupRecoveryMode::ImuPropagatedOnly: return "imu_propagated_only";
    }
    return "unknown";
}

HiltiFrontendOptions LoadHiltiFrontendOptions(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) throw std::invalid_argument("cannot open Hilti startup configuration");
    int window = -1, enabled = -1, stride = -1, maximum = -1;
    file["startup_window_frames"] >> window;
    file["startup_retry_enabled"] >> enabled;
    file["startup_retry_stride_frames"] >> stride;
    file["startup_maximum_end_frame"] >> maximum;
    if (window <= 0 || (enabled != 0 && enabled != 1) || stride <= 0 || maximum < 0)
        throw std::invalid_argument("malformed Hilti startup configuration");
    HiltiFrontendOptions options;
    options.startup_frames = size_t(window);
    options.startup_retry_enabled = enabled != 0;
    options.startup_retry_stride = size_t(stride);
    options.startup_max_end_frame = size_t(maximum);
    const cv::FileNode backfill = file["startup_backfill_enabled"];
    const cv::FileNode boundary = file["startup_backfill_start_seconds"];
    if (!backfill.empty()) {
        int enabled_backfill = -1; backfill >> enabled_backfill;
        if (enabled_backfill != 0 && enabled_backfill != 1)
            throw std::invalid_argument("malformed Hilti startup backfill configuration");
        options.startup_backfill_enabled = enabled_backfill != 0;
    }
    if (!boundary.empty()) boundary >> options.startup_backfill_start_s;
    CheckOptions(options);
    return options;
}

HiltiPoseCorrection RefineHiltiBodyPose(const NavState& prediction, const HiltiImageEpoch& epoch,
        const HiltiPoseObservations& observations, const HiltiCameras& cameras,
        const ImuCalibration& calibration, const Vec3& gravity, const HiltiFrontendOptions& options) {
    CheckOptions(options); CheckEpoch(epoch, calibration);
    if (!prediction.Pose().matrix().allFinite() || !prediction.velocity_w.allFinite() ||
        !prediction.bias.gyroscope.allFinite() || !prediction.bias.accelerometer.allFinite() || !gravity.allFinite())
        throw std::invalid_argument("pose correction requires a finite predicted navigation state and gravity");
    HiltiPoseCorrection result; result.state = prediction;
    Prepared prepared;
    std::array<Exposure, 2> exposures;
    for (int k = 0; k < 2; ++k) {
        if (!cameras[k]) throw std::invalid_argument("pose correction requires both calibrated cameras");
        exposures[k] = AtExposure(epoch.imu_samples, epoch.timestamp_ns, epoch.camera_imu_timestamps_ns[k], calibration, prediction.bias);
    }
    std::set<std::pair<int, uint64_t>> unique;
    for (const auto& o : observations) {
        if (o.camera_id < 0 || o.camera_id > 1 || !o.point_w.allFinite() || !o.bearing.allFinite() ||
            std::abs(o.bearing.norm() - 1) > 1e-8 || !unique.insert({o.camera_id, o.feature_id}).second)
            throw std::invalid_argument("invalid/duplicate pose observation or camera ID");
        Vec2 pixel;
        if (!cameras[o.camera_id]->Project(o.bearing, pixel)) continue;
        const auto& e = exposures[o.camera_id];
        PreparedObservation p; p.camera = o.camera_id; p.ray = o.bearing;
        p.point = o.point_w - prediction.velocity_w * e.dt - 0.5 * gravity * e.dt * e.dt;
        p.T_camera_reference_body = cameras[o.camera_id]->pose() * SE3(e.rotation, e.displacement).inverse();
        prepared.push_back(p); ++result.observations[o.camera_id];
    }
    if (prepared.size() < options.min_visual_inliers) { result.reason = "insufficient fixed-reference observations"; return result; }
    SE3 pose = prediction.Pose().inverse(); // Only this body's six pose parameters vary.
    const double huber = 0.5 * options.max_bearing_error_rad;
    double damping = 1e-5;
    for (int iteration = 0; iteration < 20; ++iteration) {
        Mat66 h = Mat66::Zero(); Vec6 rhs = Vec6::Zero();
        for (const auto& o : prepared) {
            Vec3 p; const auto j = Jacobian(pose, o, p);
            if (p.norm() < 1e-10 || !j.allFinite()) continue;
            const Vec3 r = p.normalized() - o.ray;
            const double weight = std::min(1.0, huber / std::max(r.norm(), 1e-12));
            h += weight * j.transpose() * j; rhs -= weight * j.transpose() * r;
        }
        Mat66 damped = h; damped.diagonal() += damping * h.diagonal().cwiseMax(1e-8);
        const Vec6 step = damped.ldlt().solve(rhs);
        if (!step.allFinite()) { result.reason = "nonfinite pose correction"; return result; }
        const SE3 candidate = SE3::exp(step) * pose;
        if (Cost(candidate, prepared, huber) <= Cost(pose, prepared, huber)) {
            pose = candidate; damping = std::max(1e-9, damping / 3);
            if (step.norm() < 1e-8) break;
        } else { damping *= 10; if (damping > 1e5) break; }
    }
    Mat66 h = Mat66::Zero(); double squared_error = 0;
    for (const auto& o : prepared) {
        Vec3 p; const auto j = Jacobian(pose, o, p);
        const double error = p.norm() > 1e-10 && p.dot(o.ray) > 0 ? Angle(p.normalized(), o.ray) : M_PI;
        if (error <= options.max_bearing_error_rad) {
            ++result.inliers[o.camera]; squared_error += error * error; h += j.transpose() * j;
        }
    }
    const size_t inliers = result.inliers[0] + result.inliers[1];
    if (inliers < options.min_visual_inliers || inliers < 0.6 * prepared.size()) {
        result.reason = "insufficient visual inlier support"; return result;
    }
    const Vec6 norm = h.diagonal().cwiseMax(1e-15).cwiseSqrt();
    const Mat66 normalized = norm.cwiseInverse().asDiagonal() * h * norm.cwiseInverse().asDiagonal();
    Eigen::JacobiSVD<Mat66> svd(normalized);
    const auto& values = svd.singularValues();
    result.condition = std::sqrt(values[0] / std::max(values[5], 1e-30));
    result.bearing_rms_rad = std::sqrt(squared_error / inliers);
    if (result.condition > options.max_pose_condition) { result.reason = "pose-only visual system is ill-conditioned"; return result; }
    const SE3 corrected = pose.inverse();
    result.correction_m = (corrected.translation() - prediction.position_wb).norm();
    result.correction_rad = (prediction.rotation_wb.inverse() * corrected.so3()).log().norm();
    if (result.correction_m > options.max_correction_m || result.correction_rad > options.max_correction_rad) {
        result.reason = "visual correction exceeds prediction consistency limits";
        result.correction_m = 0; result.correction_rad = 0; return result;
    }
    result.state.rotation_wb = corrected.so3(); result.state.position_wb = corrected.translation();
    // World velocity is the IMU-predicted velocity; initialized biases/gravity
    // remain fixed. This is prediction + pose correction, not window VI-BA.
    result.accepted = true; result.reason = "current body pose refined from fixed temporal references";
    return result;
}

class HiltiFrontendRuntime {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    HiltiFrontendRuntime(const HiltiCameras& cameras, const ImuCalibration& calibration, const HiltiFrontendOptions& options)
        : cameras_(cameras), calibration_(calibration), options_(options) {
        CheckOptions(options);
        for (int k = 0; k < 2; ++k) {
            if (!cameras[k]) throw std::invalid_argument("Hilti frontend requires two calibrated EUCM cameras");
            trackers_[k].reset(new HiltiTemporalTracker(k ? SensorStream::Cam1 : SensorStream::Cam0, cameras[k], options.tracking));
        }
    }
    HiltiFrontendResult Process(const HiltiImageEpoch& epoch, const std::array<HiltiTrackingResult, 2>* external) {
        CheckEpoch(epoch, calibration_);
        if (epochs_ && epoch.timestamp_ns <= last_timestamp_) throw std::invalid_argument("frontend epochs must be consecutive and increasing");
        const int mode = external ? 2 : 1;
        if (input_mode_ && input_mode_ != mode) throw std::logic_error("do not mix internal and external Hilti trackers");
        input_mode_ = mode;
        if (last_frame_ && epoch.imu_samples.front().timestamp_ns > last_timestamp_)
            throw std::invalid_argument("epoch IMU does not cover the previous frame");
        HiltiFrontendResult result; result.backend_update = ConsumeBackend();
        result.frame = Frame::CreateFrame();
        const std::array<int64_t, 2> raw{{epoch.cameras[0].timestamp_ns, epoch.cameras[1].timestamp_ns}};
        result.frame->ConfigureHiltiEpoch(epoch.timestamp_ns, raw, epoch.camera_imu_timestamps_ns);
        for (int k = 0; k < 2; ++k) {
            if (external) result.tracking[k] = (*external)[k];
            else trackers_[k]->Process(epoch.cameras[k], result.tracking[k]);
            const auto& current = result.tracking[k];
            const auto stream = k ? SensorStream::Cam1 : SensorStream::Cam0;
            if (current.stream != stream || current.timestamp_ns != raw[k] || (epochs_ == 0 && current.has_previous))
                throw std::invalid_argument("observations do not match the epoch/camera");
            std::set<uint64_t> ids;
            for (const auto& feature : current.features)
                if (!feature.age || !ids.insert(feature.id).second) throw std::invalid_argument("invalid feature bookkeeping");
            if (epochs_) MakeBearingMatches(previous_tracking_[k], current, *cameras_[k]);
        }
        ++epochs_; result.epochs = epochs_;
        if (epochs_ == 1) startup_recording_start_raw_ns_ = raw[0];
        if (status_ == HiltiFrontendStatus::Buffering) {
            for (const auto& sample : epoch.imu_samples) {
                auto at = std::lower_bound(startup_samples_.begin(), startup_samples_.end(), sample.timestamp_ns,
                    [](const ImuSample& s, int64_t t) { return s.timestamp_ns < t; });
                if (at == startup_samples_.end()) startup_samples_.push_back(sample);
                else if (at->timestamp_ns == sample.timestamp_ns) {
                    if (at->angular_velocity != sample.angular_velocity || at->linear_acceleration != sample.linear_acceleration)
                        throw std::invalid_argument("overlapping epoch IMU measurements disagree");
                } else startup_samples_.insert(at, sample);
            }
            startup_frames_.push_back(result.frame);
            for (int k = 0; k < 2; ++k) {
                if (!startup_tracks_[k].empty()) {
                    const auto bearings = MakeBearingMatches(startup_tracks_[k].back(), result.tracking[k], *cameras_[k]);
                    startup_motions_[k].push_back(EstimateBearingMotion(bearings.matches));
                }
                startup_tracks_[k].push_back(result.tracking[k]);
            }
            if (options_.startup_backfill_enabled &&
                raw[0] >= startup_recording_start_raw_ns_ +
                    int64_t(std::llround(options_.startup_backfill_start_s * 1e9))) {
                StartupBuffered buffered;
                buffered.frame_index = epochs_ - 1; buffered.frame = result.frame;
                buffered.epoch = epoch; buffered.tracks = result.tracking;
                for (auto& camera : buffered.epoch.cameras) camera.image.data.clear();
                startup_backfill_buffer_.push_back(std::move(buffered));
            }
            if (startup_frames_.size() > options_.startup_frames) {
                startup_frames_.erase(startup_frames_.begin());
                for (int k = 0; k < 2; ++k) {
                    startup_tracks_[k].erase(startup_tracks_[k].begin());
                    startup_motions_[k].erase(startup_motions_[k].begin());
                }
                const auto first = startup_frames_.front()->HiltiNavigation();
                const int64_t begin = std::min(first.camera_imu_timestamps_ns[0], first.camera_imu_timestamps_ns[1]);
                auto keep = std::lower_bound(startup_samples_.begin(), startup_samples_.end(), begin,
                    [](const ImuSample& sample, int64_t time) { return sample.timestamp_ns < time; });
                if (keep != startup_samples_.begin()) --keep; // Retain the interpolation boundary neighbor.
                startup_samples_.erase(startup_samples_.begin(), keep);
            }
            const size_t frame_index = epochs_ - 1;
            if (startup_frames_.size() == options_.startup_frames &&
                (frame_index == options_.startup_frames - 1 ||
                 (options_.startup_retry_enabled && frame_index > options_.startup_frames - 1 &&
                  (frame_index - (options_.startup_frames - 1)) % options_.startup_retry_stride == 0)))
                Initialize(result, frame_index);
        } else if (status_ != HiltiFrontendStatus::InitializationFailed) {
            const auto previous = last_frame_->HiltiNavigation();
            const auto interval = std::shared_ptr<const ImuPreintegration>(new ImuPreintegration(
                PreintegrateImu(epoch.imu_samples, last_timestamp_, epoch.timestamp_ns, calibration_, previous.state->bias)));
            result.prediction = PropagateNavState(*previous.state, *interval, initialization_.gravity_w);
            
            
            result.has_prediction = true;
            HiltiPoseObservations observations;
            if (local_map_) observations = local_map_->PoseObservations(result.tracking);
            else for (int k = 0; k < 2; ++k) for (const auto& feature : result.tracking[k].features) {
                const auto reference = references_[k].find(feature.id); Vec3 bearing;
                if (reference == references_[k].end() || !cameras_[k]->Unproject(Vec2(feature.pixel.x, feature.pixel.y), bearing)) continue;
                HiltiPoseObservation o; o.camera_id = k; o.feature_id = feature.id;
                o.point_w = reference->second; o.bearing = bearing; observations.push_back(o);
            }
            result.visual = RefineHiltiBodyPose(result.prediction, epoch, observations, cameras_, calibration_, initialization_.gravity_w, options_);
            
            
            status_ = result.visual.accepted ? HiltiFrontendStatus::Tracking : HiltiFrontendStatus::PredictionOnly;
            result.frame->SetHiltiNavigation(result.visual.state, result.visual.accepted, interval, backend_version_);
            result.reason = result.visual.reason;
        }
        if (status_ == HiltiFrontendStatus::InitializationFailed) result.reason = initialization_.message;
        if (status_ == HiltiFrontendStatus::Buffering && result.reason.empty())
            result.reason = "buffering startup observations; no metric state";
        result.status = status_; result.reference_points = {{references_[0].size(), references_[1].size()}};
        previous_tracking_ = result.tracking; last_frame_ = result.frame; last_timestamp_ = epoch.timestamp_ns;
        if (local_map_) {
            const auto nav=result.frame->HiltiNavigation();
            History record;record.epoch=epoch;record.tracks=result.tracking;record.state=*nav.state;record.valid=nav.tracking_valid;
            for(auto& camera:record.epoch.cameras) camera.image.data.clear(); // History needs observations/IMU, not JPEG copies.
            history_[epoch.timestamp_ns]=std::move(record);
            MergeSamples(keyframe_samples_,epoch.imu_samples);
            const bool keyframe=result.initialized_now || (epochs_-options_.startup_frames)%backend_->Options().keyframe_stride==0;
            
            if(keyframe && !result.initialized_now) {
                local_map_->AddKeyframe(epoch.timestamp_ns,*nav.state,epoch.camera_imu_timestamps_ns,result.tracking,keyframe_samples_);
                keyframe_samples_=epoch.imu_samples;
            }
            if(keyframe && loop_closing_) {
                PendingLoop p;p.input.frame.id=next_loop_id_++;
                p.input.frame.timestamp_ns=epoch.timestamp_ns;p.input.images=epoch.cameras;p.input.tracks=result.tracking;
                pending_loops_.emplace(epoch.timestamp_ns,std::move(p));
            }
            if(keyframe && !backend_->Busy() && local_map_->WindowSize()>=3) {
                const auto job=local_map_->Snapshot(++job_version_,backend_version_);
                result.backend_job_submitted=backend_->Submit(job);
            }
            result.backend_window_size=local_map_->WindowSize();
            result.archived_landmarks=local_map_->ArchiveLandmarks();result.archived_observations=local_map_->ArchiveObservations();
        }
        result.backend_version=backend_version_;
        if(global_trajectory_) {
            global_trajectory_->RequestUpdate(epoch.timestamp_ns);
            result.global_navigation=GlobalOutput(true);
        }
        return result;
    }
    const VioInitializationResult& Initialization() const { return initialization_; }
    const std::vector<HiltiStartupAttempt>& StartupAttempts() const { return startup_attempts_; }
    HiltiFrontendStatus Status() const { return status_; }
    void SetBackend(std::shared_ptr<HiltiBackend> backend) {
        if(epochs_ || backend_ || !backend) throw std::logic_error("attach one Hilti backend before the first epoch");
        backend_=std::move(backend);
    }
    void SetLoopClosing(std::shared_ptr<LoopClosing> loop) {
        if(epochs_ || loop_closing_ || !loop || !backend_)throw std::logic_error("attach loop closing after Hilti backend, before first epoch");
        loop_closing_=std::move(loop);
    }
    void SetGlobalTrajectory(std::shared_ptr<GlobalTrajectory> global) {
        if(epochs_ || global_trajectory_ || !global || !loop_closing_ || global->LoopSource()!=loop_closing_.get())
            throw std::logic_error("attach one matching global trajectory after Hilti loop closing, before epochs");
        global_trajectory_=std::move(global);
    }
    std::shared_ptr<const GlobalNavigation> GlobalOutput(bool smooth_live) const {
        if(!global_trajectory_ || !last_frame_)return nullptr;
        const auto local=last_frame_->HiltiNavigation();if(!local.state)return nullptr;
        return std::make_shared<const GlobalNavigation>(global_trajectory_->Corrections().Project(
            *local.state,initialization_.gravity_w,local.timestamp_ns,smooth_live));
    }
    void FlushGlobalTrajectory() {
        if(!global_trajectory_)return;
        FlushLoops();
        if(!loop_closing_->WaitUntilIdle(std::chrono::seconds(30)))throw std::runtime_error("global loop drain timed out");
        if(!global_trajectory_->RequestUpdate(last_timestamp_,true) ||
            !global_trajectory_->WaitUntilIdle(std::chrono::seconds(30)))throw std::runtime_error("global graph drain timed out");
        const auto statistics=global_trajectory_->Statistics();
        if(!statistics.last_error.empty())throw std::runtime_error(statistics.last_error);
    }
    void FlushLoops() {
        Poll(true);
        PublishLoops(std::numeric_limits<int64_t>::max());
    }
    std::shared_ptr<const HiltiBackendResult> Poll(bool wait) {
        if(!backend_) return nullptr;
        if(wait && !backend_->WaitUntilIdle(std::chrono::seconds(30))) throw std::runtime_error("Hilti backend wait timed out");
        return ConsumeBackend();
    }
   private:
    struct PendingLoop {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        LoopInput input;bool optimized=false;
    };
    void PublishLoops(int64_t before) {
        for(auto it=pending_loops_.begin();it!=pending_loops_.end() && it->first<before;) {
            // Only accepted local solutions enter the loop database. Rejected
            // or unoptimized keyframes cannot invent metric loop constraints.
            if(it->second.optimized && !loop_closing_->Submit(it->second.input))
                throw std::runtime_error("loop closing rejected an archival keyframe");
            it=pending_loops_.erase(it);
        }
    }
    struct History {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        HiltiImageEpoch epoch;std::array<HiltiTrackingResult,2> tracks;NavState state;bool valid=false;
    };
    struct StartupBuffered {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        size_t frame_index = 0;
        HiltiImageEpoch epoch;
        std::array<HiltiTrackingResult,2> tracks;
        Frame::Ptr frame;
    };
    static void MergeSamples(std::vector<ImuSample>& target,const std::vector<ImuSample>& source) {
        for(const auto& s:source) {
            auto it=std::lower_bound(target.begin(),target.end(),s.timestamp_ns,[](const ImuSample& a,int64_t t){return a.timestamp_ns<t;});
            if(it==target.end()) target.push_back(s);
            else if(it->timestamp_ns!=s.timestamp_ns) target.insert(it,s);
            else if(it->angular_velocity!=s.angular_velocity || it->linear_acceleration!=s.linear_acceleration)
                throw std::invalid_argument("backend history IMU measurements disagree");
        }
    }
    std::shared_ptr<const HiltiBackendResult> ConsumeBackend() {
        if(!backend_) return nullptr;
        const auto correction=backend_->TakeResult();if(!correction) return nullptr;
        
        if(!correction->diagnostics.accepted) return correction;
        if(!local_map_ || correction->base_version!=backend_version_ || correction->version<=backend_version_ || correction->keyframes.empty())
            throw std::logic_error("stale/incoherent Hilti backend publication");
        const int64_t anchor=correction->keyframes.back().timestamp_ns;
        
        if(!history_.count(anchor)) throw std::logic_error("backend correction anchor absent from frontend history");
        // Prepare the entire landmark/state/replay batch before its frontend
        // commit. Readers get one immutable BackendResult; worker owns no Frames.
        auto next=std::unique_ptr<HiltiLocalMap>(new HiltiLocalMap(*local_map_));
        next->Apply(*correction,{});
        HiltiAlignedVector<HiltiKeyframeState> replayed;
        NavState state=correction->keyframes.back().state;int64_t previous=anchor;
        bool valid=history_.at(anchor).valid;
        for(auto it=history_.upper_bound(anchor);it!=history_.end();++it) {
            const auto& record=it->second;
            const auto d=PreintegrateImu(record.epoch.imu_samples,previous,it->first,calibration_,state.bias);
            const auto predicted=PropagateNavState(state,d,initialization_.gravity_w);
            
            
            const auto fit=RefineHiltiBodyPose(predicted,record.epoch,next->PoseObservations(record.tracks),
                cameras_,calibration_,initialization_.gravity_w,options_);
            state=fit.state;valid=fit.accepted;previous=it->first;
            
            HiltiKeyframeState f;f.timestamp_ns=it->first;f.state=state;f.exposure_timestamps_ns=record.epoch.camera_imu_timestamps_ns;
            replayed.push_back(f);
        }
        next->Apply(*correction,replayed);
        const auto old=last_frame_->HiltiNavigation();std::shared_ptr<const ImuPreintegration> attached;
        if(old.from_previous) attached.reset(new ImuPreintegration(PreintegrateImu(history_.at(last_timestamp_).epoch.imu_samples,
            old.from_previous->begin_timestamp_ns,last_timestamp_,calibration_,state.bias)));
        last_frame_->SetHiltiNavigation(state,valid,attached,correction->version);
        local_map_.swap(next);backend_version_=correction->version;
        
        
        if(loop_closing_) {
            for(const auto& f:correction->keyframes) {
                auto it=pending_loops_.find(f.timestamp_ns);if(it==pending_loops_.end())continue;
                auto& p=it->second;p.optimized=true;p.input.frame.backend_version=correction->version;
                p.input.frame.optimized_body_pose=f.state.Pose();p.input.landmarks.clear();
                for(int k=0;k<2;++k)p.input.frame.camera_from_body[k]=
                    HiltiCameraWorldPose(f,k,cameras_,initialization_.gravity_w).inverse()*f.state.Pose();
                for(const auto& l:correction->landmarks) {
                    bool observed=false;for(const auto& o:l.observations)if(o.timestamp_ns==f.timestamp_ns){observed=true;break;}
                    if(observed)p.input.landmarks.push_back(l);
                }
            }
            PublishLoops(correction->keyframes.front().timestamp_ns);
        }
        for(const auto& f:correction->keyframes) {auto it=history_.find(f.timestamp_ns);if(it!=history_.end())it->second.state=f.state;}
        for(const auto& f:replayed) history_.at(f.timestamp_ns).state=f.state;
        status_=valid?HiltiFrontendStatus::Tracking:HiltiFrontendStatus::PredictionOnly;
        // Preserve enough replay history for the only in-flight job, without
        // deleting any landmark's archival observations.
        history_.erase(history_.begin(),history_.lower_bound(anchor));
        return correction;
    }
    std::shared_ptr<HiltiStartupRecovery> RecoverStartup(
            const VioInitializationResult& candidate,
            const std::array<ReferenceMap,2>& references,
            size_t start_frame, size_t end_frame) {
        if (!options_.startup_backfill_enabled || startup_backfill_buffer_.empty() ||
            startup_backfill_buffer_.front().frame_index >= start_frame) return nullptr;
        auto recovery = std::make_shared<HiltiStartupRecovery>(); recovery->attempted = true;
        recovery->first_eligible_frame = startup_backfill_buffer_.front().frame_index;
        recovery->anchor_frame = start_frame; recovery->initialization_end_frame = end_frame;
        recovery->eligible_buffered_frames = end_frame - recovery->first_eligible_frame + 1;
        auto fail = [&](const std::string& reason) {
            recovery->accepted = false; recovery->reason = reason; return recovery;
        };
        if (candidate.states.size() != end_frame-start_frame+1 ||
            startup_backfill_buffer_.back().frame_index != end_frame)
            return fail("accepted initializer and eligibility buffer do not share the same end frame");
        for (size_t i=0;i<startup_backfill_buffer_.size();++i)
            if (startup_backfill_buffer_[i].frame_index != recovery->first_eligible_frame+i)
                return fail("eligibility buffer contains a frame gap or duplicate");
        auto record = [&](size_t frame) -> const StartupBuffered& {
            if (frame < recovery->first_eligible_frame || frame > end_frame)
                throw std::logic_error("startup recovery frame is outside the eligibility buffer");
            return startup_backfill_buffer_.at(frame-recovery->first_eligible_frame);
        };
        auto observations = [&](const StartupBuffered& buffered) {
            HiltiPoseObservations result;
            for(int k=0;k<2;++k) for(const auto& feature:buffered.tracks[k].features) {
                const auto point=references[k].find(feature.id);Vec3 ray;
                if(point==references[k].end() ||
                   !cameras_[k]->Unproject(Vec2(feature.pixel.x,feature.pixel.y),ray))continue;
                HiltiPoseObservation observation;observation.camera_id=k;observation.feature_id=feature.id;
                observation.point_w=point->second;observation.bearing=ray;result.push_back(observation);
            }
            return result;
        };
        auto reconstruct = [&](size_t frame, const NavState& next, size_t next_frame, bool overlap) {
            const auto& current=record(frame);const auto& following=record(next_frame);
            const auto delta=PreintegrateImu(following.epoch.imu_samples,
                current.epoch.timestamp_ns,following.epoch.timestamp_ns,calibration_,next.bias);
            const NavState predicted=BackwardPropagateNavState(next,delta,candidate.gravity_w);
            const auto visible=observations(current);
            const auto fit=RefineHiltiBodyPose(predicted,current.epoch,visible,cameras_,calibration_,
                candidate.gravity_w,options_);
            HiltiStartupRecoveryFrame output;output.frame_index=frame;
            output.raw_cam0_timestamp_ns=current.epoch.cameras[0].timestamp_ns;
            output.reference_timestamp_ns=current.epoch.timestamp_ns;
            output.initializer_overlap=overlap;output.metric_landmark_observations=visible.size();
            output.pose_only_residual_rad=fit.bearing_rms_rad;output.pose_condition=fit.condition;
            output.mode=fit.accepted?HiltiStartupRecoveryMode::VisualRefined:
                HiltiStartupRecoveryMode::ImuPropagatedOnly;
            output.state=fit.accepted?fit.state:predicted;
            const auto forward=PropagateNavState(output.state,delta,candidate.gravity_w);
            output.imu_forward_position_error_m=(forward.position_wb-next.position_wb).norm();
            output.imu_forward_rotation_error_rad=(forward.rotation_wb.inverse()*next.rotation_wb).log().norm();
            output.imu_forward_velocity_error_m_s=(forward.velocity_w-next.velocity_w).norm();
            return output;
        };

        // Shadow the exact recovery algorithm across the accepted window. This
        // never replaces initializer states; it only validates backward overlap.
        HiltiStartupRecoveryFrames shadow;
        HiltiStartupRecoveryFrame end;const auto& end_record=record(end_frame);
        end.frame_index=end_frame;end.raw_cam0_timestamp_ns=end_record.epoch.cameras[0].timestamp_ns;
        end.reference_timestamp_ns=end_record.epoch.timestamp_ns;end.initializer_overlap=true;
        end.mode=HiltiStartupRecoveryMode::InitializerAnchor;end.state=candidate.states.back();
        end.metric_landmark_observations=observations(end_record).size();
        end.initializer_position_disagreement_m=0;end.initializer_rotation_disagreement_rad=0;
        end.initializer_velocity_disagreement_m_s=0;end.initializer_bias_disagreement=0;
        shadow.push_back(end);NavState next=end.state;size_t next_frame=end_frame;
        for(size_t frame=end_frame;frame-- > start_frame;) {
            auto output=reconstruct(frame,next,next_frame,true);
            const auto& accepted=candidate.states.at(frame-start_frame);
            output.initializer_position_disagreement_m=(output.state.position_wb-accepted.position_wb).norm();
            output.initializer_rotation_disagreement_rad=(output.state.rotation_wb.inverse()*accepted.rotation_wb).log().norm();
            output.initializer_velocity_disagreement_m_s=(output.state.velocity_w-accepted.velocity_w).norm();
            output.initializer_bias_disagreement=BiasDifference(output.state.bias,accepted.bias);
            shadow.push_back(output);next=output.state;next_frame=frame;
        }
        std::reverse(shadow.begin(),shadow.end());

        // The publishable segment starts from the initializer's exact first
        // state. Shadow overlap error therefore cannot leak into the backfill.
        HiltiStartupRecoveryFrames backfill;next=candidate.states.front();next_frame=start_frame;
        for(size_t frame=start_frame;frame-- > recovery->first_eligible_frame;) {
            auto output=reconstruct(frame,next,next_frame,false);
            backfill.push_back(output);next=output.state;next_frame=frame;
        }
        std::reverse(backfill.begin(),backfill.end());
        recovery->frames=backfill;
        recovery->frames.insert(recovery->frames.end(),shadow.begin(),shadow.end());
        recovery->overlap_frames=shadow.size();
        recovery->overlap_max_position_disagreement_m=0;
        recovery->overlap_max_rotation_disagreement_rad=0;
        recovery->overlap_max_velocity_disagreement_m_s=0;
        recovery->overlap_max_bias_disagreement=0;
        std::set<size_t> frame_ids;
        for(const auto& output:recovery->frames) {
            recovery->duplicate_frames+=!frame_ids.insert(output.frame_index).second;
            recovery->nonfinite_states+=!Finite(output.state);
            if(output.mode==HiltiStartupRecoveryMode::VisualRefined)++recovery->visual_refined_frames;
            if(output.mode==HiltiStartupRecoveryMode::ImuPropagatedOnly)++recovery->imu_only_frames;
            if(output.initializer_overlap) {
                recovery->overlap_max_position_disagreement_m=std::max(
                    recovery->overlap_max_position_disagreement_m,output.initializer_position_disagreement_m);
                recovery->overlap_max_rotation_disagreement_rad=std::max(
                    recovery->overlap_max_rotation_disagreement_rad,output.initializer_rotation_disagreement_rad);
                recovery->overlap_max_velocity_disagreement_m_s=std::max(
                    recovery->overlap_max_velocity_disagreement_m_s,output.initializer_velocity_disagreement_m_s);
                recovery->overlap_max_bias_disagreement=std::max(
                    recovery->overlap_max_bias_disagreement,output.initializer_bias_disagreement);
                const size_t local=output.frame_index-start_frame;
                const auto metadata=startup_frames_.at(local)->HiltiNavigation();
                recovery->timestamp_mismatches+=metadata.timestamp_ns!=output.reference_timestamp_ns ||
                    metadata.raw_camera_timestamps_ns[0]!=output.raw_cam0_timestamp_ns;
            }
        }
        recovery->eligible_published_frames=backfill.size()+candidate.states.size();
        recovery->eligible_coverage_percent=recovery->eligible_buffered_frames?
            100.0*recovery->eligible_published_frames/recovery->eligible_buffered_frames:0;
        // Initializer velocities are a joint least-squares trajectory rather
        // than exact per-interval propagation. Keep a bounded 0.5 m/s overlap
        // check while retaining the native 0.15 m / 0.1 rad pose gates.
        constexpr double kOverlapVelocityLimitM_S=0.5;
        const bool consistent=recovery->eligible_coverage_percent>=99 && !recovery->timestamp_mismatches &&
            !recovery->duplicate_frames && !recovery->nonfinite_states &&
            recovery->overlap_max_position_disagreement_m<=options_.max_correction_m &&
            recovery->overlap_max_rotation_disagreement_rad<=options_.max_correction_rad &&
            recovery->overlap_max_velocity_disagreement_m_s<=kOverlapVelocityLimitM_S &&
            recovery->overlap_max_bias_disagreement<=1e-12;
        recovery->accepted=consistent;
        recovery->reason=consistent?"coverage and initializer-overlap consistency gates passed":
            "coverage or initializer-overlap consistency gate failed";
        return recovery;
    }
    void Initialize(HiltiFrontendResult& result, size_t end_frame) {
        VioInitializationResult candidate;
        if (options_.diagnostic_legacy_startup) {
            const auto window = MakeVioVisualWindow(startup_tracks_[0], startup_motions_[0], *cameras_[0], calibration_, options_.initialization);
            VioImuIntervals zero_bias;
            for (size_t i = 1; i < window.frames.size(); ++i)
                zero_bias.push_back(PreintegrateImu(startup_samples_, window.frames[i - 1].timestamp_ns,
                    window.frames[i].timestamp_ns, calibration_));
            candidate = VioInitializer(options_.initialization).Initialize(
                window, zero_bias, startup_samples_, calibration_, cameras_[0]->pose());
            candidate.path = "diagnostic_legacy_startup/" + candidate.path;
        } else {
            candidate = InitializeHiltiStartup(startup_tracks_[0], startup_motions_[0], *cameras_[0],
                startup_samples_, calibration_, options_.initialization);
        }
        HiltiStartupAttempt attempt;
        attempt.attempt = startup_attempts_.size() + 1;
        attempt.start_frame = end_frame + 1 - startup_frames_.size(); attempt.end_frame = end_frame;
        const auto first_metadata = startup_frames_.front()->HiltiNavigation();
        const auto last_metadata = startup_frames_.back()->HiltiNavigation();
        attempt.raw_cam0_start_ns = first_metadata.raw_camera_timestamps_ns[0];
        attempt.raw_cam0_end_ns = last_metadata.raw_camera_timestamps_ns[0];
        attempt.imu_samples = startup_samples_.size();
        if (!startup_samples_.empty()) {
            attempt.imu_sample_start_ns = startup_samples_.front().timestamp_ns;
            attempt.imu_sample_end_ns = startup_samples_.back().timestamp_ns;
        }
        attempt.status = candidate.status;
        attempt.recoverable = !candidate.Success() && RecoverableInitializationFailure(candidate.status);
        attempt.reason = candidate.message;
        startup_attempts_.push_back(attempt);
        if (!candidate.Success()) {
            result.reason = candidate.message;
            const bool exhausted = end_frame >= options_.startup_max_end_frame ||
                options_.startup_max_end_frame - end_frame < options_.startup_retry_stride;
            if (!attempt.recoverable || !options_.startup_retry_enabled || exhausted) {
                if (!attempt.recoverable)
                    candidate.message = "fatal initialization failure (not retried): " + candidate.message;
                else if (exhausted)
                    candidate.message = "startup retry horizon exhausted at frame " + std::to_string(end_frame) +
                        " after " + std::to_string(startup_attempts_.size()) + " attempts; last failure: " + candidate.message;
                initialization_ = std::move(candidate);
                status_ = HiltiFrontendStatus::InitializationFailed;
            }
            return;
        }
        if (candidate.states.size() != startup_frames_.size()) throw std::logic_error("initializer state count mismatch");
        // Prepare corrected-bias intervals and references BEFORE publishing any
        // buffered states. A failed/partial initializer never publishes NavState.
        std::vector<std::shared_ptr<const ImuPreintegration>> intervals(startup_frames_.size());
        for (size_t i = 1; i < startup_frames_.size(); ++i)
            intervals[i].reset(new ImuPreintegration(PreintegrateImu(startup_samples_, startup_frames_[i - 1]->HiltiNavigation().timestamp_ns,
                startup_frames_[i]->HiltiNavigation().timestamp_ns, calibration_, candidate.bias)));
        std::array<ReferenceMap, 2> candidate_references;
        for (int k = 0; k < 2; ++k) {
            std::map<uint64_t, ReferenceTrack> tracks;
            for (size_t i = 0; i < startup_frames_.size(); ++i) {
                const auto metadata = startup_frames_[i]->HiltiNavigation();
                const auto e = AtExposure(startup_samples_, metadata.timestamp_ns, metadata.camera_imu_timestamps_ns[k], calibration_, candidate.bias);
                const SE3 T_wc = ExposurePose(candidate.states[i], e, candidate.gravity_w) * cameras_[k]->pose().inverse();
                for (const auto& feature : startup_tracks_[k][i].features) {
                    Vec3 ray;
                    if (cameras_[k]->Unproject(Vec2(feature.pixel.x, feature.pixel.y), ray)) tracks[feature.id].push_back({T_wc, ray});
                }
            }
            for (const auto& track : tracks) {
                Vec3 point;
                if (MakeReference(track.second, point, options_.max_bearing_error_rad, options_.max_pose_condition))
                    candidate_references[k][track.first] = point;
            }
        }
        std::shared_ptr<HiltiStartupRecovery> recovery;
        try {
            recovery=RecoverStartup(candidate,candidate_references,attempt.start_frame,attempt.end_frame);
        } catch(const std::exception& error) {
            recovery=std::make_shared<HiltiStartupRecovery>();recovery->attempted=true;
            recovery->anchor_frame=attempt.start_frame;recovery->initialization_end_frame=attempt.end_frame;
            recovery->reason=std::string("startup recovery exception: ")+error.what();
        }
        std::vector<std::shared_ptr<const ImuPreintegration>> recovery_intervals;
        if(recovery && recovery->accepted) {
            size_t publishable=0;while(publishable<recovery->frames.size() &&
                !recovery->frames[publishable].initializer_overlap)++publishable;
            recovery_intervals.resize(publishable);
            for(size_t i=1;i<publishable;++i) {
                const auto& previous=recovery->frames[i-1];const auto& current=recovery->frames[i];
                const auto& buffered=startup_backfill_buffer_.at(
                    current.frame_index-startup_backfill_buffer_.front().frame_index);
                recovery_intervals[i].reset(new ImuPreintegration(PreintegrateImu(
                    buffered.epoch.imu_samples,previous.reference_timestamp_ns,current.reference_timestamp_ns,
                    calibration_,current.state.bias)));
            }
        }
        std::unique_ptr<HiltiLocalMap> candidate_map;
        if(backend_) {
            candidate_map.reset(new HiltiLocalMap(cameras_,calibration_,backend_->Options(),candidate.gravity_w));
            for(int k=0;k<2;++k) for(const auto& p:candidate_references[k]) candidate_map->Seed({k,p.first},p.second);
            std::vector<size_t> selected;
            for(size_t i=startup_frames_.size()-1;;) {
                selected.push_back(i);
                if(selected.size()==backend_->Options().window_size || i<backend_->Options().keyframe_stride) break;
                i-=backend_->Options().keyframe_stride;
            }
            std::reverse(selected.begin(),selected.end());
            for(size_t i:selected) {
                const auto metadata=startup_frames_[i]->HiltiNavigation();
                const std::array<HiltiTrackingResult,2> tracks{{startup_tracks_[0][i],startup_tracks_[1][i]}};
                candidate_map->AddKeyframe(metadata.timestamp_ns,candidate.states[i],metadata.camera_imu_timestamps_ns,tracks,startup_samples_);
            }
        }
        auto publication = std::make_shared<HiltiStartupPublication>();
        publication->start_frame = attempt.start_frame; publication->end_frame = attempt.end_frame;
        publication->states = candidate.states;publication->recovery=recovery;
        for (const auto& frame : startup_frames_) {
            const auto metadata = frame->HiltiNavigation();
            publication->raw_cam0_timestamps_ns.push_back(metadata.raw_camera_timestamps_ns[0]);
            publication->reference_timestamps_ns.push_back(metadata.timestamp_ns);
        }
        for (size_t i = 0; i < publication->states.size(); ++i) {
            if (!publication->states[i].Pose().matrix().allFinite() || !publication->states[i].velocity_w.allFinite() ||
                !publication->states[i].bias.gyroscope.allFinite() || !publication->states[i].bias.accelerometer.allFinite() ||
                (i && (publication->raw_cam0_timestamps_ns[i] <= publication->raw_cam0_timestamps_ns[i-1] ||
                       publication->reference_timestamps_ns[i] <= publication->reference_timestamps_ns[i-1])))
                throw std::logic_error("invalid accepted startup trajectory publication");
        }
        // Commit exactly once, after every candidate object and invariant is ready.
        if(recovery && recovery->accepted) {
            size_t i=0;for(const auto& recovered:recovery->frames) {
                if(recovered.initializer_overlap)break;
                auto& buffered=startup_backfill_buffer_.at(
                    recovered.frame_index-startup_backfill_buffer_.front().frame_index);
                buffered.frame->SetHiltiNavigation(recovered.state,
                    recovered.mode==HiltiStartupRecoveryMode::VisualRefined,recovery_intervals.at(i++));
            }
        }
        for (size_t i = 0; i < startup_frames_.size(); ++i)
            startup_frames_[i]->SetHiltiNavigation(candidate.states[i], true, intervals[i]);
        initialization_ = std::move(candidate);
        references_ = std::move(candidate_references);
        local_map_ = std::move(candidate_map);
        result.startup_publication = publication;
        status_ = HiltiFrontendStatus::Initialized; result.initialized_now = true;
        
        for(size_t i=0;i<startup_frames_.size();++i)
        result.reason = initialization_.message;
        startup_frames_.clear(); startup_samples_.clear();
        startup_backfill_buffer_.clear();
        for (int k = 0; k < 2; ++k) { startup_tracks_[k].clear(); startup_motions_[k].clear(); }
    }
    HiltiCameras cameras_; ImuCalibration calibration_; HiltiFrontendOptions options_;
    std::array<std::unique_ptr<HiltiTemporalTracker>, 2> trackers_;
    HiltiFrontendStatus status_ = HiltiFrontendStatus::Buffering;
    int input_mode_ = 0; size_t epochs_ = 0; int64_t last_timestamp_ = 0;
    Frame::Ptr last_frame_;
    std::array<HiltiTrackingResult, 2> previous_tracking_;
    std::vector<Frame::Ptr> startup_frames_;
    std::array<std::vector<HiltiTrackingResult>, 2> startup_tracks_;
    std::array<std::vector<BearingMotionResult>, 2> startup_motions_;
    std::vector<ImuSample> startup_samples_;
    std::vector<HiltiStartupAttempt> startup_attempts_;
    std::vector<StartupBuffered, Eigen::aligned_allocator<StartupBuffered>> startup_backfill_buffer_;
    int64_t startup_recording_start_raw_ns_ = 0;
    std::array<ReferenceMap, 2> references_;
    VioInitializationResult initialization_;
    std::shared_ptr<HiltiBackend> backend_;
    std::shared_ptr<LoopClosing> loop_closing_;
    std::shared_ptr<GlobalTrajectory> global_trajectory_; // Reports only; never read by propagation/local optimization.
    uint64_t next_loop_id_=0;
    std::map<int64_t,PendingLoop,std::less<int64_t>,Eigen::aligned_allocator<std::pair<const int64_t,PendingLoop>>> pending_loops_;
    std::unique_ptr<HiltiLocalMap> local_map_;
    uint64_t job_version_=0,backend_version_=0;
    std::vector<ImuSample> keyframe_samples_;
    std::map<int64_t,History,std::less<int64_t>,Eigen::aligned_allocator<std::pair<const int64_t,History>>> history_;
};

Frontend::Frontend(const HiltiCameras& cameras, const ImuCalibration& calibration, const HiltiFrontendOptions& options)
    : hilti_runtime_(new HiltiFrontendRuntime(cameras, calibration, options)) {}
HiltiFrontendResult Frontend::AddHiltiEpoch(const HiltiImageEpoch& epoch) {
    if (!hilti_runtime_) throw std::logic_error("Frontend was not constructed for Hilti");
    auto result = hilti_runtime_->Process(epoch, nullptr);
    status_ = result.status == HiltiFrontendStatus::Buffering ? FrontendStatus::INITING :
              result.status == HiltiFrontendStatus::InitializationFailed ? FrontendStatus::LOST :
              result.status == HiltiFrontendStatus::PredictionOnly ? FrontendStatus::TRACKING_BAD : FrontendStatus::TRACKING_GOOD;
    return result;
}
HiltiFrontendResult Frontend::AddHiltiTrackedEpoch(const HiltiImageEpoch& epoch, const std::array<HiltiTrackingResult, 2>& tracking) {
    if (!hilti_runtime_) throw std::logic_error("Frontend was not constructed for Hilti");
    auto result = hilti_runtime_->Process(epoch, &tracking);
    status_ = result.status == HiltiFrontendStatus::Buffering ? FrontendStatus::INITING :
              result.status == HiltiFrontendStatus::InitializationFailed ? FrontendStatus::LOST :
              result.status == HiltiFrontendStatus::PredictionOnly ? FrontendStatus::TRACKING_BAD : FrontendStatus::TRACKING_GOOD;
    return result;
}
const VioInitializationResult& Frontend::HiltiInitialization() const {
    if (!hilti_runtime_) throw std::logic_error("Frontend was not constructed for Hilti");
    return hilti_runtime_->Initialization();
}
const std::vector<HiltiStartupAttempt>& Frontend::HiltiStartupAttempts() const {
    if (!hilti_runtime_) throw std::logic_error("Frontend was not constructed for Hilti");
    return hilti_runtime_->StartupAttempts();
}
void Frontend::SetHiltiBackend(std::shared_ptr<HiltiBackend> backend) {
    if(!hilti_runtime_) throw std::logic_error("Hilti backend requires Hilti frontend");
    hilti_runtime_->SetBackend(std::move(backend));
}
std::shared_ptr<const HiltiBackendResult> Frontend::PollHiltiBackend(bool wait) {
    if(!hilti_runtime_) throw std::logic_error("Hilti backend requires Hilti frontend");
    auto result=hilti_runtime_->Poll(wait);
    const auto status=hilti_runtime_->Status();
    status_=status==HiltiFrontendStatus::Buffering?FrontendStatus::INITING:
        status==HiltiFrontendStatus::InitializationFailed?FrontendStatus::LOST:
        status==HiltiFrontendStatus::PredictionOnly?FrontendStatus::TRACKING_BAD:FrontendStatus::TRACKING_GOOD;
    return result;
}
void Frontend::SetHiltiLoopClosing(std::shared_ptr<LoopClosing> loop) {
    if(!hilti_runtime_)throw std::logic_error("loop closing requires Hilti frontend");
    hilti_runtime_->SetLoopClosing(std::move(loop));
}
void Frontend::FlushHiltiLoopClosing() {
    if(!hilti_runtime_)throw std::logic_error("loop closing requires Hilti frontend");
    hilti_runtime_->FlushLoops();
}
void Frontend::SetHiltiGlobalTrajectory(std::shared_ptr<GlobalTrajectory> global) {
    if(!hilti_runtime_)throw std::logic_error("global trajectory requires Hilti frontend");
    hilti_runtime_->SetGlobalTrajectory(std::move(global));
}
std::shared_ptr<const GlobalNavigation> Frontend::HiltiGlobalNavigation(bool smooth_live) const {
    if(!hilti_runtime_)throw std::logic_error("global output requires Hilti frontend");
    return hilti_runtime_->GlobalOutput(smooth_live);
}
void Frontend::FlushHiltiGlobalTrajectory() {
    if(!hilti_runtime_)throw std::logic_error("global trajectory requires Hilti frontend");
    hilti_runtime_->FlushGlobalTrajectory();
}
}  // namespace myslam
