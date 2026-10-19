#include "myslam/startup_visual_trajectory.h"

#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace myslam {
namespace {
// All residuals use oriented unit rays, including rays with negative camera Z.
// Structure is temporary and never published to Map. No inertial measurements
// are used here; calibration is used only for body rotations and clock conversion.
struct Observation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t frame;
    Vec3 ray;
};
struct Track {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t id;
    std::vector<Observation, Eigen::aligned_allocator<Observation>> observations;
    Vec3 point = Vec3::Zero();
    bool initialized = false;
};
using Tracks = std::vector<Track, Eigen::aligned_allocator<Track>>;
using Poses = std::vector<SE3, Eigen::aligned_allocator<SE3>>; // T_camera_world
Mat33 Cross(const Vec3& v) {
    Mat33 m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}
double Angle(const Vec3& a, const Vec3& b) {
    return std::atan2(a.cross(b).norm(), a.dot(b));
}
double Median(std::vector<double> x) {
    if (x.empty()) return 0;
    std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
    return x[x.size() / 2];
}
const Observation* At(const Track& track, size_t frame) {
    for (const auto& o : track.observations) if (o.frame == frame) return &o;
    return nullptr;
}
Vec3 Centre(const SE3& pose) { return -(pose.so3().inverse() * pose.translation()); }
double Residual(const SE3& pose, const Vec3& point, const Vec3& ray) {
    const Vec3 p = pose * point;
    if (p.norm() < 1e-12 || p.dot(ray) <= 0) return M_PI;
    return Angle(p.normalized(), ray);
}

// Oriented-ray intersection, followed by robust angular refinement. The two
// excluded observations are not used when validating a consecutive pair's sign.
bool Triangulate(const Track& track, const Poses& poses, Vec3& point,
                 double limit, int exclude1 = -1, int exclude2 = -1) {
    Mat33 h = Mat33::Zero(); Vec3 rhs = Vec3::Zero();
    std::vector<const Observation*> obs;
    for (const auto& o : track.observations) {
        if (int(o.frame) == exclude1 || int(o.frame) == exclude2) continue;
        const Vec3 ray = poses[o.frame].so3().inverse() * o.ray;
        const Mat33 a = Mat33::Identity() - ray * ray.transpose();
        h += a; rhs += a * Centre(poses[o.frame]); obs.push_back(&o);
    }
    if (obs.size() < 2) return false;
    Eigen::JacobiSVD<Mat33> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()[2] < 1e-9 * svd.singularValues()[0]) return false;
    point = svd.solve(rhs);
    if (!point.allFinite()) return false;
    for (int iteration = 0; iteration < 8; ++iteration) {
        h.setZero(); rhs.setZero();
        for (const auto* o : obs) {
            const auto& pose = poses[o->frame];
            const Vec3 p = pose * point;
            if (p.norm() < 1e-10) return false;
            const Vec3 u = p.normalized();
            const Vec3 r = u - o->ray;
            const Mat33 j = (Mat33::Identity() - u * u.transpose()) / p.norm() * pose.rotationMatrix();
            const double weight = std::min(1.0, limit * 0.5 / std::max(r.norm(), 1e-12));
            h += weight * j.transpose() * j; rhs -= weight * j.transpose() * r;
        }
        const Vec3 step = h.ldlt().solve(rhs);
        if (!step.allFinite()) return false;
        point += step;
        if (step.norm() < 1e-9) break;
    }
    size_t positive = 0; std::vector<double> errors;
    for (const auto* o : obs) {
        const Vec3 p = poses[o->frame] * point;
        positive += p.dot(o->ray) > 0;
        errors.push_back(Residual(poses[o->frame], point, o->ray));
    }
    return positive >= 0.8 * obs.size() && Median(errors) <= limit;
}

using JPose = Eigen::Matrix<double, 3, 6>;
JPose PoseJacobian(const Vec3& p) {
    const Vec3 u = p.normalized();
    const Mat33 ju = (Mat33::Identity() - u * u.transpose()) / p.norm();
    JPose j; j.leftCols<3>() = ju; j.rightCols<3>() = -ju * Cross(p);
    return j;
}
bool FitPose(size_t frame, const Tracks& tracks, SE3& pose, double limit, std::string& reason) {
    size_t count = 0;
    for (const auto& t : tracks) count += t.initialized && At(t, frame);
    if (count < 15) {
        reason = "only " + std::to_string(count) + " shared points (need 15)"; return false;
    }
    for (int iteration = 0; iteration < 25; ++iteration) {
        Mat66 h = Mat66::Zero(); Vec6 rhs = Vec6::Zero();
        for (const auto& t : tracks) {
            const auto* o = At(t, frame);
            if (!t.initialized || !o) continue;
            const Vec3 p = pose * t.point;
            if (p.norm() < 1e-10) continue;
            const Vec3 r = p.normalized() - o->ray;
            const auto j = PoseJacobian(p);
            const double weight = std::min(1.0, limit * 0.5 / std::max(r.norm(), 1e-12));
            h += weight * j.transpose() * j; rhs -= weight * j.transpose() * r;
        }
        const Vec6 step = (h + 1e-9 * Mat66::Identity()).ldlt().solve(rhs);
        if (!step.allFinite() || step.tail<3>().norm() > 0.5) {
            reason = "unstable bearing-pose update with " + std::to_string(count) + " points"; return false;
        }
        pose = SE3::exp(step) * pose;
        if (step.norm() < 1e-8) break;
    }
    std::vector<double> errors;
    for (const auto& t : tracks) {
        const auto* o = At(t, frame);
        if (t.initialized && o) errors.push_back(Residual(pose, t.point, o->ray));
    }
    const double median = Median(errors);
    reason = "median bearing residual " + std::to_string(median) + " rad exceeds " +
             std::to_string(limit) + " with " + std::to_string(count) + " shared points";
    return median < limit;
}

double Cost(const Tracks& tracks, const Poses& poses, double huber) {
    double sum = 0;
    for (const auto& t : tracks) if (t.initialized) for (const auto& o : t.observations) {
        const Vec3 p = poses[o.frame] * t.point;
        const double e = p.norm() < 1e-10 ? 2 : (p.normalized() - o.ray).norm();
        sum += e < huber ? e * e : huber * (2 * e - huber);
    }
    return sum;
}

// Small, visual-only startup structure/pose refinement. The first camera is
// fixed; normalizing the long anchor baseline fixes the remaining scale gauge.
// This is isolated from the runtime frontend, backend, and metric alignment.
void Refine(Tracks& tracks, Poses& poses, size_t anchor, double limit) {
    const int columns = 6 * int(poses.size() - 1);
    const double huber = limit * 0.5;
    double damping = 1e-5;
    for (int iteration = 0; iteration < 25; ++iteration) {
        MatXX h = MatXX::Zero(columns, columns); VecX rhs = VecX::Zero(columns);
        struct PointSystem {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            Mat33 inverse = Mat33::Zero(); Vec3 rhs = Vec3::Zero();
            MatXX cross;
        };
        std::vector<PointSystem, Eigen::aligned_allocator<PointSystem>> systems(tracks.size());
        for (size_t k = 0; k < tracks.size(); ++k) {
            const auto& t = tracks[k];
            if (!t.initialized) continue;
            auto& system = systems[k]; system.cross = MatXX::Zero(columns, 3);
            Mat33 hp = Mat33::Zero();
            for (const auto& o : t.observations) {
                const auto& pose = poses[o.frame]; const Vec3 p = pose * t.point;
                if (p.norm() < 1e-10) continue;
                const Vec3 u = p.normalized(), r = u - o.ray;
                const double w = std::min(1.0, huber / std::max(r.norm(), 1e-12));
                const Mat33 jp = (Mat33::Identity() - u * u.transpose()) / p.norm() * pose.rotationMatrix();
                hp += w * jp.transpose() * jp; system.rhs -= w * jp.transpose() * r;
                if (!o.frame) continue;
                const int offset = 6 * int(o.frame - 1); const JPose jc = PoseJacobian(p);
                h.block<6, 6>(offset, offset) += w * jc.transpose() * jc;
                rhs.segment<6>(offset) -= w * jc.transpose() * r;
                system.cross.block<6, 3>(offset, 0) += w * jc.transpose() * jp;
            }
            hp.diagonal() += damping * hp.diagonal().cwiseMax(1e-8);
            system.inverse = hp.ldlt().solve(Mat33::Identity());
            h.noalias() -= system.cross * system.inverse * system.cross.transpose();
            rhs.noalias() -= system.cross * system.inverse * system.rhs;
        }
        h.diagonal() += damping * h.diagonal().cwiseMax(1e-8);
        const VecX step = h.ldlt().solve(rhs);
        if (!step.allFinite()) break;
        Poses candidate = poses; Tracks points = tracks;
        for (size_t i = 1; i < poses.size(); ++i) candidate[i] = SE3::exp(step.segment<6>(6 * (i - 1))) * poses[i];
        for (size_t k = 0; k < points.size(); ++k) if (points[k].initialized) {
            const auto& s = systems[k];
            points[k].point += s.inverse * (s.rhs - s.cross.transpose() * step);
        }
        if (Cost(points, candidate, huber) < Cost(tracks, poses, huber)) {
            poses.swap(candidate); tracks.swap(points); damping = std::max(1e-9, damping / 3);
            const double gauge = Centre(poses[anchor]).norm();
            if (!std::isfinite(gauge) || gauge < 1e-8) break;
            for (auto& p : poses) p.translation() /= gauge;
            for (auto& t : tracks) if (t.initialized) t.point /= gauge;
            if (step.norm() < 1e-7) break;
        } else {
            damping *= 10;
            if (damping > 1e4) break;
        }
    }
}

void FailPairs(VioVisualWindow& window, const std::string& reason) {
    window.translation_failure = reason;
    for (auto& d : window.pair_diagnostics) { d.status = StartupPairStatus::Rejected; d.reason = reason; }
}
}  // namespace

const char* StartupPairStatusName(StartupPairStatus status) {
    switch (status) {
        case StartupPairStatus::Accepted: return "accepted";
        case StartupPairStatus::Ambiguous: return "ambiguous";
        case StartupPairStatus::Rejected: return "rejected";
    }
    return "unknown";
}

VioVisualWindow BuildStartupVisualTrajectory(const std::vector<HiltiTrackingResult>& frames,
        const std::vector<BearingMotionResult>& motions, const Camera& camera,
        const ImuCalibration& calibration, const VioInitializerOptions& options) {
    VioVisualWindow window;
    const size_t n = frames.size();
    if (n < 4 || motions.size() + 1 != n) throw std::invalid_argument("startup frame/pair count invalid");
    const int camera_id = frames[0].stream == SensorStream::Cam0 ? 0 : 1;
    if (frames[0].stream != SensorStream::Cam0 && frames[0].stream != SensorStream::Cam1)
        throw std::invalid_argument("startup requires a camera stream");
    Poses poses(n); Tracks tracks; std::map<uint64_t, size_t> ids;
    window.pair_diagnostics.resize(n - 1);
    double largest_step = 0, largest_rotation = 0;
    for (size_t i = 0; i < n; ++i) {
        if (frames[i].stream != frames[0].stream) throw std::invalid_argument("mixed camera startup tracks");
        if (i) {
            const auto matches = MakeBearingMatches(frames[i - 1], frames[i], camera);
            if (!motions[i - 1].rotation_valid) {
                FailPairs(window, "missing visual rotation"); return window;
            }
            const SO3 relative(motions[i - 1].rotation);
            poses[i] = SE3(relative * poses[i - 1].so3(), Vec3::Zero());
            largest_rotation = std::max(largest_rotation, relative.log().norm());
            for (const auto& m : matches.matches) largest_step = std::max(largest_step, Angle(m.previous, m.current));
            window.pair_diagnostics[i - 1].pair = i;
        }
        VioVisualFrame f;
        f.timestamp_ns = CameraTimeOnImuClock(calibration, camera_id, frames[i].timestamp_ns);
        f.rotation_wb = poses[i].so3().inverse() * camera.pose().so3();
        window.frames.push_back(f);
        std::set<uint64_t> unique;
        for (const auto& feature : frames[i].features) {
            if (!unique.insert(feature.id).second) throw std::invalid_argument("duplicate startup feature ID");
            Vec3 ray;
            if (!camera.Unproject(Vec2(feature.pixel.x, feature.pixel.y), ray)) continue;
            if (!ids.count(feature.id)) { ids[feature.id] = tracks.size(); Track t; t.id = feature.id; tracks.push_back(t); }
            tracks[ids[feature.id]].observations.push_back({i, ray});
        }
    }
    window.stationary_visual_evidence = largest_step <= options.stationary_bearing_step_rad &&
                                         largest_rotation <= options.stationary_rotation_rad;
    // Long-baseline anchor candidates, never an inertial or consecutive-t sign prior.
    size_t anchor = 0; BearingMotionResult seed; double best = 0;
    for (size_t j = std::max(size_t(2), n / 6); j < n; ++j) {
        if (j != n - 1 && j % std::max(size_t(1), n / 6)) continue;
        BearingCorrespondences matches;
        for (const auto& t : tracks) {
            const auto* a = At(t, 0); const auto* b = At(t, j);
            if (a && b) { BearingCorrespondence m; m.feature_id = t.id; m.previous = a->ray; m.current = b->ray; matches.push_back(m); }
        }
        const auto m = EstimateBearingMotion(matches);
        const double score = m.median_parallax_rad * std::sqrt(double(m.inliers));
        if (m.translation_valid && m.rotation_valid && m.inliers >= 15 && m.median_parallax_rad >= 0.005 && score > best) {
            best = score; anchor = j; seed = m;
        }
    }
    if (!anchor) { FailPairs(window, "no reliable long-baseline anchor: pure rotation or low parallax"); return window; }
    window.anchor_frame = anchor;
    const SO3 correction = SO3(seed.rotation) * poses[anchor].so3().inverse();
    for (size_t i = 1; i < n; ++i) {
        poses[i].so3() = SO3::exp(double(i) / anchor * correction.log()) * poses[i].so3();
        const Vec3 centre = -(SO3(seed.rotation).inverse() * seed.translation_direction) * (double(i) / anchor);
        poses[i].translation() = -(poses[i].so3() * centre);
    }
    poses[anchor] = SE3(SO3(seed.rotation), seed.translation_direction);
    // Triangulate only the seed pair before bearing-PnP. The sign comes from
    // oriented positive rays; the subsequent shared structure checks every pair.
    for (auto& t : tracks) {
        const auto* a = At(t, 0); const auto* b = At(t, anchor);
        if (!a || !b) continue;
        Track pair; pair.observations = {*a, *b};
        const Vec3 wa = a->ray, wb = poses[anchor].so3().inverse() * b->ray;
        if (Angle(wa, wb) >= 0.005 && Triangulate(pair, poses, t.point, options.max_bearing_rms_rad)) t.initialized = true;
    }
    size_t seeds = 0;
    for (const auto& t : tracks) seeds += t.initialized;
    if (seeds < 15) { FailPairs(window, "insufficient positive-ray anchor structure"); return window; }
    for (size_t i = 1; i < n; ++i) {
        if (i > anchor) {
            // Extend the visual seed using the last visual velocity. This is an
            // initial guess only; bearing residuals determine the final pose.
            const double ratio = double(frames[i].timestamp_ns - frames[i - 1].timestamp_ns) /
                                 double(frames[i - 1].timestamp_ns - frames[i - 2].timestamp_ns);
            const Vec3 centre = Centre(poses[i - 1]) + ratio * (Centre(poses[i - 1]) - Centre(poses[i - 2]));
            poses[i].so3() = SO3(motions[i - 1].rotation) * poses[i - 1].so3();
            poses[i].translation() = -(poses[i].so3() * centre);
        }
        std::string pose_reason;
        if (i != anchor && !FitPose(i, tracks, poses[i], options.max_bearing_rms_rad, pose_reason)) {
            FailPairs(window, "bearing pose at frame " + std::to_string(i) + ": " + pose_reason); return window;
        }
        // New features need not survive all the way from frame zero. Expand
        // structure using only already recovered poses and the anchor pose.
        for (auto& t : tracks) if (i >= anchor && !t.initialized) {
            Track known;
            for (const auto& o : t.observations) if (o.frame <= i || o.frame == anchor) known.observations.push_back(o);
            Vec3 point;
            if (known.observations.size() >= 3 && Triangulate(known, poses, point, options.max_bearing_rms_rad)) {
                t.point = point; t.initialized = true;
            }
        }
    }
    for (auto& t : tracks) if (t.observations.size() >= 3) {
        Vec3 p;
        if (Triangulate(t, poses, p, options.max_bearing_rms_rad)) { t.point = p; t.initialized = true; }
    }
    Refine(tracks, poses, anchor, options.max_bearing_rms_rad);
    // Drop gross outlier tracks as a whole, then refine once more.
    for (auto& t : tracks) if (t.initialized) {
        size_t good = 0;
        for (const auto& o : t.observations) good += Residual(poses[o.frame], t.point, o.ray) < options.max_bearing_rms_rad;
        if (good < 0.8 * t.observations.size()) { t.initialized = false; ++window.rejected_tracks; }
    }
    Refine(tracks, poses, anchor, options.max_bearing_rms_rad);
    double sum = 0; size_t count = 0;
    for (const auto& t : tracks) if (t.initialized) {
        ++window.reconstruction_tracks;
        for (const auto& o : t.observations) {
            const double r = Residual(poses[o.frame], t.point, o.ray);
            if (r <= options.max_bearing_rms_rad) { sum += r * r; ++count; }
        }
    }
    window.bearing_rms_rad = count ? std::sqrt(sum / count) : std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) {
        window.frames[i].rotation_wb = poses[i].so3().inverse() * camera.pose().so3();
        window.frames[i].camera_position = Centre(poses[i]);
    }
    bool accepted = true;
    for (size_t i = 1; i < n; ++i) {
        auto& d = window.pair_diagnostics[i - 1];
        const Vec3 ci = Centre(poses[i - 1]), cj = Centre(poses[i]);
        const SE3 opposite(poses[i].so3(), -(poses[i].so3() * (2 * ci - cj)));
        std::vector<double> errors, opposite_errors, margins, parallax;
        size_t votes = 0;
        for (const auto& t : tracks) if (t.initialized && t.observations.size() >= 5) {
            const auto* a = At(t, i - 1); const auto* b = At(t, i);
            if (!a || !b) continue;
            Vec3 point;
            if (!Triangulate(t, poses, point, options.max_bearing_rms_rad, int(i - 1), int(i)) ||
                Residual(poses[i - 1], point, a->ray) > options.max_bearing_rms_rad) continue;
            const double e = Residual(poses[i], point, b->ray), other = Residual(opposite, point, b->ray);
            ++d.supporting_tracks;
            d.positive_rays += (poses[i] * point).dot(b->ray) > 0;
            d.opposite_positive_rays += (opposite * point).dot(b->ray) > 0;
            errors.push_back(e); opposite_errors.push_back(other); margins.push_back(other - e);
            votes += other > e;
            parallax.push_back(Angle(poses[i - 1].so3().inverse() * a->ray, poses[i].so3().inverse() * b->ray));
        }
        d.baseline = (cj - ci).norm(); d.parallax_rad = Median(parallax);
        d.residual_rad = Median(errors); d.opposite_residual_rad = Median(opposite_errors);
        d.residual_margin_rad = Median(margins);
        d.sign_vote_fraction = d.supporting_tracks ? double(votes) / d.supporting_tracks : 0;
        std::vector<double> deviation;
        for (double m : margins) deviation.push_back(std::abs(m - d.residual_margin_rad));
        const double uncertainty = d.supporting_tracks ? 3 * 1.4826 * Median(deviation) / std::sqrt(double(d.supporting_tracks)) : 1;
        d.status = StartupPairStatus::Ambiguous;
        if (d.supporting_tracks < 15) d.reason = "fewer than 15 independent multi-frame tracks";
        else if (d.baseline < 1e-6 || d.parallax_rad < 1e-4) d.reason = "pure/near-pure rotation or unresolved parallax";
        else if (d.positive_rays < 0.8 * d.supporting_tracks || d.residual_rad > options.max_bearing_rms_rad) {
            d.status = StartupPairStatus::Rejected; d.reason = "positive-ray or held-out bearing residual check failed";
        } else if (d.sign_vote_fraction < 0.7 || d.residual_margin_rad <= std::max(1e-5, uncertainty))
            d.reason = "opposite sign not distinguishable from visual residual uncertainty";
        else {
            d.status = StartupPairStatus::Accepted; d.reason = "held-out shared structure supports sign";
            Vec3 axis = motions[i - 1].translation_direction;
            if (!motions[i - 1].translation_valid || axis.norm() < 1e-10) {
                if (motions[i - 1].essential.norm() > 1e-10) {
                    Eigen::JacobiSVD<Mat33> svd(motions[i - 1].essential, Eigen::ComputeFullU);
                    axis = svd.matrixU().col(2);
                } else {
                    axis = -(poses[i].so3() * (cj - ci)).normalized();
                    Eigen::Index component;
                    axis.cwiseAbs().maxCoeff(&component);
                    if (axis[component] < 0) axis *= -1;
                }
            }
            d.selected_sign = axis.dot(-(poses[i].so3() * (cj - ci))) >= 0 ? 1 : -1;
        }
        if (i > 1) {
            const double dt = double(frames[i].timestamp_ns - frames[i - 1].timestamp_ns) * 1e-9;
            const double prev_dt = double(frames[i - 1].timestamp_ns - frames[i - 2].timestamp_ns) * 1e-9;
            const Vec3 v = (cj - ci) / dt, prev_v = (ci - Centre(poses[i - 2])) / prev_dt;
            d.continuity_score = (v - prev_v).norm() / std::max(v.norm() + prev_v.norm(), 1e-9);
        }
        if (d.status != StartupPairStatus::Accepted) {
            accepted = false;
            if (window.translation_failure.empty()) window.translation_failure = "pair " + std::to_string(i) + ": " + d.reason;
        }
    }
    window.positions_in_common_scale = accepted && window.reconstruction_tracks >= 15;
    // Geometric conditioning of the recovered long-baseline structure (angular
    // ray-intersection normal matrices), not an inertial alignment condition.
    window.visual_condition = 0;
    for (const auto& t : tracks) if (t.initialized) {
        Mat33 h = Mat33::Zero();
        for (const auto& o : t.observations) {
            const Vec3 ray = poses[o.frame].so3().inverse() * o.ray;
            h += Mat33::Identity() - ray * ray.transpose();
        }
        Eigen::JacobiSVD<Mat33> svd(h);
        window.visual_condition = std::max(window.visual_condition,
            std::sqrt(svd.singularValues()[0] / std::max(svd.singularValues()[2], 1e-15)));
    }
    if (window.visual_condition > options.max_visual_condition) {
        window.positions_in_common_scale = false;
        window.translation_failure = "shared structure exceeds visual condition limit";
    }
    return window;
}
}  // namespace myslam
