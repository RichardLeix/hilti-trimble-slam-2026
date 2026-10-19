#include "myslam/bearing_geometry.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace myslam {
namespace {
constexpr double kPi = 3.14159265358979323846;

bool UnitBearing(const Vec3& bearing) {
    return bearing.allFinite() && std::abs(bearing.norm() - 1) < 1e-6;
}

double Angle(const Vec3& a, const Vec3& b) {
    return std::atan2(a.cross(b).norm(), a.dot(b));
}

double MedianParallax(const Mat33& rotation, const BearingCorrespondences& matches,
                      const std::vector<uint8_t>& mask) {
    std::vector<double> angles;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (mask[i]) angles.push_back(Angle(rotation * matches[i].previous, matches[i].current));
    }
    if (angles.empty()) return 0;
    std::sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

struct Consensus {
    Mat33 model = Mat33::Zero();
    std::vector<uint8_t> mask;
    size_t count = 0;
    double cost = std::numeric_limits<double>::infinity();
};

Consensus Score(const Mat33& model, const BearingCorrespondences& matches,
                double threshold, bool rotation_only) {
    Consensus score;
    score.model = model;
    score.mask.resize(matches.size(), 0);
    score.cost = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& pair = matches[i];
        const double error = rotation_only ? Angle(model * pair.previous, pair.current)
                                           : BearingEpipolarError(model, pair.previous, pair.current);
        if (error <= threshold) {
            score.mask[i] = 1;
            ++score.count;
        }
        score.cost += std::min(error * error, threshold * threshold);
    }
    return score;
}

bool Better(const Consensus& candidate, const Consensus& best) {
    return candidate.count > best.count ||
           (candidate.count == best.count && candidate.cost < best.cost);
}

std::vector<size_t> InlierIndices(const std::vector<uint8_t>& mask) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < mask.size(); ++i) if (mask[i]) indices.push_back(i);
    return indices;
}

bool FitEssential(const BearingCorrespondences& matches,
                  const std::vector<size_t>& indices, Mat33& essential) {
    if (indices.size() < 8) return false;
    Eigen::MatrixXd design(indices.size(), 9);
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& pair = matches[indices[i]];
        // b_current^T E b_previous = 0, with row-major coefficients of E.
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                design(i, 3 * r + c) = pair.current[r] * pair.previous[c];
            }
        }
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> fit(design, Eigen::ComputeFullV);
    // Eight independent constraints are needed. Pure rotation, repeated rays
    // and exact planar configurations can leave a larger nullspace.
    if (fit.singularValues()[7] <= fit.singularValues()[0] * 1e-8) return false;
    const Eigen::VectorXd coefficients = fit.matrixV().col(8);
    Mat33 linear;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) linear(r, c) = coefficients[3 * r + c];
    }
    const Eigen::JacobiSVD<Mat33> svd(linear, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()[1] < 1e-10) return false;
    // Enforce two equal nonzero singular values and one zero singular value.
    essential = svd.matrixU() * Vec3(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
    essential.normalize();
    return essential.allFinite();
}

bool FitRotation(const BearingCorrespondences& matches,
                 const std::vector<size_t>& indices, Mat33& rotation) {
    if (indices.size() < 2) return false;
    Mat33 covariance = Mat33::Zero();
    for (size_t i : indices) covariance += matches[i].current * matches[i].previous.transpose();
    const Eigen::JacobiSVD<Mat33> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()[1] <= svd.singularValues()[0] * 1e-8) return false;
    Mat33 correction = Mat33::Identity();
    correction(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0 ? -1 : 1;
    rotation = svd.matrixU() * correction * svd.matrixV().transpose();
    return rotation.allFinite();
}

Consensus Ransac(const BearingCorrespondences& matches,
                 const BearingGeometryOptions& options, bool rotation_only) {
    const size_t sample_size = rotation_only ? 2 : 8;
    const double threshold = rotation_only ? options.rotation_only_threshold_rad
                                           : options.epipolar_threshold_rad;
    std::mt19937 generator(options.random_seed);
    std::uniform_int_distribution<size_t> random_index(0, matches.size() - 1);
    int limit = options.max_iterations;
    Consensus best;
    for (int iteration = 0; iteration < limit; ++iteration) {
        std::vector<size_t> sample;
        while (sample.size() < sample_size) {
            const size_t index = random_index(generator);
            if (std::find(sample.begin(), sample.end(), index) == sample.end()) sample.push_back(index);
        }
        Mat33 model;
        const bool fitted = rotation_only ? FitRotation(matches, sample, model)
                                          : FitEssential(matches, sample, model);
        if (!fitted) continue;
        auto score = Score(model, matches, threshold, rotation_only);
        if (!Better(score, best)) continue;
        best = std::move(score);
        const double ratio = double(best.count) / matches.size();
        const double probability = std::pow(ratio, sample_size);
        if (probability > 0) {
            const double required = probability >= 1 ? 1 :
                std::ceil(std::log(1 - options.confidence) / std::log1p(-probability));
            // Keep some sampling even for near-perfect consensus on noisy data.
            if (required < limit) limit = std::max(std::min(50, options.max_iterations), int(required));
        }
    }
    // Refit using the consensus rather than trusting a noisy minimal sample.
    for (int refit = 0; refit < 3 && best.count >= sample_size; ++refit) {
        Mat33 model;
        const auto indices = InlierIndices(best.mask);
        const bool fitted = rotation_only ? FitRotation(matches, indices, model)
                                          : FitEssential(matches, indices, model);
        if (!fitted) break;
        auto score = Score(model, matches, threshold, rotation_only);
        if (!Better(score, best)) break;
        best = std::move(score);
    }
    return best;
}

void ValidateOptions(const BearingGeometryOptions& options) {
    if (!std::isfinite(options.epipolar_threshold_rad) || options.epipolar_threshold_rad <= 0 ||
        options.epipolar_threshold_rad >= kPi / 2 ||
        !std::isfinite(options.rotation_only_threshold_rad) || options.rotation_only_threshold_rad <= 0 ||
        options.rotation_only_threshold_rad > options.epipolar_threshold_rad ||
        !std::isfinite(options.min_parallax_rad) || options.min_parallax_rad <= 0 ||
        options.min_parallax_rad >= kPi / 2 || !std::isfinite(options.confidence) ||
        options.confidence <= 0 || options.confidence >= 1 || options.max_iterations <= 0 ||
        options.min_inliers < 8 || !std::isfinite(options.min_positive_ray_fraction) ||
        options.min_positive_ray_fraction <= 0.5 || options.min_positive_ray_fraction > 1) {
        throw std::invalid_argument("invalid bearing geometry options");
    }
}
}  // namespace

BearingMatchSet MakeBearingMatches(const HiltiTrackingResult& previous,
                                   const HiltiTrackingResult& current,
                                   const Camera& camera) {
    if (previous.stream != current.stream || current.stream == SensorStream::Imu ||
        !current.has_previous || current.previous_timestamp_ns != previous.timestamp_ns ||
        current.timestamp_ns <= previous.timestamp_ns) {
        throw std::invalid_argument("bearing matches require consecutive results from the same camera");
    }
    std::map<uint64_t, const TemporalFeatureTrack*> old;
    for (const auto& track : previous.features) {
        if (!old.emplace(track.id, &track).second) throw std::invalid_argument("duplicate previous feature ID");
    }
    std::set<uint64_t> seen;
    BearingMatchSet result;
    for (const auto& track : current.features) {
        if (!seen.insert(track.id).second) throw std::invalid_argument("duplicate current feature ID");
        if (track.age == 1) continue;  // Replenished corners have no temporal match.
        const auto found = old.find(track.id);
        if (found == old.end() || track.age != found->second->age + 1) {
            throw std::invalid_argument("inconsistent persistent feature ID or age");
        }
        const auto& source = found->second->pixel;
        BearingCorrespondence pair;
        pair.feature_id = track.id;
        if (!camera.Unproject(Vec2(source.x, source.y), pair.previous) ||
            !camera.Unproject(Vec2(track.pixel.x, track.pixel.y), pair.current)) {
            ++result.rejected_pixels;
            continue;
        }
        result.matches.push_back(pair);
    }
    return result;
}

double BearingEpipolarError(const Mat33& essential, const Vec3& previous,
                            const Vec3& current) {
    const double infinity = std::numeric_limits<double>::infinity();
    if (!essential.allFinite() || !UnitBearing(previous) || !UnitBearing(current)) return infinity;
    const double scale = essential.cwiseAbs().maxCoeff();
    if (scale == 0) return infinity;
    const Mat33 normalized = essential / scale;
    const Vec3 current_plane = normalized * previous;
    const Vec3 previous_plane = normalized.transpose() * current;
    const double denominator = std::min(current_plane.norm(), previous_plane.norm());
    if (denominator <= 1e-12) return infinity;
    const double sine = std::abs(current.dot(current_plane)) / denominator;
    return std::asin(std::min(1.0, sine));
}

const char* BearingMotionStatusName(BearingMotionStatus status) {
    switch (status) {
        case BearingMotionStatus::Success: return "success";
        case BearingMotionStatus::RotationOnly: return "rotation_only_translation_unobservable";
        case BearingMotionStatus::InsufficientCorrespondences: return "insufficient_correspondences";
        case BearingMotionStatus::Degenerate: return "degenerate";
        case BearingMotionStatus::AmbiguousMotion: return "ambiguous_motion";
    }
    return "unknown";
}

BearingMotionResult EstimateBearingMotion(const BearingCorrespondences& matches,
                                          const BearingGeometryOptions& options) {
    ValidateOptions(options);
    std::set<uint64_t> ids;
    for (const auto& pair : matches) {
        if (!UnitBearing(pair.previous) || !UnitBearing(pair.current) ||
            !ids.insert(pair.feature_id).second) {
            throw std::invalid_argument("geometry requires finite unit bearings and unique IDs");
        }
    }
    BearingMotionResult result;
    result.inlier_mask.resize(matches.size(), 0);
    if (matches.size() < options.min_inliers) return result;
    const Consensus essential = Ransac(matches, options, false);
    const Consensus rotation_only = Ransac(matches, options, true);
    // A stricter rotation model explaining nearly the whole epipolar consensus
    // signals that translation cannot be distinguished reliably at this noise level.
    if (rotation_only.count >= options.min_inliers && rotation_only.count * 2 >= matches.size() &&
        rotation_only.count >= 0.95 * std::max(essential.count, options.min_inliers)) {
        result.status = BearingMotionStatus::RotationOnly;
        result.rotation_valid = true;
        result.rotation = rotation_only.model;
        result.inlier_mask = rotation_only.mask;
        result.inliers = rotation_only.count;
        return result;
    }
    result.status = BearingMotionStatus::Degenerate;
    if (essential.count < options.min_inliers) return result;
    result.essential = essential.model;
    result.inlier_mask = essential.mask;
    result.inliers = essential.count;

    const Eigen::JacobiSVD<Mat33> svd(essential.model, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat33 u = svd.matrixU(), v = svd.matrixV();
    if (u.determinant() < 0) u.col(2) *= -1;
    if (v.determinant() < 0) v.col(2) *= -1;
    Mat33 w;
    w << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    const std::array<Mat33, 2> rotations{{u * w * v.transpose(), u * w.transpose() * v.transpose()}};
    struct Candidate {
        Vec3 translation = Vec3::Zero();
        std::vector<uint8_t> mask;
        size_t count = 0;
        size_t parallax_count = 0;
    };
    std::array<Candidate, 4> candidates;
    std::array<size_t, 2> rotation_support{{0, 0}};
    for (size_t r = 0; r < rotations.size(); ++r) {
        const auto& rotation = rotations[r];
        for (size_t s = 0; s < 2; ++s) {
            auto& candidate = candidates[2 * r + s];
            candidate.translation = (s == 0 ? 1 : -1) * u.col(2);
            candidate.mask.resize(matches.size(), 0);
            for (size_t i = 0; i < matches.size(); ++i) {
                if (!essential.mask[i]) continue;
                const Vec3 a = rotation * matches[i].previous;
                const Vec3& b = matches[i].current;
                const Vec3 normal = a.cross(b);
                // Positive oriented rays: signs of the two range numerators in
                // lambda_current*b = lambda_previous*a + t. No ranges or 3D
                // point coordinates are reconstructed. Use ray range, not Z.
                if (normal.norm() <= 1e-12 ||
                    -normal.dot(candidate.translation.cross(b)) <= 0 ||
                    -normal.dot(candidate.translation.cross(a)) <= 0) continue;
                candidate.mask[i] = 1;
                ++candidate.count;
                if (normal.norm() >= std::sin(options.min_parallax_rad)) ++candidate.parallax_count;
            }
            rotation_support[r] += candidate.count;
        }
    }
    result.status = BearingMotionStatus::AmbiguousMotion;
    const size_t r = rotation_support[1] > rotation_support[0] ? 1 : 0;
    const double margin = std::max(1.0, 0.05 * essential.count);
    if (rotation_support[r] < options.min_positive_ray_fraction * essential.count ||
        rotation_support[r] - rotation_support[1 - r] < margin) return result;
    result.rotation = rotations[r];
    result.rotation_valid = true;
    result.inliers = rotation_support[r];
    for (size_t i = 0; i < matches.size(); ++i) {
        result.inlier_mask[i] = candidates[2 * r].mask[i] || candidates[2 * r + 1].mask[i];
    }
    result.median_parallax_rad = MedianParallax(result.rotation, matches, result.inlier_mask);
    const size_t winner = candidates[2 * r + 1].count > candidates[2 * r].count ? 1 : 0;
    const auto& best = candidates[2 * r + winner];
    const auto& other = candidates[2 * r + 1 - winner];
    result.parallax_inliers = best.parallax_count;
    // Sign support uses ALL positive rays, including distant features. The
    // separate parallax count checks whether enough rays constrain translation.
    // A large number of distant, valid features must not veto the nearby ones.
    if (best.count < options.min_positive_ray_fraction * essential.count ||
        best.count - other.count < margin || best.parallax_count < options.min_inliers) {
        return result;
    }
    result.inlier_mask = best.mask;
    result.inliers = best.count;
    result.median_parallax_rad = MedianParallax(result.rotation, matches, result.inlier_mask);
    result.translation_direction = best.translation.normalized();
    result.status = BearingMotionStatus::Success;
    result.rotation_valid = true;
    result.translation_valid = true;
    return result;
}

}  // namespace myslam
