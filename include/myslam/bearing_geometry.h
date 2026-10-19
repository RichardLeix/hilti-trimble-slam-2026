#ifndef MYSLAM_BEARING_GEOMETRY_H
#define MYSLAM_BEARING_GEOMETRY_H

#include "myslam/hilti_temporal_tracker.h"

namespace myslam {

struct BearingCorrespondence {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t feature_id = 0;
    Vec3 previous = Vec3::Zero();
    Vec3 current = Vec3::Zero();
};
using BearingCorrespondences = std::vector<
    BearingCorrespondence, Eigen::aligned_allocator<BearingCorrespondence>>;

struct BearingMatchSet {
    BearingCorrespondences matches;
    size_t rejected_pixels = 0;
};

// Match persistent IDs from consecutive results for ONE camera, then call its
// checked Unproject API. New corners have no correspondence. Invalid pixels are
// excluded. Throws for mixed cameras, nonconsecutive timestamps or duplicate IDs.
BearingMatchSet MakeBearingMatches(const HiltiTrackingResult& previous,
                                   const HiltiTrackingResult& current,
                                   const Camera& camera);

// Symmetric spherical point-to-epipolar-plane error, in radians:
// max(asin(|b2^T E b1| / ||E b1||), asin(|b2^T E b1| / ||E^T b2||)).
// Does not divide bearings by Z; supports lateral and rear-facing rays.
// Invalid input / undefined epipolar planes return infinity.
double BearingEpipolarError(const Mat33& essential, const Vec3& previous,
                            const Vec3& current);

struct BearingGeometryOptions {
    double epipolar_threshold_rad = 0.003;
    double rotation_only_threshold_rad = 0.001;
    double min_parallax_rad = 0.001;
    double confidence = 0.999;
    int max_iterations = 2000;
    size_t min_inliers = 12;  // At least eight are necessary for the linear fit.
    double min_positive_ray_fraction = 0.6;
    uint32_t random_seed = 20261004;
};

enum class BearingMotionStatus {
    Success,
    RotationOnly,  // Translation is below visual resolution, or pure rotation.
    InsufficientCorrespondences,
    Degenerate,
    AmbiguousMotion
};
const char* BearingMotionStatusName(BearingMotionStatus status);

struct BearingMotionResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    BearingMotionStatus status = BearingMotionStatus::InsufficientCorrespondences;
    bool rotation_valid = false;
    bool translation_valid = false;
    // Convention: p_current = R_current_previous * p_previous + scale * t.
    // t is a UNIT direction in the CURRENT camera frame, not a metric displacement.
    Mat33 rotation = Mat33::Identity();
    Vec3 translation_direction = Vec3::Zero();
    Mat33 essential = Mat33::Zero();  // E = [t]_x R, up to nonzero scale.
    std::vector<uint8_t> inlier_mask;  // Same indices as input correspondences.
    size_t inliers = 0;
    size_t parallax_inliers = 0;  // Positive-ray inliers above min_parallax_rad.
    double median_parallax_rad = 0;
};

// Bearing-space eight-point RANSAC, essential-manifold projection and consensus
// refit. Pose selection uses ray-orientation signs, without constructing 3D
// points. A robust rotation-only check prevents inventing an unobservable t.
// Finite unit vectors are required; invalid bearings/options throw.
BearingMotionResult EstimateBearingMotion(
    const BearingCorrespondences& matches,
    const BearingGeometryOptions& options = BearingGeometryOptions());

}  // namespace myslam

#endif  // MYSLAM_BEARING_GEOMETRY_H
