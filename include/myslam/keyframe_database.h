#ifndef MYSLAM_KEYFRAME_DATABASE_H
#define MYSLAM_KEYFRAME_DATABASE_H

#include "myslam/hilti_backend.h"

namespace myslam {
struct LoopObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cv::Point2f pixel;
    Vec3 bearing = Vec3::Zero();
    bool has_track = false;
    uint64_t track_id = 0; // Camera-local identity; survives active-window eviction.
    bool has_point = false;
    Vec3 point_body = Vec3::Zero(); // Metric, in THIS archived reference body.
};
struct LoopKeyframe {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t id = 0, backend_version = 0;
    int64_t timestamp_ns = 0;
    SE3 optimized_body_pose; // T_wb; archive metadata, never a correction target.
    std::array<SE3,2> camera_from_body; // Includes calibrated exposure delta.
    std::array<HiltiAlignedVector<LoopObservation>,2> observations;
    std::array<cv::Mat,2> descriptors; // ORB CV_8UC1, 32 columns; rows match observations.
};
struct LoopCandidate {
    uint64_t keyframe_id = 0;
    double score = 0, time_separation_s = 0;
    uint64_t keyframe_separation = 0;
};
struct LoopPoseMetadata {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t id = 0, backend_version = 0;
    int64_t timestamp_ns = 0;
    SE3 optimized_body_pose; // Frozen local T_wb; no descriptor/landmark aliases.
};

// Owns full value snapshots, independent of HiltiLocalMap and its factor archive.
// All methods serialize vocabulary/index access. Reads return deep descriptor
// copies so a caller cannot mutate archival cv::Mat storage through an alias.
class KeyframeDatabase {
   public:
    explicit KeyframeDatabase(const std::string& vocabulary_path);
    ~KeyframeDatabase();
    KeyframeDatabase(const KeyframeDatabase&) = delete;
    KeyframeDatabase& operator=(const KeyframeDatabase&) = delete;
    void Insert(const LoopKeyframe& frame);
    void UpdateGeometry(const LoopKeyframe& frame); // Identity/appearance immutable.
    std::shared_ptr<const LoopKeyframe> Get(uint64_t id) const;
    std::vector<LoopCandidate> Query(const LoopKeyframe& query, size_t maximum,
        uint64_t minimum_keyframes, double minimum_seconds, double minimum_score) const;
    size_t Size() const;
    HiltiAlignedVector<LoopPoseMetadata> PoseMetadata() const;
   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace myslam
#endif
