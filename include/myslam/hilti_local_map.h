#ifndef MYSLAM_HILTI_LOCAL_MAP_H
#define MYSLAM_HILTI_LOCAL_MAP_H
#include "myslam/hilti_backend.h"

namespace myslam {
// Frontend-owned Hilti lifecycle. No legacy MapPoint/Feature weak observations.
// Chronological eviction removes ONLY active keyframe membership. Full bearing
// archives and landmark identities survive independently of marginalization.
class HiltiLocalMap {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    HiltiLocalMap(const HiltiCameras& cameras, const ImuCalibration& calibration,
                  const HiltiBackendOptions& options, const Vec3& gravity_w);
    void Seed(HiltiLandmarkId id, const Vec3& point_w);
    void AddKeyframe(int64_t timestamp, const NavState& state,
        const std::array<int64_t,2>& exposures, const std::array<HiltiTrackingResult,2>& tracks,
        const std::vector<ImuSample>& imu);
    HiltiBackendJob Snapshot(uint64_t version, uint64_t base_version);
    void Apply(const HiltiBackendResult& correction,
               const HiltiAlignedVector<HiltiKeyframeState>& replayed);
    HiltiPoseObservations PoseObservations(const std::array<HiltiTrackingResult,2>& tracks) const;
    size_t WindowSize() const { return window_.size(); }
    size_t ArchiveObservations() const;
    size_t ArchiveLandmarks() const { return archive_.size(); }
    const HiltiAlignedVector<HiltiKeyframeState>& Window() const { return window_; }
    const HiltiAlignedVector<HiltiKeyframeState>& ArchivedKeyframes() const { return archived_keyframes_; }
   private:
    void Triangulate(HiltiLandmark& landmark);
    void PruneImu();
    HiltiCameras cameras_;
    ImuCalibration calibration_;
    HiltiBackendOptions options_;
    Vec3 gravity_w_;
    HiltiAlignedVector<HiltiKeyframeState> window_;
    HiltiAlignedVector<HiltiKeyframeState> archived_keyframes_; // Pose/clock metadata retained after active eviction.
    std::map<HiltiLandmarkId, HiltiLandmark, std::less<HiltiLandmarkId>,
        Eigen::aligned_allocator<std::pair<const HiltiLandmarkId,HiltiLandmark>>> archive_;
    std::vector<ImuSample> samples_; // Also brackets quarantined states until a correction commits.
    std::shared_ptr<const HiltiBackendJob> factor_model_;
    std::set<HiltiObservationId> absorbed_observations_;
    std::set<HiltiLandmarkId> retired_landmarks_;
};
} // namespace myslam
#endif
