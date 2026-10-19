#include "myslam/hilti_local_map.h"
#include "myslam/hilti_rolling_shutter.h"
#include <iomanip>
#include <sstream>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace myslam {
HiltiLocalMap::HiltiLocalMap(const HiltiCameras& cameras, const ImuCalibration& calibration,
        const HiltiBackendOptions& options, const Vec3& gravity)
    : cameras_(cameras), calibration_(calibration), options_(options), gravity_w_(gravity) {}
void HiltiLocalMap::Seed(HiltiLandmarkId id, const Vec3& point) {
    if (id.first < 0 || id.first > 1 || !point.allFinite()) throw std::invalid_argument("invalid Hilti landmark seed");
    auto& landmark = archive_[id]; landmark.id = id; landmark.position_w = point;
    landmark.initialized = true; landmark.usable_for_tracking = true;
}
void HiltiLocalMap::AddKeyframe(int64_t timestamp, const NavState& state,
        const std::array<int64_t,2>& exposures, const std::array<HiltiTrackingResult,2>& tracks,
        const std::vector<ImuSample>& imu) {
    if (!window_.empty() && timestamp <= window_.back().timestamp_ns)
        throw std::invalid_argument("Hilti keyframes must be strictly chronological");
    for (const auto& sample : imu) {
        auto it = std::lower_bound(samples_.begin(), samples_.end(), sample.timestamp_ns,
            [](const ImuSample& s,int64_t t){return s.timestamp_ns<t;});
        if (it == samples_.end()) samples_.push_back(sample);
        else if (it->timestamp_ns == sample.timestamp_ns) {
            if (it->angular_velocity != sample.angular_velocity || it->linear_acceleration != sample.linear_acceleration)
                throw std::invalid_argument("Hilti local IMU snapshots disagree");
        } else samples_.insert(it,sample);
    }
    HiltiKeyframeState frame; frame.timestamp_ns = timestamp; frame.state = state; frame.exposure_timestamps_ns = exposures;
    for(int k=0;k<2;++k) if(exposures[k]!=timestamp)
        frame.exposure_deltas[k] = PreintegrateImu(samples_,std::min(timestamp,exposures[k]),
            std::max(timestamp,exposures[k]),calibration_,state.bias);
    if(options_.center_row_rolling_shutter)for(int k=0;k<2;++k)
        frame.exposure_angular_velocity[k]=HiltiInterpolateGyro(samples_,exposures[k]);
    window_.push_back(frame);
    archived_keyframes_.push_back(frame);
    if(window_.size()>options_.window_size) window_.erase(window_.begin()); // Chronology only.
    PruneImu();
    for(int k=0;k<2;++k) for(const auto& feature:tracks[k].features) {
        Vec3 bearing;
        if(!cameras_[k]->Unproject(Vec2(feature.pixel.x,feature.pixel.y),bearing)) continue;
        const HiltiLandmarkId id{k,feature.id}; auto& landmark=archive_[id]; landmark.id=id;
        HiltiBearingObservation o; o.timestamp_ns=timestamp; o.bearing=bearing;
        landmark.observations.push_back(o);
        if(!landmark.initialized) Triangulate(landmark);
    }
}
void HiltiLocalMap::Triangulate(HiltiLandmark& landmark) {
    HiltiAlignedVector<SE3> poses; HiltiAlignedVector<Vec3> rays;
    for(const auto& frame:window_) {
        auto o=std::lower_bound(landmark.observations.begin(),landmark.observations.end(),frame.timestamp_ns,
            [](const HiltiBearingObservation& o,int64_t t){return o.timestamp_ns<t;});
        if(o==landmark.observations.end() || o->timestamp_ns!=frame.timestamp_ns) continue;
        poses.push_back(HiltiCameraWorldPose(frame,landmark.id.first,cameras_,gravity_w_)); rays.push_back(o->bearing);
    }
    if(poses.size()<3) return;
    Mat33 h=Mat33::Zero(); Vec3 rhs=Vec3::Zero(); double parallax=0;
    const Vec3 first=poses[0].so3()*rays[0];
    for(size_t i=0;i<poses.size();++i) {
        const Vec3 ray=poses[i].so3()*rays[i];
        parallax=std::max(parallax,std::atan2(first.cross(ray).norm(),first.dot(ray)));
        const Mat33 a=Mat33::Identity()-ray*ray.transpose(); h+=a; rhs+=a*poses[i].translation();
    }
    if(parallax<0.005) return;
    Eigen::SelfAdjointEigenSolver<Mat33> eigen(h);
    if(eigen.info()!=Eigen::Success || eigen.eigenvalues()[0]<1e-9 ||
        eigen.eigenvalues()[2]/eigen.eigenvalues()[0]>1e8) return;
    const Vec3 point=h.ldlt().solve(rhs); if(!point.allFinite()) return;
    size_t good=0;
    for(size_t i=0;i<poses.size();++i) {
        const Vec3 p=poses[i].inverse()*point; Vec2 pixel;
        if(p.norm()>1e-8 && p.dot(rays[i])>0 && cameras_[landmark.id.first]->Project(p,pixel) &&
            std::atan2(p.normalized().cross(rays[i]).norm(),p.normalized().dot(rays[i]))<options_.max_landmark_bearing_rad) ++good;
    }
    if(good<3 || good<0.8*poses.size()) return;
    landmark.position_w=point; landmark.initialized=true;landmark.usable_for_tracking=true;
    
}
void HiltiLocalMap::PruneImu() {
    if(window_.empty())return;
    const auto& oldest=options_.enable_marginalization?(factor_model_?factor_model_->keyframes.front():archived_keyframes_.front()):window_.front();
    auto first=std::lower_bound(samples_.begin(),samples_.end(),
        std::min(oldest.timestamp_ns,oldest.exposure_timestamps_ns[1]),
        [](const ImuSample& s,int64_t t){return s.timestamp_ns<t;});
    if(first!=samples_.begin())--first;
    samples_.erase(samples_.begin(),first);
}

void HiltiLocalMap::Apply(const HiltiBackendResult& correction,const HiltiAlignedVector<HiltiKeyframeState>& replayed) {
    if(options_.enable_marginalization && correction.factor_snapshot) {
        factor_model_=correction.factor_snapshot;
        absorbed_observations_.insert(correction.absorbed_observations.begin(),correction.absorbed_observations.end());
        retired_landmarks_.insert(correction.retired_landmarks.begin(),correction.retired_landmarks.end());
    }
    // Archive membership does not imply a trustworthy live frontend reference.
    // Startup points that were not optimized with this state batch must not be
    // mixed with the newly corrected local geometry. They remain BA candidates.
    for(auto& entry:archive_) entry.second.usable_for_tracking=false;
    for(const auto& landmark:correction.landmarks) {
        auto it=archive_.find(landmark.id);
        if(it!=archive_.end()) {
            it->second.position_w=landmark.position_w;it->second.initialized=true;size_t good=0;
            for(const auto& obs:landmark.observations) {
                auto f=std::lower_bound(correction.keyframes.begin(),correction.keyframes.end(),obs.timestamp_ns,
                    [](const HiltiKeyframeState& f,int64_t t){return f.timestamp_ns<t;});
                if(f==correction.keyframes.end() || f->timestamp_ns!=obs.timestamp_ns)continue;
                const Vec3 p=HiltiCameraWorldPose(*f,landmark.id.first,cameras_,gravity_w_).inverse()*landmark.position_w;Vec2 pixel;
                if(p.norm()>1e-8 && p.dot(obs.bearing)>0 && cameras_[landmark.id.first]->Project(p,pixel) &&
                    std::atan2(p.normalized().cross(obs.bearing).norm(),p.normalized().dot(obs.bearing))<=options_.max_landmark_bearing_rad)++good;
            }
            it->second.usable_for_tracking=good>=2 && good>=0.8*landmark.observations.size();
        }
    }
    for(auto& frame:archived_keyframes_) {
        for(const auto& optimized:correction.keyframes)if(optimized.timestamp_ns==frame.timestamp_ns)frame=optimized;
        for(const auto& replay:replayed)if(replay.timestamp_ns==frame.timestamp_ns)frame.state=replay.state;
    }
    for(auto& frame:window_) {
        for(const auto& optimized:correction.keyframes) if(optimized.timestamp_ns==frame.timestamp_ns) frame.state=optimized.state;
        for(const auto& replay:replayed) if(replay.timestamp_ns==frame.timestamp_ns) frame.state=replay.state;
        for(int k=0;k<2;++k) if(frame.exposure_timestamps_ns[k]!=frame.timestamp_ns)
            frame.exposure_deltas[k]=PreintegrateImu(samples_,std::min(frame.timestamp_ns,frame.exposure_timestamps_ns[k]),
                std::max(frame.timestamp_ns,frame.exposure_timestamps_ns[k]),calibration_,frame.state.bias);
    }
    PruneImu();
}
HiltiPoseObservations HiltiLocalMap::PoseObservations(const std::array<HiltiTrackingResult,2>& tracks) const {
    HiltiPoseObservations observations;
    for(int k=0;k<2;++k) for(const auto& f:tracks[k].features) {
        auto it=archive_.find({k,f.id}); Vec3 ray;
        if(it==archive_.end() || !it->second.initialized || !it->second.usable_for_tracking ||
            !cameras_[k]->Unproject(Vec2(f.pixel.x,f.pixel.y),ray)) continue;
        HiltiPoseObservation o;o.camera_id=k;o.feature_id=f.id;o.point_w=it->second.position_w;o.bearing=ray;observations.push_back(o);
    }
    return observations;
}
size_t HiltiLocalMap::ArchiveObservations() const {
    size_t count=0;for(const auto& landmark:archive_) count+=landmark.second.observations.size();return count;
}
} // namespace myslam
