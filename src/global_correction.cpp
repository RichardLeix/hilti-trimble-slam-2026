#include "myslam/global_correction.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace myslam {
namespace {
bool ValidPose(const SE3& pose) {
    return pose.matrix().allFinite() && std::abs(pose.unit_quaternion().squaredNorm()-1)<1e-10;
}
bool Chronological(const PoseGraphPose& a,const PoseGraphPose& b) { return a.timestamp_ns<b.timestamp_ns; }
}
SE3 GlobalCorrectionBatch::LiveCorrection(int64_t timestamp_ns) const {
    if(!has_solution || live_smoothing_seconds==0)return local_to_global;
    const double elapsed=double((static_cast<long double>(timestamp_ns)-publication_timestamp_ns)*1e-9L);
    double fraction=std::max(0.0,std::min(1.0,elapsed/live_smoothing_seconds));
    if(fraction==0)return live_start_correction;
    if(fraction==1)return local_to_global;
    fraction=fraction*fraction*(3-2*fraction); // C1 endpoints, user-facing output only.
    const SO3 rotation=live_start_correction.so3()*SO3::exp(
        fraction*(live_start_correction.so3().inverse()*local_to_global.so3()).log());
    return SE3(rotation,(1-fraction)*live_start_correction.translation()+fraction*local_to_global.translation());
}
const PoseGraphPose* GlobalCorrectionBatch::FindGlobalKeyframe(uint64_t keyframe_id) const {
    for(const auto& p:global_keyframes)if(p.keyframe_id==keyframe_id)return &p;
    return nullptr;
}
GlobalNavigation ProjectGlobalNavigation(const std::shared_ptr<const GlobalCorrectionBatch>& batch,
    const NavState& local,const Vec3& gravity_w,int64_t timestamp_ns,bool smooth_live) {
    if(!batch || !ValidPose(local.Pose()) || !local.velocity_w.allFinite() || !gravity_w.allFinite() ||
        !local.bias.gyroscope.allFinite() || !local.bias.accelerometer.allFinite())
        throw std::invalid_argument("global output requires a finite local state, gravity and correction batch");
    GlobalNavigation output;output.timestamp_ns=timestamp_ns;output.correction_version=batch->version;
    output.correction=batch;output.smoothed=smooth_live && batch->live_smoothing_seconds>0 && batch->has_solution;
    const SE3 correction=smooth_live?batch->LiveCorrection(timestamp_ns):batch->local_to_global;
    output.body_pose_global=correction*local.Pose();
    output.velocity_global=correction.so3()*local.velocity_w;
    output.gravity_global=correction.so3()*gravity_w;
    output.bias_body=local.bias;
    if(!ValidPose(output.body_pose_global) || !output.velocity_global.allFinite() || !output.gravity_global.allFinite())
        throw std::overflow_error("nonfinite global trajectory output");
    return output;
}
GlobalCorrection::GlobalCorrection(const GlobalCorrectionOptions& options):options_(options) {
    if(!std::isfinite(options.live_smoothing_seconds) || options.live_smoothing_seconds<0 || options.live_smoothing_seconds>3600)
        throw std::invalid_argument("invalid global live smoothing duration");
    current_=std::make_shared<const GlobalCorrectionBatch>();
}
std::shared_ptr<const GlobalCorrectionBatch> GlobalCorrection::Snapshot() const {
    return std::atomic_load_explicit(&current_,std::memory_order_acquire);
}
bool GlobalCorrection::Publish(const PoseGraphInput& local,const PoseGraphResult& global,
    uint64_t graph_generation,int64_t publication_timestamp_ns) {
    if(!global.success)return false;
    if(!graph_generation || local.poses.empty() || local.poses.size()!=global.poses.size())
        throw std::invalid_argument("global publication requires a complete successful graph");
    auto batch=std::make_shared<GlobalCorrectionBatch>();
    batch->local_keyframes=local.poses;batch->global_keyframes=global.poses;
    std::sort(batch->local_keyframes.begin(),batch->local_keyframes.end(),Chronological);
    std::sort(batch->global_keyframes.begin(),batch->global_keyframes.end(),Chronological);
    std::set<uint64_t> ids;
    for(size_t k=0;k<batch->local_keyframes.size();++k) {
        const auto& a=batch->local_keyframes[k];const auto& b=batch->global_keyframes[k];
        if(!ValidPose(a.body_pose) || !ValidPose(b.body_pose) || a.keyframe_id!=b.keyframe_id || a.timestamp_ns!=b.timestamp_ns ||
            !ids.insert(a.keyframe_id).second || (k && a.timestamp_ns<=batch->local_keyframes[k-1].timestamp_ns))
            throw std::invalid_argument("invalid/incomplete local-to-global keyframe correspondence");
    }
    const auto& anchor_local=batch->local_keyframes.back();const auto& anchor_global=batch->global_keyframes.back();
    batch->local_to_global=anchor_global.body_pose*anchor_local.body_pose.inverse();
    if(!ValidPose(batch->local_to_global))throw std::invalid_argument("nonfinite local-to-global correction");
    batch->anchor_keyframe=anchor_local.keyframe_id;batch->anchor_timestamp_ns=anchor_local.timestamp_ns;
    batch->graph_generation=graph_generation;batch->publication_timestamp_ns=publication_timestamp_ns;
    batch->has_solution=true;batch->live_smoothing_seconds=options_.live_smoothing_seconds;
    std::lock_guard<std::mutex> lock(publication_mutex_);
    const auto previous=Snapshot();
    if(graph_generation<=previous->graph_generation || (previous->has_solution &&
        (publication_timestamp_ns<previous->publication_timestamp_ns || anchor_local.timestamp_ns<previous->anchor_timestamp_ns)))return false;
    // Keep every previously optimized historical vertex and its immutable LOCAL
    // reference. A change of local gauge or a truncated history is not a refresh.
    std::map<uint64_t,const PoseGraphPose*> locals;
    for(const auto& p:batch->local_keyframes)locals.emplace(p.keyframe_id,&p);
    for(const auto& p:previous->local_keyframes) {
        const auto found=locals.find(p.keyframe_id);
        if(found==locals.end() || found->second->timestamp_ns!=p.timestamp_ns ||
            (p.body_pose.inverse()*found->second->body_pose).log().norm()>1e-10)
            throw std::invalid_argument("global refresh changed the local frame or dropped historical keyframes");
    }
    if(previous->version==std::numeric_limits<uint64_t>::max())throw std::overflow_error("global correction version overflow");
    batch->version=previous->version+1;
    batch->live_start_correction=previous->LiveCorrection(publication_timestamp_ns);
    std::shared_ptr<const GlobalCorrectionBatch> immutable=batch;
    std::atomic_store_explicit(&current_,immutable,std::memory_order_release);
    
    return true;
}
GlobalNavigation GlobalCorrection::Project(const NavState& local,const Vec3& gravity_w,
    int64_t timestamp_ns,bool smooth_live) const {
    return ProjectGlobalNavigation(Snapshot(),local,gravity_w,timestamp_ns,smooth_live);
}
} // namespace myslam
