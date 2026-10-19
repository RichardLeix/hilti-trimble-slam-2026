#include "myslam/hilti_landmark_quarantine.h"
#include "myslam/hilti_rolling_shutter.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
namespace myslam {
HiltiLandmarkValidity InspectHiltiLandmark(const HiltiBackendJob& job,const HiltiLandmark& point,
        const HiltiBackendOptions& options,const Vec3* previous_point,double previous_rms) {
    HiltiLandmarkValidity out;out.min_depth=std::numeric_limits<double>::infinity();
    out.nonfinite=!point.position_w.allFinite();double squared=0;size_t count=0;
    for(const auto& obs:point.observations) {
        auto frame=std::lower_bound(job.keyframes.begin(),job.keyframes.end(),obs.timestamp_ns,
            [](const HiltiKeyframeState& f,int64_t t){return f.timestamp_ns<t;});
        if(frame==job.keyframes.end() || frame->timestamp_ns!=obs.timestamp_ns)continue;
        const SE3 tcw=HiltiVisualCameraWorldPose(*frame,point.id.first,obs.bearing,job.cameras,job.gravity_w,options).inverse();
        const Vec3 p=tcw*point.position_w;const double range=p.norm(),depth=p.dot(obs.bearing);
        out.nonfinite=out.nonfinite || !p.allFinite() || !std::isfinite(range);
        out.min_depth=std::min(out.min_depth,depth);out.max_range=std::max(out.max_range,range);
        out.nonpositive=out.nonpositive || depth<=0 || range<1e-8;
        out.too_far=out.too_far || depth>500 || range>500;
        if(previous_point) {
            const double old_depth=(tcw*(*previous_point)).dot(obs.bearing);
            if(depth>1e-6 && old_depth>1e-6)out.ratio=std::max(out.ratio,std::max(depth/old_depth,old_depth/depth));
        }
        Vec2 pixel;const bool valid=range>1e-8 && depth>0 && job.cameras[point.id.first]->Project(p,pixel);
        const double angle=valid?std::atan2(p.normalized().cross(obs.bearing).norm(),p.normalized().dot(obs.bearing)):M_PI;
        squared+=angle*angle;++count;
    }
    // A prior-only point has no bearing with which to define signed depth.
    // Still enforce the same metric range bound at the current camera poses.
    if(!count)for(const auto& frame:job.keyframes) {
        const Vec3 p=HiltiCameraWorldPose(frame,point.id.first,job.cameras,job.gravity_w).inverse()*point.position_w;
        const double range=p.norm();out.nonfinite=out.nonfinite || !p.allFinite() || !std::isfinite(range);
        out.max_range=std::max(out.max_range,range);out.too_far=out.too_far || range>500;
    }
    out.rms=count?std::sqrt(squared/count):0;
    out.jump=out.ratio>10;
    out.residual=out.rms>std::max(0.02,previous_rms>=0?4*previous_rms:0.0);
    return out;
}

std::shared_ptr<const MarginalizationPrior> QuarantineHiltiPriorLandmarks(const MarginalizationPrior& old,
        const std::set<HiltiLandmarkId>& bad,double tolerance) {
    MarginalizationInfo info;info.variables=old.linearization;info.hessian=old.hessian;info.gradient=old.gradient;
    for(const auto& v:old.linearization)if(v.key.kind==MarginalizationKey::Kind::Landmark && bad.count({v.key.camera_id,v.key.feature_id}))
        info.eliminate.insert(v.key);
    if(info.eliminate.empty())return std::make_shared<const MarginalizationPrior>(old);
    auto clean=info.Marginalize(tolerance);
    clean.has_fixed_pose=old.has_fixed_pose;clean.fixed_pose_timestamp_ns=old.fixed_pose_timestamp_ns;
    clean.marginalizations=old.marginalizations;clean.marginalized_landmarks=old.marginalized_landmarks+info.eliminate.size();
    return std::make_shared<const MarginalizationPrior>(std::move(clean));
}

HiltiLandmarkQuarantine::HiltiLandmarkQuarantine(const std::string& path) {
    if(path.empty())return;
    log_.open(path);
    if(!log_)throw std::runtime_error("cannot open B landmark quarantine log");
    log_<<std::setprecision(17)<<"version,timestamp_ns,variant,stage,camera,id,nonfinite,nonpositive,too_far,jump,residual,min_signed_depth_m,max_range_m,depth_ratio,visual_rms_rad\n";
}
void HiltiLandmarkQuarantine::Inspect(const HiltiBackendJob& graph,const HiltiBackendOptions& options,
        const char* stage,uint64_t version,int64_t time) {
    for(const auto& point:graph.landmarks) {
        auto old=history_.find(point.id);
        const auto g=InspectHiltiLandmark(graph,point,options,old==history_.end()?nullptr:&old->second.point,old==history_.end()?-1:old->second.rms);
        if(g.Invalid()) {
            const bool first=bad_.insert(point.id).second;
            if(first && log_)log_<<version<<','<<time<<",B,"<<stage<<','<<point.id.first<<','<<point.id.second<<','
                <<g.nonfinite<<','<<g.nonpositive<<','<<g.too_far<<','<<g.jump<<','<<g.residual<<','
                <<g.min_depth<<','<<g.max_range<<','<<g.ratio<<','<<g.rms<<'\n';
        }
    }
}
void HiltiLandmarkQuarantine::Erase(HiltiBackendJob& graph) {
    graph.landmarks.erase(std::remove_if(graph.landmarks.begin(),graph.landmarks.end(),
        [&](const HiltiLandmark& p){return bad_.count(p.id);}),graph.landmarks.end());
}
void HiltiLandmarkQuarantine::Apply(HiltiBackendJob& current,const HiltiBackendOptions& options,
        std::set<HiltiLandmarkId>& retired) {
    Inspect(current,options,"active_input",current.version,current.keyframes.back().timestamp_ns);
    if(current.marginalization_source) {
        auto source=std::make_shared<HiltiBackendJob>(*current.marginalization_source);
        Inspect(*source,options,"marginalization_source",current.version,current.keyframes.back().timestamp_ns);
        if(source->prior)source->prior=QuarantineHiltiPriorLandmarks(*source->prior,bad_,options.marginalization_rank_tolerance);
        Erase(*source);current.marginalization_source=source;
    }
    if(current.prior)current.prior=QuarantineHiltiPriorLandmarks(*current.prior,bad_,options.marginalization_rank_tolerance);
    Erase(current);retired.insert(bad_.begin(),bad_.end());
    for(const auto& point:current.landmarks) {
        const auto g=InspectHiltiLandmark(current,point,options);
        History h;h.point=point.position_w;h.rms=g.rms;history_[point.id]=h;
    }
    if(log_)log_.flush();
}
} // namespace myslam
