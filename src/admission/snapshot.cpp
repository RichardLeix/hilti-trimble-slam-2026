#include "camera_floor.h"
#include "myslam/hilti_rolling_shutter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace myslam {
HiltiBackendJob HiltiLocalMap::Snapshot(uint64_t version,uint64_t base_version) {
    HiltiBackendJob job;job.version=version;job.base_version=base_version;job.keyframes=window_;
    job.cameras=cameras_;job.calibration=calibration_;job.gravity_w=gravity_w_;
    for(size_t i=1;i<window_.size();++i)job.intervals.push_back(PreintegrateImu(samples_,
        window_[i-1].timestamp_ns,window_[i].timestamp_ns,calibration_,window_[i-1].state.bias));
    std::set<HiltiLandmarkId> previous_points,prior_points;
    if(options_.enable_marginalization && factor_model_) {
        for(const auto& p:factor_model_->landmarks)previous_points.insert(p.id);
        if(factor_model_->prior)for(const auto& v:factor_model_->prior->linearization)
            if(v.key.kind==MarginalizationKey::Kind::Landmark)prior_points.insert({v.key.camera_id,v.key.feature_id});
    }
    // Keep still-visible prior variables before admitting new points. Retiring
    // a track consumes its remaining factors, while its archive stays intact.
    std::array<HiltiAlignedVector<HiltiLandmark>,2> candidates;
    for(const auto& entry:archive_) {
        if(!entry.second.initialized || (options_.enable_marginalization && retired_landmarks_.count(entry.first)))continue;
        HiltiLandmark local=entry.second;local.observations.clear();bool visible=false;
        for(const auto& frame:window_) {
            auto obs=std::lower_bound(entry.second.observations.begin(),entry.second.observations.end(),frame.timestamp_ns,
                [](const HiltiBearingObservation& o,int64_t t){return o.timestamp_ns<t;});
            if(obs==entry.second.observations.end() || obs->timestamp_ns!=frame.timestamp_ns)continue;
            if(frame.timestamp_ns==window_.back().timestamp_ns)visible=true;
            if(!absorbed_observations_.count({entry.first,obs->timestamp_ns}))local.observations.push_back(*obs);
        }
        const bool enough=local.observations.size()>=2 || (prior_points.count(local.id) && !local.observations.empty());
        if(enough && (!options_.enable_marginalization || !factor_model_ || visible))candidates[local.id.first].push_back(std::move(local));
    }

    std::vector<B3Admission::Entry> pool;
    for(int camera=0;camera<2;++camera)for(size_t i=0;i<candidates[camera].size();++i) {
        const auto& landmark=candidates[camera][i];
        B3Admission::Entry e;auto& c=e.core;c.id=landmark.id;c.camera_index=i;
        e.previous=previous_points.count(c.id);e.raw_observations=landmark.observations.size();
        c.mandatory=prior_points.count(c.id);
        // Evaluate a private filtered copy using precisely the native ready
        // bearing-validity gate. Candidate factors themselves are unmodified.
        const auto valid=CAdmission::ValidCopy(job,landmark,options_);
        c.geometry=HiltiLandmarkWeightGeometry(job,valid,options_);
        c.visible=std::any_of(valid.observations.begin(),valid.observations.end(),[&](const HiltiBearingObservation& b){return b.timestamp_ns==window_.back().timestamp_ns;});
        c.quality=CAdmission::Quality(c.geometry);
        const auto& archived=archive_.at(c.id);
        c.age_s=archived.observations.empty()?0:(window_.back().timestamp_ns-archived.observations.front().timestamp_ns)*1e-9;
        pool.push_back(e);
    }
    const auto decision=B3Admission::Select(pool,options_.max_landmarks);
    for(const auto& e:pool)if(e.core.selected)
        job.landmarks.push_back(candidates[e.core.id.first][e.core.camera_index]);
    if(options_.enable_marginalization && factor_model_) {
        job.has_initial_anchor=true;job.initial_anchor=factor_model_->initial_anchor;
        auto source=std::make_shared<HiltiBackendJob>();source->cameras=cameras_;source->calibration=calibration_;
        source->gravity_w=gravity_w_;source->prior=factor_model_->prior;
        source->has_initial_anchor=true;source->initial_anchor=factor_model_->initial_anchor;
        // Accepted graph plus every pending chronological keyframe: busy or
        // rejected jobs cannot cause an IMU link to disappear before elimination.
        const int64_t begin=factor_model_->keyframes.front().timestamp_ns;
        for(const auto& frame:archived_keyframes_)if(frame.timestamp_ns>=begin)source->keyframes.push_back(frame);
        for(size_t i=1;i<source->keyframes.size();++i)source->intervals.push_back(PreintegrateImu(samples_,
            source->keyframes[i-1].timestamp_ns,source->keyframes[i].timestamp_ns,calibration_,source->keyframes[i-1].state.bias));
        std::set<HiltiLandmarkId> source_ids=previous_points;
        for(const auto& point:job.landmarks)source_ids.insert(point.id);
        for(const auto& id:source_ids) {
            HiltiLandmark point=archive_.at(id);point.observations.clear();
            for(const auto& obs:archive_.at(id).observations) {
                if(obs.timestamp_ns<begin || absorbed_observations_.count({id,obs.timestamp_ns}))continue;
                auto frame=std::lower_bound(source->keyframes.begin(),source->keyframes.end(),obs.timestamp_ns,
                    [](const HiltiKeyframeState& f,int64_t t){return f.timestamp_ns<t;});
                if(frame==source->keyframes.end() || frame->timestamp_ns!=obs.timestamp_ns)continue;
                // Factors accepted in the previous graph survive relinearization.
                bool existing=false;
                for(const auto& old:factor_model_->landmarks)if(old.id==id)
                    for(const auto& o:old.observations)if(o.timestamp_ns==obs.timestamp_ns)existing=true;
                const Vec3 p=HiltiCameraWorldPose(*frame,id.first,cameras_,gravity_w_).inverse()*point.position_w;Vec2 pixel;
                if(existing || (p.norm()>1e-8 && p.dot(obs.bearing)>0 && cameras_[id.first]->Project(p,pixel) &&
                    std::atan2(p.normalized().cross(obs.bearing).norm(),p.normalized().dot(obs.bearing))<=options_.max_initial_bearing_rad))
                    point.observations.push_back(obs);
            }
            source->landmarks.push_back(std::move(point));
        }
        job.marginalization_source=source;
    } else if(options_.enable_marginalization) {
        // Quarantine the first submitted graph even if it is rejected or still
        // in flight. Its original gauge and sensor factors remain available.
        job.has_initial_anchor=true;job.initial_anchor=job.keyframes.front().state;
        factor_model_=std::make_shared<const HiltiBackendJob>(job);
    }
    return job;
}
}
