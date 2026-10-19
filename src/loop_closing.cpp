#include "myslam/loop_closing.h"
#include "myslam/pose_graph.h"
#include <opencv2/features2d.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace myslam {
void ValidateLoopClosingOptions(const LoopClosingOptions& o) {
    if(!o.minimum_keyframe_separation || !std::isfinite(o.minimum_time_separation_s) || o.minimum_time_separation_s<=0 ||
        !o.maximum_candidates || o.orb_features_per_camera<50 || o.minimum_inliers<12 || !o.ransac_iterations ||
        o.max_hamming_distance<1 || o.max_hamming_distance>256)
        throw std::invalid_argument("invalid loop support/temporal limits");
    for(double x:{o.minimum_score,o.descriptor_ratio,o.minimum_inlier_ratio,o.ransac_confidence})
        if(!std::isfinite(x) || x<=0 || x>=1)throw std::invalid_argument("invalid loop probability/score/ratio");
    for(double x:{o.inlier_threshold_rad,o.maximum_rms_rad,o.maximum_pose_condition})
        if(!std::isfinite(x) || x<=0)throw std::invalid_argument("invalid loop residual/condition limit");
    if(o.maximum_rms_rad>o.inlier_threshold_rad || o.inlier_threshold_rad>=0.1 || o.maximum_pose_condition<=1)
        throw std::invalid_argument("incoherent loop geometric limits");
}
LoopClosingOptions LoadLoopClosingOptions(const std::string& path,const std::string& vocabulary_override) {
    cv::FileStorage file(path,cv::FileStorage::READ);if(!file.isOpened())throw std::runtime_error("cannot open loop configuration: "+path);
    LoopClosingOptions o;file["vocabulary_path"]>>o.vocabulary_path;
    auto number=[&](const char* key,double current){return file[key].empty()?current:double(file[key]);};
    auto count=[&](const char* key,size_t current) {
        const double x=number(key,double(current));
        if(!std::isfinite(x) || x<1 || x>1e7 || std::floor(x)!=x)throw std::invalid_argument(std::string("invalid loop count: ")+key);
        return size_t(x);
    };
    o.minimum_keyframe_separation=count("minimum_keyframe_separation",o.minimum_keyframe_separation);
    o.minimum_time_separation_s=number("minimum_time_separation_s",o.minimum_time_separation_s);
    o.maximum_candidates=count("maximum_candidates",o.maximum_candidates);
    o.orb_features_per_camera=int(count("orb_features_per_camera",o.orb_features_per_camera));
    o.minimum_score=number("minimum_score",o.minimum_score);
    o.max_hamming_distance=int(count("max_hamming_distance",o.max_hamming_distance));
    o.descriptor_ratio=number("descriptor_ratio",o.descriptor_ratio);
    o.minimum_inliers=count("minimum_inliers",o.minimum_inliers);
    o.minimum_inlier_ratio=number("minimum_inlier_ratio",o.minimum_inlier_ratio);
    o.inlier_threshold_rad=number("inlier_threshold_rad",o.inlier_threshold_rad);
    o.maximum_rms_rad=number("maximum_rms_rad",o.maximum_rms_rad);
    o.maximum_pose_condition=number("maximum_pose_condition",o.maximum_pose_condition);
    o.ransac_iterations=count("ransac_iterations",o.ransac_iterations);
    o.ransac_confidence=number("ransac_confidence",o.ransac_confidence);
    if(!vocabulary_override.empty())o.vocabulary_path=vocabulary_override;
    ValidateLoopClosingOptions(o);return o;
}
LoopClosing::LoopClosing(const HiltiCameras& cameras,const LoopClosingOptions& options)
    :cameras_(cameras),options_(options),database_(options.vocabulary_path) {
    ValidateLoopClosingOptions(options);
    for(int k=0;k<2;++k) {
        if(!cameras[k])throw std::invalid_argument("loop closing needs two calibrated EUCM cameras");
        valid_masks_[k]=cv::Mat::zeros(cameras[k]->Height(),cameras[k]->Width(),CV_8UC1);
        for(int y=0;y<cameras[k]->Height();++y)for(int x=0;x<cameras[k]->Width();++x)
            if(cameras[k]->IsValidPixel(Vec2(x,y)))valid_masks_[k].at<uint8_t>(y,x)=255;
    }
    worker_=std::thread(&LoopClosing::Run,this);
}
LoopClosing::~LoopClosing(){Stop();}
bool LoopClosing::Submit(const LoopInput& input) {
    auto copy=std::make_shared<LoopInput>(input);
    for(int k=0;k<2;++k)copy->frame.descriptors[k]=input.frame.descriptors[k].clone();
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopping_ || (have_id_ && input.frame.id<=last_id_))return false;
    Work work;work.raw=copy;queue_.push_back(work);last_id_=input.frame.id;have_id_=true;wake_.notify_one();return true;
}
bool LoopClosing::SubmitPrepared(const LoopKeyframe& frame) {
    auto copy=std::make_shared<LoopKeyframe>(frame);
    for(int k=0;k<2;++k)copy->descriptors[k]=frame.descriptors[k].clone();
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopping_ || (have_id_ && frame.id<=last_id_))return false;
    Work work;work.prepared=copy;queue_.push_back(work);last_id_=frame.id;have_id_=true;wake_.notify_one();return true;
}
bool LoopClosing::WaitUntilIdle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);return idle_.wait_for(lock,timeout,[this]{return queue_.empty() && !working_;});
}
void LoopClosing::Stop() {
    std::lock_guard<std::mutex> stop_lock(stop_mutex_);
    {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;wake_.notify_all();}
    if(worker_.joinable())worker_.join();
}
LoopStatistics LoopClosing::Statistics() const {std::lock_guard<std::mutex> lock(mutex_);return statistics_;}
HiltiAlignedVector<LoopDiagnostic> LoopClosing::Diagnostics() const {std::lock_guard<std::mutex> lock(mutex_);return diagnostics_;}
HiltiAlignedVector<LoopConstraint> LoopClosing::Constraints() const {std::lock_guard<std::mutex> lock(mutex_);return constraints_;}
PoseGraphInput LoopClosing::GraphSnapshot() const {
    // Process inserts into the DB before committing its diagnostics under this
    // mutex. Include only committed query IDs; an extra inserted frame may be
    // present while the next query is in flight. Archived body poses are frozen.
    std::lock_guard<std::mutex> lock(mutex_);
    if(statistics_.failed_inputs)throw std::runtime_error("cannot graph an archive with failed loop inputs");
    std::set<uint64_t> committed;
    std::map<std::pair<uint64_t,uint64_t>,const LoopDiagnostic*> accepted;
    for(const auto& d:diagnostics_) {
        committed.insert(d.query_keyframe);
        if(d.accepted)accepted.emplace(std::make_pair(d.query_keyframe,d.candidate_keyframe),&d);
    }
    PoseGraphInput input;
    for(const auto& frame:database_.PoseMetadata())if(committed.count(frame.id)) {
        PoseGraphPose p;p.keyframe_id=frame.id;p.timestamp_ns=frame.timestamp_ns;p.body_pose=frame.optimized_body_pose;
        input.poses.push_back(p);
    }
    if(input.poses.size()!=statistics_.keyframes)throw std::runtime_error("incomplete committed pose archive");
    for(const auto& c:constraints_) {
        const auto found=accepted.find({c.query_keyframe,c.candidate_keyframe});
        if(found==accepted.end() || !found->second->verification_attempted ||
            !committed.count(c.query_keyframe) || !committed.count(c.candidate_keyframe))
            throw std::runtime_error("loop constraint lacks committed geometric verification");
        PoseGraphLoop l;l.query_keyframe=c.query_keyframe;l.candidate_keyframe=c.candidate_keyframe;
        l.relative_body_transform=c.relative_body_transform;l.inliers=c.inliers;
        l.bearing_rms_rad=c.bearing_rms_rad;l.pose_condition=found->second->pose_condition;input.loops.push_back(l);
    }
    return input;
}
LoopKeyframe LoopClosing::Extract(const LoopInput& input) const {
    LoopKeyframe frame=input.frame;
    auto orb=cv::ORB::create(options_.orb_features_per_camera);
    std::map<HiltiLandmarkId,const HiltiLandmark*> points;
    for(const auto& l:input.landmarks)if(l.initialized)points[l.id]=&l;
    for(int k=0;k<2;++k) {
        if(input.images[k].stream!=(k?SensorStream::Cam1:SensorStream::Cam0))throw std::invalid_argument("loop image/camera mismatch");
        const auto image=DecodeHiltiJpeg(input.images[k].image);
        if(image.cols!=cameras_[k]->Width() || image.rows!=cameras_[k]->Height())throw std::invalid_argument("loop image resolution mismatch");
        std::vector<cv::KeyPoint> keypoints;orb->detect(image,keypoints,valid_masks_[k]);
        for(auto& key:keypoints)key.class_id=-1;
        const auto& tracks=input.tracks[k].features;
        for(size_t i=0;i<tracks.size();++i) {
            const auto pixel=tracks[i].pixel;
            if(!cameras_[k]->IsValidPixel(Vec2(pixel.x,pixel.y)) || pixel.x<32 || pixel.y<32 ||
                pixel.x>=image.cols-32 || pixel.y>=image.rows-32)continue;
            bool nearby=false;
            for(auto& key:keypoints)if(cv::norm(key.pt-pixel)<4) {
                if(key.class_id<0)key.class_id=int(i);
                nearby=true;break;
            }
            if(nearby)continue;
            // ORB::compute expects supplied angles for caller-owned corners.
            // Intensity-centroid orientation uses ORB's circular 31px patch.
            const int x=cvRound(pixel.x),y=cvRound(pixel.y);double m10=0,m01=0;
            for(int dy=-15;dy<=15;++dy)for(int dx=-15;dx<=15;++dx)if(dx*dx+dy*dy<=225) {
                const double value=image.at<uint8_t>(y+dy,x+dx);m10+=dx*value;m01+=dy*value;
            }
            cv::KeyPoint key(pixel,31,float(cv::fastAtan2(float(m01),float(m10))),0,0,int(i));keypoints.push_back(key);
        }
        cv::Mat descriptors;orb->compute(image,keypoints,descriptors);
        frame.descriptors[k]=cv::Mat();frame.observations[k].clear();
        for(size_t i=0;i<keypoints.size();++i) {
            const auto& key=keypoints[i];LoopObservation o;o.pixel=key.pt;
            if(!cameras_[k]->Unproject(Vec2(key.pt.x,key.pt.y),o.bearing))continue;
            if(key.class_id>=0) {
                o.has_track=true;o.track_id=tracks.at(size_t(key.class_id)).id;
                const auto point=points.find({k,o.track_id});
                if(point!=points.end()) {
                    const Vec3 body=frame.optimized_body_pose.inverse()*point->second->position_w;
                    const Vec3 p=frame.camera_from_body[k]*body;
                    if(p.norm()>1e-8 && p.dot(o.bearing)>0 &&
                        std::atan2(p.cross(o.bearing).norm(),p.dot(o.bearing))<options_.inlier_threshold_rad) {
                        o.has_point=true;o.point_body=body;
                    }
                }
            }
            frame.observations[k].push_back(o);frame.descriptors[k].push_back(descriptors.row(int(i)));
        }
    }
    return frame;
}
void LoopClosing::EnrichGeometry(LoopKeyframe& current) {
    // Extra ORB points are triangulated ONLY across recent optimized local
    // keyframes, independently of loop hypotheses and without changing BA.
    // This supplies metric support when persistent frontend tracks have expired.
    for(uint64_t distance=1;distance<=3 && distance<=current.id;++distance) {
        const auto archived=database_.Get(current.id-distance);if(!archived)continue;LoopKeyframe previous=*archived;
        const auto matches=MatchLoopDescriptors(current,previous,options_);bool changed=false;
        for(const auto& match:matches) {
            auto& a=previous.observations[match.candidate_camera][match.candidate_row];
            auto& b=current.observations[match.query_camera][match.query_row];if(a.has_point && b.has_point)continue;
            const SE3 wa=previous.optimized_body_pose*previous.camera_from_body[match.candidate_camera].inverse();
            const SE3 wb=current.optimized_body_pose*current.camera_from_body[match.query_camera].inverse();
            const Vec3 ra=wa.so3()*a.bearing,rb=wb.so3()*b.bearing;
            const double angle=std::atan2(ra.cross(rb).norm(),ra.dot(rb));
            if(angle<0.008 || angle>0.5)continue;
            const Mat33 aa=Mat33::Identity()-ra*ra.transpose(),ab=Mat33::Identity()-rb*rb.transpose();
            const Mat33 h=aa+ab;
            const Vec3 point=h.ldlt().solve(aa*wa.translation()+ab*wb.translation());
            const Vec3 pa=wa.inverse()*point,pb=wb.inverse()*point;
            if(!point.allFinite() || pa.norm()<0.15 || pb.norm()<0.15 || pa.norm()>80 || pb.norm()>80 ||
                pa.dot(a.bearing)<=0 || pb.dot(b.bearing)<=0 ||
                std::atan2(pa.cross(a.bearing).norm(),pa.dot(a.bearing))>0.003 ||
                std::atan2(pb.cross(b.bearing).norm(),pb.dot(b.bearing))>0.003)continue;
            if(!a.has_point){a.point_body=previous.optimized_body_pose.inverse()*point;a.has_point=true;changed=true;}
            if(!b.has_point){b.point_body=current.optimized_body_pose.inverse()*point;b.has_point=true;}
        }
        if(changed)database_.UpdateGeometry(previous);
    }
}
void LoopClosing::Process(LoopKeyframe frame) {
    EnrichGeometry(frame);
    const auto candidates=database_.Query(frame,options_.maximum_candidates,
        options_.minimum_keyframe_separation,options_.minimum_time_separation_s,options_.minimum_score);
    HiltiAlignedVector<LoopDiagnostic> diagnostics;HiltiAlignedVector<LoopConstraint> constraints;
    size_t verified=0;
    for(const auto& candidate:candidates) {
        const auto archived=database_.Get(candidate.keyframe_id);
        auto d=VerifyLoopGeometry(frame,*archived,options_);
        d.score=candidate.score;d.temporal_separation_s=candidate.time_separation_s;d.keyframe_separation=candidate.keyframe_separation;
        if(d.verification_attempted)++verified;
        if(d.accepted) {
            LoopConstraint c;c.query_keyframe=frame.id;c.candidate_keyframe=archived->id;
            c.query_timestamp_ns=frame.timestamp_ns;c.candidate_timestamp_ns=archived->timestamp_ns;
            c.relative_body_transform=d.relative_body_transform;c.inliers=d.geometric_inliers;c.bearing_rms_rad=d.relative_pose_rms_rad;
            constraints.push_back(c);
        }
        diagnostics.push_back(d);
    }
    if(candidates.empty()) {
        LoopDiagnostic d;d.query_keyframe=frame.id;d.candidate_keyframe=std::numeric_limits<uint64_t>::max();
        d.reason="no eligible place-recognition candidate";diagnostics.push_back(d);
    }
    database_.Insert(frame);
    std::lock_guard<std::mutex> lock(mutex_);
    ++statistics_.keyframes;++statistics_.queries;statistics_.candidates+=candidates.size();statistics_.geometrically_verified+=verified;
    statistics_.accepted_constraints+=constraints.size();
    
    diagnostics_.insert(diagnostics_.end(),diagnostics.begin(),diagnostics.end());constraints_.insert(constraints_.end(),constraints.begin(),constraints.end());
}
void LoopClosing::Run() {
    for(;;) {
        Work work;
        {std::unique_lock<std::mutex> lock(mutex_);wake_.wait(lock,[this]{return stopping_ || !queue_.empty();});
            if(queue_.empty() && stopping_)break;
            work=queue_.front();queue_.pop_front();working_=true;}
        try {Process(work.prepared?*work.prepared:Extract(*work.raw));}
        catch(const std::exception& e) {
            LoopDiagnostic d;d.query_keyframe=work.prepared?work.prepared->id:work.raw->frame.id;
            d.candidate_keyframe=std::numeric_limits<uint64_t>::max();d.reason=std::string("loop input failure: ")+e.what();
            std::lock_guard<std::mutex> lock(mutex_);++statistics_.failed_inputs;diagnostics_.push_back(d);
        }
        {std::lock_guard<std::mutex> lock(mutex_);working_=false;idle_.notify_all();}
    }
}
} // namespace myslam
