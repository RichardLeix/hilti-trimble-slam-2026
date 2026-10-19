#include "myslam/keyframe_database.h"
#include "fbow.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace myslam {
namespace {
LoopKeyframe Copy(const LoopKeyframe& frame) {
    LoopKeyframe copy=frame;
    for(int k=0;k<2;++k) copy.descriptors[k]=frame.descriptors[k].clone();
    return copy;
}
void Check(const LoopKeyframe& frame) {
    if(frame.timestamp_ns<=0 || !frame.backend_version || !frame.optimized_body_pose.matrix().allFinite())
        throw std::invalid_argument("loop archive needs an optimized timestamped body pose");
    for(int k=0;k<2;++k) {
        const auto& d=frame.descriptors[k];
        if(!frame.camera_from_body[k].matrix().allFinite() ||
            size_t(d.rows)!=frame.observations[k].size() ||
            (!d.empty() && (d.type()!=CV_8UC1 || d.cols!=32)))
            throw std::invalid_argument("invalid loop descriptors/calibration");
        for(const auto& o:frame.observations[k])
            if(!o.bearing.allFinite() || std::abs(o.bearing.norm()-1)>1e-6 ||
                !std::isfinite(o.pixel.x) || !std::isfinite(o.pixel.y) ||
                (o.has_point && !o.point_body.allFinite()))
                throw std::invalid_argument("invalid archived EUCM observation");
    }
}
cv::Mat Combined(const LoopKeyframe& f) {
    cv::Mat result;
    for(int k=0;k<2;++k)if(!f.descriptors[k].empty())result.push_back(f.descriptors[k]);
    return result;
}
}
struct KeyframeDatabase::Impl {
    mutable std::mutex mutex;
    fbow::Vocabulary vocabulary;
    struct Entry {
        std::shared_ptr<const LoopKeyframe> frame;
        fbow::fBow bow;
    };
    std::map<uint64_t,Entry> entries;
    std::map<uint32_t,std::set<uint64_t>> inverted;
};
KeyframeDatabase::KeyframeDatabase(const std::string& path):impl_(new Impl) {
    impl_->vocabulary.readFromFile(path);
    if(!impl_->vocabulary.isValid() || impl_->vocabulary.getDescType()!=CV_8UC1 || impl_->vocabulary.getDescSize()!=32)
        throw std::invalid_argument("loop vocabulary must contain 32-byte ORB words");
}
KeyframeDatabase::~KeyframeDatabase()=default;
void KeyframeDatabase::Insert(const LoopKeyframe& frame) {
    Check(frame);
    auto copy=std::make_shared<const LoopKeyframe>(Copy(frame));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(impl_->entries.count(frame.id))throw std::invalid_argument("duplicate loop keyframe ID");
    const auto descriptors=Combined(frame);
    fbow::fBow bow;if(!descriptors.empty())bow=impl_->vocabulary.transform(descriptors);
    Impl::Entry entry;entry.frame=copy;entry.bow=bow;
    impl_->entries.emplace(frame.id,std::move(entry));
    for(const auto& word:bow)impl_->inverted[word.first].insert(frame.id);
}
void KeyframeDatabase::UpdateGeometry(const LoopKeyframe& frame) {
    Check(frame);auto copy=std::make_shared<const LoopKeyframe>(Copy(frame));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it=impl_->entries.find(frame.id);
    if(it==impl_->entries.end())throw std::invalid_argument("missing loop keyframe");
    const auto& old=*it->second.frame;
    if(old.timestamp_ns!=frame.timestamp_ns || old.backend_version!=frame.backend_version ||
        (old.optimized_body_pose.matrix()-frame.optimized_body_pose.matrix()).norm()>1e-12)
        throw std::invalid_argument("archived loop identity/pose is frozen");
    for(int k=0;k<2;++k) {
        if((old.camera_from_body[k].matrix()-frame.camera_from_body[k].matrix()).norm()>1e-12 ||
            old.descriptors[k].size()!=frame.descriptors[k].size() ||
            (!old.descriptors[k].empty() && cv::norm(old.descriptors[k],frame.descriptors[k],cv::NORM_INF)!=0))
            throw std::invalid_argument("archived loop appearance/calibration is immutable");
        for(size_t i=0;i<old.observations[k].size();++i) {
            const auto& a=old.observations[k][i];const auto& b=frame.observations[k][i];
            if(a.pixel!=b.pixel || a.has_track!=b.has_track || a.track_id!=b.track_id || (a.bearing-b.bearing).norm()>1e-12)
                throw std::invalid_argument("archived loop observations are immutable");
        }
    }
    it->second.frame=copy;
}
std::shared_ptr<const LoopKeyframe> KeyframeDatabase::Get(uint64_t id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);auto it=impl_->entries.find(id);
    if(it==impl_->entries.end())return nullptr;
    return std::make_shared<const LoopKeyframe>(Copy(*it->second.frame));
}
size_t KeyframeDatabase::Size() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->entries.size();
}
HiltiAlignedVector<LoopPoseMetadata> KeyframeDatabase::PoseMetadata() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    HiltiAlignedVector<LoopPoseMetadata> poses;
    poses.reserve(impl_->entries.size());
    for(const auto& entry:impl_->entries) {
        const auto& frame=*entry.second.frame;
        LoopPoseMetadata p;p.id=frame.id;p.backend_version=frame.backend_version;
        p.timestamp_ns=frame.timestamp_ns;p.optimized_body_pose=frame.optimized_body_pose;
        poses.push_back(p);
    }
    return poses;
}
std::vector<LoopCandidate> KeyframeDatabase::Query(const LoopKeyframe& query,size_t maximum,
        uint64_t minimum_keyframes,double minimum_seconds,double minimum_score) const {
    Check(query);
    if(!maximum || !std::isfinite(minimum_seconds) || minimum_seconds<0 ||
        !std::isfinite(minimum_score) || minimum_score<0 || minimum_score>1)
        throw std::invalid_argument("invalid loop retrieval limits");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<LoopCandidate> result;const auto descriptors=Combined(query);
    if(descriptors.empty())return result;
    const auto bow=impl_->vocabulary.transform(descriptors);std::set<uint64_t> ids;
    for(const auto& word:bow) {
        auto it=impl_->inverted.find(word.first);
        if(it!=impl_->inverted.end())ids.insert(it->second.begin(),it->second.end());
    }
    for(uint64_t id:ids) {
        const auto& entry=impl_->entries.at(id);const auto& f=*entry.frame;
        if(id==query.id)continue;
        LoopCandidate c;c.keyframe_id=id;c.keyframe_separation=id>query.id?id-query.id:query.id-id;
        c.time_separation_s=std::abs(double((long double)query.timestamp_ns-f.timestamp_ns)*1e-9);
        if(c.keyframe_separation<minimum_keyframes || c.time_separation_s<minimum_seconds)continue;
        c.score=fbow::fBow::score(bow,entry.bow);
        if(std::isfinite(c.score) && c.score>=minimum_score)result.push_back(c);
    }
    std::sort(result.begin(),result.end(),[](const LoopCandidate& a,const LoopCandidate& b){
        return a.score!=b.score?a.score>b.score:a.keyframe_id<b.keyframe_id;
    });
    if(result.size()>maximum)result.resize(maximum);
    return result;
}
} // namespace myslam
