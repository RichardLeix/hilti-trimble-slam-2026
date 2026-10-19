#ifndef MYSLAM_HILTI_LANDMARK_QUARANTINE_H
#define MYSLAM_HILTI_LANDMARK_QUARANTINE_H
#include "myslam/hilti_backend.h"
#include <fstream>
#include <map>
namespace myslam {
// Fixed B validity policy. No intra-LM factor/proposal gating.
struct HiltiLandmarkValidity {
    bool nonfinite=false, nonpositive=false, too_far=false, jump=false, residual=false;
    double min_depth=0,max_range=0,rms=0,ratio=1;
    bool Invalid() const {return nonfinite || nonpositive || too_far || jump || residual;}
};
HiltiLandmarkValidity InspectHiltiLandmark(const HiltiBackendJob&,const HiltiLandmark&,
    const HiltiBackendOptions&,const Vec3* previous_point=nullptr,double previous_rms=-1);
std::shared_ptr<const MarginalizationPrior> QuarantineHiltiPriorLandmarks(
    const MarginalizationPrior&,const std::set<HiltiLandmarkId>&,double tolerance);

// One context per backend instance. Evidence persists across rejected windows;
// retirement/prior publication still happens only in accepted native results.
class HiltiLandmarkQuarantine {
public:
    explicit HiltiLandmarkQuarantine(const std::string& log_path="");
    void Apply(HiltiBackendJob&,const HiltiBackendOptions&,std::set<HiltiLandmarkId>&);
private:
    struct History {EIGEN_MAKE_ALIGNED_OPERATOR_NEW;Vec3 point=Vec3::Zero();double rms=0;};
    using HistoryMap=std::map<HiltiLandmarkId,History,std::less<HiltiLandmarkId>,
        Eigen::aligned_allocator<std::pair<const HiltiLandmarkId,History>>>;
    HistoryMap history_;
    std::set<HiltiLandmarkId> bad_;
    std::ofstream log_;
    void Inspect(const HiltiBackendJob&,const HiltiBackendOptions&,const char*,uint64_t,int64_t);
    void Erase(HiltiBackendJob&);
};
}
#endif
