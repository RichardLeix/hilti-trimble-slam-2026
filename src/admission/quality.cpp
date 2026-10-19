#include "quality.h"
#include "myslam/hilti_rolling_shutter.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace CAdmission {
double Quality(const LandmarkWeightQuality& q) {
    // Explicit age-free subset of frozen C_quality, never a fabricated age.
    if(!q.observations || !std::isfinite(q.residual_sigma))return 0;
    const double r=std::max(0.,q.residual_sigma/2.5-1.);
    const double u=!std::isfinite(q.relative_depth_sigma)?1.:std::max(0.,1.-.1/std::max(.1,q.relative_depth_sigma));
    const double p=std::max(0.,1.-q.parallax_sigma/15.);
    const double n=std::max(0.,1.-double(q.observations)/4.);
    const double g=(u+p+n)/3.;
    return .25+.75/(1.+r*r+g*g);
}
bool Better(const Candidate& a,const Candidate& b) {
    if(a.mandatory!=b.mandatory)return a.mandatory;
    if(a.visible!=b.visible)return a.visible;
    if(bool(a.geometry.observations)!=bool(b.geometry.observations))return bool(a.geometry.observations);
    if(a.quality!=b.quality)return a.quality>b.quality;
    if(a.geometry.parallax_sigma!=b.geometry.parallax_sigma)return a.geometry.parallax_sigma>b.geometry.parallax_sigma;
    if(a.geometry.observations!=b.geometry.observations)return a.geometry.observations>b.geometry.observations;
    if(a.age_s!=b.age_s)return a.age_s>b.age_s;
    return a.id<b.id;
}
void Select(std::vector<Candidate>& cs,size_t cap) {
    if(size_t(std::count_if(cs.begin(),cs.end(),[](const Candidate& c){return c.mandatory;}))>cap)
        throw std::runtime_error("prior-required candidates exceed unchanged landmark cap");
    std::sort(cs.begin(),cs.end(),Better);
    for(size_t i=0;i<cs.size();++i)cs[i].selected=i<cap;
}
HiltiLandmark ValidCopy(const HiltiBackendJob& j,const HiltiLandmark& l,const HiltiBackendOptions& o) {
    auto copy=l;
    auto& obs=copy.observations;
    obs.erase(std::remove_if(obs.begin(),obs.end(),[&](const HiltiBearingObservation& b){
        auto f=std::lower_bound(j.keyframes.begin(),j.keyframes.end(),b.timestamp_ns,[](const HiltiKeyframeState& f,int64_t t){return f.timestamp_ns<t;});
        if(f==j.keyframes.end()||f->timestamp_ns!=b.timestamp_ns)return true;
        const Vec3 p=HiltiVisualCameraWorldPose(*f,l.id.first,b.bearing,j.cameras,j.gravity_w,o).inverse()*l.position_w;Vec2 pixel;
        return !p.allFinite() || p.norm()<1e-8 || p.dot(b.bearing)<=0 || !j.cameras[l.id.first]->Project(p,pixel) ||
            std::atan2(p.normalized().cross(b.bearing).norm(),p.normalized().dot(b.bearing))>o.max_initial_bearing_rad;
    }),obs.end());
    return copy;
}
}
