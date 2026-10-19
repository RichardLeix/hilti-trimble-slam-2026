#include "myslam/hilti_landmark_weighting.h"
#include "myslam/hilti_rolling_shutter.h"
#include "myslam/hilti_camera_information.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
namespace myslam {namespace {
double Excess(double x){return std::max(0.,x-1.);}
double BoundedDeficit(double value,double reference){return !std::isfinite(value)?1.:std::max(0.,1.-reference/std::max(reference,value));}
struct Track { int64_t first=0; };
struct History { std::map<HiltiLandmarkId,Track> tracks; };
History& Log(){static thread_local History h;return h;}
struct Context {uint64_t version=0;std::map<HiltiLandmarkId,double> weights;};
thread_local Context* active=nullptr;
}
double HiltiLandmarkQualityWeight(const LandmarkWeightQuality& q,bool combined){
    if(!q.observations)return 1.; // No live factor: the stored prior is untouched.
    if(!std::isfinite(q.residual_sigma))return .25;
    const double r=Excess(q.residual_sigma/2.5);double penalty=r*r;
    if(combined){
        const double u=BoundedDeficit(q.relative_depth_sigma,.1);
        const double p=std::isfinite(q.parallax_sigma)?std::max(0.,1.-q.parallax_sigma/15.):1.;
        const double n=std::max(0.,1.-double(q.observations)/4.);
        const double g=(u+p+n)/3.;
        const double age=std::max(0.,q.age_s)/(std::max(0.,q.age_s)+std::max(1e-9,q.window_span_s));
        penalty=penalty*(1.+age)+g*g;
    }
    return .25+.75/(1.+penalty);
}
LandmarkWeightQuality HiltiLandmarkWeightGeometry(const HiltiBackendJob& j,const HiltiLandmark& l,const HiltiBackendOptions& o){
    LandmarkWeightQuality q;q.window_span_s=(j.keyframes.back().timestamp_ns-j.keyframes.front().timestamp_ns)*1e-9;
    Mat33 h=Mat33::Zero();Vec3 derivative=Vec3::Zero();double depth=0,square=0;std::vector<Vec3,Eigen::aligned_allocator<Vec3>> rays;
    for(const auto& obs:l.observations){
        auto f=std::lower_bound(j.keyframes.begin(),j.keyframes.end(),obs.timestamp_ns,[](const HiltiKeyframeState& f,int64_t t){return f.timestamp_ns<t;});
        if(f==j.keyframes.end()||f->timestamp_ns!=obs.timestamp_ns)throw std::invalid_argument("weighting bearing outside native window");
        const SE3 tcw=HiltiVisualCameraWorldPose(*f,l.id.first,obs.bearing,j.cameras,j.gravity_w,o).inverse();
        const Vec3 p=tcw*l.position_w;const double range=p.norm();Vec2 projected;
        const bool valid=range>1e-10&&p.dot(obs.bearing)>0&&j.cameras[l.id.first]->Project(p,projected);
        const double angle=valid?std::atan2(p.normalized().cross(obs.bearing).norm(),p.normalized().dot(obs.bearing)):M_PI;
        square+=angle*angle;++q.observations;rays.push_back(tcw.so3().inverse()*obs.bearing);
        if(range<=1e-10)continue;const Vec3 u=p.normalized(),r=(u-obs.bearing)/o.bearing_sigma_rad;
        const Mat33 jp=(Mat33::Identity()-u*u.transpose())/(range*o.bearing_sigma_rad)*tcw.rotationMatrix();
        const double w=HiltiOfflineCameraInformation(l.id.first)*std::min(1.,o.visual_huber_sigma/std::max(r.norm(),1e-12));
        h.noalias()+=w*jp.transpose()*jp;
        if(derivative.isZero()){depth=p.dot(obs.bearing);derivative=tcw.rotationMatrix().transpose()*obs.bearing;}
    }
    q.residual_sigma=q.observations?std::sqrt(square/q.observations)/o.bearing_sigma_rad:0;
    for(size_t i=1;i<rays.size();++i)q.parallax_sigma=std::max(q.parallax_sigma,std::atan2(rays[0].cross(rays[i]).norm(),rays[0].dot(rays[i]))/o.bearing_sigma_rad);
    Eigen::SelfAdjointEigenSolver<Mat33> e(.5*(h+h.transpose()));if(e.info()!=Eigen::Success)throw std::runtime_error("landmark weight uncertainty eigen failure");
    const auto z=e.eigenvectors().transpose()*derivative;const double cutoff=1e-10*std::max(1.,e.eigenvalues().maxCoeff());double variance=0,null=0;
    for(int i=0;i<3;++i)if(e.eigenvalues()[i]>cutoff)variance+=z[i]*z[i]/e.eigenvalues()[i];else null+=z[i]*z[i];
    q.relative_depth_sigma=!q.observations||null>1e-10*std::max(1e-30,derivative.squaredNorm())?INFINITY:std::sqrt(std::max(0.,variance))/std::max(1e-12,std::abs(depth));
    return q;
}
struct HiltiLandmarkWeightWindow::Impl {
 Context context;bool combined=false,ready=false,finished=false;int64_t timestamp=0;
 std::vector<std::pair<HiltiLandmarkId,LandmarkWeightQuality>> qualities;
 Impl(const HiltiBackendJob& j,bool c):combined(c){context.version=j.version;timestamp=j.keyframes.back().timestamp_ns;auto& l=Log();for(const auto& p:j.landmarks){auto& t=l.tracks[p.id];if(!t.first)t.first=timestamp;}}

};
HiltiLandmarkWeightWindow::HiltiLandmarkWeightWindow(const HiltiBackendJob& j){
 if(active)throw std::logic_error("nested landmark weight context");impl_.reset(new Impl(j,true));
}
HiltiLandmarkWeightWindow::~HiltiLandmarkWeightWindow(){if(impl_&&active==&impl_->context)active=nullptr;}
void HiltiLandmarkWeightWindow::Ready(const HiltiBackendJob& j,const HiltiBackendOptions& o){
 if(!impl_)return;auto& x=*impl_;auto& logger=Log();
 for(const auto& l:j.landmarks){auto q=HiltiLandmarkWeightGeometry(j,l,o);q.age_s=(x.timestamp-logger.tracks.at(l.id).first)*1e-9;
    x.context.weights[l.id]=HiltiLandmarkQualityWeight(q,x.combined);x.qualities.push_back({l.id,q});}
 x.ready=true;active=&x.context;
}
void HiltiLandmarkWeightWindow::Finish(bool accepted){
 if(!impl_||!impl_->ready)return;auto& x=*impl_;if(x.finished)throw std::logic_error("duplicate weighting window completion");x.finished=true;

}
double HiltiLandmarkLiveInformation(const HiltiBackendJob& j,const HiltiLandmarkId& id,bool connected_source){
 if(!active||connected_source||j.version!=active->version)return 1.;const auto it=active->weights.find(id);if(it==active->weights.end())throw std::logic_error("landmark changed within frozen weight window");return it->second;
}
}
