#include "myslam/hilti_landmark_weighting.h"
#include "myslam/hilti_camera_information.h"
#include "myslam/hilti_vi_linearization.h"
#include "myslam/hilti_rolling_shutter.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace myslam {
namespace {
using M15=NavLocalMatrix;
using V15=NavLocalVector;
using J315=Eigen::Matrix<double,3,15>;
struct CameraLinearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    SE3 T_cw;std::array<Mat33,15> rotation_derivatives;std::array<Vec3,15> translation_derivatives;
};
using CameraCache=HiltiAlignedVector<std::array<CameraLinearization,2>>;
CameraCache Prepare(const HiltiBackendJob& job) {
    CameraCache cache(job.keyframes.size());const double eps=1e-6;
    for(size_t i=0;i<job.keyframes.size();++i)for(int k=0;k<2;++k) {
        auto& c=cache[i][k];c.T_cw=HiltiCameraWorldPose(job.keyframes[i],k,job.cameras,job.gravity_w).inverse();
        for(int j=0;j<15;++j) {
            auto a=job.keyframes[i],b=a;V15 d=V15::Zero();d[j]=eps;
            a.state=RetractNavState(a.state,d);b.state=RetractNavState(b.state,-d);
            const SE3 plus=HiltiCameraWorldPose(a,k,job.cameras,job.gravity_w).inverse();
            const SE3 minus=HiltiCameraWorldPose(b,k,job.cameras,job.gravity_w).inverse();
            c.rotation_derivatives[j]=(plus.rotationMatrix()-minus.rotationMatrix())/(2*eps);
            c.translation_derivatives[j]=(plus.translation()-minus.translation())/(2*eps);
        }
    }
    return cache;
}
std::map<MarginalizationKey,int> Offsets(const MarginalizationVariables& variables) {
    std::map<MarginalizationKey,int> result;int offset=0;
    for(const auto& v:variables){if(!result.emplace(v.key,offset).second)throw std::invalid_argument("duplicate Hilti graph variable");offset+=v.Dimension();}
    return result;
}

}
Eigen::Matrix<double,15,15> HiltiImuWhitening(const ImuPreintegration& delta) {
    const M15 p=0.5*(delta.covariance+delta.covariance.transpose());
    if(!p.allFinite() || p.diagonal().minCoeff()<=0)throw std::invalid_argument("IMU covariance must be positive definite");
    const V15 scale=p.diagonal().cwiseSqrt().cwiseInverse();const M15 normalized=scale.asDiagonal()*p*scale.asDiagonal();
    Eigen::LLT<M15> llt(normalized);if(llt.info()!=Eigen::Success)throw std::invalid_argument("IMU covariance Cholesky failed");
    return llt.matrixL().solve(M15(scale.asDiagonal()));
}
MarginalizationVariables HiltiGraphVariables(const HiltiBackendJob& job) {
    MarginalizationVariables variables;
    for(const auto& f:job.keyframes){MarginalizationVariable v;v.key=MarginalizationKey::State(f.timestamp_ns);v.state=f.state;variables.push_back(v);}
    for(const auto& l:job.landmarks){MarginalizationVariable v;v.key=MarginalizationKey::Landmark(l.id.first,l.id.second);v.point_w=l.position_w;variables.push_back(v);}
    return variables;
}
MarginalizationVariables HiltiPriorVariables(const HiltiBackendJob& job) {
    MarginalizationVariables current;if(!job.prior)return current;
    const auto all=HiltiGraphVariables(job);std::map<MarginalizationKey,MarginalizationVariable> by_key;
    for(const auto& v:all)by_key.emplace(v.key,v);
    for(const auto& anchor:job.prior->linearization) {
        auto it=by_key.find(anchor.key);if(it==by_key.end())throw std::invalid_argument("prior references a missing active state/landmark");
        current.push_back(it->second);
    }
    return current;
}
HiltiNormalEquations LinearizeHiltiGraph(const HiltiBackendJob& job,const HiltiBackendOptions& o,
        const std::set<MarginalizationKey>& connected) {
    HiltiNormalEquations normal;normal.variables=HiltiGraphVariables(job);const auto offsets=Offsets(normal.variables);
    const int ns=int(job.keyframes.size())*15,n=ns+int(job.landmarks.size())*3;
    normal.hessian=MatXX::Zero(n,n);normal.gradient=VecX::Zero(n);
    
    const auto cameras=o.center_row_rolling_shutter?CameraCache():Prepare(job);
    std::map<int64_t,size_t> indices;for(size_t i=0;i<job.keyframes.size();++i)indices.emplace(job.keyframes[i].timestamp_ns,i);
    for(size_t l=0;l<job.landmarks.size();++l) {
        const auto& landmark=job.landmarks[l];const int ip=ns+int(l)*3;
        const bool selected_landmark=connected.count(MarginalizationKey::Landmark(landmark.id.first,landmark.id.second));
        for(const auto& obs:landmark.observations) {
            if(!connected.empty() && !selected_landmark && !connected.count(MarginalizationKey::State(obs.timestamp_ns)))continue;
            const auto at=indices.find(obs.timestamp_ns);if(at==indices.end())throw std::invalid_argument("bearing outside Hilti linearization graph");
            const int i=int(at->second),ix=15*i;
            CameraLinearization row_camera;
            if(o.center_row_rolling_shutter) {
                // Measured row and raw gyro stay fixed while the full NavState
                // is perturbed. This same visual normal forms outgoing priors.
                const int k=landmark.id.first;Vec2 pixel;
                if(!job.cameras[k]->Project(obs.bearing,pixel))throw std::invalid_argument("invalid Hilti row-time bearing");
                const auto& frame=job.keyframes[i];
                row_camera.T_cw=HiltiRowCameraWorldPose(frame,k,job.cameras,job.gravity_w,pixel.y()).inverse();
                for(int a=0;a<15;++a) {
                    V15 dx=V15::Zero();dx[a]=1e-6;auto plus_frame=frame,minus_frame=frame;
                    plus_frame.state=RetractNavState(frame.state,dx);minus_frame.state=RetractNavState(frame.state,-dx);
                    const SE3 plus=HiltiRowCameraWorldPose(plus_frame,k,job.cameras,job.gravity_w,pixel.y()).inverse();
                    const SE3 minus=HiltiRowCameraWorldPose(minus_frame,k,job.cameras,job.gravity_w,pixel.y()).inverse();
                    row_camera.rotation_derivatives[a]=(plus.rotationMatrix()-minus.rotationMatrix())/2e-6;
                    row_camera.translation_derivatives[a]=(plus.translation()-minus.translation())/2e-6;
                }
            }
            const auto& c=o.center_row_rolling_shutter?row_camera:cameras[i][landmark.id.first];
            const Vec3 p=c.T_cw*landmark.position_w;if(p.norm()<1e-10)throw std::runtime_error("marginalization landmark at camera centre");
            const Vec3 u=p.normalized(),r=(u-obs.bearing)/o.bearing_sigma_rad;
            const Mat33 unit=(Mat33::Identity()-u*u.transpose())/(p.norm()*o.bearing_sigma_rad);
            J315 j;for(int a=0;a<15;++a)j.col(a)=unit*(c.rotation_derivatives[a]*landmark.position_w+c.translation_derivatives[a]);
            const Mat33 jp=unit*c.T_cw.rotationMatrix();const double information=HiltiOfflineCameraInformation(landmark.id.first)*HiltiLandmarkLiveInformation(job,landmark.id,!connected.empty());const double w=information*std::min(1.0,o.visual_huber_sigma/std::max(r.norm(),1e-12));
            normal.hessian.block<15,15>(ix,ix).noalias()+=w*j.transpose()*j;
            normal.hessian.block<3,3>(ip,ip).noalias()+=w*jp.transpose()*jp;
            normal.hessian.block<15,3>(ix,ip).noalias()+=w*j.transpose()*jp;
            normal.hessian.block<3,15>(ip,ix).noalias()+=w*jp.transpose()*j;
            if(o.center_row_rolling_shutter) {
                // Preserve the validated prototype's accumulation arithmetic.
                const V15 gs=w*j.transpose()*r;const Vec3 gp=w*jp.transpose()*r;
                normal.gradient.segment<15>(ix)+=gs;normal.gradient.segment<3>(ip)+=gp;
                
            } else {
                normal.gradient.segment<15>(ix).noalias()+=w*j.transpose()*r;
                normal.gradient.segment<3>(ip).noalias()+=w*jp.transpose()*r;
                
            }
        }
    }
    
    for(size_t i=0;i<job.intervals.size();++i) {
        if(!connected.empty() && !connected.count(MarginalizationKey::State(job.keyframes[i].timestamp_ns)) &&
            !connected.count(MarginalizationKey::State(job.keyframes[i+1].timestamp_ns)))continue;
        const auto& a=job.keyframes[i].state;const auto& b=job.keyframes[i+1].state;const auto& d=job.intervals[i];
        const M15 w=HiltiImuWhitening(d);const V15 r=w*HiltiImuResidual(a,b,d,job.gravity_w);M15 ja,jb;const double eps=1e-6;
        for(int c=0;c<15;++c) {
            V15 increment=V15::Zero();increment[c]=eps;
            ja.col(c)=w*(HiltiImuResidual(RetractNavState(a,increment),b,d,job.gravity_w)-HiltiImuResidual(RetractNavState(a,-increment),b,d,job.gravity_w))/(2*eps);
            jb.col(c)=w*(HiltiImuResidual(a,RetractNavState(b,increment),d,job.gravity_w)-HiltiImuResidual(a,RetractNavState(b,-increment),d,job.gravity_w))/(2*eps);
        }
        const int ia=int(i)*15,ib=ia+15;
        normal.hessian.block<15,15>(ia,ia).noalias()+=ja.transpose()*ja;normal.hessian.block<15,15>(ib,ib).noalias()+=jb.transpose()*jb;
        normal.hessian.block<15,15>(ia,ib).noalias()+=ja.transpose()*jb;normal.hessian.block<15,15>(ib,ia).noalias()+=jb.transpose()*ja;
        normal.gradient.segment<15>(ia).noalias()+=ja.transpose()*r;normal.gradient.segment<15>(ib).noalias()+=jb.transpose()*r;
    }
    
    if(job.prior) {
        const auto evaluation=job.prior->Evaluate(HiltiPriorVariables(job));int row=0;
        for(const auto& a:job.prior->linearization) {
            const int ia=offsets.at(a.key),na=a.Dimension();normal.gradient.segment(ia,na)+=evaluation.gradient.segment(row,na);
            int col=0;for(const auto& b:job.prior->linearization) {
                normal.hessian.block(ia,offsets.at(b.key),na,b.Dimension())+=evaluation.hessian.block(row,col,na,b.Dimension());col+=b.Dimension();
            }
            row+=na;
        }
    } else {
        const auto& state=job.keyframes.front().state;
        const NavState& anchor=job.has_initial_anchor?job.initial_anchor:state;
        const Vec9 residual=(NavStateLocalDelta(anchor,state)).tail<9>();
        for(int c=0;c<9;++c) {
            const double sigma=c<3?o.velocity_prior_sigma:c<6?o.gyro_prior_sigma:o.accel_prior_sigma;
            normal.hessian(6+c,6+c)+=1/(sigma*sigma);normal.gradient[6+c]+=residual[c]/(sigma*sigma);
        }
    }
    
    return normal;
}
std::shared_ptr<const MarginalizationPrior> MarginalizeHiltiGraph(const HiltiBackendJob& source,
        const HiltiBackendJob& retained,const HiltiBackendOptions& options,
        std::set<HiltiObservationId>& absorbed,std::set<HiltiLandmarkId>& retired) {
    const auto keep_variables=HiltiGraphVariables(retained);std::set<MarginalizationKey> keep;
    for(const auto& v:keep_variables)keep.insert(v.key);
    std::set<MarginalizationKey> drop;size_t old_states=0,old_points=0;
    for(const auto& v:HiltiGraphVariables(source))if(!keep.count(v.key)) {
        drop.insert(v.key);
        if(v.Dimension()==15)++old_states;else {++old_points;retired.insert({v.key.camera_id,v.key.feature_id});}
    }
    if(drop.empty())return source.prior;
    auto normal=LinearizeHiltiGraph(source,options,drop);MarginalizationInfo info;
    info.variables=std::move(normal.variables);info.hessian=std::move(normal.hessian);info.gradient=std::move(normal.gradient);info.eliminate=drop;
    const bool fixed=!source.prior || source.prior->has_fixed_pose;
    const int64_t gauge=source.prior?source.prior->fixed_pose_timestamp_ns:source.keyframes.front().timestamp_ns;
    const bool gauge_leaves=fixed && drop.count(MarginalizationKey::State(gauge));
    if(gauge_leaves) {
        auto it=std::find_if(source.keyframes.begin(),source.keyframes.end(),[&](const HiltiKeyframeState& f){return f.timestamp_ns==gauge;});
        if(it==source.keyframes.end())throw std::invalid_argument("fixed gauge missing from marginalization source");
        const int offset=int(it-source.keyframes.begin())*15;for(int i=0;i<6;++i)info.fixed_coordinates.insert(offset+i);
    }
    for(const auto& l:source.landmarks)for(const auto& obs:l.observations)
        if(drop.count(MarginalizationKey::State(obs.timestamp_ns)) || drop.count(MarginalizationKey::Landmark(l.id.first,l.id.second)))
            absorbed.insert({l.id,obs.timestamp_ns});
    
    auto prior=info.Marginalize(options.marginalization_rank_tolerance);
    
    prior.has_fixed_pose=fixed && !gauge_leaves;prior.fixed_pose_timestamp_ns=prior.has_fixed_pose?gauge:0;
    prior.marginalizations=(source.prior?source.prior->marginalizations:0)+old_states;
    prior.marginalized_landmarks=(source.prior?source.prior->marginalized_landmarks:0)+old_points;
    if(prior.rank==0)throw std::runtime_error("marginalization produced an empty-information prior");
    return std::make_shared<const MarginalizationPrior>(std::move(prior));
}
} // namespace myslam
