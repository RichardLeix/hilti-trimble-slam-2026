#include "myslam/hilti_landmark_weighting.h"
#include "myslam/hilti_camera_information.h"
#include "myslam/hilti_vi_linearization.h"
#include "myslam/hilti_rolling_shutter.h"
#include "myslam/hilti_landmark_quarantine.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

namespace myslam {
namespace {
using V15=Eigen::Matrix<double,15,1>;
using M15=Eigen::Matrix<double,15,15>;
using J315=Eigen::Matrix<double,3,15>;
NavState Perturb(const NavState& state,int column,double amount) {
    NavState s=state; Vec3 d=Vec3::Zero(); d[column%3]=amount;
    if(column<3) s.rotation_wb=s.rotation_wb*SO3::exp(d);
    else if(column<6) s.position_wb+=d;
    else if(column<9) s.velocity_w+=d;
    else if(column<12) s.bias.gyroscope+=d;
    else s.bias.accelerometer+=d;
    return s;
}
ImuPreintegration Corrected(const ImuPreintegration& delta,const ImuBias& bias) {
    ImuPreintegration d=delta;
    Vec6 db;db.head<3>()=bias.gyroscope-d.bias.gyroscope;db.tail<3>()=bias.accelerometer-d.bias.accelerometer;
    const Vec9 correction=d.bias_jacobian*db;
    d.delta_rotation=d.delta_rotation*SO3::exp(correction.head<3>());
    d.delta_velocity+=correction.segment<3>(3);d.delta_position+=correction.tail<3>();
    return d;
}
M15 Whitening(const ImuPreintegration& delta) {
    const M15 p=0.5*(delta.covariance+delta.covariance.transpose());
    if(!p.allFinite() || p.diagonal().minCoeff()<=0) throw std::invalid_argument("IMU covariance must be positive definite");
    const V15 scale=p.diagonal().cwiseSqrt().cwiseInverse();
    M15 normalized=scale.asDiagonal()*p*scale.asDiagonal();
    Eigen::LLT<M15> llt(normalized);
    if(llt.info()!=Eigen::Success) throw std::invalid_argument("IMU covariance Cholesky failed");
    return llt.matrixL().solve(M15(scale.asDiagonal()));
}
struct CameraLinearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    SE3 T_cw;
    std::array<Mat33,15> rotation_derivatives;
    std::array<Vec3,15> translation_derivatives;
};
using CameraCache=HiltiAlignedVector<std::array<CameraLinearization,2>>;
CameraCache PrepareCameras(const HiltiBackendJob& job,bool derivatives) {
    CameraCache cache(job.keyframes.size()); const double eps=1e-6;
    for(size_t i=0;i<job.keyframes.size();++i) for(int k=0;k<2;++k) {
        auto& c=cache[i][k]; c.T_cw=HiltiCameraWorldPose(job.keyframes[i],k,job.cameras,job.gravity_w).inverse();
        if(!derivatives) continue;
        for(int j=0;j<15;++j) {
            auto a=job.keyframes[i],b=a;
            a.state=Perturb(a.state,j,eps); b.state=Perturb(b.state,j,-eps);
            const SE3 plus=HiltiCameraWorldPose(a,k,job.cameras,job.gravity_w).inverse();
            const SE3 minus=HiltiCameraWorldPose(b,k,job.cameras,job.gravity_w).inverse();
            c.rotation_derivatives[j]=(plus.rotationMatrix()-minus.rotationMatrix())/(2*eps);
            c.translation_derivatives[j]=(plus.translation()-minus.translation())/(2*eps);
        }
    }
    return cache;
}
size_t FrameIndex(const HiltiBackendJob& job,int64_t timestamp) {
    auto it=std::lower_bound(job.keyframes.begin(),job.keyframes.end(),timestamp,
        [](const HiltiKeyframeState& f,int64_t t){return f.timestamp_ns<t;});
    if(it==job.keyframes.end() || it->timestamp_ns!=timestamp) throw std::invalid_argument("observation outside keyframe snapshot");
    return size_t(it-job.keyframes.begin());
}
double Robust(double norm,double huber) {return norm<=huber?norm*norm:huber*(2*norm-huber);}
Vec9 Prior(const NavState& state,const NavState& anchor,const HiltiBackendOptions& o) {
    Vec9 r; r.head<3>()=(state.velocity_w-anchor.velocity_w)/o.velocity_prior_sigma;
    r.segment<3>(3)=(state.bias.gyroscope-anchor.bias.gyroscope)/o.gyro_prior_sigma;
    r.tail<3>()=(state.bias.accelerometer-anchor.bias.accelerometer)/o.accel_prior_sigma;return r;
}
HiltiBackendDiagnostics Measure(const HiltiBackendJob& job,const HiltiBackendOptions& o,
        const std::vector<M15,Eigen::aligned_allocator<M15>>& whitening,const NavState& anchor) {
    HiltiBackendDiagnostics d;d.window_size=job.keyframes.size();d.landmarks=job.landmarks.size();
    const auto cameras=PrepareCameras(job,false); double visual=0,imu=0,bias=0;
    for(const auto& landmark:job.landmarks) for(const auto& obs:landmark.observations) {
        const size_t i=FrameIndex(job,obs.timestamp_ns);
        const SE3 tcw=o.center_row_rolling_shutter?
            HiltiVisualCameraWorldPose(job.keyframes[i],landmark.id.first,obs.bearing,job.cameras,job.gravity_w,o).inverse():
            cameras[i][landmark.id.first].T_cw;
        const Vec3 p=tcw*landmark.position_w;
        Vec2 pixel;
        const bool valid=p.norm()>1e-10 && p.dot(obs.bearing)>0 && job.cameras[landmark.id.first]->Project(p,pixel);
        const double chord=valid?(p.normalized()-obs.bearing).norm():2;
        const double angle=valid?std::atan2(p.normalized().cross(obs.bearing).norm(),p.normalized().dot(obs.bearing)):M_PI;
        d.cost_before+=HiltiOfflineCameraInformation(landmark.id.first)*HiltiLandmarkLiveInformation(job,landmark.id)*Robust(chord/o.bearing_sigma_rad,o.visual_huber_sigma);
        visual+=angle*angle;++d.visual_observations;++d.camera_observations[landmark.id.first];
    }
    for(size_t i=0;i<job.intervals.size();++i) {
        const V15 r=whitening[i]*HiltiImuResidual(job.keyframes[i].state,job.keyframes[i+1].state,job.intervals[i],job.gravity_w);
        d.cost_before+=r.squaredNorm();imu+=r.head<9>().squaredNorm();bias+=r.tail<6>().squaredNorm();
    }
    d.cost_before+=job.prior?job.prior->Evaluate(HiltiPriorVariables(job)).residual.squaredNorm():Prior(job.keyframes[0].state,anchor,o).squaredNorm();
    d.visual_before_rad=d.visual_observations?std::sqrt(visual/d.visual_observations):0;
    d.imu_before=std::sqrt(imu/(9*job.intervals.size()));d.bias_before=std::sqrt(bias/(6*job.intervals.size()));
    return d;
}
HiltiBackendJob Step(const HiltiBackendJob& current,const HiltiBackendOptions& o,
        double damping,double& step_norm) {
    auto normal=LinearizeHiltiGraph(current,o);
    const int n=int(current.keyframes.size())*15,np=int(current.landmarks.size())*3;
    
    MatXX h=std::move(normal.hessian);VecX rhs=-normal.gradient;
    h.diagonal()+=damping*h.diagonal().cwiseMax(1e-8);
    MatXX reduced=h.topLeftCorner(n,n);VecX reduced_rhs=rhs.head(n);
    const MatXX cross=h.topRightCorner(n,np);
    MatXX inverse_cross(np,n);VecX inverse_rhs(np);
    if(current.prior) {
        // A marginal prior correlates retained landmarks. Eliminating each 3x3
        // block independently would discard those cross terms.
        Eigen::LDLT<MatXX> points(h.bottomRightCorner(np,np));
        inverse_cross=points.solve(cross.transpose());inverse_rhs=points.solve(rhs.tail(np));
    } else {
        for(int i=0;i<np;i+=3) {
            const auto inverse=h.block<3,3>(n+i,n+i).ldlt().solve(Mat33::Identity()).eval();
            inverse_cross.middleRows(i,3)=inverse*cross.middleCols(i,3).transpose();
            inverse_rhs.segment<3>(i)=inverse*rhs.segment<3>(n+i);
        }
    }
    reduced.noalias()-=cross*inverse_cross;reduced_rhs.noalias()-=cross*inverse_rhs;
    // The initial pose is fixed only until its gauge is transferred into the
    // prior. Later windows have no additional hard pose anchor.
    if(!current.prior || current.prior->has_fixed_pose) {
        const int64_t timestamp=current.prior?current.prior->fixed_pose_timestamp_ns:current.keyframes.front().timestamp_ns;
        const int offset=int(FrameIndex(current,timestamp))*15;
        reduced.middleRows(offset,6).setZero();reduced.middleCols(offset,6).setZero();
        reduced.block<6,6>(offset,offset).setIdentity();reduced_rhs.segment<6>(offset).setZero();
    }
    const VecX dx=reduced.ldlt().solve(reduced_rhs);
    const VecX dp=inverse_rhs-inverse_cross*dx;
    if(!dx.allFinite() || !dp.allFinite())throw std::runtime_error("nonfinite Hilti Schur solve");
    HiltiBackendJob candidate=current;step_norm=dx.norm();
    for(size_t i=0;i<candidate.keyframes.size();++i)
        candidate.keyframes[i].state=RetractNavState(candidate.keyframes[i].state,dx.segment<15>(int(i)*15));
    for(size_t l=0;l<candidate.landmarks.size();++l)candidate.landmarks[l].position_w+=dp.segment<3>(int(l)*3);
    
    return candidate;
}
void Validate(const HiltiBackendJob& job,const HiltiBackendOptions& options) {
    if(job.keyframes.size()<3 || job.keyframes.size()>options.window_size || job.intervals.size()+1!=job.keyframes.size())
        throw std::invalid_argument("invalid chronological Hilti window/interval count");
    if(!job.gravity_w.allFinite() || !job.cameras[0] || !job.cameras[1]) throw std::invalid_argument("missing Hilti calibration/gravity");
    for(size_t i=0;i<job.keyframes.size();++i) {
        const auto& f=job.keyframes[i];
        if(options.center_row_rolling_shutter)for(int k=0;k<2;++k)
            if(!f.exposure_angular_velocity[k].allFinite() || job.cameras[k]->Height()!=HiltiReadoutImageHeight)
                throw std::invalid_argument("missing Hilti row-time gyro or incompatible image height");
        if(!f.state.Pose().matrix().allFinite() || !f.state.velocity_w.allFinite() ||
            !f.state.bias.gyroscope.allFinite() || !f.state.bias.accelerometer.allFinite() ||
            f.exposure_timestamps_ns[0]!=f.timestamp_ns || (i && f.timestamp_ns<=job.keyframes[i-1].timestamp_ns))
            throw std::invalid_argument("invalid Hilti keyframe state/ordering/exposure");
        if(i && (job.intervals[i-1].begin_timestamp_ns!=job.keyframes[i-1].timestamp_ns ||
            job.intervals[i-1].end_timestamp_ns!=f.timestamp_ns || job.intervals[i-1].dt<=0))
            throw std::invalid_argument("disconnected Hilti IMU interval");
    }
    std::set<HiltiLandmarkId> ids;
    for(const auto& l:job.landmarks) {
        if(l.id.first<0 || l.id.first>1 || !l.position_w.allFinite() || !l.initialized || !ids.insert(l.id).second)
            throw std::invalid_argument("invalid/duplicate Hilti landmark");
        int64_t last=0;bool first=true;
        for(const auto& obs:l.observations) {
            FrameIndex(job,obs.timestamp_ns);
            if(!obs.bearing.allFinite() || std::abs(obs.bearing.norm()-1)>1e-8 || (!first && obs.timestamp_ns<=last))
                throw std::invalid_argument("invalid Hilti bearing archive");
            Vec2 pixel;if(!job.cameras[l.id.first]->Project(obs.bearing,pixel)) throw std::invalid_argument("bearing outside EUCM domain/mask");
            last=obs.timestamp_ns;first=false;
        }
    }
}
} // namespace

Eigen::Matrix<double,15,1> HiltiImuResidual(const NavState& i,const NavState& j,
        const ImuPreintegration& delta,const Vec3& gravity) {
    const auto d=Corrected(delta,i.bias);V15 r;
    r.head<3>()=(d.delta_rotation.inverse()*i.rotation_wb.inverse()*j.rotation_wb).log();
    r.segment<3>(3)=i.rotation_wb.inverse()*(j.velocity_w-i.velocity_w-gravity*d.dt)-d.delta_velocity;
    r.segment<3>(6)=i.rotation_wb.inverse()*(j.position_wb-i.position_wb-i.velocity_w*d.dt-0.5*gravity*d.dt*d.dt)-d.delta_position;
    r.segment<3>(9)=j.bias.gyroscope-i.bias.gyroscope;r.tail<3>()=j.bias.accelerometer-i.bias.accelerometer;
    return r;
}
SE3 HiltiCameraWorldPose(const HiltiKeyframeState& frame,int camera,const HiltiCameras& cameras,const Vec3& gravity) {
    if(camera<0 || camera>1 || !cameras[camera]) throw std::invalid_argument("invalid Hilti camera ID");
    const double h=double(frame.exposure_timestamps_ns[camera]-frame.timestamp_ns)*1e-9;
    const auto d=Corrected(frame.exposure_deltas[camera],frame.state.bias);
    SO3 a;Vec3 b=Vec3::Zero();
    if(h>0) {a=d.delta_rotation;b=d.delta_position;}
    else if(h<0) {a=d.delta_rotation.inverse();b=a*(d.delta_velocity*d.dt-d.delta_position);}
    const auto& s=frame.state;
    return SE3(s.rotation_wb*a,s.position_wb+s.velocity_w*h+0.5*gravity*h*h+s.rotation_wb*b)*cameras[camera]->pose().inverse();
}
Vec3 HiltiBearingResidual(const HiltiKeyframeState& frame,const Vec3& point,int camera,
        const Vec3& ray,const HiltiCameras& cameras,const Vec3& gravity) {
    const Vec3 p=HiltiCameraWorldPose(frame,camera,cameras,gravity).inverse()*point;
    if(p.norm()<1e-10) throw std::invalid_argument("landmark at camera centre");
    return p.normalized()-ray;
}
HiltiBackendResult OptimizeHiltiWindow(const HiltiBackendJob& input,const HiltiBackendOptions& options,
        HiltiLandmarkQuarantine* quarantine) {
    const auto start=std::chrono::steady_clock::now();Validate(input,options);
    HiltiLandmarkWeightWindow landmark_weights(input);
    
    
    
    HiltiBackendResult result;result.version=input.version;result.base_version=input.base_version;
    HiltiBackendJob current=input;
    if(options.experimental_landmark_quarantine_B) {
        if(!quarantine)throw std::invalid_argument("experimental B requires a backend-owned quarantine context");
        quarantine->Apply(current,options,result.retired_landmarks);
    }
    double marginalization_ms=0;
    if(options.enable_marginalization && current.marginalization_source) {
        const auto begin=std::chrono::steady_clock::now();
        
        current.prior=MarginalizeHiltiGraph(*current.marginalization_source,current,options,
            result.absorbed_observations,result.retired_landmarks);
        
        marginalization_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
    if(!options.enable_marginalization)current.prior.reset();
    current.marginalization_source.reset();
    if(!current.has_initial_anchor){current.initial_anchor=current.keyframes.front().state;current.has_initial_anchor=true;}
    std::set<HiltiLandmarkId> prior_points;
    if(current.prior)for(const auto& v:current.prior->linearization)
        if(v.key.kind==MarginalizationKey::Kind::Landmark)prior_points.insert({v.key.camera_id,v.key.feature_id});
    // Reject bad ray correspondences locally, preserving the frontend archive.
    for(auto& landmark:current.landmarks) {
        auto& obs=landmark.observations;
        obs.erase(std::remove_if(obs.begin(),obs.end(),[&](const HiltiBearingObservation& o){
            const auto& f=current.keyframes[FrameIndex(current,o.timestamp_ns)];
            const Vec3 p=HiltiVisualCameraWorldPose(f,landmark.id.first,o.bearing,current.cameras,current.gravity_w,options).inverse()*landmark.position_w;
            Vec2 pixel;return p.norm()<1e-8 || p.dot(o.bearing)<=0 || !current.cameras[landmark.id.first]->Project(p,pixel) ||
                std::atan2(p.normalized().cross(o.bearing).norm(),p.normalized().dot(o.bearing))>options.max_initial_bearing_rad;
        }),obs.end());
    }
    current.landmarks.erase(std::remove_if(current.landmarks.begin(),current.landmarks.end(),
        [&](const HiltiLandmark& l){return l.observations.size()<2 && !prior_points.count(l.id);}),current.landmarks.end());
    if(current.landmarks.size()<options.min_landmarks) {
        result.diagnostics.window_size=current.keyframes.size();result.diagnostics.landmarks=current.landmarks.size();
        result.diagnostics.reason="insufficient persistent multi-keyframe landmarks";
    } else {
        std::vector<M15,Eigen::aligned_allocator<M15>> whitening;
        for(const auto& d:current.intervals) whitening.push_back(Whitening(d));
        const NavState anchor=current.initial_anchor;
        
        
        landmark_weights.Ready(current,options);
        result.diagnostics=Measure(current,options,whitening,anchor);
        double cost=result.diagnostics.cost_before,damping=1e-4;
        for(size_t iteration=0;iteration<options.max_iterations;++iteration) {
            double norm=0;auto candidate=Step(current,options,damping,norm);
            const auto score=Measure(candidate,options,whitening,anchor);++result.diagnostics.iterations;
            
            if(std::isfinite(score.cost_before) && score.cost_before<=cost) {
                const double improvement=cost-score.cost_before;
                current=std::move(candidate);cost=score.cost_before;damping=std::max(1e-9,damping/3);
                if(norm<1e-7 || improvement<1e-8*std::max(1.0,cost)) break;
            } else damping*=10;
        }
        const auto after=Measure(current,options,whitening,anchor);auto& d=result.diagnostics;
        
        
        d.cost_after=after.cost_before;d.visual_after_rad=after.visual_before_rad;d.imu_after=after.imu_before;d.bias_after=after.bias_before;
        for(size_t i=0;i<current.keyframes.size();++i) {
            const auto& a=input.keyframes[i].state;const auto& b=current.keyframes[i].state;
            d.pose_update_m=std::max(d.pose_update_m,(b.position_wb-a.position_wb).norm());
            d.rotation_update_rad=std::max(d.rotation_update_rad,(a.rotation_wb.inverse()*b.rotation_wb).log().norm());
            d.velocity_update=std::max(d.velocity_update,(b.velocity_w-a.velocity_w).norm());
            d.gyro_update=std::max(d.gyro_update,(b.bias.gyroscope-a.bias.gyroscope).norm());
            d.accel_update=std::max(d.accel_update,(b.bias.accelerometer-a.bias.accelerometer).norm());
        }
        d.accepted=std::isfinite(d.cost_after) && d.cost_after<=d.cost_before+1e-8 &&
            d.pose_update_m<=options.max_pose_update_m && d.rotation_update_rad<=options.max_rotation_update_rad &&
            d.velocity_update<=options.max_velocity_update && d.gyro_update<=options.max_gyro_update && d.accel_update<=options.max_accel_update;
        if(d.accepted)d.reason=current.prior?"joint optimization accepted with fixed-lag prior":"joint optimization accepted; initial pose fixed";
        else {
            std::ostringstream reason;reason<<"window rejected:";
            if(!std::isfinite(d.cost_after))reason<<" nonfinite cost;";
            if(d.cost_after>d.cost_before+1e-8)reason<<" increasing cost;";
            if(d.pose_update_m>options.max_pose_update_m)reason<<" position correction "<<d.pose_update_m<<" > "<<options.max_pose_update_m<<" m;";
            if(d.rotation_update_rad>options.max_rotation_update_rad)reason<<" rotation correction "<<d.rotation_update_rad<<" > "<<options.max_rotation_update_rad<<" rad;";
            if(d.velocity_update>options.max_velocity_update)reason<<" velocity correction "<<d.velocity_update<<" > "<<options.max_velocity_update<<" m/s;";
            if(d.gyro_update>options.max_gyro_update)reason<<" gyro correction "<<d.gyro_update<<" > "<<options.max_gyro_update<<" rad/s;";
            if(d.accel_update>options.max_accel_update)reason<<" accel correction "<<d.accel_update<<" > "<<options.max_accel_update<<" m/s^2;";
            d.reason=reason.str();
        }
        if(d.accepted) {
            result.keyframes=current.keyframes;result.landmarks=current.landmarks;
            result.prior=current.prior;result.factor_snapshot=std::make_shared<const HiltiBackendJob>(current);
        }
    }
    if(current.prior) {
        auto& d=result.diagnostics;d.marginalizations=current.prior->marginalizations;
        d.marginalized_landmarks=current.prior->marginalized_landmarks;
        d.prior_dimension=current.prior->Dimension();d.prior_rank=current.prior->rank;d.prior_condition=current.prior->condition;
    }
    landmark_weights.Finish(result.diagnostics.accepted);
    result.diagnostics.marginalization_ms=marginalization_ms;
    result.diagnostics.runtime_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return result;
}
} // namespace myslam
