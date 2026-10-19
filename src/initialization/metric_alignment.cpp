#include "metric_alignment.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace myslam {
namespace metric_audit {
namespace {
struct LinearFit {
    VecX x;
    int rank = 0;
    double condition = INFINITY;
    bool Usable(int columns, double limit) const {
        return rank == columns && condition <= limit && x.allFinite();
    }
};
// Verbatim production solve normalization/rank threshold, isolated from runtime.
LinearFit Solve(const MatXX& a, const VecX& b) {
    if (!a.allFinite() || !b.allFinite()) throw std::invalid_argument("nonfinite initialization system");
    VecX norms(a.cols());
    for (int i=0;i<a.cols();++i) norms[i]=std::max(a.col(i).norm(),1e-15);
    const MatXX normalized=a*norms.cwiseInverse().asDiagonal();
    Eigen::JacobiSVD<MatXX> svd(normalized,Eigen::ComputeThinU|Eigen::ComputeThinV);
    svd.setThreshold(1e-8);
    LinearFit result;result.rank=int(svd.rank());const auto&values=svd.singularValues();
    if(values.size()==a.cols()&&values[values.size()-1]>1e-15)result.condition=values[0]/values[values.size()-1];
    result.x=svd.solve(b).cwiseQuotient(norms);return result;
}
VioImuIntervals Integrate(const VioVisualFrames&frames,const std::vector<ImuSample>&samples,
                         const ImuCalibration&calibration,const ImuBias&bias) {
    VioImuIntervals result;
    for(size_t i=1;i<frames.size();++i)result.push_back(PreintegrateImu(samples,frames[i-1].timestamp_ns,frames[i].timestamp_ns,calibration,bias));
    return result;
}
Eigen::Matrix<double,3,2> GravityBasis(const Vec3&gravity) {
    const Vec3 direction=gravity.normalized();Eigen::Matrix<double,3,2>basis;
    basis.col(0)=direction.unitOrthogonal();basis.col(1)=direction.cross(basis.col(0));return basis;
}
}
System BuildSystem(const VioVisualFrames&frames,const VioImuIntervals&corrected,
                   const SE3&T_camera_body,bool fit_accel_bias) {
    const size_t n=frames.size();
    if(n<2||corrected.size()+1!=n)throw std::invalid_argument("invalid diagnostic frame/interval count");
    System s;s.velocity_columns=3*n;s.gravity_column=3*n;s.scale_column=3*n+3;
    s.accel_column=fit_accel_bias?s.scale_column+1:-1;
    const int columns=s.scale_column+1+(fit_accel_bias?3:0);
    s.a=MatXX::Zero(6*(n-1),columns);s.b=VecX::Zero(6*(n-1));
    const Vec3 p_body_camera=T_camera_body.inverse().translation();
    // The blocks and weighting below are the production make_system equations.
    for(size_t i=0;i+1<n;++i) {
        const double h=corrected[i].dt;
        if(h<=0)throw std::invalid_argument("nonpositive preintegration interval");
        const Mat33 ri=frames[i].rotation_wb.matrix(),rj=frames[i+1].rotation_wb.matrix();
        const Vec3 dc=frames[i+1].camera_position-frames[i].camera_position;const int row=6*i;
        s.a.block<3,3>(row,3*i)=h*Mat33::Identity();
        s.a.block<3,3>(row,s.gravity_column)=0.5*h*h*Mat33::Identity();
        s.a.block<3,1>(row,s.scale_column)=-dc;
        s.b.segment<3>(row)=-ri*corrected[i].delta_position-(rj-ri)*p_body_camera;
        s.a.block<3,3>(row+3,3*i)=-Mat33::Identity();
        s.a.block<3,3>(row+3,3*(i+1))=Mat33::Identity();
        s.a.block<3,3>(row+3,s.gravity_column)=-h*Mat33::Identity();
        s.b.segment<3>(row+3)=ri*corrected[i].delta_velocity;
        if(fit_accel_bias) {
            s.a.block<3,3>(row,s.accel_column)=ri*corrected[i].bias_jacobian.block<3,3>(6,3);
            s.a.block<3,3>(row+3,s.accel_column)=-ri*corrected[i].bias_jacobian.block<3,3>(3,3);
        }
        s.a.middleRows(row,3)/=h;s.b.segment<3>(row)/=h;
        s.a.middleRows(row,6)/=std::sqrt(h);s.b.segment(row,6)/=std::sqrt(h);
    }
    return s;
}
Fit SolveAndDiagnose(const MatXX&a,const VecX&b,int nuisance) {
    const auto linear=Solve(a,b);Fit f;f.x=linear.x;f.rank=linear.rank;f.condition=linear.condition;
    f.column_norms.resize(a.cols());for(int i=0;i<a.cols();++i)f.column_norms[i]=std::max(a.col(i).norm(),1e-15);
    const MatXX normalized=a*f.column_norms.cwiseInverse().asDiagonal();
    Eigen::JacobiSVD<MatXX>svd(normalized,Eigen::ComputeThinU|Eigen::ComputeThinV);
    svd.setThreshold(1e-8);f.singular_values=svd.singularValues();
    Eigen::JacobiSVD<MatXX>raw(a);f.raw_singular_values=raw.singularValues();
    VecX inverse=VecX::Zero(svd.singularValues().size());
    for(int i=0;i<inverse.size();++i)if(svd.singularValues()[i]>1e-8*svd.singularValues()[0])inverse[i]=1/std::pow(svd.singularValues()[i],2);
    f.normalized_covariance=svd.matrixV()*inverse.asDiagonal()*svd.matrixV().transpose();
    f.covariance_unit_noise=f.column_norms.cwiseInverse().asDiagonal()*f.normalized_covariance*f.column_norms.cwiseInverse().asDiagonal();
    const VecX residual=a*f.x-b;f.weighted_rms=std::sqrt(residual.squaredNorm()/a.rows());
    f.residual_variance=residual.squaredNorm()/std::max(1,int(a.rows())-f.rank);
    // Project out velocities with an SVD rather than subtracting large normal
    // blocks. Remaining physical columns are g,s,ba (or g,s / s).
    const MatXX av=a.leftCols(nuisance),theta=a.rightCols(a.cols()-nuisance);
    Eigen::JacobiSVD<MatXX>vs(av,Eigen::ComputeThinU|Eigen::ComputeThinV);
    const MatXX profile=theta-av*vs.solve(theta);
    f.profiled_normal=profile.transpose()*profile;
    VecX norms(profile.cols());for(int i=0;i<norms.size();++i)norms[i]=std::max(profile.col(i).norm(),1e-15);
    const MatXX pn=profile*norms.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<MatXX>eigen(pn.transpose()*pn);
    f.profiled_normal_eigenvalues=eigen.eigenvalues();
    Eigen::JacobiSVD<MatXX>ps(profile,Eigen::ComputeThinU|Eigen::ComputeThinV);
    ps.setThreshold(1e-8);VecX pi=VecX::Zero(ps.singularValues().size());
    for(int i=0;i<pi.size();++i)if(ps.singularValues()[i]>1e-8*ps.singularValues()[0])pi[i]=1/std::pow(ps.singularValues()[i],2);
    f.profiled_covariance=ps.matrixV()*pi.asDiagonal()*ps.matrixV().transpose();return f;
}
VecX PhysicalResidual(const VioVisualFrames&frames,const VioImuIntervals&deltas,
                     const SE3&T_camera_body,const VecX&velocities,const Vec3&gravity,
                     double scale,const Vec3&accel_bias) {
    VecX residual(6*(frames.size()-1));const Vec3 lever=T_camera_body.inverse().translation();
    for(size_t i=0;i+1<frames.size();++i) {
        const auto&d=deltas[i];const double h=d.dt;const Mat33 ri=frames[i].rotation_wb.matrix(),rj=frames[i+1].rotation_wb.matrix();
        const Vec3 pi=scale*frames[i].camera_position-ri*lever,pj=scale*frames[i+1].camera_position-rj*lever;
        const Vec3 db=accel_bias-d.bias.accelerometer;
        const Vec3 dp=d.delta_position+d.bias_jacobian.block<3,3>(6,3)*db;
        const Vec3 dv=d.delta_velocity+d.bias_jacobian.block<3,3>(3,3)*db;
        const Vec3 predicted=pi+velocities.segment<3>(3*i)*h+0.5*gravity*h*h+ri*dp;
        residual.segment<3>(6*i)=(predicted-pj)/(h*std::sqrt(h));
        residual.segment<3>(6*i+3)=(velocities.segment<3>(3*(i+1))-velocities.segment<3>(3*i)-gravity*h-ri*dv)/std::sqrt(h);
    }
    return residual;
}
Result Run(const VioVisualWindow&visual,const std::vector<ImuSample>&samples,
           const ImuCalibration&calibration,const SE3&T_camera_body,const Vec3&gyro,
           Variant variant,const Vec3&baseline_bias,const Vec3&reference_gravity,
           const VioInitializerOptions&o) {
    Result r;r.bias.gyroscope=gyro;
    r.bias.accelerometer=variant==Variant::BaselineAccelBias?baseline_bias:Vec3::Zero();
    const auto&frames=visual.frames;const int n=frames.size(),gcol=3*n,scol=gcol+3;
    if(!visual.positions_in_common_scale||n<int(o.min_frames)) {r.reason="visual structure or frame count unavailable";return r;}
    const double duration=double(frames.back().timestamp_ns-frames.front().timestamp_ns)*1e-9;
    if(duration<o.min_duration_s){r.reason="startup window is too short";return r;}
    auto corrected=Integrate(frames,samples,calibration,r.bias);
    double rotation_squared=0,maximum_rotation=0;Vec3 mean_force=Vec3::Zero();
    std::vector<Vec3,Eigen::aligned_allocator<Vec3>>forces;
    for(int i=0;i+1<n;++i) {
        const double e=(corrected[i].delta_rotation.inverse()*frames[i].rotation_wb.inverse()*frames[i+1].rotation_wb).log().norm();
        rotation_squared+=e*e;maximum_rotation=std::max(maximum_rotation,e);
        forces.push_back(frames[i].rotation_wb*corrected[i].delta_velocity/corrected[i].dt);mean_force+=forces.back();
    }
    r.rotation_rms=std::sqrt(rotation_squared/(n-1));mean_force/=forces.size();
    double excitation_squared=0;for(const auto&f:forces)excitation_squared+=(f-mean_force).squaredNorm();
    r.excitation=std::sqrt(excitation_squared/forces.size());
    if(gyro.norm()>o.max_gyro_bias){r.reason="gyro bias exceeds configured limit";return r;}
    if(r.rotation_rms>o.max_rotation_rms_rad||maximum_rotation>o.max_rotation_error_rad){r.reason="no constant gyro bias reconciles visual/IMU rotations";return r;}
    if(r.excitation<o.min_acceleration_excitation){r.reason="world specific force has insufficient variation for scale";return r;}
    const bool full=variant==Variant::Full;
    auto system=BuildSystem(frames,corrected,T_camera_body,full);
    auto fit=Solve(system.a,system.b);
    r.accel_bias_fitted=full&&fit.Usable(system.a.cols(),o.max_condition);
    if(full&&!r.accel_bias_fitted){system=BuildSystem(frames,corrected,T_camera_body,false);fit=Solve(system.a,system.b);}
    const bool fixed_gravity=variant==Variant::FixedGravityZeroAccelBias;
    if(fixed_gravity) {
        const Vec3 g=reference_gravity.normalized()*o.gravity_magnitude;
        MatXX a(system.a.rows(),gcol+1);a.leftCols(gcol)=system.a.leftCols(gcol);a.rightCols(1)=system.a.middleCols(scol,1);
        const VecX b=system.b-system.a.middleCols(gcol,3)*g;
        r.initial_system=system;r.initial_system.a=a;r.initial_system.b=b;
        r.initial_system.gravity_column=-1;r.initial_system.scale_column=gcol;r.initial_system.accel_column=-1;
        r.initial_fit=SolveAndDiagnose(a,b,gcol);fit=Solve(a,b);
        r.candidate_gravity=g;r.candidate_scale=fit.x[gcol];r.candidate_accel_bias=r.bias.accelerometer;
        r.gravity=g;
    } else {
        r.initial_system=system;r.initial_fit=SolveAndDiagnose(system.a,system.b,gcol);
        r.candidate_gravity=fit.x.segment<3>(gcol);r.candidate_scale=fit.x[scol];
        r.candidate_accel_bias=r.accel_bias_fitted?Vec3(fit.x.segment<3>(scol+1)):r.bias.accelerometer;
        if(!fit.Usable(system.a.cols(),o.max_condition)){r.reason="gravity/velocity/scale alignment lacks rank/conditioning";return r;}
        if(!r.candidate_gravity.allFinite()||std::abs(r.candidate_gravity.norm()-o.gravity_magnitude)>o.max_gravity_magnitude_error){r.reason="unconstrained gravity magnitude is inconsistent with calibration";return r;}
        Vec3 gravity=r.candidate_gravity.normalized()*o.gravity_magnitude;
        for(int iteration=0;iteration<6;++iteration) {
            const auto basis=GravityBasis(gravity);MatXX a(system.a.rows(),system.a.cols()-1);
            a.leftCols(gcol)=system.a.leftCols(gcol);a.middleCols(gcol,2)=system.a.middleCols(gcol,3)*basis;
            a.rightCols(system.a.cols()-scol)=system.a.rightCols(system.a.cols()-scol);
            const VecX b=system.b-system.a.middleCols(gcol,3)*gravity;const auto refined=Solve(a,b);
            r.final_condition=refined.condition;
            if(!refined.Usable(a.cols(),o.max_condition)){r.reason="gravity tangent alignment lacks conditioning";return r;}
            const Vec3 change=basis*refined.x.segment<2>(gcol);gravity=(gravity+change).normalized()*o.gravity_magnitude;
            if(change.norm()<1e-8)break;
        }
        r.gravity=gravity;
    }
    MatXX fixed;VecX rhs;
    if(fixed_gravity){fixed=r.initial_system.a;rhs=r.initial_system.b;r.final_condition=fit.condition;}
    else {fixed.resize(system.a.rows(),system.a.cols()-3);fixed.leftCols(gcol)=system.a.leftCols(gcol);fixed.rightCols(system.a.cols()-scol)=system.a.rightCols(system.a.cols()-scol);rhs=system.b-system.a.middleCols(gcol,3)*r.gravity;}
    const auto final=Solve(fixed,rhs);
    if(!final.Usable(fixed.cols(),o.max_condition)){r.reason="final velocity/scale solve lacks conditioning";return r;}
    r.candidate_final_scale=final.x[gcol];
    r.candidate_final_accel_bias=r.accel_bias_fitted?Vec3(final.x.segment<3>(gcol+1)):r.bias.accelerometer;
    r.linear_final_rms=std::sqrt((fixed*final.x-rhs).squaredNorm()/fixed.rows());
    r.bias.accelerometer=r.candidate_final_accel_bias;
    if(!std::isfinite(r.candidate_final_scale)||r.candidate_final_scale<=1e-6||!r.bias.accelerometer.allFinite()||r.bias.accelerometer.norm()>o.max_accel_bias){r.reason="nonpositive scale or implausible accelerometer bias";return r;}
    corrected=Integrate(frames,samples,calibration,r.bias);const Vec3 lever=T_camera_body.inverse().translation();
    VioNavStates states;
    for(int i=0;i<n;++i){NavState s;s.rotation_wb=frames[i].rotation_wb;s.position_wb=r.candidate_final_scale*(frames[i].camera_position-frames.front().camera_position)-(frames[i].rotation_wb.matrix()-frames.front().rotation_wb.matrix())*lever;s.velocity_w=final.x.segment<3>(3*i);s.bias=r.bias;states.push_back(s);}
    double p=0,v=0;
    for(int i=0;i+1<n;++i){const auto predicted=PropagateNavState(states[i],corrected[i],r.gravity);p+=(predicted.position_wb-states[i+1].position_wb).squaredNorm();v+=(predicted.velocity_w-states[i+1].velocity_w).squaredNorm();}
    r.position_rms=std::sqrt(p/(n-1));r.velocity_rms=std::sqrt(v/(n-1));
    if(r.position_rms>o.max_position_rms_m||r.velocity_rms>o.max_velocity_rms_m_s){r.reason="visual trajectory and corrected IMU translation disagree";return r;}
    r.success=true;r.scale=r.candidate_final_scale;r.states=std::move(states);
    r.reason=r.accel_bias_fitted?"dynamic metric initialization, both biases observable":
        (full?"dynamic metric initialization, accelerometer bias held at explicit zero prior (unobservable)":
              "diagnostic constrained metric initialization");return r;
}
} // namespace metric_audit
} // namespace myslam
