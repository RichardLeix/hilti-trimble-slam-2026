#include "uncertainty_metric_initializer.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
namespace myslam {
namespace uncertainty_metric {
namespace {
Mat33 Cross(const Vec3& p) {Mat33 m;m<<0,-p.z(),p.y(),p.z(),0,-p.x(),-p.y(),p.x(),0;return m;}
double Condition(const MatXX& matrix) {
    Eigen::SelfAdjointEigenSolver<MatXX> e(matrix,Eigen::EigenvaluesOnly);
    if(e.info()!=Eigen::Success||e.eigenvalues().minCoeff()<=0)return INFINITY;
    return e.eigenvalues().maxCoeff()/e.eigenvalues().minCoeff();
}
}
MatXX VisualResidualJacobian(const BootstrapResult& v,const VioImuIntervals& deltas,
                            const SE3& extrinsic,double scale) {
    const int m=6*(v.poses.size()-1);MatXX j=MatXX::Zero(m,m);
    if(deltas.size()+1!=v.poses.size())throw std::invalid_argument("visual/delta size mismatch");
    const Vec3 lever=extrinsic.inverse().translation();const Mat33 cb=extrinsic.so3().matrix();
    for(size_t i=0;i<deltas.size();++i) {
        const int row=6*i;const Mat33 wi=v.poses[i].so3().inverse().matrix(),wj=v.poses[i+1].so3().inverse().matrix();
        if(i) {
            const int col=6*(i-1);
            j.block<3,3>(row,col)=-scale*wi;
            j.block<3,3>(row,col+3)=wi*Cross(cb*(deltas[i].delta_position-lever));
            j.block<3,3>(row+3,col+3)=-wi*Cross(cb*deltas[i].delta_velocity);
        }
        const int col=6*i;
        j.block<3,3>(row,col)=scale*wj;
        j.block<3,3>(row,col+3)=wj*Cross(cb*lever);
    }
    return j;
}
Covariance BuildCovariance(const BootstrapResult& v,const visual_uncertainty::Estimate& u,
    const VioImuIntervals& deltas,const SE3& extrinsic,double scale) {
    if(!u.usable)throw std::invalid_argument("unusable visual covariance: "+u.reason);
    if(!(scale>0&&std::isfinite(scale)))throw std::invalid_argument("invalid covariance scale");
    Covariance c;const int m=6*deltas.size();c.imu=MatXX::Zero(m,m);
    if(m==0||v.window.frames.size()!=deltas.size()+1||v.poses.size()!=v.window.frames.size()||
       u.pose_covariance.rows()!=m||u.pose_covariance.cols()!=m||!u.pose_covariance.allFinite()||
       !u.pose_covariance.isApprox(u.pose_covariance.transpose(),1e-12))
        throw std::invalid_argument("invalid visual covariance dimensions/values/symmetry");
    for(size_t i=0;i<deltas.size();++i)c.imu.block<6,6>(6*i,6*i)=staged_metric::ResidualCovariance(deltas[i],v.window.frames[i].rotation_wb);
    if(!c.imu.allFinite()||!c.imu.isApprox(c.imu.transpose(),1e-12))
        throw std::invalid_argument("invalid IMU preintegration covariance values/symmetry");
    for(size_t i=0;i<deltas.size();++i) {
        Eigen::SelfAdjointEigenSolver<staged_metric::Mat66> eigen(c.imu.block<6,6>(6*i,6*i),Eigen::EigenvaluesOnly);
        if(eigen.info()!=Eigen::Success||eigen.eigenvalues().minCoeff()<0)
            throw std::invalid_argument("indefinite IMU preintegration covariance");
    }
    const MatXX j=VisualResidualJacobian(v,deltas,extrinsic,scale);
    c.visual=j*u.pose_covariance*j.transpose();c.combined=c.imu+c.visual;
    c.combined=(.5*(c.combined+c.combined.transpose())).eval();
    if(!c.combined.allFinite())throw std::invalid_argument("nonfinite combined residual covariance");
    Eigen::LLT<MatXX> factor(c.combined);
    if(factor.info()!=Eigen::Success)throw std::invalid_argument("combined residual covariance is not positive definite");
    c.whitening=factor.matrixL().solve(MatXX::Identity(m,m));
    c.raw_condition=Condition(c.combined);
    const VecX d=c.combined.diagonal().cwiseSqrt().cwiseInverse();
    c.normalized_condition=Condition(d.asDiagonal()*c.combined*d.asDiagonal());
    if(!c.whitening.allFinite()||!std::isfinite(c.raw_condition)||!std::isfinite(c.normalized_condition))
        throw std::invalid_argument("combined residual covariance is rank deficient or invalid");
    c.position_visual_variance=c.position_imu_variance=c.velocity_visual_variance=c.velocity_imu_variance=0;
    MatXX cross=c.visual;
    for(size_t i=0;i<deltas.size();++i) {
        c.position_visual_variance+=c.visual.block<3,3>(6*i,6*i).trace();
        c.position_imu_variance+=c.imu.block<3,3>(6*i,6*i).trace();
        c.velocity_visual_variance+=c.visual.block<3,3>(6*i+3,6*i+3).trace();
        c.velocity_imu_variance+=c.imu.block<3,3>(6*i+3,6*i+3).trace();
        cross.block<6,6>(6*i,6*i).setZero();
    }
    const double axes=3*deltas.size();
    c.position_visual_variance/=axes;c.position_imu_variance/=axes;
    c.velocity_visual_variance/=axes;c.velocity_imu_variance/=axes;c.cross_interval_norm=cross.norm();
    return c;
}
metric_audit::System RawFixedBiasSystem(const VioVisualFrames& f,const VioImuIntervals& d,const SE3& extrinsic) {
    auto system=metric_audit::BuildSystem(f,d,extrinsic,false);
    for(size_t i=0;i<d.size();++i) {
        const double h=d[i].dt;
        system.a.middleRows(6*i,3)*=h*std::sqrt(h);system.b.segment(6*i,3)*=h*std::sqrt(h);
        system.a.middleRows(6*i+3,3)*=std::sqrt(h);system.b.segment(6*i+3,3)*=std::sqrt(h);
    }
    return system;
}
Result Run(const BootstrapResult& v,const visual_uncertainty::Estimate& u,
    const std::vector<ImuSample>& samples,const ImuCalibration& cal,const SE3& extrinsic,
    const Vec3& gyro,double seed,const VioInitializerOptions& options) {
    Result result;
    if(!u.usable) {result.metric.reason="visual covariance: "+u.reason;return result;}
    if(!(seed>0&&std::isfinite(seed))) {result.metric.reason="no positive heuristic seed for feasible GLS";return result;}
    ImuBias bias;bias.gyroscope=gyro;VioImuIntervals deltas;
    for(size_t i=1;i<v.window.frames.size();++i)deltas.push_back(PreintegrateImu(samples,v.window.frames[i-1].timestamp_ns,v.window.frames[i].timestamp_ns,cal,bias));
    const auto raw=RawFixedBiasSystem(v.window.frames,deltas,extrinsic);
    double scale=seed;
    // Predeclared numerical fixed-point stopping rule. No GT or physical gates changed.
    for(int iteration=0;iteration<10;++iteration) {
        result.covariance=BuildCovariance(v,u,deltas,extrinsic,scale);
        auto weighted=raw;weighted.a=result.covariance.whitening*raw.a;weighted.b=result.covariance.whitening*raw.b;
        result.metric=staged_metric::Run(v.window,samples,cal,extrinsic,gyro,staged_metric::Weighting::Heuristic,
            options,staged_metric::FinalStage::FixedAccelBias,&weighted,true);
        result.reweighting_iterations=iteration+1;
        if(!result.metric.success)return result;
        const double next=result.metric.stages[1].scale;
        result.relative_scale_step=std::abs(next-scale)/scale;
        if(result.relative_scale_step<=1e-6) {result.converged=true;return result;}
        scale=next;
    }
    result.metric.success=false;result.metric.reason="feasible GLS scale-dependent covariance did not converge in 10 iterations";
    result.metric.stages[1].success=false;result.metric.stages[1].states.clear();return result;
}
} // namespace uncertainty_metric
} // namespace myslam
