#include "staged_metric_initializer.h"
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace myslam {
namespace staged_metric {
namespace {
struct LinearFit {
    VecX x;
    int rank = 0;
    double condition = INFINITY;
    bool Usable(int cols, const VioInitializerOptions& o) const {
        return rank == cols && condition <= o.max_condition && x.allFinite();
    }
};
LinearFit Solve(const MatXX& a, const VecX& b, bool fast = false) {
    if (!a.allFinite() || !b.allFinite()) throw std::invalid_argument("nonfinite staged system");
    VecX norms(a.cols());
    for (int i = 0; i < a.cols(); ++i) norms[i] = std::max(a.col(i).norm(), 1e-15);
    if(fast) {
        const MatXX normalized=a*norms.cwiseInverse().asDiagonal();
        Eigen::SelfAdjointEigenSolver<MatXX> eigen(normalized.transpose()*normalized,Eigen::EigenvaluesOnly);
        // Below condition 1e4, the normal spectrum is safely resolved in
        // double precision. Near a rank/gate boundary retain the original SVD.
        if(eigen.info()==Eigen::Success&&eigen.eigenvalues().minCoeff()>1e-8*eigen.eigenvalues().maxCoeff()) {
            LinearFit fit;fit.rank=a.cols();fit.condition=std::sqrt(eigen.eigenvalues().maxCoeff()/eigen.eigenvalues().minCoeff());
            fit.x=normalized.colPivHouseholderQr().solve(b).cwiseQuotient(norms);return fit;
        }
    }
    Eigen::JacobiSVD<MatXX> svd(a * norms.cwiseInverse().asDiagonal(),
                               Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(1e-8);
    LinearFit fit; fit.rank = int(svd.rank());
    const auto& singular = svd.singularValues();
    if (singular.size() == a.cols() && singular.tail<1>()[0] > 1e-15)
        fit.condition = singular[0] / singular.tail<1>()[0];
    fit.x = svd.solve(b).cwiseQuotient(norms);
    return fit;
}
metric_audit::Fit FastDiagnostics(const MatXX& a,const VecX& b,int nuisance) {
    VecX norms(a.cols());for(int i=0;i<a.cols();++i)norms[i]=std::max(a.col(i).norm(),1e-15);
    const MatXX normalized=a*norms.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<MatXX> eigen(normalized.transpose()*normalized);
    if(eigen.info()!=Eigen::Success||eigen.eigenvalues().minCoeff()<=1e-8*eigen.eigenvalues().maxCoeff())
        return metric_audit::SolveAndDiagnose(a,b,nuisance);
    metric_audit::Fit f;f.column_norms=norms;f.rank=a.cols();
    f.condition=std::sqrt(eigen.eigenvalues().maxCoeff()/eigen.eigenvalues().minCoeff());
    f.x=normalized.colPivHouseholderQr().solve(b).cwiseQuotient(norms);
    f.singular_values=eigen.eigenvalues().reverse().cwiseSqrt();
    f.normalized_covariance=eigen.eigenvectors()*eigen.eigenvalues().cwiseInverse().asDiagonal()*eigen.eigenvectors().transpose();
    f.covariance_unit_noise=norms.cwiseInverse().asDiagonal()*f.normalized_covariance*norms.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<MatXX> raw(a.transpose()*a,Eigen::EigenvaluesOnly);
    f.raw_singular_values=raw.eigenvalues().reverse().cwiseMax(0).cwiseSqrt();
    const VecX residual=a*f.x-b;f.weighted_rms=std::sqrt(residual.squaredNorm()/a.rows());
    f.residual_variance=residual.squaredNorm()/std::max(1,int(a.rows())-f.rank);
    const MatXX av=a.leftCols(nuisance),theta=a.rightCols(a.cols()-nuisance);
    const MatXX profile=theta-av*av.colPivHouseholderQr().solve(theta);
    f.profiled_normal=profile.transpose()*profile;
    VecX pn(profile.cols());for(int i=0;i<pn.size();++i)pn[i]=std::max(profile.col(i).norm(),1e-15);
    const MatXX scaled=profile*pn.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<MatXX> pe(scaled.transpose()*scaled);
    f.profiled_normal_eigenvalues=pe.eigenvalues();
    // Only the 3 remaining gravity-tangent/scale columns need this SVD.
    Eigen::JacobiSVD<MatXX> ps(profile,Eigen::ComputeThinU|Eigen::ComputeThinV);
    VecX inverse=VecX::Zero(ps.singularValues().size());
    for(int i=0;i<inverse.size();++i)if(ps.singularValues()[i]>1e-8*ps.singularValues()[0])inverse[i]=1/std::pow(ps.singularValues()[i],2);
    f.profiled_covariance=ps.matrixV()*inverse.asDiagonal()*ps.matrixV().transpose();return f;
}
Eigen::Matrix<double, 3, 2> GravityBasis(const Vec3& g) {
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = g.normalized().unitOrthogonal();
    basis.col(1) = g.normalized().cross(basis.col(0));
    return basis;
}
VioImuIntervals Integrate(const VioVisualFrames& frames, const std::vector<ImuSample>& samples,
                         const ImuCalibration& calibration, const ImuBias& bias) {
    VioImuIntervals result;
    for (size_t i = 1; i < frames.size(); ++i)
        result.push_back(PreintegrateImu(samples, frames[i-1].timestamp_ns,
                                       frames[i].timestamp_ns, calibration, bias));
    return result;
}
struct Reduced {
    MatXX a;
    VecX b;
};
Reduced Tangent(const metric_audit::System& system, const Vec3& gravity, bool release) {
    const int v = system.velocity_columns;
    Reduced r; r.a = MatXX::Zero(system.a.rows(), v + 3 + (release ? 3 : 0));
    r.a.leftCols(v) = system.a.leftCols(v);
    r.a.middleCols(v, 2) = system.a.middleCols(system.gravity_column, 3) * GravityBasis(gravity);
    r.a.middleCols(v+2, 1) = system.a.middleCols(system.scale_column, 1);
    if (release) r.a.rightCols(3) = system.a.middleCols(system.accel_column, 3);
    r.b = system.b - system.a.middleCols(system.gravity_column, 3) * gravity;
    return r;
}
Reduced FixedGravity(const metric_audit::System& system, const Vec3& gravity, bool release) {
    const int v = system.velocity_columns;
    Reduced r; r.a = MatXX::Zero(system.a.rows(), v + 1 + (release ? 3 : 0));
    r.a.leftCols(v) = system.a.leftCols(v);
    r.a.middleCols(v, 1) = system.a.middleCols(system.scale_column, 1);
    if (release) r.a.rightCols(3) = system.a.middleCols(system.accel_column, 3);
    r.b = system.b - system.a.middleCols(system.gravity_column, 3) * gravity;
    return r;
}
Reduced AddPrior(const Reduced& data, bool release, double sigma) {
    if (!release) return data;
    Reduced r; r.a = MatXX::Zero(data.a.rows()+3, data.a.cols());
    r.a.topRows(data.a.rows()) = data.a;
    r.a.bottomRightCorner(3, 3) = Mat33::Identity() / sigma;
    r.b = VecX::Zero(data.b.size()+3); r.b.head(data.b.size()) = data.b;
    return r; // Absolute body accel bias has a zero-mean Gaussian prior.
}
void Diagnostics(Stage& stage, const Reduced& data, const Reduced& posterior, bool fast = false) {
    stage.data_a = data.a; stage.posterior_a = posterior.a;
    stage.data_fit = fast?FastDiagnostics(data.a,data.b,stage.velocity_columns):
        metric_audit::SolveAndDiagnose(data.a, data.b, stage.velocity_columns);
    // Fixed-bias stages have no prior: these are the same diagnostic system.
    stage.posterior_fit = posterior.a.rows()==data.a.rows() ? stage.data_fit :
        metric_audit::SolveAndDiagnose(posterior.a, posterior.b, stage.velocity_columns);
}
void Evaluate(Stage& stage, const VioVisualFrames& frames, const VioImuIntervals& at_zero,
              const std::vector<ImuSample>& samples, const ImuCalibration& calibration,
              const SE3& extrinsic, const Vec3& gyro, const metric_audit::System& system,
              double sigma, const VioInitializerOptions& o) {
    if (!stage.gravity.allFinite() || !std::isfinite(stage.scale) || !stage.accel_bias.allFinite()) {
        stage.reason = "nonfinite metric candidate"; return;
    }
    ImuBias bias; bias.gyroscope = gyro; bias.accelerometer = stage.accel_bias;
    const auto corrected = Integrate(frames, samples, calibration, bias);
    const Vec3 lever = extrinsic.inverse().translation();
    VioNavStates states;
    for (size_t i = 0; i < frames.size(); ++i) {
        NavState s; s.rotation_wb = frames[i].rotation_wb;
        s.position_wb = stage.scale*(frames[i].camera_position-frames[0].camera_position)
            - (frames[i].rotation_wb.matrix()-frames[0].rotation_wb.matrix())*lever;
        s.velocity_w = stage.velocities.segment<3>(3*i); s.bias = bias;
        states.push_back(s);
    }
    double p = 0, v = 0;
    for (size_t i = 0; i+1 < frames.size(); ++i) {
        const auto predicted = PropagateNavState(states[i], corrected[i], stage.gravity);
        p += (predicted.position_wb-states[i+1].position_wb).squaredNorm();
        v += (predicted.velocity_w-states[i+1].velocity_w).squaredNorm();
    }
    stage.position_rms = std::sqrt(p/(frames.size()-1)); stage.velocity_rms = std::sqrt(v/(frames.size()-1));
    VecX x(system.a.cols()); x.head(system.velocity_columns) = stage.velocities;
    x.segment<3>(system.gravity_column) = stage.gravity; x[system.scale_column] = stage.scale;
    if (system.accel_column >= 0) x.tail<3>() = stage.accel_bias;
    stage.weighted_data_rms = std::sqrt((system.a*x-system.b).squaredNorm()/system.a.rows());
    const VecX heuristic = metric_audit::PhysicalResidual(frames, at_zero, extrinsic,
        stage.velocities, stage.gravity, stage.scale, stage.accel_bias);
    stage.heuristic_rms = std::sqrt(heuristic.squaredNorm()/heuristic.size());
    stage.prior_cost = stage.number == 3 ? stage.accel_bias.squaredNorm()/(sigma*sigma) : 0;
    if (std::abs(stage.gravity.norm()-o.gravity_magnitude) > o.max_gravity_magnitude_error) {
        stage.reason = "gravity magnitude is inconsistent with calibration"; return;
    }
    if (stage.scale <= 1e-6) { stage.reason = "nonpositive scale"; return; }
    if (stage.accel_bias.norm() > o.max_accel_bias) {
        stage.reason = "implausible accelerometer bias"; return;
    }
    if (stage.position_rms > o.max_position_rms_m || stage.velocity_rms > o.max_velocity_rms_m_s) {
        stage.reason = "visual trajectory and corrected IMU translation disagree"; return;
    }
    stage.success = true; stage.reason = "accepted offline stage"; stage.states = std::move(states);
}
} // namespace

Mat66 ResidualCovariance(const ImuPreintegration& delta, const SO3& rotation_wb) {
    Mat66 pv;
    pv.topLeftCorner<3,3>() = delta.covariance.block<3,3>(6,6);
    pv.topRightCorner<3,3>() = delta.covariance.block<3,3>(6,3);
    pv.bottomLeftCorner<3,3>() = delta.covariance.block<3,3>(3,6);
    pv.bottomRightCorner<3,3>() = delta.covariance.block<3,3>(3,3);
    Mat66 transform = Mat66::Zero();
    transform.topLeftCorner<3,3>() = rotation_wb.matrix();
    transform.bottomRightCorner<3,3>() = -rotation_wb.matrix();
    return transform * pv * transform.transpose();
}
Mat66 CovarianceWhitening(const Mat66& covariance) {
    if (!covariance.allFinite() || !covariance.isApprox(covariance.transpose(), 1e-12))
        throw std::invalid_argument("nonfinite/asymmetric IMU p/v covariance");
    Eigen::LLT<Mat66> llt(covariance);
    if (llt.info() != Eigen::Success) throw std::invalid_argument("IMU p/v covariance is not positive definite");
    return llt.matrixL().solve(Mat66::Identity());
}
double BiasPriorSigma(const ImuCalibration& cal, double duration, const VioInitializerOptions& o) {
    // chi(3) 99.73% radius, derived before replay from its analytic Maxwell CDF.
    const double sigma0 = o.max_accel_bias / 3.7624795681227154;
    if (duration <= 0 || !std::isfinite(duration) || sigma0 <= 0 || !std::isfinite(sigma0) ||
        cal.accelerometer_random_walk < 0 || !std::isfinite(cal.accelerometer_random_walk))
        throw std::invalid_argument("invalid engineering bias-prior inputs");
    return std::sqrt(sigma0*sigma0 + cal.accelerometer_random_walk*cal.accelerometer_random_walk*duration);
}
metric_audit::System WeightedSystem(const VioVisualFrames& frames, const VioImuIntervals& corrected,
                                   const SE3& extrinsic, bool release, const ImuCalibration& cal, Weighting weighting) {
    auto system = metric_audit::BuildSystem(frames, corrected, extrinsic, release);
    if (cal.accelerometer_noise_density <= 0 || !std::isfinite(cal.accelerometer_noise_density))
        throw std::invalid_argument("positive calibrated accel noise required for noise-unit weighting");
    if (weighting == Weighting::Heuristic) {
        system.a /= cal.accelerometer_noise_density; system.b /= cal.accelerometer_noise_density;
    } else {
        for (size_t i = 0; i < corrected.size(); ++i) {
            const int row = 6*i; const double h = corrected[i].dt;
            // Undo production heuristic weights before applying the full 6x6 covariance.
            system.a.middleRows(row,3) *= h*std::sqrt(h); system.b.segment<3>(row) *= h*std::sqrt(h);
            system.a.middleRows(row+3,3) *= std::sqrt(h); system.b.segment<3>(row+3) *= std::sqrt(h);
            const Mat66 w = CovarianceWhitening(ResidualCovariance(corrected[i], frames[i].rotation_wb));
            system.a.middleRows(row,6) = (w*system.a.middleRows(row,6)).eval();
            system.b.segment<6>(row) = w*system.b.segment<6>(row);
        }
    }
    return system;
}
Result Run(const VioVisualWindow& visual, const std::vector<ImuSample>& samples,
           const ImuCalibration& cal, const SE3& extrinsic, const Vec3& gyro,
           Weighting weighting, const VioInitializerOptions& o, FinalStage final_stage,
           const metric_audit::System* override_system, bool fast) {
    Result result; result.gyro_bias = gyro;
    for (int i=0;i<3;++i) result.stages[i].number=i+1;
    const auto& frames = visual.frames;
    if (!visual.positions_in_common_scale || frames.size() < o.min_frames) {
        result.reason="visual structure or frame count unavailable"; return result;
    }
    const double duration = double(frames.back().timestamp_ns-frames.front().timestamp_ns)*1e-9;
    if (duration < o.min_duration_s) { result.reason="startup window is too short"; return result; }
    const bool release_bias=final_stage==FinalStage::ReleasedAccelBias;
    if(fast&&(!override_system||release_bias))throw std::invalid_argument("fast solver is restricted to the offline fixed-bias covariance override");
    if(override_system&&release_bias)throw std::invalid_argument("offline weighting override requires fixed accelerometer bias");
    if(release_bias) result.bias_prior_sigma = BiasPriorSigma(cal,duration,o);
    else result.stages[2].reason="not requested: accelerometer bias remains fixed";
    ImuBias zero; zero.gyroscope=gyro; const auto corrected=Integrate(frames,samples,cal,zero);
    double rot=0, max_rot=0; Vec3 mean=Vec3::Zero();
    std::vector<Vec3,Eigen::aligned_allocator<Vec3>> forces;
    for(size_t i=0;i<corrected.size();++i) {
        const double error=(corrected[i].delta_rotation.inverse()*frames[i].rotation_wb.inverse()*frames[i+1].rotation_wb).log().norm();
        rot+=error*error; max_rot=std::max(max_rot,error);
        forces.push_back(frames[i].rotation_wb*corrected[i].delta_velocity/corrected[i].dt); mean+=forces.back();
    }
    result.rotation_rms=std::sqrt(rot/corrected.size()); mean/=forces.size();
    double excitation=0; for(const auto& f:forces) excitation+=(f-mean).squaredNorm();
    result.excitation=std::sqrt(excitation/forces.size());
    if(gyro.norm()>o.max_gyro_bias || result.rotation_rms>o.max_rotation_rms_rad || max_rot>o.max_rotation_error_rad) {
        result.reason="native gyro/rotation plausibility failure"; return result;
    }
    if(result.excitation<o.min_acceleration_excitation) {result.reason="insufficient acceleration excitation"; return result;}
    const auto fixed_bias=override_system?*override_system:WeightedSystem(frames,corrected,extrinsic,false,cal,weighting);
    if(fixed_bias.velocity_columns!=int(3*frames.size())||fixed_bias.gravity_column!=fixed_bias.velocity_columns||
       fixed_bias.scale_column!=fixed_bias.velocity_columns+3||fixed_bias.accel_column!=-1||
       fixed_bias.a.rows()!=int(6*(frames.size()-1))||fixed_bias.a.cols()!=fixed_bias.velocity_columns+4||
       fixed_bias.b.size()!=fixed_bias.a.rows())throw std::invalid_argument("invalid offline fixed-bias system override");
    const int velocities=fixed_bias.velocity_columns;
    auto& first=result.stages[0]; first.velocity_columns=velocities; first.gravity_dimensions=3;
    first.scale_column=fixed_bias.scale_column; first.gravity_basis=Mat33::Identity();
    const auto fit=Solve(fixed_bias.a,fixed_bias.b,fast);
    Diagnostics(first,{fixed_bias.a,fixed_bias.b},{fixed_bias.a,fixed_bias.b},fast);
    if(!fit.Usable(fixed_bias.a.cols(),o)) {first.reason="stage 1 rank/conditioning failure"; result.reason=first.reason; return result;}
    first.velocities=fit.x.head(velocities); first.gravity=fit.x.segment<3>(velocities);
    first.scale=fit.x[first.scale_column]; first.accel_bias=Vec3::Zero();
    Evaluate(first,frames,corrected,samples,cal,extrinsic,gyro,fixed_bias,result.bias_prior_sigma,o);
    if(!first.success) {result.reason="stage 1: "+first.reason; return result;}
    Vec3 gravity=first.gravity.normalized()*o.gravity_magnitude;
    for(int index=1;index<(release_bias?3:2);++index) {
        const bool release=index==2; const auto system=release?WeightedSystem(frames,corrected,extrinsic,true,cal,weighting):fixed_bias;
        auto& stage=result.stages[index]; stage.velocity_columns=velocities; stage.gravity_dimensions=2;
        stage.scale_column=velocities+2; stage.bias_column=release?velocities+3:-1;
        for(int iteration=0;iteration<6;++iteration) {
            const auto data=Tangent(system,gravity,release); const auto posterior=AddPrior(data,release,result.bias_prior_sigma);
            const auto refined=Solve(posterior.a,posterior.b,fast); ++stage.iterations;
            if(!refined.Usable(posterior.a.cols(),o)) {stage.reason="gravity tangent rank/conditioning failure"; result.reason=stage.reason; return result;}
            const Vec3 change=GravityBasis(gravity)*refined.x.segment<2>(velocities);
            gravity=(gravity+change).normalized()*o.gravity_magnitude;
            if(change.norm()<1e-8) break;
        }
        const auto fixed=AddPrior(FixedGravity(system,gravity,release),release,result.bias_prior_sigma);
        const auto final=Solve(fixed.a,fixed.b,fast);
        if(!final.Usable(fixed.a.cols(),o)) {stage.reason="final velocity/scale rank/conditioning failure"; result.reason=stage.reason; return result;}
        stage.gravity=gravity; stage.gravity_basis=GravityBasis(gravity); stage.velocities=final.x.head(velocities);
        stage.scale=final.x[velocities]; stage.accel_bias=release?Vec3(final.x.tail<3>()):Vec3::Zero();
        const auto tangent=Tangent(system,gravity,release);
        Diagnostics(stage,tangent,AddPrior(tangent,release,result.bias_prior_sigma),fast);
        stage.remaining_gravity_step=stage.posterior_fit.x.segment<2>(velocities).norm();
        Evaluate(stage,frames,corrected,samples,cal,extrinsic,gyro,system,result.bias_prior_sigma,o);
        if(!stage.success) {result.reason="stage "+std::to_string(index+1)+": "+stage.reason; return result;}
    }
    result.success=true; result.reason=release_bias?"accepted offline staged metric initialization":
        "accepted offline fixed-bias stages 1/2"; return result;
}
double Correlation(const Stage& stage, bool posterior, bool gravity_bias) {
    const auto& fit=posterior?stage.posterior_fit:stage.data_fit;
    const auto& covariance=fit.covariance_unit_noise;
    if(covariance.size()==0 || fit.rank<covariance.cols() || (gravity_bias&&stage.bias_column<0)) return NAN;
    const int v=stage.velocity_columns;
    MatXX left,right,cross;
    if(gravity_bias) {
        left=stage.gravity_basis*covariance.block(v,v,stage.gravity_dimensions,stage.gravity_dimensions)*stage.gravity_basis.transpose();
        right=covariance.block(stage.bias_column,stage.bias_column,3,3);
        cross=stage.gravity_basis*covariance.block(v,stage.bias_column,stage.gravity_dimensions,3);
    } else {
        left=covariance.topLeftCorner(v,v); right=covariance.block(stage.scale_column,stage.scale_column,1,1);
        cross=covariance.block(0,stage.scale_column,v,1);
    }
    double maximum=0;
    for(int i=0;i<cross.rows();++i) for(int j=0;j<cross.cols();++j)
        if(left(i,i)*right(j,j)>0) maximum=std::max(maximum,std::abs(cross(i,j))/std::sqrt(left(i,i)*right(j,j)));
    return maximum;
}
} // namespace staged_metric
} // namespace myslam
