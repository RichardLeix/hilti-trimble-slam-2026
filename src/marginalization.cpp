#include "myslam/marginalization.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace myslam {
namespace {
Mat33 Cross(const Vec3& v) {Mat33 m;m<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return m;}
struct Spectrum {
    VecX inverse_scale, values;
    MatXX vectors;
    double cutoff;
};
Spectrum Decompose(const MatXX& input,double tolerance,const char* audit_stage) {
    Spectrum s;const MatXX h=(0.5*(input+input.transpose())).eval();
    s.inverse_scale=h.diagonal().cwiseMax(1e-18).cwiseSqrt().cwiseInverse();
    const MatXX normalized=s.inverse_scale.asDiagonal()*h*s.inverse_scale.asDiagonal();
    Eigen::SelfAdjointEigenSolver<MatXX> eigen(normalized);
    if(eigen.info()!=Eigen::Success)throw std::runtime_error("marginalization eigendecomposition failed");
    s.values=eigen.eigenvalues();s.vectors=eigen.eigenvectors();
    
    s.cutoff=tolerance*std::max(1.0,s.values.maxCoeff());
    
    if(s.values.minCoeff() < -1e-7*std::max(1.0,s.values.maxCoeff()))
        throw std::runtime_error("marginalization Hessian has a significant negative eigenvalue");
    return s;
}
MatXX PseudoInverse(const MatXX& input,double tolerance) {
    if(input.rows()==0)return MatXX(0,0);
    const auto s=Decompose(input,tolerance,"marginalization_eliminated");VecX inverse=s.values;
    for(int i=0;i<inverse.size();++i)inverse[i]=s.values[i]>s.cutoff?1/s.values[i]:0;
    const MatXX normalized=s.vectors*inverse.asDiagonal()*s.vectors.transpose();
    return s.inverse_scale.asDiagonal()*normalized*s.inverse_scale.asDiagonal();
}
}
NavState RetractNavState(const NavState& state,const NavLocalVector& d) {
    NavState s=state;s.rotation_wb=s.rotation_wb*SO3::exp(d.head<3>());
    s.position_wb+=d.segment<3>(3);s.velocity_w+=d.segment<3>(6);
    s.bias.gyroscope+=d.segment<3>(9);s.bias.accelerometer+=d.tail<3>();return s;
}
NavLocalVector NavStateLocalDelta(const NavState& a,const NavState& b) {
    NavLocalVector d;d.head<3>()=(a.rotation_wb.inverse()*b.rotation_wb).log();
    d.segment<3>(3)=b.position_wb-a.position_wb;d.segment<3>(6)=b.velocity_w-a.velocity_w;
    d.segment<3>(9)=b.bias.gyroscope-a.bias.gyroscope;d.tail<3>()=b.bias.accelerometer-a.bias.accelerometer;return d;
}
NavLocalMatrix NavStateLocalJacobian(const NavState& a,const NavState& b) {
    NavLocalMatrix j=NavLocalMatrix::Identity();const Vec3 phi=(a.rotation_wb.inverse()*b.rotation_wb).log();
    const double t=phi.norm();const Mat33 k=Cross(phi);
    const double coefficient=t<1e-4?1.0/12+t*t/720:(1-0.5*t/std::tan(0.5*t))/(t*t);
    j.topLeftCorner<3,3>()=Mat33::Identity()+0.5*k+coefficient*k*k;return j;
}
MarginalizationKey MarginalizationKey::State(int64_t timestamp) {MarginalizationKey k;k.timestamp_ns=timestamp;return k;}
MarginalizationKey MarginalizationKey::Landmark(int camera,uint64_t feature) {
    MarginalizationKey k;k.kind=Kind::Landmark;k.camera_id=camera;k.feature_id=feature;return k;
}
bool MarginalizationKey::operator<(const MarginalizationKey& o) const {
    return std::tie(kind,timestamp_ns,camera_id,feature_id)<std::tie(o.kind,o.timestamp_ns,o.camera_id,o.feature_id);
}
bool MarginalizationKey::operator==(const MarginalizationKey& o) const {return !(*this<o) && !(o<*this);}
MarginalizationEvaluation MarginalizationPrior::Evaluate(const MarginalizationVariables& current) const {
    if(current.size()!=linearization.size())throw std::invalid_argument("marginalization prior variable count mismatch");
    MarginalizationEvaluation e;e.delta=VecX::Zero(Dimension());
    MatXX tangent=MatXX::Identity(Dimension(),Dimension());int offset=0;
    for(size_t i=0;i<current.size();++i) {
        if(!(current[i].key==linearization[i].key))throw std::invalid_argument("marginalization prior variable ordering mismatch");
        if(current[i].Dimension()==15) {
            e.delta.segment<15>(offset)=NavStateLocalDelta(linearization[i].state,current[i].state);
            tangent.block<15,15>(offset,offset)=NavStateLocalJacobian(linearization[i].state,current[i].state);
        }else e.delta.segment<3>(offset)=current[i].point_w-linearization[i].point_w;
        offset+=current[i].Dimension();
    }
    e.residual=residual_at_linearization+sqrt_information*e.delta;
    // Multiply by the block-diagonal tangent map, retaining dense landmark
    // correlations. Cached H avoids a cubic J^T J product on every LM iteration.
    e.jacobian=sqrt_information;
    MatXX ht=hessian;offset=0;
    for(const auto& v:current) {
        const int n=v.Dimension();
        if(n==15) {
            const Mat33 r=tangent.block<3,3>(offset,offset);
            e.jacobian.middleCols(offset,3)=(e.jacobian.middleCols(offset,3)*r).eval();
            ht.middleCols(offset,3)=(ht.middleCols(offset,3)*r).eval();
        }
        offset+=n;
    }
    e.hessian=ht;e.gradient=gradient+hessian*e.delta;offset=0;
    for(const auto& v:current) {
        if(v.Dimension()==15) {
            const Mat33 r=tangent.block<3,3>(offset,offset);
            e.hessian.middleRows(offset,3)=(r.transpose()*ht.middleRows(offset,3)).eval();
            e.gradient.segment<3>(offset)=(r.transpose()*e.gradient.segment<3>(offset)).eval();
        }
        offset+=v.Dimension();
    }
    if(!e.residual.allFinite() || !e.hessian.allFinite() || !e.gradient.allFinite())throw std::runtime_error("nonfinite marginalization prior evaluation");
    return e;
}
MarginalizationPrior MarginalizationInfo::Marginalize(double tolerance) const {
    int dimension=0;std::set<MarginalizationKey> keys;
    for(const auto& v:variables){dimension+=v.Dimension();if(!keys.insert(v.key).second)throw std::invalid_argument("duplicate marginalization key");}
    if(dimension==0 || hessian.rows()!=dimension || hessian.cols()!=dimension || gradient.size()!=dimension ||
        !hessian.allFinite() || !gradient.allFinite() || !std::isfinite(tolerance) || tolerance<=0 || tolerance>=1)
        throw std::invalid_argument("invalid marginalization normal system");
    for(const auto& k:eliminate)if(!keys.count(k))throw std::invalid_argument("marginalized variable absent");
    for(int c:fixed_coordinates)if(c<0 || c>=dimension)throw std::invalid_argument("invalid fixed marginalization coordinate");
    std::vector<int> marginal,retain;MarginalizationPrior prior;int offset=0;
    for(const auto& v:variables) {
        const bool drop=eliminate.count(v.key);
        if(!drop)prior.linearization.push_back(v);
        for(int c=0;c<v.Dimension();++c) {
            if(fixed_coordinates.count(offset+c)) {if(!drop)throw std::invalid_argument("fixed coordinate must belong to outgoing variable");}
            else (drop?marginal:retain).push_back(offset+c);
        }
        offset+=v.Dimension();
    }
    if(retain.empty())throw std::invalid_argument("cannot create an empty retained prior");
    MatXX hmm(marginal.size(),marginal.size()),hmr(marginal.size(),retain.size()),hrr(retain.size(),retain.size());
    VecX gm(marginal.size()),gr(retain.size());
    for(size_t i=0;i<marginal.size();++i) {
        gm[i]=gradient[marginal[i]];
        for(size_t j=0;j<marginal.size();++j)hmm(i,j)=hessian(marginal[i],marginal[j]);
        for(size_t j=0;j<retain.size();++j)hmr(i,j)=hessian(marginal[i],retain[j]);
    }
    for(size_t i=0;i<retain.size();++i) {
        gr[i]=gradient[retain[i]];for(size_t j=0;j<retain.size();++j)hrr(i,j)=hessian(retain[i],retain[j]);
    }
    // Explicit Schur partition, gradient sign follows H dx = -g. No LM damping
    // or artificial diagonal enters the retained information.
    const MatXX inverse=PseudoInverse(hmm,tolerance);
    MatXX schur=hrr-hmr.transpose()*inverse*hmr;VecX reduced=gr-hmr.transpose()*inverse*gm;
    schur=(0.5*(schur+schur.transpose())).eval();
    const auto spectrum=Decompose(schur,tolerance,"marginalization_schur");
    for(int i=0;i<spectrum.values.size();++i)if(spectrum.values[i]>spectrum.cutoff)++prior.rank;
    prior.sqrt_information=MatXX::Zero(prior.rank,retain.size());prior.residual_at_linearization=VecX::Zero(prior.rank);
    size_t row=0;double smallest=std::numeric_limits<double>::infinity(),largest=0;
    const VecX normalized_gradient=spectrum.inverse_scale.asDiagonal()*reduced;
    for(int i=0;i<spectrum.values.size();++i)if(spectrum.values[i]>spectrum.cutoff) {
        const double value=spectrum.values[i];smallest=std::min(smallest,value);largest=std::max(largest,value);
        prior.sqrt_information.row(row)=std::sqrt(value)*spectrum.vectors.col(i).transpose()*spectrum.inverse_scale.cwiseInverse().asDiagonal();
        prior.residual_at_linearization[row]=spectrum.vectors.col(i).dot(normalized_gradient)/std::sqrt(value);++row;
    }
    prior.hessian=prior.sqrt_information.transpose()*prior.sqrt_information;
    prior.gradient=prior.sqrt_information.transpose()*prior.residual_at_linearization;
    prior.condition=prior.rank?largest/smallest:0;
    if(!prior.hessian.allFinite() || !prior.gradient.allFinite())throw std::runtime_error("nonfinite Schur marginalization prior");
    return prior;
}
} // namespace myslam
