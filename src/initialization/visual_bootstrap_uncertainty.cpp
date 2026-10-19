#include "visual_bootstrap_uncertainty.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace myslam {
namespace visual_uncertainty {
namespace {
Mat33 Cross(const Vec3& p) {
    Mat33 m;m<<0,-p.z(),p.y(),p.z(),0,-p.x(),-p.y(),p.x(),0;return m;
}
// Numerical rank tolerance, not a geometric/initializer acceptance threshold.
constexpr double kRankRelative = 1e-10;
}
MatXX CenterJacobian(const std::vector<SE3,Eigen::aligned_allocator<SE3>>& poses,size_t i) {
    if(i>=poses.size())throw std::out_of_range("visual center index");
    MatXX j=MatXX::Zero(3,6*(poses.size()-1));
    if(i)j.block<3,3>(0,6*(i-1))=-poses[i].so3().inverse().matrix();
    return j;
}
Mat33 RelativePositionCovariance(const Estimate& e,
    const std::vector<SE3,Eigen::aligned_allocator<SE3>>& poses,size_t i,size_t j) {
    if(!e.usable)throw std::invalid_argument("unusable visual covariance: "+e.reason);
    const MatXX d=CenterJacobian(poses,j)-CenterJacobian(poses,i);
    return d*e.pose_covariance*d.transpose();
}
Estimate Compute(const BootstrapResult& b,double limit) {
    Estimate e;
    if(!b.window.positions_in_common_scale||b.poses.size()<3||b.landmarks.empty()) {
        e.reason="missing accepted joint visual reconstruction";return e;
    }
    if(!(limit>0))throw std::invalid_argument("invalid bearing limit");
    const int m=6*(b.poses.size()-1);e.pose_columns=m;e.conditional_columns=m-1;
    MatXX h=MatXX::Zero(m,m);double sse=0;
    for(const auto& point:b.landmarks) {
        MatXX jp_all(3*point.observations.size(),3);
        std::vector<Eigen::Matrix<double,3,6>,Eigen::aligned_allocator<Eigen::Matrix<double,3,6>>> jc_all;
        for(size_t index=0;index<point.observations.size();++index) {
            const auto& o=point.observations[index];
            if(o.frame>=b.poses.size())throw std::invalid_argument("invalid bootstrap observation");
            const auto& pose=b.poses[o.frame];const Vec3 p=pose*point.point;
            if(!p.allFinite()||p.norm()<1e-12) {e.reason="invalid optimized visual landmark";return e;}
            const Vec3 u=p.normalized(),r=u-o.ray;
            const double weight=std::min(1.,.5*limit/std::max(r.norm(),1e-12));
            const Mat33 d=(Mat33::Identity()-u*u.transpose())/p.norm();
            jp_all.middleRows(3*index,3)=std::sqrt(weight)*d*pose.so3().matrix();
            Eigen::Matrix<double,3,6> jc;jc.leftCols<3>()=d;jc.rightCols<3>()=-d*Cross(p);jc*=std::sqrt(weight);
            jc_all.push_back(jc);
            if(o.frame) {
                const int k=6*(o.frame-1);
                h.block<6,6>(k,k).noalias()+=jc.transpose()*jc;
            }
            sse+=weight*r.squaredNorm();++e.observations;
        }
        // Eliminate using the original weighted Jacobian's column space, not
        // an inverse of its squared-condition normal matrix. This is exactly
        // H_cc - H_cp H_pp^+ H_pc, with a stable orthogonal projection.
        Eigen::JacobiSVD<MatXX> point_svd(jp_all,Eigen::ComputeThinU|Eigen::ComputeFullV);
        point_svd.setThreshold(kRankRelative);const int rank=point_svd.rank();
        e.landmark_rank+=rank;
        if(rank<3) {
            ++e.deficient_landmarks;
            for(int k=rank;k<3;++k) {
                e.deficient_landmark_ids.push_back(point.id);
                e.landmark_null_directions.push_back(point_svd.matrixV().col(k));
            }
        }
        MatXX projected=MatXX::Zero(m,rank);
        for(size_t index=0;index<point.observations.size();++index)if(point.observations[index].frame) {
            const int offset=6*(point.observations[index].frame-1);
            projected.middleRows(offset,6).noalias()+=jc_all[index].transpose()*point_svd.matrixU().block(3*index,0,3,rank);
        }
        h.noalias()-=projected*projected.transpose();
    }
    h=(.5*(h+h.transpose())).eval();e.schur_information=h;
    const size_t anchor=b.window.anchor_frame;
    if(!anchor||anchor>=b.poses.size()) {e.reason="invalid visual gauge anchor";return e;}
    const Vec3 c=b.poses[anchor].inverse().translation();
    e.anchor_constraint=VecX::Zero(m);
    e.anchor_constraint.segment<3>(6*(anchor-1))=-b.poses[anchor].so3().matrix()*c.normalized();
    VecX scale_direction=VecX::Zero(m);
    for(size_t i=1;i<b.poses.size();++i)scale_direction.segment<3>(6*(i-1))=b.poses[i].translation();
    e.scale_null_leakage=(h*scale_direction).norm()/std::max(1e-30,h.norm()*scale_direction.norm());
    // One Householder reflection supplies an orthonormal basis of ker(anchor J).
    VecX v=e.anchor_constraint.normalized();v[0]+=(v[0]>=0?1.:-1.);v.normalize();
    MatXX q=MatXX::Identity(m,m).rightCols(m-1);
    q.noalias()-=2*v*v.tail(m-1).transpose();
    MatXX conditional=q.transpose()*h*q;conditional=(.5*(conditional+conditional.transpose())).eval();
    VecX norms=conditional.diagonal().cwiseMax(0).cwiseSqrt();
    // Keep zero columns as explicit nulls. Unit normalization adds no information.
    for(int k=0;k<norms.size();++k)if(norms[k]<=0)norms[k]=1;
    const VecX inverse_norms=norms.cwiseInverse();
    MatXX normalized=inverse_norms.asDiagonal()*conditional*inverse_norms.asDiagonal();
    Eigen::SelfAdjointEigenSolver<MatXX> solver(normalized);
    if(solver.info()!=Eigen::Success) {e.reason="visual information eigendecomposition failed";return e;}
    e.normalized_eigenvalues=solver.eigenvalues();const double largest=e.normalized_eigenvalues.maxCoeff();
    if(e.normalized_eigenvalues.minCoeff()<-kRankRelative*largest) {e.reason="indefinite undamped visual information";return e;}
    VecX inverse=VecX::Zero(m-1);std::vector<int> nulls;
    for(int k=0;k<m-1;++k) {
        if(e.normalized_eigenvalues[k]>kRankRelative*largest) {inverse[k]=1/e.normalized_eigenvalues[k];++e.rank;}
        else nulls.push_back(k);
    }
    e.null_directions.resize(m,nulls.size());
    for(size_t k=0;k<nulls.size();++k)e.null_directions.col(k)=(q*inverse_norms.asDiagonal()*solver.eigenvectors().col(nulls[k])).normalized();
    e.dof=2*e.observations-e.landmark_rank-e.rank;e.weighted_sse=sse;
    if(e.dof<=0) {e.reason="visual residual variance has no degrees of freedom";return e;}
    e.bearing_variance=sse/e.dof;
    e.condition=e.rank?largest/e.normalized_eigenvalues[m-1-e.rank]:INFINITY;
    const MatXX transform=q*inverse_norms.asDiagonal()*solver.eigenvectors();
    e.pose_covariance=e.bearing_variance*transform*inverse.asDiagonal()*transform.transpose();
    e.pose_covariance=(.5*(e.pose_covariance+e.pose_covariance.transpose())).eval();
    e.anchor_constraint_error=(e.pose_covariance*e.anchor_constraint).norm();
    for(size_t i=0;i<b.poses.size();++i) {
        const MatXX jc=CenterJacobian(b.poses,i);e.center_covariances.push_back(jc*e.pose_covariance*jc.transpose());
    }
    if(e.rank!=m-1||e.deficient_landmarks) {
        e.reason="additional visual rank deficiency (conditional pseudocovariance is not physical certainty)";return e;
    }
    e.usable=true;e.reason="full rank after seven visual gauge definitions";return e;
}
} // namespace visual_uncertainty
} // namespace myslam
