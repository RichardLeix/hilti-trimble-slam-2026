#include "myslam/loop_closing.h"
#include <opencv2/features2d.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace myslam {
namespace {
using Vec6=Eigen::Matrix<double,6,1>;
Mat33 Cross(const Vec3& v) {
    Mat33 m;m<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return m;
}
struct Row { int camera;size_t row; };
cv::Mat Flatten(const LoopKeyframe& f,std::vector<Row>& rows) {
    cv::Mat d;
    for(int k=0;k<2;++k) {
        d.push_back(f.descriptors[k]);
        for(size_t i=0;i<f.observations[k].size();++i)rows.push_back({k,i});
    }
    return d;
}
struct Correspondence {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Vec3 point, bearing;
    int camera;
};
using Correspondences=HiltiAlignedVector<Correspondence>;
double Error(const Correspondence& c,const SE3& pose,const LoopKeyframe& query) {
    const Vec3 p=query.camera_from_body[c.camera]*(pose*c.point);
    if(!p.allFinite() || p.norm()<1e-8 || p.dot(c.bearing)<=0)return std::numeric_limits<double>::infinity();
    return std::atan2(p.cross(c.bearing).norm(),p.dot(c.bearing));
}
std::vector<size_t> Consensus(const Correspondences& cs,const SE3& pose,
        const LoopKeyframe& query,double threshold,double* squared=nullptr) {
    std::vector<size_t> ids;double cost=0;
    for(size_t i=0;i<cs.size();++i) {
        const double e=Error(cs[i],pose,query);
        if(e<=threshold){ids.push_back(i);cost+=e*e;}
    }
    if(squared)*squared=cost;
    return ids;
}
// Six-point bearing DLT with centered/scaled metric coordinates. The cross
// product equations retain the entire sphere; no pinhole/Z division is used.
bool BearingDlt(const Correspondences& cs,const std::vector<size_t>& ids,SE3& camera_pose) {
    if(ids.size()<6)return false;
    Vec3 center=Vec3::Zero();for(size_t i:ids)center+=cs[i].point;center/=ids.size();
    double scale=0;for(size_t i:ids)scale+=(cs[i].point-center).squaredNorm();scale=std::sqrt(scale/ids.size());
    if(scale<1e-5)return false;
    MatXX a=MatXX::Zero(3*ids.size(),12);
    for(size_t n=0;n<ids.size();++n) {
        const auto& c=cs[ids[n]];const Vec3 p=(c.point-center)/scale;const Mat33 cross=Cross(c.bearing);
        for(int r=0;r<3;++r)for(int j=0;j<3;++j) {
            a.block<1,3>(3*n+r,4*j)=cross(r,j)*p.transpose();a(3*n+r,4*j+3)=cross(r,j);
        }
    }
    Eigen::JacobiSVD<MatXX> svd(a,Eigen::ComputeFullV);
    const auto s=svd.singularValues();
    if(s.size()!=12 || s[10]<s[0]*1e-7)return false; // Collinear/planar/ambiguous samples.
    const auto v=svd.matrixV().col(11);Mat33 m;Vec3 t;
    for(int r=0;r<3;++r){m.row(r)=v.segment<3>(4*r).transpose();t[r]=v[4*r+3];}
    const double sign=m.determinant()<0?-1:1;m*=sign;t*=sign;
    Eigen::JacobiSVD<Mat33> rotation(m,Eigen::ComputeFullU|Eigen::ComputeFullV);
    const double factor=rotation.singularValues().mean();if(factor<1e-10)return false;
    Mat33 correction=Mat33::Identity();correction(2,2)=(rotation.matrixU()*rotation.matrixV().transpose()).determinant();
    const Mat33 r=rotation.matrixU()*correction*rotation.matrixV().transpose();
    camera_pose=SE3(SO3(r),scale*t/factor-r*center);
    return camera_pose.matrix().allFinite();
}
double Linearize(const Correspondences& cs,const std::vector<size_t>& ids,
        const SE3& pose,const LoopKeyframe& query,Mat66* h=nullptr,Vec6* g=nullptr) {
    if(h)h->setZero();
    if(g)g->setZero();
    double cost=0;
    for(size_t i:ids) {
        const auto& c=cs[i];const Vec3 body=pose*c.point;
        const auto& extrinsic=query.camera_from_body[c.camera];const Vec3 p=extrinsic*body;
        if(p.norm()<1e-8)return std::numeric_limits<double>::infinity();
        const Vec3 ray=p.normalized(),residual=ray-c.bearing;cost+=residual.squaredNorm();
        if(h && g) {
            Eigen::Matrix<double,3,6> dp;dp.leftCols<3>()=-Cross(body);dp.rightCols<3>()=Mat33::Identity();
            const Eigen::Matrix<double,3,6> j=(Mat33::Identity()-ray*ray.transpose())/p.norm()*extrinsic.rotationMatrix()*dp;
            h->noalias()+=j.transpose()*j;g->noalias()+=j.transpose()*residual;
        }
    }
    return cost;
}
bool Refine(const Correspondences& cs,const std::vector<size_t>& ids,
        const LoopKeyframe& query,SE3& pose) {
    if(ids.size()<6)return false;
    double damping=1e-5;
    for(int iteration=0;iteration<20;++iteration) {
        Mat66 h;Vec6 g;const double cost=Linearize(cs,ids,pose,query,&h,&g);
        Mat66 system=h;for(int i=0;i<6;++i)system(i,i)+=damping*std::max(1e-9,h(i,i));
        const Vec6 delta=-system.ldlt().solve(g);if(!delta.allFinite())return false;
        // Our tangent order is rotation, then translation (not Sophus order).
        const SE3 trial=SE3(SO3::exp(delta.head<3>()),delta.tail<3>())*pose;
        if(Linearize(cs,ids,trial,query)<cost) {
            pose=trial;damping=std::max(1e-10,damping*0.25);if(delta.norm()<1e-9)break;
        } else {damping*=10;if(damping>1e8)break;}
    }
    return pose.matrix().allFinite();
}
}
std::vector<LoopDescriptorMatch> MatchLoopDescriptors(const LoopKeyframe& query,
        const LoopKeyframe& candidate,const LoopClosingOptions& options) {
    std::vector<Row> qrows,crows;const auto q=Flatten(query,qrows),c=Flatten(candidate,crows);
    std::vector<LoopDescriptorMatch> result;if(q.empty() || c.rows<2)return result;
    cv::BFMatcher matcher(cv::NORM_HAMMING);std::vector<std::vector<cv::DMatch>> forward;
    std::vector<cv::DMatch> backward;matcher.knnMatch(q,c,forward,2);matcher.match(c,q,backward);
    for(size_t i=0;i<forward.size();++i) {
        const auto& matches=forward[i];if(matches.size()<2)continue;const auto& best=matches[0];
        if(best.distance>options.max_hamming_distance || best.distance>=options.descriptor_ratio*matches[1].distance ||
            backward[best.trainIdx].trainIdx!=best.queryIdx)continue;
        const auto& qr=qrows[best.queryIdx];const auto& cr=crows[best.trainIdx];
        LoopDescriptorMatch m;m.query_camera=qr.camera;m.query_row=qr.row;m.candidate_camera=cr.camera;m.candidate_row=cr.row;
        result.push_back(m);
    }
    return result;
}
LoopDiagnostic VerifyLoopGeometry(const LoopKeyframe& query,const LoopKeyframe& candidate,
        const LoopClosingOptions& options) {
    ValidateLoopClosingOptions(options);
    LoopDiagnostic d;d.query_keyframe=query.id;d.candidate_keyframe=candidate.id;
    const auto matches=MatchLoopDescriptors(query,candidate,options);d.raw_matches=matches.size();
    Correspondences cs;std::array<std::vector<size_t>,2> per_camera;
    for(const auto& match:matches) {
        const auto& c=candidate.observations[match.candidate_camera][match.candidate_row];
        const auto& q=query.observations[match.query_camera][match.query_row];
        if(!c.has_point || !c.point_body.allFinite() || !q.bearing.allFinite() || std::abs(q.bearing.norm()-1)>1e-6)continue;
        Correspondence o;o.point=c.point_body;o.bearing=q.bearing;o.camera=match.query_camera;
        per_camera[o.camera].push_back(cs.size());cs.push_back(o);
    }
    d.metric_matches=cs.size();
    if(cs.size()<options.minimum_inliers){d.reason="insufficient metric correspondences";return d;}
    if(per_camera[0].size()<6 && per_camera[1].size()<6){d.reason="insufficient per-camera RANSAC support";return d;}
    d.verification_attempted=true;
    SE3 best;std::vector<size_t> best_ids;double best_cost=std::numeric_limits<double>::infinity();
    std::mt19937 rng(options.random_seed^uint32_t(query.id)^uint32_t(candidate.id<<1));
    for(int camera=0;camera<2;++camera) {
        if(per_camera[camera].size()<6)continue;
        size_t iterations=options.ransac_iterations;
        for(size_t iteration=0;iteration<iterations;++iteration) {
            std::vector<size_t> sample;
            std::uniform_int_distribution<size_t> choose(0,per_camera[camera].size()-1);
            while(sample.size()<6) {
                const size_t id=per_camera[camera][choose(rng)];
                if(std::find(sample.begin(),sample.end(),id)==sample.end())sample.push_back(id);
            }
            SE3 camera_pose;if(!BearingDlt(cs,sample,camera_pose))continue;
            SE3 body=query.camera_from_body[camera].inverse()*camera_pose;
            double cost;auto ids=Consensus(cs,body,query,options.inlier_threshold_rad,&cost);
            if(ids.size()>=6 && (ids.size()>best_ids.size() || (ids.size()==best_ids.size() && cost<best_cost))) {
                Refine(cs,ids,query,body);ids=Consensus(cs,body,query,options.inlier_threshold_rad,&cost);
                if(ids.size()>best_ids.size() || (ids.size()==best_ids.size() && cost<best_cost)) {
                    best=body;best_ids=ids;best_cost=cost;
                    size_t camera_inliers=0;for(size_t i:ids)if(cs[i].camera==camera)++camera_inliers;
                    const double fraction=double(camera_inliers)/per_camera[camera].size();
                    const double failure=1-std::pow(fraction,6);
                    if(failure<1) {
                        const double needed=failure<=0?1:std::ceil(std::log(1-options.ransac_confidence)/std::log(failure));
                        iterations=std::min(iterations,std::max(iteration+1,size_t(std::max(1.0,needed))));
                    }
                }
            }
        }
    }
    for(int pass=0;pass<3 && best_ids.size()>=6;++pass) {
        Refine(cs,best_ids,query,best);best_ids=Consensus(cs,best,query,options.inlier_threshold_rad,&best_cost);
    }
    d.geometric_inliers=best_ids.size();d.relative_body_transform=best;
    for(size_t i:best_ids)++d.camera_inliers[cs[i].camera];
    if(!best_ids.empty())d.relative_pose_rms_rad=std::sqrt(best_cost/best_ids.size());
    if(best_ids.size()<options.minimum_inliers){d.reason="insufficient geometric inliers";return d;}
    // Compare against all appearance matches, including those without metric
    // support, so a tiny valid subset cannot justify a repetitive-looking place.
    if(double(best_ids.size())/matches.size()<options.minimum_inlier_ratio){d.reason="low geometric inlier ratio";return d;}
    if(!std::isfinite(d.relative_pose_rms_rad) || d.relative_pose_rms_rad>options.maximum_rms_rad){d.reason="excessive spherical relative-pose residual";return d;}
    Mat66 h;Vec6 g;Linearize(cs,best_ids,best,query,&h,&g);
    Vec6 scaling;for(int i=0;i<6;++i)scaling[i]=1/std::sqrt(std::max(1e-15,h(i,i)));
    const Mat66 normalized=scaling.asDiagonal()*h*scaling.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Mat66> eigen(normalized);
    if(eigen.info()==Eigen::Success && eigen.eigenvalues()[0]>1e-12)
        d.pose_condition=eigen.eigenvalues()[5]/eigen.eigenvalues()[0];
    if(!std::isfinite(d.pose_condition) || d.pose_condition>options.maximum_pose_condition){d.reason="degenerate relative body pose";return d;}
    d.accepted=true;d.reason="metric body transform verified by spherical RANSAC";
    return d;
}
} // namespace myslam
