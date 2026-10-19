#include "startup_visual_bootstrap.h"
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
namespace myslam {
namespace bootstrap_experiment {
// Geometry helpers copied verbatim from src/startup_visual_trajectory.cpp.
// Deliberately isolated prototype: production implementation is untouched.
// All residuals use oriented unit rays, including rays with negative camera Z.
// Structure is temporary and never published to Map. No inertial measurements
// are used here; calibration is used only for body rotations and clock conversion.
struct Observation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t frame;
    Vec3 ray;
};
struct Track {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    uint64_t id;
    std::vector<Observation, Eigen::aligned_allocator<Observation>> observations;
    Vec3 point = Vec3::Zero();
    bool initialized = false;
};
using Tracks = std::vector<Track, Eigen::aligned_allocator<Track>>;
using Poses = std::vector<SE3, Eigen::aligned_allocator<SE3>>; // T_camera_world
Mat33 Cross(const Vec3& v) {
    Mat33 m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}
double Angle(const Vec3& a, const Vec3& b) {
    return std::atan2(a.cross(b).norm(), a.dot(b));
}
double Median(std::vector<double> x) {
    if (x.empty()) return 0;
    std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
    return x[x.size() / 2];
}
const Observation* At(const Track& track, size_t frame) {
    for (const auto& o : track.observations) if (o.frame == frame) return &o;
    return nullptr;
}
Vec3 Centre(const SE3& pose) { return -(pose.so3().inverse() * pose.translation()); }
double Residual(const SE3& pose, const Vec3& point, const Vec3& ray) {
    const Vec3 p = pose * point;
    if (p.norm() < 1e-12 || p.dot(ray) <= 0) return M_PI;
    return Angle(p.normalized(), ray);
}

// Oriented-ray intersection, followed by robust angular refinement. The two
// excluded observations are not used when validating a consecutive pair's sign.
bool Triangulate(const Track& track, const Poses& poses, Vec3& point,
                 double limit, int exclude1 = -1, int exclude2 = -1) {
    Mat33 h = Mat33::Zero(); Vec3 rhs = Vec3::Zero();
    std::vector<const Observation*> obs;
    for (const auto& o : track.observations) {
        if (int(o.frame) == exclude1 || int(o.frame) == exclude2) continue;
        const Vec3 ray = poses[o.frame].so3().inverse() * o.ray;
        const Mat33 a = Mat33::Identity() - ray * ray.transpose();
        h += a; rhs += a * Centre(poses[o.frame]); obs.push_back(&o);
    }
    if (obs.size() < 2) return false;
    Eigen::JacobiSVD<Mat33> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()[2] < 1e-9 * svd.singularValues()[0]) return false;
    point = svd.solve(rhs);
    if (!point.allFinite()) return false;
    for (int iteration = 0; iteration < 8; ++iteration) {
        h.setZero(); rhs.setZero();
        for (const auto* o : obs) {
            const auto& pose = poses[o->frame];
            const Vec3 p = pose * point;
            if (p.norm() < 1e-10) return false;
            const Vec3 u = p.normalized();
            const Vec3 r = u - o->ray;
            const Mat33 j = (Mat33::Identity() - u * u.transpose()) / p.norm() * pose.rotationMatrix();
            const double weight = std::min(1.0, limit * 0.5 / std::max(r.norm(), 1e-12));
            h += weight * j.transpose() * j; rhs -= weight * j.transpose() * r;
        }
        const Vec3 step = h.ldlt().solve(rhs);
        if (!step.allFinite()) return false;
        point += step;
        if (step.norm() < 1e-9) break;
    }
    size_t positive = 0; std::vector<double> errors;
    for (const auto* o : obs) {
        const Vec3 p = poses[o->frame] * point;
        positive += p.dot(o->ray) > 0;
        errors.push_back(Residual(poses[o->frame], point, o->ray));
    }
    return positive >= 0.8 * obs.size() && Median(errors) <= limit;
}

using JPose = Eigen::Matrix<double, 3, 6>;
JPose PoseJacobian(const Vec3& p) {
    const Vec3 u = p.normalized();
    const Mat33 ju = (Mat33::Identity() - u * u.transpose()) / p.norm();
    JPose j; j.leftCols<3>() = ju; j.rightCols<3>() = -ju * Cross(p);
    return j;
}
bool FitPose(size_t frame, const Tracks& tracks, SE3& pose, double limit, std::string& reason) {
    size_t count = 0;
    for (const auto& t : tracks) count += t.initialized && At(t, frame);
    if (count < 15) {
        reason = "only " + std::to_string(count) + " shared points (need 15)"; return false;
    }
    for (int iteration = 0; iteration < 25; ++iteration) {
        Mat66 h = Mat66::Zero(); Vec6 rhs = Vec6::Zero();
        for (const auto& t : tracks) {
            const auto* o = At(t, frame);
            if (!t.initialized || !o) continue;
            const Vec3 p = pose * t.point;
            if (p.norm() < 1e-10) continue;
            const Vec3 r = p.normalized() - o->ray;
            const auto j = PoseJacobian(p);
            const double weight = std::min(1.0, limit * 0.5 / std::max(r.norm(), 1e-12));
            h += weight * j.transpose() * j; rhs -= weight * j.transpose() * r;
        }
        const Vec6 step = (h + 1e-9 * Mat66::Identity()).ldlt().solve(rhs);
        if (!step.allFinite() || step.tail<3>().norm() > 0.5) {
            reason = "unstable bearing-pose update with " + std::to_string(count) + " points"; return false;
        }
        pose = SE3::exp(step) * pose;
        if (step.norm() < 1e-8) break;
    }
    std::vector<double> errors;
    for (const auto& t : tracks) {
        const auto* o = At(t, frame);
        if (t.initialized && o) errors.push_back(Residual(pose, t.point, o->ray));
    }
    const double median = Median(errors);
    reason = "median bearing residual " + std::to_string(median) + " rad exceeds " +
             std::to_string(limit) + " with " + std::to_string(count) + " shared points";
    return median < limit;
}

double Cost(const Tracks& tracks, const Poses& poses, double huber) {
    double sum = 0;
    for (const auto& t : tracks) if (t.initialized) for (const auto& o : t.observations) {
        const Vec3 p = poses[o.frame] * t.point;
        const double e = p.norm() < 1e-10 ? 2 : (p.normalized() - o.ray).norm();
        sum += e < huber ? e * e : huber * (2 * e - huber);
    }
    return sum;
}

// Small, visual-only startup structure/pose refinement. The first camera is
// fixed; normalizing the long anchor baseline fixes the remaining scale gauge.
// This is isolated from the runtime frontend, backend, and metric alignment.
void Refine(Tracks& tracks, Poses& poses, size_t anchor, double limit) {
    const int columns = 6 * int(poses.size() - 1);
    const double huber = limit * 0.5;
    double damping = 1e-5;
    for (int iteration = 0; iteration < 25; ++iteration) {
        MatXX h = MatXX::Zero(columns, columns); VecX rhs = VecX::Zero(columns);
        struct PointSystem {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            Mat33 inverse = Mat33::Zero(); Vec3 rhs = Vec3::Zero();
            MatXX cross;
        };
        std::vector<PointSystem, Eigen::aligned_allocator<PointSystem>> systems(tracks.size());
        for (size_t k = 0; k < tracks.size(); ++k) {
            const auto& t = tracks[k];
            if (!t.initialized) continue;
            auto& system = systems[k]; system.cross = MatXX::Zero(columns, 3);
            Mat33 hp = Mat33::Zero();
            for (const auto& o : t.observations) {
                const auto& pose = poses[o.frame]; const Vec3 p = pose * t.point;
                if (p.norm() < 1e-10) continue;
                const Vec3 u = p.normalized(), r = u - o.ray;
                const double w = std::min(1.0, huber / std::max(r.norm(), 1e-12));
                const Mat33 jp = (Mat33::Identity() - u * u.transpose()) / p.norm() * pose.rotationMatrix();
                hp += w * jp.transpose() * jp; system.rhs -= w * jp.transpose() * r;
                if (!o.frame) continue;
                const int offset = 6 * int(o.frame - 1); const JPose jc = PoseJacobian(p);
                h.block<6, 6>(offset, offset) += w * jc.transpose() * jc;
                rhs.segment<6>(offset) -= w * jc.transpose() * r;
                system.cross.block<6, 3>(offset, 0) += w * jc.transpose() * jp;
            }
            hp.diagonal() += damping * hp.diagonal().cwiseMax(1e-8);
            system.inverse = hp.ldlt().solve(Mat33::Identity());
            h.noalias() -= system.cross * system.inverse * system.cross.transpose();
            rhs.noalias() -= system.cross * system.inverse * system.rhs;
        }
        h.diagonal() += damping * h.diagonal().cwiseMax(1e-8);
        const VecX step = h.ldlt().solve(rhs);
        if (!step.allFinite()) break;
        Poses candidate = poses; Tracks points = tracks;
        for (size_t i = 1; i < poses.size(); ++i) candidate[i] = SE3::exp(step.segment<6>(6 * (i - 1))) * poses[i];
        for (size_t k = 0; k < points.size(); ++k) if (points[k].initialized) {
            const auto& s = systems[k];
            points[k].point += s.inverse * (s.rhs - s.cross.transpose() * step);
        }
        if (Cost(points, candidate, huber) < Cost(tracks, poses, huber)) {
            poses.swap(candidate); tracks.swap(points); damping = std::max(1e-9, damping / 3);
            const double gauge = Centre(poses[anchor]).norm();
            if (!std::isfinite(gauge) || gauge < 1e-8) break;
            for (auto& p : poses) p.translation() /= gauge;
            for (auto& t : tracks) if (t.initialized) t.point /= gauge;
            if (step.norm() < 1e-7) break;
        } else {
            damping *= 10;
            if (damping > 1e4) break;
        }
    }
}

struct Anchor { size_t frame; BearingMotionResult motion; double score; };
Tracks Collect(const std::vector<HiltiTrackingResult>& frames, const Camera& camera) {
    Tracks tracks; std::map<uint64_t,size_t> ids;
    for(size_t i=0;i<frames.size();++i) {
        if(frames[i].stream!=frames[0].stream || (i && frames[i].timestamp_ns<=frames[i-1].timestamp_ns))
            throw std::invalid_argument("mixed camera or nonincreasing startup time");
        std::set<uint64_t> unique;
        for(const auto& f:frames[i].features) {
            if(!unique.insert(f.id).second) throw std::invalid_argument("duplicate startup feature ID");
            Vec3 ray; if(!camera.Unproject(Vec2(f.pixel.x,f.pixel.y),ray)) continue;
            if(!ids.count(f.id)){ids[f.id]=tracks.size();Track t;t.id=f.id;tracks.push_back(t);}
            tracks[ids[f.id]].observations.push_back({i,ray});
        }
    }
    return tracks;
}
std::vector<Anchor> Anchors(const Tracks& tracks,size_t n) {
    std::vector<Anchor> result;
    for(size_t j=std::max(size_t(2),n/6);j<n;++j) {
        if(j!=n-1 && j%std::max(size_t(1),n/6)) continue;
        BearingCorrespondences matches;
        for(const auto& t:tracks){const auto* a=At(t,0);const auto* b=At(t,j);if(a&&b){BearingCorrespondence m;m.feature_id=t.id;m.previous=a->ray;m.current=b->ray;matches.push_back(m);}}
        const auto m=EstimateBearingMotion(matches);
        if(m.rotation_valid&&m.translation_valid&&m.inliers>=15&&m.median_parallax_rad>=0.005)
            result.push_back({j,m,m.median_parallax_rad*std::sqrt(double(m.inliers))});
    }
    std::stable_sort(result.begin(),result.end(),[](const Anchor&a,const Anchor&b){return a.score>b.score;});
    return result;
}
Poses Rotations(const std::vector<BearingMotionResult>& motions) {
    Poses poses(motions.size()+1);
    for(size_t i=1;i<poses.size();++i){if(!motions[i-1].rotation_valid)throw std::invalid_argument("missing pair rotation");poses[i].so3()=SO3(motions[i-1].rotation)*poses[i-1].so3();}
    return poses;
}
size_t Seed(Tracks& tracks,Poses& poses,const Anchor& seed,double limit) {
    const size_t anchor=seed.frame;const auto& m=seed.motion;
    const SO3 correction=SO3(m.rotation)*poses[anchor].so3().inverse();
    for(size_t i=1;i<poses.size();++i){poses[i].so3()=SO3::exp(double(i)/anchor*correction.log())*poses[i].so3();const Vec3 c=-(SO3(m.rotation).inverse()*m.translation_direction)*(double(i)/anchor);poses[i].translation()=-(poses[i].so3()*c);}
    poses[anchor]=SE3(SO3(m.rotation),m.translation_direction);
    size_t count=0;
    for(auto& t:tracks){const auto* a=At(t,0);const auto* b=At(t,anchor);if(!a||!b)continue;Track pair;pair.observations={*a,*b};if(Angle(a->ray,poses[anchor].so3().inverse()*b->ray)>=0.005&&Triangulate(pair,poses,t.point,limit)){t.initialized=true;++count;}}
    return count;
}
// Same provisional geometry as production; only the treatment of a provisional
// median residual differs. It is NOT a final acceptance test before joint BA.
size_t InitializePoses(Tracks& tracks,Poses& poses,const Anchor& seed,
        const std::vector<HiltiTrackingResult>& frames,const std::vector<BearingMotionResult>& motions,
        double limit,bool stop_on_failure,size_t& failures) {
    for(size_t i=1;i<poses.size();++i){
        if(i>seed.frame){const double ratio=double(frames[i].timestamp_ns-frames[i-1].timestamp_ns)/double(frames[i-1].timestamp_ns-frames[i-2].timestamp_ns);const Vec3 c=Centre(poses[i-1])+ratio*(Centre(poses[i-1])-Centre(poses[i-2]));poses[i].so3()=SO3(motions[i-1].rotation)*poses[i-1].so3();poses[i].translation()=-(poses[i].so3()*c);}
        std::string reason;
        if(i!=seed.frame&&!FitPose(i,tracks,poses[i],limit,reason)){++failures;if(stop_on_failure)return i+1;}
        if(!poses[i].matrix().allFinite())return i;
        for(auto& t:tracks)if(i>=seed.frame&&!t.initialized){Track known;for(const auto&o:t.observations)if(o.frame<=i||o.frame==seed.frame)known.observations.push_back(o);Vec3 p;if(known.observations.size()>=3&&Triangulate(known,poses,p,limit)){t.point=p;t.initialized=true;}}
    }
    return poses.size();
}
double Condition(const Track& t,const Poses& poses) {
    Mat33 h=Mat33::Zero();for(const auto&o:t.observations){const Vec3 r=poses[o.frame].so3().inverse()*o.ray;h+=Mat33::Identity()-r*r.transpose();}
    Eigen::JacobiSVD<Mat33> s(h);return std::sqrt(s.singularValues()[0]/std::max(s.singularValues()[2],1e-15));
}
double Parallax(const Track& t,const Poses& poses) {
    double angle=0;if(t.observations.empty())return NAN;
    // Full track span relative to its first ray; no pinhole/z-axis approximation.
    const Vec3 first=poses[t.observations.front().frame].so3().inverse()*t.observations.front().ray;
    for(const auto&o:t.observations) angle=std::max(angle,Angle(first,poses[o.frame].so3().inverse()*o.ray));
    return angle;
}
void Summarize(BootstrapResult& result,const Tracks& tracks,const Poses& poses,
        const std::vector<HiltiTrackingResult>& frames,const std::vector<BearingMotionResult>& motions,
        const Camera& camera,double limit,size_t available) {
    result.poses=poses;result.audited_pose_count=available;
    std::vector<double> parallaxes,conditions;Vec3 mean=Vec3::Zero();size_t points=0,positive=0,total=0;
    for(const auto&t:tracks)if(t.initialized){parallaxes.push_back(Parallax(t,poses));conditions.push_back(Condition(t,poses));mean+=t.point;++points;for(const auto&o:t.observations)if(o.frame<available){++total;positive+=(poses[o.frame]*t.point).dot(o.ray)>0;}}
    result.track_parallax_median=points?Median(parallaxes):NAN;result.track_condition_median=points?Median(conditions):NAN;result.positive_fraction=total?double(positive)/total:NAN;
    if(points>=3){mean/=points;Mat33 covariance=Mat33::Zero();for(const auto&t:tracks)if(t.initialized){const Vec3 d=t.point-mean;covariance+=d*d.transpose();}Eigen::JacobiSVD<Mat33>s(covariance);result.cloud_planarity_ratio=s.singularValues()[2]/std::max(s.singularValues()[0],1e-15);}
    // Raw PCA can be dominated by a single poorly determined far landmark.
    // Diagnostic-only 90%-range trimmed PCA; no landmark/state is changed.
    if(points>=15) {
        std::vector<double> ranges;
        for(const auto&t:tracks) if(t.initialized) ranges.push_back(t.point.norm());
        std::sort(ranges.begin(),ranges.end());
        const double cutoff=ranges[(ranges.size()-1)*9/10];
        Vec3 centre=Vec3::Zero();size_t retained=0;
        for(const auto&t:tracks) if(t.initialized&&t.point.norm()<=cutoff){centre+=t.point;++retained;}
        centre/=retained;Mat33 covariance=Mat33::Zero();
        for(const auto&t:tracks) if(t.initialized&&t.point.norm()<=cutoff){const Vec3 d=t.point-centre;covariance+=d*d.transpose();}
        Eigen::JacobiSVD<Mat33> svd(covariance);
        result.trimmed_planarity_ratio=svd.singularValues()[2]/std::max(svd.singularValues()[0],1e-15);
        result.trimmed_linearity_ratio=svd.singularValues()[1]/std::max(svd.singularValues()[0],1e-15);
    }
    SO3 accumulated;
    for(size_t i=1;i<frames.size();++i){
        BootstrapPairAudit d;d.frame=i;const auto&m=motions[i-1];d.input_translation_valid=m.translation_valid;
        if(m.translation_valid)d.pair_axis=m.translation_direction;
        else if(m.essential.norm()>1e-10){Eigen::JacobiSVD<Mat33>s(m.essential,Eigen::ComputeFullU);d.pair_axis=s.matrixU().col(2);}
        const auto matches=MakeBearingMatches(frames[i-1],frames[i],camera).matches;d.shared=matches.size();double ep=0,rot=0;size_t count=0;
        for(size_t k=0;k<matches.size();++k){const auto&b=matches[k];if(m.inlier_mask.size()==matches.size()&&!m.inlier_mask[k])continue;
            const Vec3 a=m.rotation*b.previous,n=a.cross(b.current);
            if(d.pair_axis.allFinite()){d.pair_positive+=-n.dot(d.pair_axis.cross(b.current))>0&&-n.dot(d.pair_axis.cross(a))>0;d.pair_opposite_positive+=n.dot(d.pair_axis.cross(b.current))>0&&n.dot(d.pair_axis.cross(a))>0;}
            if(m.essential.norm()>1e-10)ep+=std::pow(BearingEpipolarError(m.essential,b.previous,b.current),2);
            rot+=std::pow(Angle(a,b.current),2);++count;
        }
        d.epipolar_rms=count&&m.essential.norm()>1e-10?std::sqrt(ep/count):NAN;d.rotation_fit_rms=count?std::sqrt(rot/count):NAN;
        if(d.pair_axis.allFinite()&&count)d.sign_confidence=double(std::max(d.pair_positive,d.pair_opposite_positive))/count;
        accumulated=SO3(m.rotation)*accumulated;d.pose_available=i<available;
        if(d.pose_available){
            d.rotation_change=(poses[i].so3()*(SO3(m.rotation)*poses[i-1].so3()).inverse()).log().norm();d.accumulated_rotation_change=(poses[i].so3()*accumulated.inverse()).log().norm();
            const Vec3 a=Centre(poses[i-1]),b=Centre(poses[i]);d.baseline=(b-a).norm();
            if(d.baseline>1e-12){d.reconstructed_direction=-(poses[i].so3()*(b-a))/d.baseline;if(d.pair_axis.allFinite())d.axis_difference=Angle(d.reconstructed_direction,d.pair_axis);}
            if(i>1){const double previous=(a-Centre(poses[i-2])).norm();if(previous>1e-12)d.baseline_ratio=d.baseline/previous;}
            const SE3 opposite(poses[i].so3(),-(poses[i].so3()*(2*a-b)));
            std::vector<double> e,p,c,margins,heldout_errors;size_t votes=0,heldout_positive=0;Mat66 h=Mat66::Zero();
            for(const auto&t:tracks)if(t.initialized){const auto*x=At(t,i-1);const auto*y=At(t,i);if(!x||!y)continue;++d.multiview_shared;d.multiview_positive+=(poses[i]*t.point).dot(y->ray)>0;
                e.push_back(Residual(poses[i],t.point,y->ray));p.push_back(Angle(poses[i-1].so3().inverse()*x->ray,poses[i].so3().inverse()*y->ray));c.push_back(Condition(t,poses));
                const Vec3 z=poses[i]*t.point;if(z.norm()>1e-10){const auto j=PoseJacobian(z);h+=j.transpose()*j;}
                Vec3 heldout;if(t.observations.size()>=5&&Triangulate(t,poses,heldout,limit,int(i-1),int(i))&&Residual(poses[i-1],heldout,x->ray)<=limit){++d.heldout_shared;const double err=Residual(poses[i],heldout,y->ray),other=Residual(opposite,heldout,y->ray);votes+=other>err;margins.push_back(other-err);heldout_errors.push_back(err);heldout_positive+=(poses[i]*heldout).dot(y->ray)>0;}
            }
            if(!e.empty()){d.bearing_median=Median(e);d.parallax=Median(p);d.depth_condition_median=Median(c);d.depth_condition_max=*std::max_element(c.begin(),c.end());d.positive_fraction=double(d.multiview_positive)/d.multiview_shared;}
            if(d.heldout_shared){d.heldout_vote=double(votes)/d.heldout_shared;d.heldout_margin=Median(margins);d.heldout_residual=Median(heldout_errors);d.heldout_positive_fraction=double(heldout_positive)/d.heldout_shared;}
            // Column-normalized bearing-PnP Hessian, diagnostic only (landmarks fixed).
            const Vec6 s=h.diagonal().cwiseMax(1e-15).cwiseSqrt().cwiseInverse();Eigen::JacobiSVD<Mat66>svd(s.asDiagonal()*h*s.asDiagonal());d.pose_condition=std::sqrt(svd.singularValues()[0]/std::max(svd.singularValues()[5],1e-15));
        }
        result.pairs.push_back(d);
    }
}
void ExportWindow(VioVisualWindow& w,const Poses& poses,const std::vector<HiltiTrackingResult>& frames,
        const Camera& camera,const ImuCalibration& calibration) {
    w.frames.clear();for(size_t i=0;i<poses.size();++i){VioVisualFrame f;f.timestamp_ns=CameraTimeOnImuClock(calibration,frames[i].stream==SensorStream::Cam0?0:1,frames[i].timestamp_ns);f.rotation_wb=poses[i].so3().inverse()*camera.pose().so3();f.camera_position=Centre(poses[i]);w.frames.push_back(f);}
}
} // namespace bootstrap_experiment

BootstrapResult AuditCurrentStartup(const std::vector<HiltiTrackingResult>& frames,
        const std::vector<BearingMotionResult>& motions,const Camera& camera,
        const ImuCalibration& calibration,const VioInitializerOptions& options) {
    using namespace bootstrap_experiment;BootstrapResult r;const auto begin=std::chrono::steady_clock::now();
    r.window=MakeVioVisualWindow(frames,motions,camera,calibration,options);Tracks tracks=Collect(frames,camera);Poses poses=Rotations(motions);size_t available=0;
    if(std::isfinite(r.window.bearing_rms_rad)){
        available=frames.size();for(size_t i=0;i<poses.size();++i){const SO3 rotation=camera.pose().so3()*r.window.frames[i].rotation_wb.inverse();poses[i]=SE3(rotation,-(rotation*r.window.frames[i].camera_position));}
        for(auto&t:tracks)if(t.observations.size()>=3&&Triangulate(t,poses,t.point,options.max_bearing_rms_rad)){size_t good=0;for(const auto&o:t.observations)good+=Residual(poses[o.frame],t.point,o.ray)<options.max_bearing_rms_rad;t.initialized=good>=0.8*t.observations.size();}
        r.audit_scope="native final poses; audit-only retriangulated landmarks";
    }else{const auto anchors=Anchors(tracks,frames.size());if(!anchors.empty()&&Seed(tracks,poses,anchors[0],options.max_bearing_rms_rad)>=15){size_t failures=0;available=InitializePoses(tracks,poses,anchors[0],frames,motions,options.max_bearing_rms_rad,true,failures);for(auto&t:tracks){Track partial;for(const auto&o:t.observations)if(o.frame<available)partial.observations.push_back(o);t.observations=partial.observations;}}r.audit_scope="provisional trace up to native early failure; not final reconstruction";}
    Summarize(r,tracks,poses,frames,motions,camera,options.max_bearing_rms_rad,available);
    r.runtime_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return r;
}
BootstrapResult BuildMultiFrameStartup(const std::vector<HiltiTrackingResult>& frames,
        const std::vector<BearingMotionResult>& motions,const Camera& camera,
        const ImuCalibration& calibration,const VioInitializerOptions& options) {
    using namespace bootstrap_experiment;const auto begin=std::chrono::steady_clock::now();
    if(frames.size()<4||motions.size()+1!=frames.size())throw std::invalid_argument("invalid startup frame/pair count");
    const Tracks original=Collect(frames,camera);const Poses rotations=Rotations(motions);auto anchors=Anchors(original,frames.size());
    BootstrapResult best;best.window.translation_failure="no reliable long-baseline anchor: pure rotation or low parallax";
    std::vector<BootstrapAttempt> attempts;
    // A fixed geometry-only hypothesis budget. No GT, inertial or metric scoring.
    if(anchors.size()>3)anchors.resize(3);
    for(const auto&anchor:anchors){
        BootstrapResult r;auto&w=r.window;Tracks tracks=original;Poses poses=rotations;BootstrapAttempt attempt;attempt.anchor=anchor.frame;attempt.anchor_parallax=anchor.motion.median_parallax_rad;w.anchor_frame=anchor.frame;
        if(Seed(tracks,poses,anchor,options.max_bearing_rms_rad)<15){attempt.reason="insufficient positive-ray anchor structure";attempts.push_back(attempt);continue;}
        const size_t available=InitializePoses(tracks,poses,anchor,frames,motions,options.max_bearing_rms_rad,false,attempt.provisional_pose_failures);
        if(available!=frames.size()){attempt.reason="nonfinite provisional pose";attempts.push_back(attempt);continue;}
        for(auto&t:tracks)if(t.observations.size()>=3){Vec3 point;if(Triangulate(t,poses,point,options.max_bearing_rms_rad)){t.point=point;t.initialized=true;}}
        Refine(tracks,poses,anchor.frame,options.max_bearing_rms_rad);
        for(auto&t:tracks)if(t.initialized){size_t good=0;for(const auto&o:t.observations)good+=Residual(poses[o.frame],t.point,o.ray)<options.max_bearing_rms_rad;
            if(good<0.8*t.observations.size()){t.initialized=false;++w.rejected_tracks;}
            else if(Condition(t,poses)>options.max_visual_condition){t.initialized=false;++r.weak_tracks;}}
        Refine(tracks,poses,anchor.frame,options.max_bearing_rms_rad);
        // Preserve the same per-track quality limits after the final solve.
        for(auto&t:tracks)if(t.initialized&&Condition(t,poses)>options.max_visual_condition){t.initialized=false;++r.weak_tracks;}
        std::vector<std::vector<double>> errors(frames.size());std::vector<size_t> positives(frames.size(),0);double sum=0;size_t count=0;w.visual_condition=0;
        for(const auto&t:tracks)if(t.initialized){++w.reconstruction_tracks;w.visual_condition=std::max(w.visual_condition,Condition(t,poses));for(const auto&o:t.observations){const double e=Residual(poses[o.frame],t.point,o.ray);errors[o.frame].push_back(e);positives[o.frame]+=(poses[o.frame]*t.point).dot(o.ray)>0;if(e<=options.max_bearing_rms_rad){sum+=e*e;++count;}}}
        w.bearing_rms_rad=count?std::sqrt(sum/count):INFINITY;r.max_pose_median=0;bool supported=w.reconstruction_tracks>=15;
        for(size_t i=0;i<errors.size();++i){if(errors[i].size()<15||positives[i]<0.8*errors[i].size()||Median(errors[i])>options.max_bearing_rms_rad)supported=false;r.max_pose_median=std::max(r.max_pose_median,errors[i].empty()?INFINITY:Median(errors[i]));}
        w.positions_in_common_scale=supported&&w.bearing_rms_rad<=options.max_bearing_rms_rad&&w.visual_condition<=options.max_visual_condition;
        w.translation_failure=w.positions_in_common_scale?"":"whole-window support, cheirality or bearing quality failed";
        ExportWindow(w,poses,frames,camera,calibration);r.audit_scope="joint multi-frame poses and actual optimized shared landmarks";
        Summarize(r,tracks,poses,frames,motions,camera,options.max_bearing_rms_rad,available);
        // Retain production's held-out support/cheirality/bearing checks.
        // Remove only the requirement to independently resolve every short
        // interval's opposite sign: the longer shared geometry estimates poses.
        for(const auto&d:r.pairs) {
            if(d.heldout_shared<15 || d.heldout_positive_fraction<0.8 ||
               !std::isfinite(d.heldout_residual) || d.heldout_residual>options.max_bearing_rms_rad) {
                w.positions_in_common_scale=false;
                w.translation_failure="held-out multi-view support, cheirality or bearing quality failed";
            }
        }
        attempt.valid=w.positions_in_common_scale;attempt.reason=w.translation_failure;attempt.tracks=w.reconstruction_tracks;attempt.weak_tracks=r.weak_tracks;attempt.rms=w.bearing_rms_rad;attempt.condition=w.visual_condition;attempts.push_back(attempt);
        // Prefer valid solutions, then greatest retained track support, then RMS.
        const bool better=(w.positions_in_common_scale&&!best.window.positions_in_common_scale)||
            (w.positions_in_common_scale==best.window.positions_in_common_scale&&
             (w.reconstruction_tracks>best.window.reconstruction_tracks||(w.reconstruction_tracks==best.window.reconstruction_tracks&&w.bearing_rms_rad<best.window.bearing_rms_rad)));
        if(better) {
            for(const auto& t:tracks)if(t.initialized) {
                BootstrapLandmark landmark;landmark.id=t.id;landmark.point=t.point;
                for(const auto& o:t.observations) {
                    BootstrapObservation observation;observation.frame=o.frame;observation.ray=o.ray;
                    landmark.observations.push_back(observation);
                }
                r.landmarks.push_back(std::move(landmark));
            }
            best=std::move(r);
        }
    }
    best.attempts=std::move(attempts);best.runtime_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return best;
}
} // namespace myslam
