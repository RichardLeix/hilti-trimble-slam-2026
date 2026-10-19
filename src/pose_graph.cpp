#include "myslam/pose_graph.h"
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace myslam {
namespace {
using Poses = HiltiAlignedVector<PoseGraphPose>;
struct Edge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    size_t from = 0, to = 0;
    SE3 measurement;
    Mat66 covariance = Mat66::Identity(), information = Mat66::Identity();
};
using Edges = HiltiAlignedVector<Edge>;

bool ValidPose(const SE3& pose) {
    return pose.translation().allFinite() && pose.unit_quaternion().coeffs().allFinite() &&
        std::abs(pose.unit_quaternion().squaredNorm()-1) < 1e-10;
}
double Seconds(int64_t a, int64_t b) {
    return double((static_cast<long double>(a)-static_cast<long double>(b))*1e-9L);
}
double Clamp(double value, double minimum, double maximum) {
    return std::max(minimum,std::min(maximum,value));
}
double Huber(double chi2, double delta) {
    return chi2 <= delta*delta ? chi2 : 2*delta*std::sqrt(chi2)-delta*delta;
}
Vec6 Error(const Edge& edge, const Poses& poses) {
    return PoseGraphResidual(edge.measurement,poses[edge.from].body_pose,poses[edge.to].body_pose);
}
double Chi2(const Edge& edge, const Poses& poses) {
    const Vec6 e=Error(edge,poses);
    if(!e.allFinite())return std::numeric_limits<double>::infinity();
    const double value=e.dot(edge.information*e);
    return std::isfinite(value)?std::max(0.0,value):std::numeric_limits<double>::infinity();
}
double Objective(const Edges& odometry, const Edges& loops, const Poses& poses,
    const std::vector<bool>& retained, double delta) {
    double cost=0;
    for(const auto& e:odometry)cost+=Chi2(e,poses);
    for(size_t k=0;k<loops.size();++k)if(retained[k])cost+=Huber(Chi2(loops[k],poses),delta);
    return cost;
}
class GraphVertex final : public g2o::BaseVertex<6,SE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void setToOriginImpl() override { _estimate=SE3(); }
    void oplusImpl(const double* update) override {
        _estimate=PoseGraphApplyUpdate(_estimate,Eigen::Map<const Vec6>(update));
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};
class GraphEdge final : public g2o::BaseBinaryEdge<6,SE3,GraphVertex,GraphVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void computeError() override {
        _error=PoseGraphResidual(_measurement,From(),To());
    }
    void linearizeOplus() override {
        Mat66 a,b;LinearizePoseGraphEdge(_measurement,From(),To(),a,b);
        _jacobianOplusXi=a;_jacobianOplusXj=b;
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
   private:
    const SE3& From() const { return static_cast<const GraphVertex*>(_vertices[0])->estimate(); }
    const SE3& To() const { return static_cast<const GraphVertex*>(_vertices[1])->estimate(); }
};
// Each pass starts from the original trajectory. A rejected loop's deformation
// cannot leak into the final graph through an optimizer warm start.
bool Solve(const Poses& original, const Edges& odometry, const Edges& loops,
    const std::vector<bool>& retained, const PoseGraphOptions& options,
    Poses& output, size_t& iterations) {
    output=original;
    if(original.size()<2 || std::count(retained.begin(),retained.end(),true)==0)return true;
    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<6,6>>;
    using Linear = g2o::LinearSolverCSparse<Block::PoseMatrixType>;
    bool force_stop=false;
    g2o::SparseOptimizerTerminateAction termination;
    termination.setGainThreshold(1e-9);termination.setMaxIterations(int(options.maximum_iterations));
    g2o::SparseOptimizer optimizer;
    optimizer.setForceStopFlag(&force_stop);
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<Block>(g2o::make_unique<Linear>())));
    std::vector<GraphVertex*> vertices;
    for(size_t k=0;k<original.size();++k) {
        auto* v=new GraphVertex;v->setId(int(k));v->setEstimate(original[k].body_pose);
        v->setFixed(k==0);optimizer.addVertex(v);vertices.push_back(v);
    }
    int edge_id=0;
    auto add=[&](const Edge& edge, bool robust) {
        auto* e=new GraphEdge;e->setId(edge_id++);
        e->setVertex(0,vertices[edge.from]);e->setVertex(1,vertices[edge.to]);
        e->setMeasurement(edge.measurement);e->setInformation(edge.information);
        if(robust) {auto* kernel=new g2o::RobustKernelHuber;kernel->setDelta(options.loop_huber_delta);e->setRobustKernel(kernel);}
        optimizer.addEdge(e);
    };
    for(const auto& edge:odometry)add(edge,false);
    for(size_t k=0;k<loops.size();++k)if(retained[k])add(loops[k],true);
    if(!optimizer.initializeOptimization())return false;
    // A single optimize call preserves LM damping between iterations. Calling
    // optimize(1) repeatedly would reset lambda and impede long-range closure.
    // The post-iteration action stops on a small relative objective decrease.
    optimizer.computeActiveErrors();const double initial=optimizer.activeRobustChi2();
    optimizer.addPostIterationAction(&termination);
    const int done=optimizer.optimize(int(options.maximum_iterations));
    optimizer.removePostIterationAction(&termination);
    if(done<=0)return false;
    iterations+=size_t(done);optimizer.computeActiveErrors();
    const double current=optimizer.activeRobustChi2();
    if(!std::isfinite(current) || current>initial+1e-7*std::max(1.0,initial))return false;
    for(size_t k=0;k<original.size();++k) {
        output[k].body_pose=vertices[k]->estimate();if(!ValidPose(output[k].body_pose))return false;
    }
    return true;
}
// Independent right-tangent odometry noise: composing P*Z transports the
// previous covariance with Ad(Z^-1). Inversion uses Ad(P). This retains the
// rotation/translation lever-arm terms missing from a scalar sqrt(N) gate.
double Innovation(const Edge& loop, const Edges& odometry, const Poses& poses) {
    const size_t begin=std::min(loop.from,loop.to),end=std::max(loop.from,loop.to);
    SE3 prediction;Mat66 covariance=Mat66::Zero();
    for(size_t k=begin;k<end;++k) {
        const auto& edge=odometry[k];const Mat66 adjoint=edge.measurement.inverse().Adj();
        covariance=adjoint*covariance*adjoint.transpose()+edge.covariance;
        prediction=prediction*edge.measurement;
    }
    if(loop.from>loop.to) {
        const Mat66 adjoint=prediction.Adj();covariance=adjoint*covariance*adjoint.transpose();
        prediction=prediction.inverse();
    }
    // Differentiate the actual Log residual, including large initial rotation.
    Mat66 jp,jm;const double step=1e-6;
    for(int axis=0;axis<6;++axis) {
        Vec6 delta=Vec6::Zero();delta[axis]=step;
        jp.col(axis)=((loop.measurement.inverse()*prediction*SE3::exp(delta)).log()-
            (loop.measurement.inverse()*prediction*SE3::exp(-delta)).log())/(2*step);
        jm.col(axis)=(((loop.measurement*SE3::exp(delta)).inverse()*prediction).log()-
            ((loop.measurement*SE3::exp(-delta)).inverse()*prediction).log())/(2*step);
    }
    Mat66 innovation=jp*covariance*jp.transpose()+jm*loop.covariance*jm.transpose();
    innovation=0.5*(innovation+innovation.transpose()).eval();
    Eigen::LDLT<Mat66> ldlt(innovation);
    if(ldlt.info()!=Eigen::Success || !ldlt.isPositive())throw std::runtime_error("invalid loop innovation covariance");
    const Vec6 e=Error(loop,poses);const double chi2=e.dot(ldlt.solve(e));
    if(!std::isfinite(chi2))throw std::runtime_error("nonfinite loop innovation");
    return std::max(0.0,chi2);
}
void ReadPose(std::istream& stream, SE3& pose) {
    double x,y,z,qx,qy,qz,qw;
    if(!(stream>>x>>y>>z>>qx>>qy>>qz>>qw))throw std::invalid_argument("truncated pose graph pose");
    Eigen::Quaterniond q(qw,qx,qy,qz);Vec3 p(x,y,z);
    if(!p.allFinite() || !q.coeffs().allFinite() || std::abs(q.norm()-1)>1e-6)
        throw std::invalid_argument("invalid pose graph quaternion/translation");
    q.normalize();pose=SE3(q,p);
}
void WritePose(std::ostream& stream, const SE3& pose) {
    const auto t=pose.translation();const auto q=pose.unit_quaternion();
    stream<<t.x()<<' '<<t.y()<<' '<<t.z()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w();
}
} // namespace

Vec6 PoseGraphResidual(const SE3& measurement, const SE3& from, const SE3& to) {
    return (measurement.inverse()*from.inverse()*to).log();
}
SE3 PoseGraphApplyUpdate(const SE3& pose, const Vec6& increment) {
    return SE3::exp(increment)*pose;
}
void LinearizePoseGraphEdge(const SE3& measurement, const SE3& from, const SE3& to,
    Mat66& from_jacobian, Mat66& to_jacobian) {
    // Central manifold differences. The same update function is used by g2o;
    // no quaternion/additive-translation approximation is mixed into the edge.
    const double step=1e-6;
    for(int axis=0;axis<6;++axis) {
        Vec6 delta=Vec6::Zero();delta[axis]=step;
        from_jacobian.col(axis)=(PoseGraphResidual(measurement,PoseGraphApplyUpdate(from,delta),to)-
            PoseGraphResidual(measurement,PoseGraphApplyUpdate(from,-delta),to))/(2*step);
        to_jacobian.col(axis)=(PoseGraphResidual(measurement,from,PoseGraphApplyUpdate(to,delta))-
            PoseGraphResidual(measurement,from,PoseGraphApplyUpdate(to,-delta)))/(2*step);
    }
}
void ValidatePoseGraphOptions(const PoseGraphOptions& o) {
    const double positive[]={o.odometry_translation_floor_m,o.odometry_translation_diffusion,
        o.odometry_rotation_floor_rad,o.odometry_rotation_diffusion,o.loop_translation_sigma_m,
        o.loop_rotation_sigma_rad,o.loop_min_translation_sigma_m,o.loop_max_translation_sigma_m,
        o.loop_min_rotation_sigma_rad,o.loop_max_rotation_sigma_rad,o.loop_reference_inliers,
        o.loop_max_effective_inliers,o.loop_reference_bearing_rms_rad,o.loop_reference_condition,
        o.loop_huber_delta,o.innovation_chi2_limit,o.postfit_chi2_limit};
    for(double v:positive)if(!std::isfinite(v) || v<=0)throw std::invalid_argument("pose graph option must be positive/finite");
    const double nonnegative[]={o.odometry_translation_relative,o.odometry_rotation_relative,o.loop_correlation_seconds};
    for(double v:nonnegative)if(!std::isfinite(v) || v<0)throw std::invalid_argument("pose graph option must be nonnegative/finite");
    if(!o.maximum_iterations || o.maximum_iterations>10000 ||
        o.loop_min_translation_sigma_m>o.loop_max_translation_sigma_m ||
        o.loop_min_rotation_sigma_rad>o.loop_max_rotation_sigma_rad || o.loop_max_effective_inliers<1)
        throw std::invalid_argument("invalid pose graph iteration limit or covariance bounds");
}
PoseGraphOptions LoadPoseGraphOptions(const std::string& path) {
    cv::FileStorage fs(path,cv::FileStorage::READ);
    if(!fs.isOpened())throw std::runtime_error("cannot open pose graph options: "+path);
    PoseGraphOptions o;
    if(!fs["maximum_iterations"].empty()) {
        double n;fs["maximum_iterations"]>>n;
        if(!std::isfinite(n) || n<1 || n>10000 || n!=std::floor(n))throw std::invalid_argument("invalid maximum_iterations");
        o.maximum_iterations=size_t(n);
    }
#define LOAD(name) if(!fs[#name].empty())fs[#name]>>o.name
    LOAD(odometry_translation_floor_m);LOAD(odometry_translation_diffusion);LOAD(odometry_translation_relative);
    LOAD(odometry_rotation_floor_rad);LOAD(odometry_rotation_diffusion);LOAD(odometry_rotation_relative);
    LOAD(loop_translation_sigma_m);LOAD(loop_rotation_sigma_rad);
    LOAD(loop_min_translation_sigma_m);LOAD(loop_max_translation_sigma_m);
    LOAD(loop_min_rotation_sigma_rad);LOAD(loop_max_rotation_sigma_rad);
    LOAD(loop_reference_inliers);LOAD(loop_max_effective_inliers);LOAD(loop_reference_bearing_rms_rad);
    LOAD(loop_reference_condition);LOAD(loop_correlation_seconds);LOAD(loop_huber_delta);
    LOAD(innovation_chi2_limit);LOAD(postfit_chi2_limit);
#undef LOAD
    ValidatePoseGraphOptions(o);return o;
}
Mat66 PoseGraphOdometryCovariance(const PoseGraphOptions& o, double dt, const SE3& relative) {
    ValidatePoseGraphOptions(o);
    if(!std::isfinite(dt) || dt<=0 || !ValidPose(relative))throw std::invalid_argument("invalid odometry interval/pose");
    const double motion=o.odometry_translation_relative*relative.translation().norm();
    const double angle=o.odometry_rotation_relative*relative.so3().log().norm();
    const double vt=o.odometry_translation_floor_m*o.odometry_translation_floor_m+
        o.odometry_translation_diffusion*o.odometry_translation_diffusion*dt+motion*motion;
    const double vr=o.odometry_rotation_floor_rad*o.odometry_rotation_floor_rad+
        o.odometry_rotation_diffusion*o.odometry_rotation_diffusion*dt+angle*angle;
    Mat66 covariance=Mat66::Zero();covariance.diagonal().head<3>().setConstant(vt);
    covariance.diagonal().tail<3>().setConstant(vr);
    if(!covariance.allFinite() || vt<=0 || vr<=0)throw std::invalid_argument("odometry covariance must be finite and positive");
    return covariance;
}
Mat66 PoseGraphLoopCovariance(const PoseGraphOptions& o, const PoseGraphLoop& loop, size_t correlated) {
    ValidatePoseGraphOptions(o);
    if(!loop.inliers || !std::isfinite(loop.bearing_rms_rad) || loop.bearing_rms_rad<0 ||
        !std::isfinite(loop.pose_condition) || loop.pose_condition<1 || !correlated)
        throw std::invalid_argument("invalid loop geometric quality");
    const double quality=std::sqrt(o.loop_reference_inliers/std::min(double(loop.inliers),o.loop_max_effective_inliers))*
        std::max(1.0,loop.bearing_rms_rad/o.loop_reference_bearing_rms_rad)*
        std::max(1.0,std::sqrt(loop.pose_condition/o.loop_reference_condition));
    const double st=Clamp(o.loop_translation_sigma_m*quality,o.loop_min_translation_sigma_m,o.loop_max_translation_sigma_m);
    const double sr=Clamp(o.loop_rotation_sigma_rad*quality,o.loop_min_rotation_sigma_rad,o.loop_max_rotation_sigma_rad);
    Mat66 covariance=Mat66::Zero();covariance.diagonal().head<3>().setConstant(st*st*correlated);
    covariance.diagonal().tail<3>().setConstant(sr*sr*correlated);
    if(!covariance.allFinite() || (covariance.diagonal().array()<=0).any())
        throw std::invalid_argument("loop covariance must be finite and positive");
    return covariance;
}
PoseGraphInput SnapshotPoseGraph(const KeyframeDatabase& db,
    const HiltiAlignedVector<LoopDiagnostic>& diagnostics,const HiltiAlignedVector<LoopConstraint>& constraints) {
    PoseGraphInput input;std::set<uint64_t> ids;
    for(const auto& d:diagnostics)ids.insert(d.query_keyframe);
    if(ids.size()!=db.Size())throw std::invalid_argument("query diagnostics do not enumerate the complete stopped archive");
    for(uint64_t id:ids) {
        const auto archived=db.Get(id);if(!archived)throw std::invalid_argument("missing archived graph vertex");
        PoseGraphPose p;p.keyframe_id=id;p.timestamp_ns=archived->timestamp_ns;p.body_pose=archived->optimized_body_pose;
        input.poses.push_back(p);
    }
    for(const auto& c:constraints) {
        const LoopDiagnostic* quality=nullptr;
        for(const auto& d:diagnostics)if(d.accepted && d.query_keyframe==c.query_keyframe && d.candidate_keyframe==c.candidate_keyframe) {
            if(quality)throw std::invalid_argument("ambiguous accepted loop diagnostic");
            quality=&d;
        }
        if(!quality || !quality->verification_attempted || !std::isfinite(quality->pose_condition))
            throw std::invalid_argument("missing accepted loop verification/conditioning");
        const auto query=db.Get(c.query_keyframe),candidate=db.Get(c.candidate_keyframe);
        if(!query || !candidate || query->timestamp_ns!=c.query_timestamp_ns || candidate->timestamp_ns!=c.candidate_timestamp_ns ||
            c.inliers!=quality->geometric_inliers || std::abs(c.bearing_rms_rad-quality->relative_pose_rms_rad)>1e-10 ||
            (c.relative_body_transform.inverse()*quality->relative_body_transform).log().norm()>1e-10)
            throw std::invalid_argument("constraint does not match archival geometric verification");
        PoseGraphLoop loop;loop.query_keyframe=c.query_keyframe;loop.candidate_keyframe=c.candidate_keyframe;
        loop.relative_body_transform=c.relative_body_transform;loop.inliers=c.inliers;
        loop.bearing_rms_rad=c.bearing_rms_rad;loop.pose_condition=quality->pose_condition;input.loops.push_back(loop);
    }
    return input;
}
void WritePoseGraphInput(std::ostream& stream,const PoseGraphInput& input) {
    const auto precision=stream.precision();stream<<std::setprecision(17)<<"HILTI_POSE_GRAPH_V1\n"<<input.poses.size()<<'\n';
    for(const auto& p:input.poses){stream<<p.keyframe_id<<' '<<p.timestamp_ns<<' ';WritePose(stream,p.body_pose);stream<<'\n';}
    stream<<input.loops.size()<<'\n';
    for(const auto& l:input.loops) {
        stream<<l.query_keyframe<<' '<<l.candidate_keyframe<<' '<<l.inliers<<' '<<l.bearing_rms_rad<<' '<<l.pose_condition<<' ';
        WritePose(stream,l.relative_body_transform);stream<<'\n';
    }
    stream.precision(precision);if(!stream)throw std::runtime_error("cannot write pose graph snapshot");
}
PoseGraphInput ReadPoseGraphInput(std::istream& stream) {
    std::string magic;size_t count;
    if(!(stream>>magic) || magic!="HILTI_POSE_GRAPH_V1" || !(stream>>count) || count>1000000)
        throw std::invalid_argument("invalid pose graph snapshot header");
    PoseGraphInput input;
    for(size_t k=0;k<count;++k) {
        PoseGraphPose p;if(!(stream>>p.keyframe_id>>p.timestamp_ns))throw std::invalid_argument("truncated graph vertex");
        ReadPose(stream,p.body_pose);input.poses.push_back(p);
    }
    if(!(stream>>count) || count>1000000)throw std::invalid_argument("invalid loop count");
    for(size_t k=0;k<count;++k) {
        PoseGraphLoop l;if(!(stream>>l.query_keyframe>>l.candidate_keyframe>>l.inliers>>l.bearing_rms_rad>>l.pose_condition))
            throw std::invalid_argument("truncated loop quality");
        ReadPose(stream,l.relative_body_transform);input.loops.push_back(l);
    }
    stream>>std::ws;if(!stream.eof())throw std::invalid_argument("extra data after pose graph snapshot");
    return input;
}
PoseGraph::PoseGraph(const PoseGraphOptions& options):options_(options) { ValidatePoseGraphOptions(options_); }
PoseGraphResult PoseGraph::Optimize(const PoseGraphInput& input) const {
    const auto started=std::chrono::steady_clock::now();
    if(input.poses.empty() || input.poses.size()>size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("pose graph requires a nonempty representable archive");
    Poses original=input.poses;
    std::sort(original.begin(),original.end(),[](const PoseGraphPose& a,const PoseGraphPose& b){return a.timestamp_ns<b.timestamp_ns;});
    std::map<uint64_t,size_t> index;Edges odometry,loops;
    for(size_t k=0;k<original.size();++k) {
        if(!ValidPose(original[k].body_pose) || !index.emplace(original[k].keyframe_id,k).second ||
            (k && original[k].timestamp_ns==original[k-1].timestamp_ns))
            throw std::invalid_argument("invalid/duplicate archived pose ID or timestamp");
        if(k) {
            Edge e;e.from=k-1;e.to=k;e.measurement=original[k-1].body_pose.inverse()*original[k].body_pose;
            e.covariance=PoseGraphOdometryCovariance(options_,Seconds(original[k].timestamp_ns,original[k-1].timestamp_ns),e.measurement);
            e.information=e.covariance.inverse();
            if(!e.information.allFinite())throw std::invalid_argument("nonfinite odometry information");
            odometry.push_back(e);
        }
    }
    PoseGraphResult r;r.poses=original;r.vertices=original.size();r.sequential_edges=odometry.size();r.loop_edges=input.loops.size();
    for(const auto& l:input.loops) {
        if(!index.count(l.query_keyframe) || !index.count(l.candidate_keyframe) || l.query_keyframe==l.candidate_keyframe || !ValidPose(l.relative_body_transform))
            throw std::invalid_argument("invalid loop endpoint/transform");
        Edge e;e.from=index.at(l.query_keyframe);e.to=index.at(l.candidate_keyframe);e.measurement=l.relative_body_transform;
        size_t correlated=0;
        for(const auto& other:input.loops) {
            if(!index.count(other.query_keyframe) || !index.count(other.candidate_keyframe))throw std::invalid_argument("missing loop vertex");
            if(std::abs(Seconds(original[e.from].timestamp_ns,original[index.at(other.query_keyframe)].timestamp_ns))<=options_.loop_correlation_seconds &&
                std::abs(Seconds(original[e.to].timestamp_ns,original[index.at(other.candidate_keyframe)].timestamp_ns))<=options_.loop_correlation_seconds)++correlated;
        }
        e.covariance=PoseGraphLoopCovariance(options_,l,correlated);e.information=e.covariance.inverse();
        if(!e.information.allFinite())throw std::invalid_argument("nonfinite loop information");
        loops.push_back(e);
        PoseGraphLoopDiagnostic d;d.query_keyframe=l.query_keyframe;d.candidate_keyframe=l.candidate_keyframe;d.correlated_loops=correlated;
        d.translation_sigma_m=std::sqrt(e.covariance(0,0));d.rotation_sigma_rad=std::sqrt(e.covariance(3,3));
        const Vec6 error=Error(e,original);d.initial_translation_residual_m=error.head<3>().norm();d.initial_rotation_residual_rad=error.tail<3>().norm();
        d.initial_chi2=Chi2(e,original);d.innovation_chi2=Innovation(e,odometry,original);
        d.retained=d.innovation_chi2<=options_.innovation_chi2_limit;
        d.reason=d.retained?"pending optimization":"rejected: odometry-path innovation";r.loops.push_back(d);
    }
    const std::vector<bool> all(loops.size(),true);std::vector<bool> retained;
    for(const auto& d:r.loops)retained.push_back(d.retained);
    r.initial_total_error=Objective(odometry,loops,original,all,options_.loop_huber_delta);
    bool solved=false;
    for(size_t pass=0;pass<=loops.size();++pass) {
        ++r.optimizer_passes;
        if(!Solve(original,odometry,loops,retained,options_,r.poses,r.optimizer_iterations))break;
        std::vector<double> prefix(odometry.size()+1,0),odometry_chi2;
        for(size_t k=0;k<odometry.size();++k) {const double c=Chi2(odometry[k],r.poses);odometry_chi2.push_back(c);prefix[k+1]=prefix[k]+c;}
        size_t worst=loops.size();double worst_score=options_.postfit_chi2_limit;
        for(size_t k=0;k<loops.size();++k)if(retained[k]) {
            const size_t begin=std::min(loops[k].from,loops[k].to),end=std::max(loops[k].from,loops[k].to);
            const double cycle=Chi2(loops[k],r.poses)+prefix[end]-prefix[begin];
            double score=cycle;for(size_t j=begin;j<end;++j)score=std::max(score,odometry_chi2[j]);
            r.loops[k].postfit_cycle_chi2=cycle;
            if(score>worst_score) {worst_score=score;worst=k;}
        }
        if(worst==loops.size()) {solved=true;break;}
        retained[worst]=false;r.loops[worst].retained=false;r.loops[worst].reason="rejected: post-fit cycle inconsistency";
    }
    if(!solved) {
        // Failure is explicit and returns the original VIO trajectory.
        r.poses=original;std::fill(retained.begin(),retained.end(),false);
        for(auto& d:r.loops) {d.retained=false;d.reason="rejected: optimizer failed; original poses restored";}
        r.reason="optimizer failed; original poses restored";
    }else {r.success=true;r.reason="standalone pose graph optimized";}
    r.final_total_error=Objective(odometry,loops,r.poses,all,options_.loop_huber_delta);
    r.initial_retained_error=Objective(odometry,loops,original,retained,options_.loop_huber_delta);
    r.final_retained_error=Objective(odometry,loops,r.poses,retained,options_.loop_huber_delta);
    for(const auto& edge:odometry) {const double c=Chi2(edge,r.poses);r.odometry_error+=c;r.maximum_odometry_edge_chi2=std::max(r.maximum_odometry_edge_chi2,c);}
    for(size_t k=0;k<loops.size();++k) {
        auto& d=r.loops[k];d.retained=retained[k];d.final_chi2=Chi2(loops[k],r.poses);const Vec6 e=Error(loops[k],r.poses);
        d.final_translation_residual_m=e.head<3>().norm();d.final_rotation_residual_rad=e.tail<3>().norm();
        if(d.retained) {++r.retained_loop_edges;d.reason="retained: robust fit and cycle consistency";}else ++r.rejected_loop_edges;
    }
    for(size_t k=0;k<original.size();++k) {
        const double translation=(r.poses[k].body_pose.translation()-original[k].body_pose.translation()).norm();
        const double rotation=(original[k].body_pose.so3().inverse()*r.poses[k].body_pose.so3()).log().norm();
        r.maximum_translation_correction_m=std::max(r.maximum_translation_correction_m,translation);
        r.maximum_rotation_correction_rad=std::max(r.maximum_rotation_correction_rad,rotation);
        if(k+1==original.size()) {r.endpoint_translation_correction_m=translation;r.endpoint_rotation_correction_rad=rotation;}
    }
    r.runtime_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return r;
}
} // namespace myslam
