#include "myslam/hilti_submission.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace myslam {
namespace {
const int64_t kSecond = 1000000000LL;
void CheckPose(const TumPose& p) {
    if(p.timestamp_ns<0 || !p.translation.allFinite() || !p.rotation.coeffs().allFinite() ||
        std::abs(p.rotation.norm()-1)>1e-9)throw std::invalid_argument("invalid/nonfinite/nonunit TUM pose");
}
void CheckTimes(const std::vector<int64_t>& times) {
    if(times.empty())throw std::invalid_argument("empty camera timestamp grid");
    for(size_t i=0;i<times.size();++i)if(times[i]<0 || (i && times[i]<=times[i-1]))
        throw std::invalid_argument("camera timestamps must be unique and strictly increasing");
}
std::vector<std::string> Fields(const std::string& line) {
    std::vector<std::string> fields;std::istringstream stream(line);std::string field;
    while(std::getline(stream,field,','))fields.push_back(field);
    return fields;
}
class Csv {
   public:
    explicit Csv(std::istream& s):stream(s) {
        std::string line;if(!std::getline(stream,line))throw std::invalid_argument("empty trajectory CSV");
        const auto fields=Fields(line);columns=fields.size();
        for(size_t i=0;i<fields.size();++i)if(!header.emplace(fields[i],i).second)throw std::invalid_argument("duplicate CSV column");
        for(const auto& name:{"timestamp_ns","px","py","pz","qx","qy","qz","qw"})Index(name);
    }
    size_t Index(const std::string& name) const {
        const auto i=header.find(name);if(i==header.end())throw std::invalid_argument("missing CSV column: "+name);return i->second;
    }
    bool Next() {
        std::string line;if(!std::getline(stream,line)){if(!stream.eof())throw std::runtime_error("trajectory CSV read failed");return false;}
        fields=Fields(line);if(fields.size()!=columns)throw std::invalid_argument("invalid CSV column count");return true;
    }
    const std::string& Get(const std::string& name) const {return fields.at(Index(name));}
    int64_t Integer(const std::string& name) const {
        const auto& text=Get(name);if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("invalid CSV integer: "+name);
        return std::stoll(text);
    }
    double Number(const std::string& name) const {
        size_t end=0;const auto& text=Get(name);const double value=std::stod(text,&end);
        if(end!=text.size() || !std::isfinite(value))throw std::invalid_argument("invalid CSV pose: "+name);
        return value;
    }
    SE3 Pose() const {
        TumPose p;p.translation=Vec3(Number("px"),Number("py"),Number("pz"));
        p.rotation=Eigen::Quaterniond(Number("qw"),Number("qx"),Number("qy"),Number("qz"));CheckPose(p);
        p.rotation.normalize();return SE3(p.rotation,p.translation);
    }
    bool MissingPose() const {
        for(const auto& name:{"px","py","pz","qx","qy","qz","qw"})if(Get(name)!="nan")return false;
        return true;
    }
   private:
    std::istream& stream;
    size_t columns;
    std::map<std::string,size_t> header;
    std::vector<std::string> fields;
};
SE3 InterpolateCorrection(const SE3& a,const SE3& b,double fraction) {
    return SE3(a.so3()*SO3::exp(fraction*(a.so3().inverse()*b.so3()).log()),
        (1-fraction)*a.translation()+fraction*b.translation());
}
}
TumPose BodyToCam0Pose(const SE3& body_to_world,const SE3& camera_from_body,int64_t raw_time) {
    if(!body_to_world.matrix().allFinite() || !camera_from_body.matrix().allFinite())
        throw std::invalid_argument("nonfinite body pose or camera extrinsic");
    const SE3 camera_to_world=body_to_world*camera_from_body.inverse();
    TumPose out;out.timestamp_ns=raw_time;out.translation=camera_to_world.translation();
    out.rotation=camera_to_world.unit_quaternion();out.rotation.normalize();CheckPose(out);return out;
}
int64_t ParseTumTimestamp(const std::string& text) {
    const auto point=text.find('.');const std::string whole=text.substr(0,point);
    std::string fraction=point==std::string::npos?"":text.substr(point+1);
    if(whole.empty() || whole.find_first_not_of("0123456789")!=std::string::npos || fraction.size()>9 ||
        (point!=std::string::npos && fraction.empty()) || fraction.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("timestamp requires nonnegative seconds and up to nine decimal digits");
    fraction.resize(9,'0');const uint64_t seconds=std::stoull(whole),nano=std::stoull(fraction);
    if(seconds>(uint64_t(std::numeric_limits<int64_t>::max())-nano)/kSecond)throw std::overflow_error("timestamp overflow");
    return int64_t(seconds*kSecond+nano);
}
std::string FormatTumTimestamp(int64_t timestamp) {
    if(timestamp<0)throw std::invalid_argument("negative submission timestamp");
    std::ostringstream line;line.imbue(std::locale::classic());
    line<<timestamp/kSecond<<'.'<<std::setw(9)<<std::setfill('0')<<timestamp%kSecond;return line.str();
}
std::string FormatTumLine(const TumPose& p) {
    CheckPose(p);std::ostringstream line;line.imbue(std::locale::classic());line<<std::setprecision(17)
        <<FormatTumTimestamp(p.timestamp_ns)<<' '<<p.translation.x()<<' '<<p.translation.y()<<' '<<p.translation.z()<<' '
        <<p.rotation.x()<<' '<<p.rotation.y()<<' '<<p.rotation.z()<<' '<<p.rotation.w();return line.str();
}
void ValidateTumTrajectory(const TumTrajectory& trajectory) {
    if(trajectory.empty())throw std::invalid_argument("empty TUM trajectory");
    for(size_t i=0;i<trajectory.size();++i){CheckPose(trajectory[i]);if(i && trajectory[i].timestamp_ns<=trajectory[i-1].timestamp_ns)
        throw std::invalid_argument("TUM timestamps duplicate or not strictly increasing");}
}
void WriteTumTrajectory(std::ostream& stream,const TumTrajectory& trajectory) {
    ValidateTumTrajectory(trajectory);for(const auto& p:trajectory)stream<<FormatTumLine(p)<<'\n';
    stream.flush();if(!stream)throw std::runtime_error("TUM trajectory write failed");
}
TumTrajectory ReadTumTrajectory(std::istream& stream) {
    TumTrajectory result;std::string line;
    while(std::getline(stream,line)) {
        const auto begin=line.find_first_not_of(" \t\r");if(begin==std::string::npos || line[begin]=='#')continue;
        std::istringstream fields(line);fields.imbue(std::locale::classic());std::vector<std::string> tokens;std::string text;
        while(fields>>text)tokens.push_back(text);
        if(tokens.size()!=8)throw std::invalid_argument("TUM line must contain exactly eight fields");
        TumPose p;p.timestamp_ns=ParseTumTimestamp(tokens[0]);double values[7];
        for(int i=0;i<7;++i){size_t end=0;values[i]=std::stod(tokens[i+1],&end);
            if(end!=tokens[i+1].size() || !std::isfinite(values[i]))throw std::invalid_argument("nonfinite/invalid TUM component");}
        p.translation=Vec3(values[0],values[1],values[2]);p.rotation=Eigen::Quaterniond(values[6],values[3],values[4],values[5]);
        result.push_back(p);
    }
    if(!stream.eof())throw std::runtime_error("TUM read failed");
    ValidateTumTrajectory(result);return result;
}
std::string HiltiSubmissionFilename(const std::string& floor,const std::string& date,unsigned run) {
    const std::set<std::string> floors{"1","2","3","4","5","6","7","EG","UG1","UG2"};
    if(!floors.count(floor) || run<1 || run>2 || !std::regex_match(date,std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}")))
        throw std::invalid_argument("invalid official sequence identifier");
    const int y=std::stoi(date.substr(0,4)),m=std::stoi(date.substr(5,2)),d=std::stoi(date.substr(8,2));
    const int days[]={0,31,28,31,30,31,30,31,31,30,31,30,31};const bool leap=y%4==0 && (y%100!=0 || y%400==0);
    if(y<1970 || m<1 || m>12 || d<1 || d>days[m]+(m==2 && leap))throw std::invalid_argument("invalid run date");
    return "floor_"+floor+"_"+date+"_run_"+std::to_string(run)+".txt";
}
std::string HiltiSubmissionFilename(const std::string& sequence) {
    std::smatch parts;
    if(!std::regex_match(sequence,parts,std::regex("floor_([^_]+)_([0-9]{4}-[0-9]{2}-[0-9]{2})_run_([0-9]+)")))
        throw std::invalid_argument("invalid Hilti sequence name");
    const auto n=std::stoull(parts[3]);if(n<1 || n>2 || parts[3].str()!=std::to_string(n))throw std::invalid_argument("invalid run number");
    return HiltiSubmissionFilename(parts[1],parts[2],unsigned(n));
}
HiltiBodyTrajectory ReadHiltiBodyCsv(std::istream& stream) {
    Csv csv(stream);csv.Index("epoch");csv.Index("status");HiltiBodyTrajectory out;
    while(csv.Next()) {
        if(csv.Integer("epoch")!=int64_t(out.size()+1))throw std::invalid_argument("body CSV epochs must be complete/consecutive");
        HiltiBodySample sample;sample.timestamp_imu_ns=csv.Integer("timestamp_ns");const auto& status=csv.Get("status");
        if(status=="buffering") {if(!csv.MissingPose())throw std::invalid_argument("buffering row contains an unexplained pose");}
        else if(status=="initialized" || status=="tracking" || status=="prediction_only"){sample.initialized=true;sample.body_pose_local=csv.Pose();}
        else throw std::invalid_argument("unknown or failed frontend state in export input");
        if(!out.empty() && sample.timestamp_imu_ns<=out.back().timestamp_imu_ns)throw std::invalid_argument("unordered body CSV");
        out.push_back(sample);
    }
    if(out.empty())throw std::invalid_argument("empty body CSV");
    return out;
}
HiltiAlignedVector<PoseGraphPose> ReadHiltiKeyframeCsv(std::istream& stream) {
    Csv csv(stream);csv.Index("id");HiltiAlignedVector<PoseGraphPose> out;std::set<uint64_t> ids;
    while(csv.Next()) {
        PoseGraphPose p;p.keyframe_id=csv.Integer("id");p.timestamp_ns=csv.Integer("timestamp_ns");p.body_pose=csv.Pose();
        if(!ids.insert(p.keyframe_id).second || (!out.empty() && p.timestamp_ns<=out.back().timestamp_ns))
            throw std::invalid_argument("unordered/duplicate keyframe CSV");
        out.push_back(p);
    }
    if(out.empty())throw std::invalid_argument("empty keyframe CSV");
    return out;
}
HiltiSubmission BuildHiltiSubmission(const std::vector<int64_t>& raw,const HiltiBodyTrajectory& body,
    const HiltiAlignedVector<PoseGraphPose>& local,const HiltiAlignedVector<PoseGraphPose>& global,
    const SE3& extrinsic,const ImuCalibration& calibration) {
    CheckTimes(raw);if(raw.size()!=body.size() || local.empty() || local.size()!=global.size())
        throw std::invalid_argument("incomplete dense or archival trajectory input");
    std::map<int64_t,size_t> samples;std::set<uint64_t> ids;HiltiAlignedVector<SE3> corrections;
    for(size_t i=0;i<body.size();++i) {
        if(CameraTimeOnImuClock(calibration,0,raw[i])!=body[i].timestamp_imu_ns)
            throw std::invalid_argument("body epoch does not correspond to recorded cam0 time");
        samples.emplace(body[i].timestamp_imu_ns,i);
        if(body[i].initialized)BodyToCam0Pose(body[i].body_pose_local,extrinsic,raw[i]);
    }
    for(size_t k=0;k<local.size();++k) {
        const auto& a=local[k];const auto& b=global[k];const auto found=samples.find(a.timestamp_ns);
        if(a.keyframe_id!=b.keyframe_id || a.timestamp_ns!=b.timestamp_ns || !ids.insert(a.keyframe_id).second ||
            (k && a.timestamp_ns<=local[k-1].timestamp_ns) || found==samples.end() || !body[found->second].initialized)
            throw std::invalid_argument("local/global archives must match initialized camera epochs");
        BodyToCam0Pose(a.body_pose,extrinsic,raw[found->second]);BodyToCam0Pose(b.body_pose,extrinsic,raw[found->second]);
        corrections.push_back(b.body_pose*a.body_pose.inverse());
    }
    HiltiSubmission result;size_t right=0;
    for(size_t i=0;i<body.size();++i) {
        if(!body[i].initialized){++result.omitted_uninitialized;continue;}
        const auto time=body[i].timestamp_imu_ns;
        while(right<local.size() && local[right].timestamp_ns<time)++right;
        SE3 local_pose=body[i].body_pose_local,global_pose;
        if(right<local.size() && local[right].timestamp_ns==time) {
            local_pose=local[right].body_pose;global_pose=global[right].body_pose;++result.exact_archived_keyframes;
        }else {
            SE3 correction;
            if(right==0)correction=corrections.front();
            else if(right==local.size())correction=corrections.back();
            else {
                const double fraction=double(time-local[right-1].timestamp_ns)/double(local[right].timestamp_ns-local[right-1].timestamp_ns);
                correction=InterpolateCorrection(corrections[right-1],corrections[right],fraction);
            }
            global_pose=correction*local_pose;
        }
        result.local_cam0.push_back(BodyToCam0Pose(local_pose,extrinsic,raw[i]));
        result.global_cam0.push_back(BodyToCam0Pose(global_pose,extrinsic,raw[i]));
    }
    ValidateTumTrajectory(result.local_cam0);ValidateTumTrajectory(result.global_cam0);return result;
}
TrajectoryCoverage CheckTrajectoryCoverage(const std::vector<int64_t>& expected,const TumTrajectory& predicted,int64_t ignore) {
    CheckTimes(expected);ValidateTumTrajectory(predicted);
    if(ignore<0 || expected.front()>std::numeric_limits<int64_t>::max()-ignore)throw std::invalid_argument("invalid startup interval");
    const int64_t cutoff=expected.front()+ignore;std::set<int64_t> available;
    for(const auto& p:predicted) {
        if(!std::binary_search(expected.begin(),expected.end(),p.timestamp_ns))throw std::invalid_argument("submission contains a non-camera timestamp");
        available.insert(p.timestamp_ns);
    }
    TrajectoryCoverage result;result.expected_poses=expected.size();result.predicted_poses=predicted.size();
    result.missing_poses=expected.size()-predicted.size();bool in_gap=false;
    for(const auto t:expected) {
        if(t>=cutoff){++result.eligible_expected;if(available.count(t))++result.eligible_predicted;}
        if(available.count(t)){in_gap=false;continue;}
        if(!in_gap){MissingPoseInterval gap;gap.begin_ns=t;result.missing_intervals.push_back(gap);in_gap=true;}
        auto& gap=result.missing_intervals.back();gap.end_ns=t;++gap.poses;
    }
    result.coverage_percent=100.0*predicted.size()/expected.size();
    result.eligible_coverage_percent=result.eligible_expected?100.0*result.eligible_predicted/result.eligible_expected:0;
    result.first_ns=predicted.front().timestamp_ns;result.last_ns=predicted.back().timestamp_ns;
    result.duration_s=double(result.last_ns-result.first_ns)/kSecond;
    for(size_t i=0;i<predicted.size();++i) {
        result.maximum_quaternion_norm_error=std::max(result.maximum_quaternion_norm_error,std::abs(predicted[i].rotation.norm()-1));
        if(i)result.largest_gap_ns=std::max(result.largest_gap_ns,predicted[i].timestamp_ns-predicted[i-1].timestamp_ns);
    }
    return result;
}
} // namespace myslam
