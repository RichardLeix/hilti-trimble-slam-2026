#include "myslam/hilti_submission.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <cerrno>

namespace {
using namespace myslam;
std::ifstream Input(const std::string& path) {
    std::ifstream file(path);if(!file)throw std::runtime_error("cannot read export input: "+path);return file;
}
void Directory(const std::string& path) {
    if(path.empty())throw std::invalid_argument("empty output directory");
    for(size_t i=1;i<=path.size();++i)if(i==path.size() || path[i]=='/') {
        const std::string prefix=path.substr(0,i);if(mkdir(prefix.c_str(),0755) && errno!=EEXIST)
            throw std::runtime_error("cannot create output directory: "+prefix);
        struct stat info;if(stat(prefix.c_str(),&info) || !S_ISDIR(info.st_mode))throw std::runtime_error("output path is not a directory");
    }
}
void Report(std::ostream& out,const char* name,const TrajectoryCoverage& c) {
    const std::string p=std::string(name)+"_";
    out<<p<<"expected_camera_poses="<<c.expected_poses<<'\n'<<p<<"exported_poses="<<c.predicted_poses<<'\n'
        <<p<<"missing_poses="<<c.missing_poses<<'\n'<<p<<"all_camera_coverage_percent="<<c.coverage_percent<<'\n'
        <<p<<"eligible_camera_poses="<<c.eligible_expected<<'\n'<<p<<"eligible_exported_poses="<<c.eligible_predicted<<'\n'
        <<p<<"eligible_camera_coverage_percent="<<c.eligible_coverage_percent<<'\n'
        <<p<<"first_timestamp="<<FormatTumTimestamp(c.first_ns)<<'\n'<<p<<"last_timestamp="<<FormatTumTimestamp(c.last_ns)<<'\n'
        <<p<<"duration_s="<<c.duration_s<<'\n'<<p<<"largest_gap_s="<<double(c.largest_gap_ns)*1e-9<<'\n'
        <<p<<"maximum_quaternion_norm_error="<<c.maximum_quaternion_norm_error<<'\n'
        <<p<<"duplicates=0\n"<<p<<"nonfinite_exported_poses=0\n";
    for(const auto& gap:c.missing_intervals)out<<p<<"missing_interval="<<FormatTumTimestamp(gap.begin_ns)<<','
        <<FormatTumTimestamp(gap.end_ns)<<','<<gap.poses<<",uninitialized: omitted explicitly\n";
}
}
int main(int argc,char** argv) {
    google::InitGoogleLogging(argv[0]);
    if(argc!=9){std::cerr<<"Usage: export_hilti_submission <rosbag> <imu.yaml> <camera.yaml> <local-epochs.csv> <local-keyframes.csv> <global-keyframes.csv> <sequence-name> <output-directory>\n";return 2;}
    try {
        const std::string filename=HiltiSubmissionFilename(argv[7]),directory=argv[8];
        const auto calibration=LoadImuCalibration(argv[2],argv[3]);HiltiDataset dataset(argv[1]);
        if(!dataset.Init() || !dataset.LoadCameraCalibration(argv[3]))throw std::runtime_error(dataset.LastError());
        const auto& info=dataset.GetStreamInfo(SensorStream::Cam0);if(!info.StrictlyIncreasing())throw std::runtime_error("invalid cam0 header timestamp order");
        std::vector<int64_t> times;SensorPacket packet;
        while(dataset.NextPacket(packet))if(packet.stream==SensorStream::Cam0)times.push_back(packet.timestamp_ns);
        auto body_file=Input(argv[4]),local_file=Input(argv[5]),global_file=Input(argv[6]);
        const auto body=ReadHiltiBodyCsv(body_file);const auto local=ReadHiltiKeyframeCsv(local_file),global=ReadHiltiKeyframeCsv(global_file);
        const auto result=BuildHiltiSubmission(times,body,local,global,dataset.GetCamera(0)->pose(),calibration);
        const auto local_coverage=CheckTrajectoryCoverage(times,result.local_cam0),global_coverage=CheckTrajectoryCoverage(times,result.global_cam0);
        Directory(directory+"/local");Directory(directory+"/global");
        std::ofstream local_output(directory+"/local/"+filename),global_output(directory+"/global/"+filename);
        if(!local_output || !global_output)throw std::runtime_error("cannot open submission outputs");
        WriteTumTrajectory(local_output,result.local_cam0);WriteTumTrajectory(global_output,result.global_cam0);
        // Verify written decimal representation independently through the reader.
        auto local_read=Input(directory+"/local/"+filename),global_read=Input(directory+"/global/"+filename);
        CheckTrajectoryCoverage(times,ReadTumTrajectory(local_read));CheckTrajectoryCoverage(times,ReadTumTrajectory(global_read));
        std::ostringstream text;text<<std::setprecision(17)<<"sequence="<<argv[7]<<"\nsubmission_filename="<<filename
            <<"\nrecording_start="<<FormatTumTimestamp(times.front())<<"\nrecording_end="<<FormatTumTimestamp(times.back())
            <<"\nrecording_duration_s="<<double(times.back()-times.front())*1e-9
            <<"\ncam0_shift_ns="<<CameraTimeOnImuClock(calibration,0,0)
            <<"\nomitted_uninitialized_poses="<<result.omitted_uninitialized
            <<"\nexact_archived_keyframes="<<result.exact_archived_keyframes
            <<"\npose_chain=T_gc0=T_gb*inverse(T_cam0_body)\nraw_camera_clock_verified=1\nscale=1\n";
        Report(text,"local",local_coverage);Report(text,"global",global_coverage);
        std::ofstream report(directory+"/export_report.txt");report<<text.str();report.flush();if(!report)throw std::runtime_error("cannot write export report");
        std::cout<<text.str();return local_coverage.eligible_coverage_percent>=99 && global_coverage.eligible_coverage_percent>=99?0:4;
    }catch(const std::exception& error){std::cerr<<"error: "<<error.what()<<'\n';return 1;}
}
