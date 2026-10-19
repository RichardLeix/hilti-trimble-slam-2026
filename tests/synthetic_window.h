// Existing backend synthetic fixture, copied without changing estimator sources.
#include "myslam/frontend.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
namespace {
using namespace myslam;
const int64_t kEpoch=10000000000000LL;
std::string Number(double value) {std::ostringstream s;s<<std::setprecision(17)<<value;return s.str();}
struct SyntheticWindow {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    HiltiBackendJob truth,perturbed;HiltiBackendOptions options;std::vector<ImuSample> samples;
    SyntheticWindow() {
        options.center_row_rolling_shutter=false; // This fixture generates nominal GS bearings.
        options.max_iterations=25;options.gyro_prior_sigma=0.1;options.accel_prior_sigma=0.5;
        truth.version=1;truth.gravity_w=Vec3(0,0,-9.81);
        auto& c=truth.calibration;c.gyroscope_noise_density=0.0005;c.accelerometer_noise_density=0.003;
        c.gyroscope_random_walk=0.0004;c.accelerometer_random_walk=0.0013;
        c.camera_time_offsets_s={{0,0.00002}};
        for(int k=0;k<2;++k)truth.cameras[k].reset(new EucmCamera(0.6,1.1,200,210,320,240,640,480,
            SE3(SO3::exp(Vec3(0.1,k?2.8:-0.15,0.05)),Vec3(0.025,-0.015,0.04))));
        ImuBias b;b.gyroscope=Vec3(0.012,-0.018,0.008);b.accelerometer=Vec3(0.08,-0.05,0.11);
        for(int i=-1;i<=1001;++i) {
            const double t=i*0.001;const Vec3 w=Vec3(0.3+0.4*std::sin(4*t),-0.25+0.3*std::cos(3*t),0.2*std::sin(5*t))+b.gyroscope;
            const Vec3 a=Vec3(1+1.5*std::sin(5*t),0.5*std::cos(4*t),9.81+std::sin(6*t))+b.accelerometer;
            ImuSample s;s.timestamp_ns=kEpoch+int64_t(i)*1000000;
            for(int axis=0;axis<3;++axis){s.angular_velocity[axis]=w[axis];s.linear_acceleration[axis]=a[axis];}samples.push_back(s);
        }
        NavState initial;initial.bias=b;initial.velocity_w=Vec3(0.8,-0.3,0.2);initial.rotation_wb=SO3::exp(Vec3(0.15,-0.1,0.2));
        for(size_t i=0;i<7;++i) {
            HiltiKeyframeState f;f.timestamp_ns=kEpoch+int64_t(i)*150000000;f.exposure_timestamps_ns={{f.timestamp_ns,f.timestamp_ns+20000}};
            if(i) {
                const auto d=PreintegrateImu(samples,truth.keyframes.back().timestamp_ns,f.timestamp_ns,c,b);
                truth.intervals.push_back(d);f.state=PropagateNavState(truth.keyframes.back().state,d,truth.gravity_w);
            } else f.state=initial;
            f.exposure_deltas[1]=PreintegrateImu(samples,f.timestamp_ns,f.timestamp_ns+20000,c,b);truth.keyframes.push_back(f);
        }
        for(int k=0;k<2;++k)for(uint64_t id=0;id<50;++id) {
            HiltiLandmark l;l.id={k,id};l.initialized=true;
            const Vec3 pc(1.5*std::cos(id*0.7),1.2*std::sin(id*0.9),6+0.06*id);
            l.position_w=HiltiCameraWorldPose(truth.keyframes[0],k,truth.cameras,truth.gravity_w)*pc;
            for(const auto& frame:truth.keyframes) {
                const Vec3 p=HiltiCameraWorldPose(frame,k,truth.cameras,truth.gravity_w).inverse()*l.position_w;Vec2 pixel;
                if(!truth.cameras[k]->Project(p,pixel))throw std::runtime_error("synthetic landmark outside EUCM domain");
                Vec3 ray;if(!truth.cameras[k]->Unproject(pixel,ray))throw std::runtime_error("synthetic ray failed");
                HiltiBearingObservation o;o.timestamp_ns=frame.timestamp_ns;o.bearing=ray;l.observations.push_back(o);
            }
            truth.landmarks.push_back(l);
        }
        perturbed=truth;
        for(size_t i=0;i<perturbed.keyframes.size();++i) {
            auto& f=perturbed.keyframes[i];
            if(i) {f.state.position_wb+=Vec3(0.012*std::sin(i),-0.01*std::cos(i),0.008);
                f.state.rotation_wb=f.state.rotation_wb*SO3::exp(Vec3(0.004,-0.003,0.002));}
            f.state.velocity_w+=Vec3(0.06,-0.04,0.03);f.state.bias.gyroscope+=Vec3(0.002,-0.001,0.0015);
            f.state.bias.accelerometer+=Vec3(0.025,-0.02,0.03);
            f.exposure_deltas[1]=PreintegrateImu(samples,f.timestamp_ns,f.exposure_timestamps_ns[1],c,f.state.bias);
            if(i)perturbed.intervals[i-1]=PreintegrateImu(samples,perturbed.keyframes[i-1].timestamp_ns,f.timestamp_ns,c,perturbed.keyframes[i-1].state.bias);
        }
        for(auto& l:perturbed.landmarks)l.position_w+=Vec3(0.015*std::sin(l.id.second),-0.02*std::cos(l.id.second),0.03);
    }
    std::array<HiltiTrackingResult,2> Tracks(size_t i) const {
        std::array<HiltiTrackingResult,2> tracks;
        for(int k=0;k<2;++k) {
            tracks[k].stream=k?SensorStream::Cam1:SensorStream::Cam0;
            tracks[k].timestamp_ns=truth.keyframes[i].timestamp_ns;
            for(const auto& l:truth.landmarks)if(l.id.first==k) {
                const Vec3 p=HiltiCameraWorldPose(truth.keyframes[i],k,truth.cameras,truth.gravity_w).inverse()*l.position_w;Vec2 pixel;
                truth.cameras[k]->Project(p,pixel);tracks[k].features.push_back({l.id.second,cv::Point2f(pixel.x(),pixel.y()),i+1});
            }
        }
        return tracks;
    }
};

}
