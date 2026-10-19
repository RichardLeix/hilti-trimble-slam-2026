#include "myslam/hilti_rolling_shutter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace myslam {
double HiltiRowTimeSeconds(double image_y) {
    if(!std::isfinite(image_y))throw std::invalid_argument("nonfinite Hilti image row");
    return (image_y / 1439.0 - 0.5) * 0.03105122184753418;
}
Vec3 HiltiInterpolateGyro(const std::vector<ImuSample>& samples,int64_t t) {
    auto i=std::lower_bound(samples.begin(),samples.end(),t,
        [](const ImuSample& s,int64_t time){return s.timestamp_ns<time;});
    if(i==samples.end())throw std::invalid_argument("Hilti row-time gyro lacks IMU coverage");
    if(i->timestamp_ns==t)return Vec3(i->angular_velocity[0],i->angular_velocity[1],i->angular_velocity[2]);
    if(i==samples.begin())throw std::invalid_argument("Hilti row-time gyro lacks IMU coverage");
    const auto& a=*(i-1);const double w=double(t-a.timestamp_ns)/double(i->timestamp_ns-a.timestamp_ns);
    Vec3 result;for(int k=0;k<3;++k)result[k]=(1-w)*a.angular_velocity[k]+w*i->angular_velocity[k];
    return result;
}
namespace {
Vec3 ExposureVelocity(const HiltiKeyframeState& frame,int camera,const HiltiCameras& cameras,const Vec3& gravity) {
    const double h=double(frame.exposure_timestamps_ns[camera]-frame.timestamp_ns)*1e-9;
    if(!h)return frame.state.velocity_w;
    const auto& d=frame.exposure_deltas[camera];Vec6 db;
    db<<frame.state.bias.gyroscope-d.bias.gyroscope,frame.state.bias.accelerometer-d.bias.accelerometer;
    const Vec3 dv=d.delta_velocity+d.bias_jacobian.middleRows<3>(3)*db;
    if(h>0)return frame.state.velocity_w+gravity*h+frame.state.rotation_wb*dv;
    const SE3 body=HiltiCameraWorldPose(frame,camera,cameras,gravity)*cameras[camera]->pose();
    return frame.state.velocity_w+gravity*h-body.so3()*dv;
}
}
SE3 HiltiRowCameraWorldPose(const HiltiKeyframeState& frame,int camera,
        const HiltiCameras& cameras,const Vec3& gravity,double image_y) {
    if(camera<0 || camera>1 || !cameras[camera] || cameras[camera]->Height()!=HiltiReadoutImageHeight)
        throw std::invalid_argument("Hilti center-row model requires calibrated 1440-row cameras");
    if(!frame.exposure_angular_velocity[camera].allFinite())
        throw std::invalid_argument("missing Hilti row-time gyro measurement");
    const SE3 base=HiltiCameraWorldPose(frame,camera,cameras,gravity);
    const double dt=HiltiRowTimeSeconds(image_y);if(dt==0)return base;
    const SE3 body=base*cameras[camera]->pose();
    const Vec3 omega=frame.exposure_angular_velocity[camera]-frame.state.bias.gyroscope;
    const SE3 moved(body.so3()*SO3::exp(omega*dt),
        body.translation()+ExposureVelocity(frame,camera,cameras,gravity)*dt);
    return moved*cameras[camera]->pose().inverse();
}
SE3 HiltiVisualCameraWorldPose(const HiltiKeyframeState& frame,int camera,const Vec3& bearing,
        const HiltiCameras& cameras,const Vec3& gravity,const HiltiBackendOptions& options) {
    if(!options.center_row_rolling_shutter)return HiltiCameraWorldPose(frame,camera,cameras,gravity);
    Vec2 pixel;if(!cameras[camera]->Project(bearing,pixel))
        throw std::invalid_argument("Hilti row-time bearing outside EUCM domain/mask");
    return HiltiRowCameraWorldPose(frame,camera,cameras,gravity,pixel.y());
}
HiltiBackendOptions LoadHiltiBackendOptions(const std::string& path) {
    cv::FileStorage file(path,cv::FileStorage::READ);
    if(!file.isOpened())throw std::invalid_argument("cannot open Hilti backend configuration");
    std::string model,reference,direction;int height=-1,verified=-1;double readout=-1;
    file["visual_model"]>>model;file["timestamp_reference"]>>reference;
    file["readout_direction"]>>direction;file["image_height"]>>height;
    file["readout_seconds"]>>readout;file["physical_reference_verified"]>>verified;
    if(model!="rolling_shutter" || reference!="center_row" || direction!="top_to_bottom" ||
        height!=HiltiReadoutImageHeight || readout!=HiltiReadoutSeconds || verified!=0)
        throw std::invalid_argument("Hilti backend requires the fixed validated, physically unverified center-row convention");
    HiltiBackendOptions options;
    const auto quarantine=file["experimental_landmark_quarantine"];
    if(!quarantine.empty()) {
        std::string mode;quarantine>>mode;
        if(mode!="disabled" && mode!="B")throw std::invalid_argument("unknown experimental landmark quarantine baseline");
        options.experimental_landmark_quarantine_B=mode=="B";
    }
    options.window_size=11; // 12 source states -> 11 retained; unchanged FIFO.
    return options;
}
} // namespace myslam
