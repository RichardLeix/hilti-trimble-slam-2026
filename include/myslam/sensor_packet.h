#ifndef MYSLAM_SENSOR_PACKET_H
#define MYSLAM_SENSOR_PACKET_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace myslam {

enum class SensorStream { Cam0, Cam1, Imu };

struct ImuSample {
    int64_t timestamp_ns = 0;      // Original ROS message header timestamp.
    int64_t bag_timestamp_ns = 0;  // Original rosbag storage timestamp.
    std::string frame_id;
    std::array<double, 3> angular_velocity{{0, 0, 0}};     // rad/s, raw axes.
    std::array<double, 3> linear_acceleration{{0, 0, 0}};  // m/s^2, raw axes.
};

struct ImagePacket {
    std::string frame_id;
    std::string format;
    std::vector<uint8_t> data;  // Original compressed bytes; no image decoding.
};

struct SensorPacket {
    SensorStream stream = SensorStream::Cam0;
    int64_t message_id = 0;
    int64_t timestamp_ns = 0;
    int64_t bag_timestamp_ns = 0;
    ImagePacket image;  // Populated for Cam0/Cam1.
    ImuSample imu;      // Populated for Imu.
};

}  // namespace myslam

#endif  // MYSLAM_SENSOR_PACKET_H
