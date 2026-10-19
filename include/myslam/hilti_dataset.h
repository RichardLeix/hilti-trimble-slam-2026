#ifndef MYSLAM_HILTI_DATASET_H
#define MYSLAM_HILTI_DATASET_H

#include "myslam/dataset.h"
#include "myslam/sensor_packet.h"

namespace myslam {

/// Statistics in recorded message-id order, before chronological sorting.
struct HiltiStreamInfo {
    std::string topic;
    std::string message_type;
    size_t count = 0;
    int64_t first_timestamp_ns = 0;
    int64_t last_timestamp_ns = 0;
    int64_t first_bag_timestamp_ns = 0;
    int64_t last_bag_timestamp_ns = 0;
    size_t duplicate_timestamps = 0;
    size_t backward_steps = 0;
    size_t bag_duplicate_timestamps = 0;
    size_t bag_backward_steps = 0;
    size_t header_bag_mismatches = 0;

    bool StrictlyIncreasing() const {
        return duplicate_timestamps == 0 && backward_steps == 0;
    }
};

/// Read-only ingestion for a single-file SQLite rosbag2 with CDR1 messages.
/// Raw packets are independent of Frame, camera calibration, and tracking.
class HiltiDataset : public Dataset {
   public:
    typedef std::shared_ptr<HiltiDataset> Ptr;

    explicit HiltiDataset(const std::string& rosbag_path);
    ~HiltiDataset() override;

    bool Init() override;
    // Explicit, independent of ingestion. Reads EUCM intrinsics, masks and
    // T_cam_imu; time offsets are not applied. Failure retains prior cameras.
    bool LoadCameraCalibration(const std::string& calibration_file);
    const std::string& LastError() const { return last_error_; }
    const HiltiStreamInfo& GetStreamInfo(SensorStream stream) const;

    /// Header timestamp order; ties retain message-id order. False means EOF.
    /// Throws on read/decode errors or use before successful initialization.
    bool NextPacket(SensorPacket& packet);
    void Rewind();

    /// Strictly open interval: begin_ns < IMU header timestamp < end_ns.
    /// Independent of the packet iterator; results are chronological.
    std::vector<ImuSample> ImuBetween(int64_t begin_ns, int64_t end_ns) const;

    // The SLAM Frame bridge is intentionally not implemented here.
    Frame::Ptr NextFrame() override { return nullptr; }
    Camera::Ptr GetCamera(int camera_id) const override {
        return cameras_.at(static_cast<size_t>(camera_id));
    }

   private:
    struct Impl;
    std::string rosbag_path_;
    std::string last_error_;
    std::unique_ptr<Impl> impl_;
    std::array<Camera::Ptr, 2> cameras_{{nullptr, nullptr}};
};

}  // namespace myslam

#endif  // MYSLAM_HILTI_DATASET_H
