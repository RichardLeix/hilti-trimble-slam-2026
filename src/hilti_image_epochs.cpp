#include "myslam/hilti_runtime.h"
#include <algorithm>
#include <stdexcept>

namespace myslam {
HiltiImageEpochReader::HiltiImageEpochReader(HiltiDataset& dataset, const ImuCalibration& calibration)
    : dataset_(dataset), calibration_(calibration) {
    for (auto stream : {SensorStream::Cam0, SensorStream::Cam1, SensorStream::Imu}) {
        if (!dataset.GetStreamInfo(stream).StrictlyIncreasing())
            throw std::invalid_argument("runtime requires strictly increasing sensor timestamps");
    }
    const auto& info = dataset.GetStreamInfo(SensorStream::Imu);
    if (info.topic != calibration.topic || info.count < 2 ||
        info.first_timestamp_ns == std::numeric_limits<int64_t>::min() ||
        info.last_timestamp_ns == std::numeric_limits<int64_t>::max())
        throw std::invalid_argument("invalid runtime IMU stream");
    samples_ = dataset.ImuBetween(info.first_timestamp_ns - 1, info.last_timestamp_ns + 1);
    if (samples_.size() != info.count) throw std::runtime_error("runtime IMU count mismatch");
}

bool HiltiImageEpochReader::Next(HiltiImageEpoch& output) {
    SensorPacket packet;
    while (!(have_[0] && have_[1])) {
        if (!dataset_.NextPacket(packet)) {
            if (have_[0] || have_[1]) throw std::runtime_error("unpaired image at end of Hilti run");
            return false;
        }
        if (packet.stream == SensorStream::Imu) continue; // Indexed raw IMU cache supplies interpolation brackets.
        const int camera = packet.stream == SensorStream::Cam0 ? 0 : 1;
        if (have_[camera]) throw std::runtime_error("missing synchronized image from the other camera");
        pending_[camera] = std::move(packet); have_[camera] = true;
    }
    if (pending_[0].timestamp_ns != pending_[1].timestamp_ns)
        throw std::runtime_error("Hilti raw image timestamps are not synchronized");
    HiltiImageEpoch epoch;
    epoch.cameras = std::move(pending_);
    for (int k = 0; k < 2; ++k)
        epoch.camera_imu_timestamps_ns[k] = CameraTimeOnImuClock(calibration_, k, epoch.cameras[k].timestamp_ns);
    epoch.timestamp_ns = epoch.camera_imu_timestamps_ns[0];
    if (have_previous_ && (epoch.timestamp_ns <= previous_time_ || epoch.cameras[0].timestamp_ns <= previous_raw_))
        throw std::runtime_error("Hilti image epochs are not strictly increasing");
    int64_t begin = std::min(epoch.camera_imu_timestamps_ns[0], epoch.camera_imu_timestamps_ns[1]);
    const int64_t end = std::max(epoch.camera_imu_timestamps_ns[0], epoch.camera_imu_timestamps_ns[1]);
    if (have_previous_) begin = std::min(begin, previous_time_);
    auto first = std::lower_bound(samples_.begin(), samples_.end(), begin,
        [](const ImuSample& s, int64_t t) { return s.timestamp_ns < t; });
    auto last = std::lower_bound(samples_.begin(), samples_.end(), end,
        [](const ImuSample& s, int64_t t) { return s.timestamp_ns < t; });
    if (first == samples_.end() || (first == samples_.begin() && first->timestamp_ns > begin) || last == samples_.end())
        throw std::runtime_error("IMU samples do not bracket the calibrated image epoch");
    if (first != samples_.begin()) --first;
    ++last;
    epoch.imu_samples.assign(first, last);
    previous_time_ = epoch.timestamp_ns; previous_raw_ = epoch.cameras[0].timestamp_ns;
    have_previous_ = true; ++epochs_read_; have_ = {{false, false}};
    output = std::move(epoch);
    return true;
}
}  // namespace myslam
