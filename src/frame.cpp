/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/frame.h"
#include <cmath>

namespace myslam {

Frame::Frame(long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right)
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}

Frame::Ptr Frame::CreateFrame() {
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

void Frame::ConfigureHiltiEpoch(int64_t timestamp_ns,
        const std::array<int64_t, 2>& raw, const std::array<int64_t, 2>& camera_times) {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    if (hilti_navigation_.is_hilti) throw std::logic_error("Hilti epoch already configured");
    if (timestamp_ns != camera_times[0] || raw[0] != raw[1])
        throw std::invalid_argument("Hilti epoch requires synchronized raw cameras and cam0 IMU reference time");
    hilti_navigation_.is_hilti = true;
    hilti_navigation_.timestamp_ns = timestamp_ns;
    hilti_navigation_.raw_camera_timestamps_ns = raw;
    hilti_navigation_.camera_imu_timestamps_ns = camera_times;
}

HiltiFrameNavigation Frame::HiltiNavigation() {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return hilti_navigation_;
}

void Frame::SetHiltiNavigation(const NavState& state, bool tracking_valid,
                               std::shared_ptr<const ImuPreintegration> interval, uint64_t backend_version) {
    if (!state.Pose().matrix().allFinite() ||
        std::abs(state.rotation_wb.unit_quaternion().squaredNorm() - 1) > 1e-12 ||
        !state.velocity_w.allFinite() || !state.bias.gyroscope.allFinite() || !state.bias.accelerometer.allFinite())
        throw std::invalid_argument("Hilti navigation requires a finite state");
    std::unique_lock<std::mutex> lock(pose_mutex_);
    if (!hilti_navigation_.is_hilti) throw std::logic_error("configure Hilti epoch before publishing navigation");
    if (interval && (interval->end_timestamp_ns != hilti_navigation_.timestamp_ns ||
        interval->begin_timestamp_ns >= interval->end_timestamp_ns || !std::isfinite(interval->dt) || interval->dt <= 0 ||
        std::abs(interval->dt - double(interval->end_timestamp_ns - interval->begin_timestamp_ns) * 1e-9) > 1e-12 ||
        interval->bias.gyroscope != state.bias.gyroscope || interval->bias.accelerometer != state.bias.accelerometer))
        throw std::invalid_argument("attached IMU interval must end at this frame and use its bias");
    hilti_navigation_.state = std::shared_ptr<const NavState>(new NavState(state));
    hilti_navigation_.from_previous = std::move(interval);
    hilti_navigation_.initialized = true;
    hilti_navigation_.tracking_valid = tracking_valid;
    hilti_navigation_.backend_version = backend_version;
}

SE3 Frame::BodyPose() {
    const auto navigation = HiltiNavigation();
    if (!navigation.initialized || !navigation.state) throw std::logic_error("Hilti metric state unavailable");
    return navigation.state->Pose();
}

SE3 Frame::CameraPose(const Camera& camera) {
    return BodyPose() * camera.pose().inverse();
}

}
