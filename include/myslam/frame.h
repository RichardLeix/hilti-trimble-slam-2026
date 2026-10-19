#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/nav_state.h"
#include <array>
#include <stdexcept>

namespace myslam {

// forward declare
struct MapPoint;
struct Feature;

// Hilti state is absent until COMPLETE metric initialization succeeds. Body is
// IMU, pose is T_wb, velocity/gravity are world-frame and biases are IMU-frame.
// A snapshot exposes const states/deltas; cameras have no independently stored pose.
struct HiltiFrameNavigation {
    bool is_hilti = false;
    int64_t timestamp_ns = 0;  // Reference epoch on the IMU clock (cam0).
    std::array<int64_t, 2> raw_camera_timestamps_ns{{0, 0}};
    std::array<int64_t, 2> camera_imu_timestamps_ns{{0, 0}};
    bool initialized = false;
    bool tracking_valid = false;  // False for buffering or IMU prediction only.
    uint64_t backend_version = 0; // Coherent immutable correction used by this snapshot.
    std::shared_ptr<const NavState> state;
    std::shared_ptr<const ImuPreintegration> from_previous;
};

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    SE3 pose_;                       // KITTI Tcw; ignored for Hilti (NavState is authoritative).
    std::mutex pose_mutex_;          // Pose数据锁
    cv::Mat left_img_, right_img_;   // stereo images

    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

   public:  // data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
          const Mat &right);

    // set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        if (hilti_navigation_.is_hilti) throw std::logic_error("Hilti uses BodyPose/CameraPose, not KITTI Pose");
        return pose_;
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        if (hilti_navigation_.is_hilti) throw std::logic_error("Hilti pose must be written through its NavState");
        pose_ = pose;
    }

    void ConfigureHiltiEpoch(int64_t timestamp_ns,
                             const std::array<int64_t, 2>& raw_camera_timestamps_ns,
                             const std::array<int64_t, 2>& camera_imu_timestamps_ns);
    HiltiFrameNavigation HiltiNavigation();
    void SetHiltiNavigation(const NavState& state, bool tracking_valid,
                            std::shared_ptr<const ImuPreintegration> from_previous = nullptr,
                            uint64_t backend_version = 0);
    SE3 BodyPose();  // T_wb; throws until initialized.
    SE3 CameraPose(const Camera& camera);  // T_wc = T_wb * T_cb^-1, at reference epoch.

    /// 设置关键帧并分配并键帧id
    void SetKeyFrame();

    /// 工厂构建模式，分配id 
    static std::shared_ptr<Frame> CreateFrame();

   private:
    HiltiFrameNavigation hilti_navigation_;
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
