#ifndef MYSLAM_HILTI_SUBMISSION_H
#define MYSLAM_HILTI_SUBMISSION_H

#include "myslam/pose_graph.h"
#include <iosfwd>

namespace myslam {
struct TumPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t timestamp_ns = 0; // Recorded cam0 header time, never the shifted IMU time.
    Vec3 translation = Vec3::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity(); // Written xyzw.
};
using TumTrajectory = HiltiAlignedVector<TumPose>;
struct HiltiBodySample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int64_t timestamp_imu_ns = 0;
    bool initialized = false;
    SE3 body_pose_local; // T_wb at cam0 exposure on the calibrated IMU clock.
};
using HiltiBodyTrajectory = HiltiAlignedVector<HiltiBodySample>;

// T_cb = camera.pose() maps body/IMU -> cam0. T_gc0 = T_gb * T_cb^-1.
// T_gb is evaluated at cam0 exposure, but the returned stamp is RAW cam0 time.
TumPose BodyToCam0Pose(const SE3& body_to_world, const SE3& camera_from_body,
    int64_t raw_cam0_timestamp_ns);
int64_t ParseTumTimestamp(const std::string& seconds);
std::string FormatTumTimestamp(int64_t timestamp_ns);
std::string FormatTumLine(const TumPose& pose);
void ValidateTumTrajectory(const TumTrajectory& trajectory);
void WriteTumTrajectory(std::ostream& stream, const TumTrajectory& trajectory);
TumTrajectory ReadTumTrajectory(std::istream& stream);
std::string HiltiSubmissionFilename(const std::string& floor, const std::string& date, unsigned run);
std::string HiltiSubmissionFilename(const std::string& sequence_name);

// Read only the existing full precision inspection format. Buffering rows with
// seven explicitly missing pose fields stay absent; all other invalid rows fail.
HiltiBodyTrajectory ReadHiltiBodyCsv(std::istream& stream);
HiltiAlignedVector<PoseGraphPose> ReadHiltiKeyframeCsv(std::istream& stream);
struct HiltiSubmission {
    TumTrajectory local_cam0, global_cam0;
    size_t omitted_uninitialized = 0, exact_archived_keyframes = 0;
};
// Final offline export only. Use optimized local/global archive poses exactly at
// keyframes, and interpolate C_i=T_gb_i*T_wb_i^-1 between them for other genuine
// local VIO poses. Clamp to the endpoint correction outside archive timestamps.
// No smoothing of graph poses, extrapolation of missing states, or SLAM writes.
HiltiSubmission BuildHiltiSubmission(const std::vector<int64_t>& raw_cam0_times,
    const HiltiBodyTrajectory& body, const HiltiAlignedVector<PoseGraphPose>& local_archive,
    const HiltiAlignedVector<PoseGraphPose>& global_archive, const SE3& camera_from_body,
    const ImuCalibration& calibration);

struct MissingPoseInterval {
    int64_t begin_ns = 0, end_ns = 0;
    size_t poses = 0;
};
struct TrajectoryCoverage {
    size_t expected_poses = 0, predicted_poses = 0, missing_poses = 0;
    size_t eligible_expected = 0, eligible_predicted = 0;
    double coverage_percent = 0, eligible_coverage_percent = 0;
    int64_t first_ns = 0, last_ns = 0, largest_gap_ns = 0;
    double duration_s = 0, maximum_quaternion_norm_error = 0;
    std::vector<MissingPoseInterval> missing_intervals;
};
// Camera-grid coverage, separate from the ground-truth association coverage.
// Exported timestamps must be an exact subset of the recorded camera times.
TrajectoryCoverage CheckTrajectoryCoverage(const std::vector<int64_t>& expected_cam0_times,
    const TumTrajectory& predicted, int64_t ignore_startup_ns = 5000000000LL);
} // namespace myslam
#endif
