#ifndef MYSLAM_GLOBAL_TRAJECTORY_H
#define MYSLAM_GLOBAL_TRAJECTORY_H

#include "myslam/global_correction.h"

namespace myslam {
struct GlobalTrajectoryOptions {
    size_t minimum_keyframe_growth = 30;
    GlobalCorrectionOptions correction;
};
GlobalTrajectoryOptions LoadGlobalTrajectoryOptions(const std::string& path);
struct GlobalTrajectoryUpdate {
    uint64_t generation = 0, correction_version = 0;
    PoseGraphInput input;
    PoseGraphResult result;
};
struct GlobalTrajectoryStatistics {
    size_t requests = 0, optimization_jobs = 0, published_versions = 0, failed_jobs = 0;
    size_t graph_keyframes = 0, input_loops = 0, retained_loops = 0;
    std::string last_error;
};
// Background Hilti pose graph/output worker. Owns only pose snapshots and its
// correction publisher. It has no access to writable local estimator objects.
// Notifications coalesce; new loops and archive growth trigger graph refreshes.
class GlobalTrajectory {
   public:
    GlobalTrajectory(std::shared_ptr<LoopClosing> loops,
        const PoseGraphOptions& graph_options = PoseGraphOptions(),
        const GlobalTrajectoryOptions& options = GlobalTrajectoryOptions());
    ~GlobalTrajectory();
    GlobalTrajectory(const GlobalTrajectory&) = delete;
    GlobalTrajectory& operator=(const GlobalTrajectory&) = delete;
    bool RequestUpdate(int64_t local_timestamp_ns, bool force = false);
    bool WaitUntilIdle(std::chrono::milliseconds timeout);
    void Stop(); // Drains requested work; does not stop LoopClosing or VIO.
    const GlobalCorrection& Corrections() const { return corrections_; }
    const LoopClosing* LoopSource() const { return loops_.get(); }
    GlobalTrajectoryStatistics Statistics() const;
    std::shared_ptr<const GlobalTrajectoryUpdate> LastUpdate() const;
   private:
    void Run();
    std::shared_ptr<LoopClosing> loops_;
    PoseGraph graph_;
    GlobalTrajectoryOptions options_;
    GlobalCorrection corrections_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_, idle_;
    bool stopping_ = false, requested_ = false, force_ = false, working_ = false;
    int64_t latest_timestamp_ns_ = 0;
    uint64_t generation_ = 0;
    size_t last_keyframes_ = 0, last_loops_ = 0;
    GlobalTrajectoryStatistics statistics_;
    std::shared_ptr<const GlobalTrajectoryUpdate> last_update_;
    std::thread worker_;
};
} // namespace myslam
#endif
