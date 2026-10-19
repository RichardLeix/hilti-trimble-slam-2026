#include "myslam/global_trajectory.h"
#include <cmath>
#include <stdexcept>

namespace myslam {
GlobalTrajectoryOptions LoadGlobalTrajectoryOptions(const std::string& path) {
    cv::FileStorage file(path,cv::FileStorage::READ);
    if(!file.isOpened())throw std::runtime_error("cannot open global trajectory configuration: "+path);
    GlobalTrajectoryOptions options;
    if(!file["minimum_keyframe_growth"].empty()) {
        const double n=double(file["minimum_keyframe_growth"]);
        if(!std::isfinite(n) || n<1 || n>1000000 || std::floor(n)!=n)throw std::invalid_argument("invalid global keyframe growth");
        options.minimum_keyframe_growth=size_t(n);
    }
    if(!file["live_smoothing_seconds"].empty())file["live_smoothing_seconds"]>>options.correction.live_smoothing_seconds;
    GlobalCorrection validate(options.correction);return options;
}
GlobalTrajectory::GlobalTrajectory(std::shared_ptr<LoopClosing> loops,const PoseGraphOptions& graph_options,
    const GlobalTrajectoryOptions& options):loops_(std::move(loops)),graph_(graph_options),options_(options),corrections_(options.correction) {
    if(!loops_ || !options.minimum_keyframe_growth)throw std::invalid_argument("global trajectory requires a loop source and positive refresh interval");
    worker_=std::thread(&GlobalTrajectory::Run,this);
}
GlobalTrajectory::~GlobalTrajectory(){Stop();}
bool GlobalTrajectory::RequestUpdate(int64_t timestamp,bool force) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopping_ || timestamp<latest_timestamp_ns_)return false;
    latest_timestamp_ns_=timestamp;requested_=true;force_=force_ || force;
    ++statistics_.requests;wake_.notify_one();return true;
}
bool GlobalTrajectory::WaitUntilIdle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_.wait_for(lock,timeout,[this]{return !requested_ && !working_;});
}
void GlobalTrajectory::Stop() {
    std::lock_guard<std::mutex> stop_lock(stop_mutex_);
    {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;wake_.notify_one();}
    if(worker_.joinable())worker_.join();
}
GlobalTrajectoryStatistics GlobalTrajectory::Statistics() const {std::lock_guard<std::mutex> lock(mutex_);return statistics_;}
std::shared_ptr<const GlobalTrajectoryUpdate> GlobalTrajectory::LastUpdate() const {std::lock_guard<std::mutex> lock(mutex_);return last_update_;}
void GlobalTrajectory::Run() {
    for(;;) {
        bool force;
        {
            std::unique_lock<std::mutex> lock(mutex_);wake_.wait(lock,[this]{return stopping_ || requested_;});
            if(!requested_ && stopping_)break;
            force=force_;force_=false;requested_=false;working_=true;
        }
        try {
            const auto source=loops_->Statistics();
            const bool refresh=force || source.accepted_constraints>last_loops_ ||
                (source.accepted_constraints && source.keyframes>=last_keyframes_ &&
                 source.keyframes-last_keyframes_>=options_.minimum_keyframe_growth);
            if(refresh && source.keyframes) {
                auto update=std::make_shared<GlobalTrajectoryUpdate>();update->input=loops_->GraphSnapshot();
                update->generation=++generation_;
                {std::lock_guard<std::mutex> lock(mutex_);++statistics_.optimization_jobs;}
                update->result=graph_.Optimize(update->input);
                if(!update->result.success)throw std::runtime_error(update->result.reason);
                int64_t effective_time;
                {std::lock_guard<std::mutex> lock(mutex_);effective_time=latest_timestamp_ns_;}
                if(!corrections_.Publish(update->input,update->result,update->generation,effective_time))
                    throw std::runtime_error("global publication rejected a stale solution");
                update->correction_version=corrections_.Snapshot()->version;
                last_keyframes_=update->input.poses.size();last_loops_=update->input.loops.size();
                std::lock_guard<std::mutex> lock(mutex_);
                last_update_=update;++statistics_.published_versions;statistics_.graph_keyframes=last_keyframes_;
                statistics_.input_loops=last_loops_;statistics_.retained_loops=update->result.retained_loop_edges;
                statistics_.last_error.clear();
            }
        }catch(const std::exception& error) {
            // Output failures retain the previous complete correction and never
            // trigger a local reset, state write, or estimator replay.
            std::lock_guard<std::mutex> lock(mutex_);++statistics_.failed_jobs;statistics_.last_error=error.what();
        }
        {std::lock_guard<std::mutex> lock(mutex_);working_=false;idle_.notify_all();}
    }
}
} // namespace myslam
