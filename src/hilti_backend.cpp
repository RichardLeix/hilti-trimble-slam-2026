#include "myslam/hilti_backend.h"
#include "myslam/hilti_landmark_quarantine.h"
#include <cmath>
#include <stdexcept>

namespace myslam {
HiltiBackend::HiltiBackend(const HiltiBackendOptions& options) : options_(options) {
    if(!std::isfinite(options.marginalization_rank_tolerance) || options.marginalization_rank_tolerance<=0 ||
        options.marginalization_rank_tolerance>=1)throw std::invalid_argument("invalid marginalization rank tolerance");
    if (options.window_size < 3 || options.window_size > 15 || !options.keyframe_stride ||
        options.min_landmarks < 6 || options.max_landmarks < options.min_landmarks || !options.max_iterations)
        throw std::invalid_argument("invalid Hilti backend window/support limits");
    for (double x : {options.bearing_sigma_rad, options.visual_huber_sigma, options.max_initial_bearing_rad, options.max_landmark_bearing_rad,
        options.velocity_prior_sigma, options.gyro_prior_sigma, options.accel_prior_sigma,
        options.max_pose_update_m, options.max_rotation_update_rad, options.max_velocity_update,
        options.max_gyro_update, options.max_accel_update})
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("invalid Hilti backend noise/update limit");
    if(options_.experimental_landmark_quarantine_B)
        quarantine_.reset(new HiltiLandmarkQuarantine(options_.landmark_quarantine_log_path));
    worker_ = std::thread(&HiltiBackend::Loop, this);
}
HiltiBackend::~HiltiBackend() { Stop(); }
bool HiltiBackend::Submit(const HiltiBackendJob& job) {
    const auto snapshot = std::make_shared<const HiltiBackendJob>(job);
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || working_ || pending_ || result_ || !job.version || job.version <= last_version_) return false;
    last_version_ = job.version; pending_ = snapshot; wake_.notify_one(); return true;
}
std::shared_ptr<const HiltiBackendResult> HiltiBackend::TakeResult() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = result_; result_.reset(); return result;
}
bool HiltiBackend::Busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return working_ || bool(pending_) || bool(result_);
}
bool HiltiBackend::WaitUntilIdle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_.wait_for(lock, timeout, [this] { return !working_ && !pending_; });
}
void HiltiBackend::Stop() {
    std::lock_guard<std::mutex> stop_lock(stop_mutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true; wake_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
}
void HiltiBackend::Loop() {
    for (;;) {
        std::shared_ptr<const HiltiBackendJob> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || bool(pending_); });
            if (!pending_ && stopping_) break;
            job = pending_; pending_.reset(); working_ = true;
        }
        HiltiBackendResult result;
        try { result = OptimizeHiltiWindow(*job, options_, quarantine_.get()); }
        catch (const std::exception& error) {
            result.version = job->version; result.base_version = job->base_version;
            result.diagnostics.reason = std::string("optimization exception: ") + error.what();
            result.diagnostics.window_size = job->keyframes.size();
            
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result_ = std::make_shared<const HiltiBackendResult>(std::move(result));
            working_ = false; idle_.notify_all();
        }
    }
    std::lock_guard<std::mutex> lock(mutex_); idle_.notify_all();
}
} // namespace myslam
