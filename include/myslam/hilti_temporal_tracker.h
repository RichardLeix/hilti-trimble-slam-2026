#ifndef MYSLAM_HILTI_TEMPORAL_TRACKER_H
#define MYSLAM_HILTI_TEMPORAL_TRACKER_H

#include "myslam/eucm_camera.h"
#include <cmath>
#include "myslam/sensor_packet.h"

namespace myslam {

// Decode the original JPEG at full resolution into an owned CV_8UC1 image.
// Throws on unsupported format or invalid compressed data.
cv::Mat DecodeHiltiJpeg(const ImagePacket& image);

struct HiltiTrackingOptions {
    int max_features = 300;
    double quality_level = 0.01;
    double min_distance = 15;
    int lk_window_size = 21;
    int pyramid_levels = 3;
    double max_forward_backward_error = 1;  // Pixels, not a geometric constraint.
};

struct TemporalFeatureTrack {
    uint64_t id = 0;  // Local to one camera's tracker; never reused.
    cv::Point2f pixel;
    size_t age = 1;  // Number of consecutive images containing this track.
    // Read-only tracker diagnostics. They do not participate in acceptance or
    // feature propagation. New detections have no LK errors yet.
    double forward_lk_error = NAN, backward_lk_error = NAN, fb_error = NAN;
    double corner_response = NAN;
};

struct HiltiTrackingResult {
    SensorStream stream = SensorStream::Cam0;
    bool has_previous = false;
    int64_t previous_timestamp_ns = 0;
    int64_t timestamp_ns = 0;  // Raw ROS header time, without calibration offset.
    size_t source_features = 0;
    size_t successfully_tracked = 0;
    size_t rejected_mask_domain = 0;  // Forward LK endpoints outside valid pixels.
    size_t rejected_flow = 0;         // LK failure or forward/backward inconsistency.
    size_t detected_features = 0;     // New corners on the current image, after LK.
    // Current active tracks: surviving IDs/ages followed by newly detected IDs.
    std::vector<TemporalFeatureTrack> features;
};

// A standalone, single-camera visual inspection path. No Frame/SLAM state.
// Use separate instances for cam0/cam1; other camera and IMU packets are ignored.
class HiltiTemporalTracker {
   public:
    HiltiTemporalTracker(SensorStream stream,
                         std::shared_ptr<const EucmCamera> camera,
                         const HiltiTrackingOptions& options = HiltiTrackingOptions());

    // True for a processed image from this camera (including the first image).
    // False for another stream, leaving result and tracking state unchanged.
    // Throws on duplicate/backward time, invalid JPEG, or resolution mismatch.
    // Failed images leave state unchanged; the caller should stop on failure.
    bool Process(const SensorPacket& packet, HiltiTrackingResult& result);

   private:
    SensorStream stream_;
    std::shared_ptr<const EucmCamera> camera_;
    HiltiTrackingOptions options_;
    cv::Mat valid_mask_;  // 255 valid; combines official mask and EUCM domain.
    cv::Mat previous_image_;
    int64_t previous_timestamp_ns_ = 0;
    uint64_t next_id_ = 0;
    std::vector<TemporalFeatureTrack> previous_features_;
};

}  // namespace myslam

#endif  // MYSLAM_HILTI_TEMPORAL_TRACKER_H
