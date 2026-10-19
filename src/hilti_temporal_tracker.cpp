#include "myslam/hilti_temporal_tracker.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace myslam {

cv::Mat DecodeHiltiJpeg(const ImagePacket& image) {
    std::string format = image.format;
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if ((format.find("jpeg") == std::string::npos &&
         format.find("jpg") == std::string::npos) ||
        image.data.size() < 4 || image.data[0] != 0xff || image.data[1] != 0xd8) {
        throw std::runtime_error("expected a compressed JPEG image");
    }
    const cv::Mat decoded = cv::imdecode(image.data, cv::IMREAD_GRAYSCALE);
    if (decoded.empty() || decoded.type() != CV_8UC1) {
        throw std::runtime_error("JPEG decoding failed");
    }
    return decoded;
}

HiltiTemporalTracker::HiltiTemporalTracker(
    SensorStream stream, std::shared_ptr<const EucmCamera> camera,
    const HiltiTrackingOptions& options)
    : stream_(stream), camera_(std::move(camera)), options_(options) {
    if ((stream != SensorStream::Cam0 && stream != SensorStream::Cam1) || !camera_) {
        throw std::invalid_argument("temporal tracker requires one calibrated camera");
    }
    if (options.max_features <= 0 || !std::isfinite(options.quality_level) ||
        options.quality_level <= 0 || options.quality_level > 1 ||
        !std::isfinite(options.min_distance) || options.min_distance <= 0 ||
        options.lk_window_size < 3 || options.lk_window_size % 2 == 0 ||
        options.pyramid_levels < 0 ||
        !std::isfinite(options.max_forward_backward_error) ||
        options.max_forward_backward_error <= 0) {
        throw std::invalid_argument("invalid temporal tracking options");
    }
    // IsValidPixel already applies the official mask (zero means valid there).
    // Rasterize once, using the same API to recheck subpixel LK endpoints below.
    valid_mask_ = cv::Mat(camera_->Height(), camera_->Width(), CV_8UC1, cv::Scalar(0));
    for (int v = 0; v < valid_mask_.rows; ++v) {
        auto* row = valid_mask_.ptr<uint8_t>(v);
        for (int u = 0; u < valid_mask_.cols; ++u) {
            if (camera_->IsValidPixel(Vec2(u, v))) row[u] = 255;
        }
    }
}

bool HiltiTemporalTracker::Process(const SensorPacket& packet,
                                   HiltiTrackingResult& result) {
    if (packet.stream != stream_) return false;
    if (!previous_image_.empty() && packet.timestamp_ns <= previous_timestamp_ns_) {
        throw std::invalid_argument("camera image timestamps must be strictly increasing");
    }
    const cv::Mat image = DecodeHiltiJpeg(packet.image);
    if (image.size() != valid_mask_.size()) {
        throw std::runtime_error("JPEG resolution differs from camera calibration");
    }

    HiltiTrackingResult current;
    current.stream = stream_;
    current.has_previous = !previous_image_.empty();
    current.previous_timestamp_ns = previous_timestamp_ns_;
    current.timestamp_ns = packet.timestamp_ns;
    current.source_features = previous_features_.size();
    if (!previous_features_.empty()) {
        std::vector<cv::Point2f> source, forward;
        for (const auto& feature : previous_features_) source.push_back(feature.pixel);
        std::vector<uint8_t> status;
        std::vector<float> error;
        const cv::Size window(options_.lk_window_size, options_.lk_window_size);
        cv::calcOpticalFlowPyrLK(previous_image_, image, source, forward, status,
                                error, window, options_.pyramid_levels);

        const auto forward_error = error;
        std::vector<size_t> candidates;
        std::vector<cv::Point2f> valid_forward, backward;
        for (size_t i = 0; i < source.size(); ++i) {
            if (!status[i]) {
                ++current.rejected_flow;
            } else if (!camera_->IsValidPixel(Vec2(forward[i].x, forward[i].y))) {
                ++current.rejected_mask_domain;
            } else {
                candidates.push_back(i);
                valid_forward.push_back(forward[i]);
            }
        }
        if (!candidates.empty()) {
            cv::calcOpticalFlowPyrLK(image, previous_image_, valid_forward,
                                    backward, status, error, window,
                                    options_.pyramid_levels);
            for (size_t j = 0; j < candidates.size(); ++j) {
                const size_t i = candidates[j];
                const double distance = cv::norm(backward[j] - source[i]);
                if (!status[j] || !std::isfinite(distance) ||
                    distance > options_.max_forward_backward_error) {
                    ++current.rejected_flow;
                    continue;
                }
                TemporalFeatureTrack track = previous_features_[i];
                track.pixel = forward[i];
                track.forward_lk_error = forward_error[i];
                track.backward_lk_error = error[j];
                track.fb_error = distance;
                ++track.age;
                current.features.push_back(track);
            }
        }
    }
    current.successfully_tracked = current.features.size();
    cv::Mat corner_response;
    const bool collect_diagnostics =
        false;
    if (collect_diagnostics) cv::cornerMinEigenVal(image, corner_response, 3, 3);

    // Replenish corners without assigning new IDs to surviving tracks.
    uint64_t next_id = next_id_;
    const int remaining = options_.max_features - int(current.features.size());
    if (remaining > 0) {
        cv::Mat detection_mask = valid_mask_.clone();
        for (const auto& track : current.features) {
            cv::circle(detection_mask, track.pixel, int(std::ceil(options_.min_distance)),
                       cv::Scalar(0), -1);
        }
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(image, corners, remaining, options_.quality_level,
                                options_.min_distance, detection_mask);
        for (const auto& pixel : corners) {
            if (!camera_->IsValidPixel(Vec2(pixel.x, pixel.y))) continue;
            TemporalFeatureTrack track;
            track.id = next_id++;
            track.pixel = pixel;
            current.features.push_back(track);
            ++current.detected_features;
        }
    }

    if (collect_diagnostics) {
        for (auto& track : current.features) {
            const int x = std::max(0, std::min(image.cols - 1, cvRound(track.pixel.x)));
            const int y = std::max(0, std::min(image.rows - 1, cvRound(track.pixel.y)));
            track.corner_response = corner_response.at<float>(y, x);
        }
    }

    // Commit state only after successful decoding, tracking, and detection.
    result = current;
    previous_features_ = std::move(current.features);
    previous_image_ = image;
    previous_timestamp_ns_ = packet.timestamp_ns;
    next_id_ = next_id;
    return true;
}

}  // namespace myslam
