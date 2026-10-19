#pragma once
namespace myslam {
// Frozen H2 live visual information weights. IMU and stored prior are unchanged.
inline double HiltiOfflineCameraInformation(int camera) {
    return camera == 1 ? 0.5 : 1.0;
}
}
