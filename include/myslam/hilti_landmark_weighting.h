#pragma once
#include "myslam/hilti_backend.h"
#include <memory>
namespace myslam {
struct LandmarkWeightQuality {
    double residual_sigma=0,relative_depth_sigma=0,parallax_sigma=0,age_s=0,window_span_s=1;
    size_t observations=0;
};
double HiltiLandmarkQualityWeight(const LandmarkWeightQuality&,bool combined);
LandmarkWeightQuality HiltiLandmarkWeightGeometry(const HiltiBackendJob&,const HiltiLandmark&,const HiltiBackendOptions&);
class HiltiLandmarkWeightWindow {
 public:
    explicit HiltiLandmarkWeightWindow(const HiltiBackendJob&);
    ~HiltiLandmarkWeightWindow();
    void Ready(const HiltiBackendJob&,const HiltiBackendOptions&);
    void Finish(bool accepted);
 private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
double HiltiLandmarkLiveInformation(const HiltiBackendJob&,const HiltiLandmarkId&,bool connected_source=false);
}
